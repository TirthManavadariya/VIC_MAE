"""
models/extended_vicmae.py
--------------------------
Extended ViC-MAE: wraps the base ViCMAE model with the SiamMAE-style
CrossFrameDecoder and provides:

  - forward_cross_recon()     -- cross-frame masked reconstruction loss
  - compute_preservation_loss() -- cosine-distance feature preservation

This module is the central interface for the Stage-1 / Stage-2 training
protocol described in the ViC-MAE temporal extension specification.
"""

from __future__ import annotations

import sys
import os
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow imports from parent directory and upstream ViC-MAE
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_PARENT)
for p in [_PARENT, os.path.join(_PARENT, "ViC-MAE"), os.path.join(_ROOT, "ViC-MAE")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from models_vicmae import ViCMAE
from models.cross_decoder import CrossFrameDecoder


# ---------------------------------------------------------------------------
# LoRA helpers (lightweight, no external dependency)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer wrapping an existing nn.Linear.

    Adds trainable low-rank matrices A (in -> r) and B (r -> out) whose
    product alpha/r * B @ A is added to the frozen base weight output.

    Args:
        linear : Frozen nn.Linear module to wrap.
        r      : LoRA rank.
        alpha  : LoRA scaling factor. Effective scale = alpha / r.
    """

    def __init__(self, linear: nn.Linear, r: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        self.linear = linear
        self.r      = r
        self.scale  = alpha / r

        # Freeze base weight
        for p in self.linear.parameters():
            p.requires_grad_(False)

        in_features  = linear.in_features
        out_features = linear.out_features
        dev = linear.weight.device
        dtype = linear.weight.dtype
        self.lora_A  = nn.Parameter(torch.randn(r, in_features, device=dev, dtype=dtype) * 0.01)
        self.lora_B  = nn.Parameter(torch.zeros(out_features, r, device=dev, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out  = self.linear(x)
        lora_out  = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out


def inject_lora(module: nn.Module, r: int = 16, alpha: float = 32.0) -> nn.Module:
    """
    Replace every ``nn.Linear`` inside ``module`` with a ``LoRALinear``,
    freezing the base weights and adding trainable low-rank adapters.

    Returns the modified module in-place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            inject_lora(child, r=r, alpha=alpha)
    return module


# ---------------------------------------------------------------------------
# ExtendedViCMAE
# ---------------------------------------------------------------------------

class ExtendedViCMAE(nn.Module):
    """
    Temporal extension of ViC-MAE combining:

      1. Base ViCMAE losses   : L_mae  (reconstruction) + L_nce (contrastive).
      2. Cross-frame loss     : L_cross (SiamMAE-style masked reconstruction of
                                Frame-j conditioned on Frame-i tokens + Delta-t).
      3. Preservation loss    : L_pres (cosine distance between active encoder
                                patch tokens and a frozen teacher's tokens on
                                image-only batches).

    Training stages
    ---------------
    Stage 1  -- decoder warm-up:
        call ``freeze_backbone()``; only cross_decoder trains (lr=1e-4).
    Stage 2  -- joint fine-tuning:
        call ``unfreeze_top_blocks(n=4)`` OR ``inject_lora_adapters()``;
        encoder uses smaller lr than cross_decoder.

    Args:
        vicmae_cfg      : kwargs forwarded to ViCMAE.__init__.
        decoder_dim     : Inner dimension of CrossFrameDecoder.
        decoder_depth   : Number of CrossAttentionBlocks.
        decoder_heads   : Attention heads per CrossAttentionBlock.
        img_size        : Image spatial size (square).
        cross_mask_ratio: Fraction of Frame-j patches to mask (default 0.95).
    """

    def __init__(
        self,
        vicmae_cfg: Optional[Dict] = None,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        img_size: int = 224,
        cross_mask_ratio: float = 0.95,
    ) -> None:
        super().__init__()

        # --- Base ViC-MAE ---
        if isinstance(vicmae_cfg, nn.Module):
            self.vicmae = vicmae_cfg
        elif isinstance(vicmae_cfg, dict):
            self.vicmae = ViCMAE(**vicmae_cfg)
        elif vicmae_cfg is None:
            self.vicmae = ViCMAE()
        else:
            self.vicmae = ViCMAE()

        embed_dim  = self.vicmae.patch_embed.proj.out_channels  # encoder dim
        patch_size = self.vicmae.patch_embed.patch_size[0]
        in_chans   = self.vicmae.patch_embed.proj.in_channels

        # --- Cross-frame decoder ---
        self.cross_decoder = CrossFrameDecoder(
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            patch_size=patch_size,
            in_chans=in_chans,
            img_size=img_size,
        )

        self.cross_mask_ratio = cross_mask_ratio
        self._patch_size      = patch_size
        self._in_chans        = in_chans

    # ------------------------------------------------------------------
    # Stage control helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Stage 1: freeze the entire ViCMAE encoder + decoder."""
        self.vicmae.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Unfreeze entire ViCMAE (used before Stage 2 full fine-tune)."""
        self.vicmae.requires_grad_(True)

    def unfreeze_top_blocks(self, n: int = 4) -> None:
        """
        Stage 2 option A: keep backbone mostly frozen; unfreeze only the
        top-``n`` Transformer blocks and the final norm layer.
        """
        self.vicmae.requires_grad_(False)
        total = len(self.vicmae.blocks)
        for blk in self.vicmae.blocks[total - n:]:
            blk.requires_grad_(True)
        self.vicmae.norm.requires_grad_(True)

    def inject_lora_adapters(self, r: int = 16, alpha: float = 32.0) -> None:
        """
        Stage 2 option B: inject LoRA adapters into all Linear layers of the
        ViCMAE encoder; freeze base weights, only adapters are trainable.
        """
        device = next(self.vicmae.parameters()).device
        self.vicmae.requires_grad_(False)
        inject_lora(self.vicmae.blocks, r=r, alpha=alpha)
        self.to(device)

    def trainable_parameters(self) -> list:
        """Return list of all parameters with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Patchification helpers
    # ------------------------------------------------------------------

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Decompose images into flattened patches (normalised per-patch).

        Args:
            imgs : (B, C, H, W)

        Returns:
            patches : (B, L, patch_size^2 * C), zero-mean unit-var per patch.
        """
        p = self._patch_size
        B, C, H, W = imgs.shape
        assert H % p == 0 and W % p == 0
        h, w = H // p, W // p

        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(B, h * w, p * p * C)

        # Per-patch normalisation (zero mean, unit variance)
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True)
        x    = (x - mean) / (var + 1e-6).sqrt()
        return x

    def _generate_mask(
        self,
        B: int,
        L: int,
        mask_ratio: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random binary mask.

        Returns:
            mask        : (B, L) float tensor; 1 = masked, 0 = visible.
            ids_restore : (B, L) permutation to unshuffle.
        """
        len_keep   = max(1, int(L * (1.0 - mask_ratio)))
        noise      = torch.rand(B, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask          = torch.ones(B, L, device=device)
        mask[:, :len_keep] = 0.0
        mask          = torch.gather(mask, dim=1, index=ids_restore)
        return mask, ids_restore

    # ------------------------------------------------------------------
    # Cross-frame reconstruction
    # ------------------------------------------------------------------

    def forward_cross_recon(
        self,
        ctx_tokens: torch.Tensor,
        target_imgs: torch.Tensor,
        delta_t: torch.Tensor,
        mask_ratio: float = 0.95,
    ) -> torch.Tensor:
        """
        SiamMAE-style cross-frame reconstruction loss.

        Procedure:
          1. Patchify and per-patch normalise target_imgs.
          2. Generate heavy mask (mask_ratio ~ 0.95) over target patches.
          3. Run CrossFrameDecoder: queries <- mask_token + pos + Delta-t,
             K/V <- ctx_tokens (Frame-i encoder output).
          4. Compute MSE loss only on masked patch indices.

        Args:
            ctx_tokens  : (B, N_ctx, embed_dim) -- Frame-i encoder tokens
                          (output of vicmae.forward_encoder, incl. CLS).
            target_imgs : (B, C, H, W) -- raw Frame-j images (not patchified).
            delta_t     : (B, 1) -- normalised time gap |t_j - t_i|.
            mask_ratio  : Fraction of Frame-j patches to mask (default 0.95).

        Returns:
            loss_cross : scalar MSE loss on masked patches.
        """
        B  = target_imgs.shape[0]
        L  = self.cross_decoder.num_patches

        # 1. Ground-truth patches (normalised)
        target = self._patchify(target_imgs)          # (B, L, p*p*C)

        # 2. Generate mask
        mask, _ = self._generate_mask(B, L, mask_ratio, target_imgs.device)

        # 3. Decode
        pred = self.cross_decoder(ctx_tokens, mask, delta_t)  # (B, L, p*p*C)

        # 4. MSE on masked patches only
        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)  # (B, L)
        loss_cross     = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)

        return loss_cross, pred, mask

    # ------------------------------------------------------------------
    # Feature preservation loss
    # ------------------------------------------------------------------

    @staticmethod
    def compute_preservation_loss(
        current_tokens: torch.Tensor,
        frozen_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine-distance preservation loss between active and frozen encoder tokens.
        """
        if current_tokens.shape[1] == frozen_tokens.shape[1] + 1:
            cur = current_tokens[:, 1:, :]
            frz = frozen_tokens
        elif frozen_tokens.shape[1] == current_tokens.shape[1] + 1:
            cur = current_tokens
            frz = frozen_tokens[:, 1:, :]
        else:
            cur = current_tokens
            frz = frozen_tokens

        # Normalise along feature dimension
        cur_norm = F.normalize(cur, dim=-1)
        frz_norm = F.normalize(frz, dim=-1)

        # Cosine similarity per patch, averaged over batch and patches
        cos_sim   = (cur_norm * frz_norm).sum(dim=-1)   # (B, L)
        loss_pres = 1.0 - cos_sim.mean()
        return loss_pres

    preservation_loss = compute_preservation_loss  # alias

    # ------------------------------------------------------------------
    # Convenience: get encoder tokens without masking
    # ------------------------------------------------------------------

    def encode_full(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run ViCMAE encoder on the FULL (unmasked) image."""
        tokens, _, _ = self.vicmae.forward_encoder(imgs, mask_ratio=0.0)
        return tokens

    def encode_frame(self, imgs: torch.Tensor, mask_ratio: float = 0.0):
        """Encode a frame with the given mask ratio."""
        return self.vicmae.forward_encoder(imgs, mask_ratio=mask_ratio)

    def forward_mae(self, imgs: torch.Tensor, mask_ratio: float = 0.5):
        """Forward pass for standard MAE reconstruction."""
        latent, mask, ids_restore = self.vicmae.forward_encoder(imgs, mask_ratio)
        pred = self.vicmae.forward_decoder(latent, ids_restore)
        loss = self.vicmae.mse_loss(imgs, pred, mask)
        return loss, pred, mask

    def forward(self, imgs):
        if isinstance(imgs, dict):
            if imgs.get("is_video", False):
                frame_i = imgs["frame_i"]
                frame_j = imgs["frame_j"]
                delta_t = imgs["delta_t"]
                mask_ratio = imgs.get("mask_ratio", 0.5)
                lambda_nce = imgs.get("lambda_nce", 0.1)
                lambda_cross = imgs.get("lambda_cross", 1.0)

                l_mae_i, _, _ = self.forward_mae(frame_i, mask_ratio)
                l_mae_j, _, _ = self.forward_mae(frame_j, mask_ratio)

                enc_i, _, _ = self.vicmae.forward_encoder(frame_i, mask_ratio)
                enc_j, _, _ = self.vicmae.forward_encoder(frame_j, mask_ratio)
                pool_i = self.vicmae.projector(enc_i[:, 1:, :].mean(dim=1))
                pool_j = self.vicmae.projector(enc_j[:, 1:, :].mean(dim=1))
                z = torch.cat([pool_i, pool_j], dim=0)
                l_nce = self.vicmae.info_nce_loss(z, self.vicmae.temperature)

                loss_cross, _, _ = self.forward_cross_recon(
                    enc_i[:, 1:, :], frame_j, delta_t, mask_ratio=self.cross_mask_ratio
                )

                total = l_mae_i + l_mae_j + lambda_nce * l_nce + lambda_cross * loss_cross
                return {
                    "loss": total,
                    "loss_mae_i": l_mae_i,
                    "loss_mae_j": l_mae_j,
                    "loss_nce": l_nce,
                    "loss_cross": loss_cross,
                }
            else:
                image = imgs["image"]
                mask_ratio = imgs.get("mask_ratio", 0.5)
                lambda_pres = imgs.get("lambda_pres", 0.5)
                frozen_tokens = imgs["frozen_tokens"]

                l_mae, _, _ = self.forward_mae(image, mask_ratio)
                act_tokens = self.encode_full(image)[:, 1:, :]
                l_pres = self.compute_preservation_loss(act_tokens, frozen_tokens)

                total = l_mae + lambda_pres * l_pres
                return {
                    "loss": total,
                    "loss_mae": l_mae,
                    "loss_pres": l_pres,
                }

        return self.vicmae(imgs)
