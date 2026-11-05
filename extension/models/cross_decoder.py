"""
models/cross_decoder.py
-----------------------
SiamMAE-style Cross-Frame Decoder for ViC-MAE temporal extension.

Architecture overview:
  - CrossAttentionBlock  : single Transformer block where queries come from
                           target (masked) patches and keys/values come from
                           the context-frame encoder tokens.
  - CrossFrameDecoder    : 4-block cross-attention decoder that reconstructs
                           heavily masked Frame-j patches conditioned on
                           Frame-i encoder tokens + a continuous Delta-t embedding.

References:
  - SiamMAE  : https://arxiv.org/abs/2212.05543
  - MAE       : https://arxiv.org/abs/2111.06377
  - ViC-MAE   : internal codebase (models_vicmae.py)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Return (grid_size*grid_size, embed_dim) 2-D sin-cos positional embedding."""
    assert embed_dim % 2 == 0, "embed_dim must be even."
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)          # (grid_size, grid_size) each
    grid = np.stack([grid_h.ravel(), grid_w.ravel()], axis=0)  # (2, L)

    # Use embed_dim // 4 frequencies per axis (4 embeddings total: sin/cos for h and w)
    half = embed_dim // 2
    omega = np.arange(half // 2, dtype=np.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)                        # (half/2,)

    pos_h = np.einsum("m,d->md", grid[0], omega)          # (L, half/2)
    pos_w = np.einsum("m,d->md", grid[1], omega)          # (L, half/2)
    emb = np.concatenate([np.sin(pos_h), np.cos(pos_h),
                          np.sin(pos_w), np.cos(pos_w)], axis=1)  # (L, embed_dim)
    return torch.from_numpy(emb).float()


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Transformer block where:
      - queries   = target (masked) patch tokens.
      - keys/vals = context-frame encoder output tokens.

    Architecture:
      1. Cross-Attention  (Q from target, K/V from context).
      2. Self-Attention   (standard, over target tokens only).
      3. Feed-Forward MLP.

    All sub-layers use pre-norm convention (LayerNorm before each sublayer).

    Args:
        dim        : Token embedding dimension (decoder_dim).
        num_heads  : Number of attention heads.
        mlp_ratio  : Hidden-to-dim ratio for the MLP.
        dropout    : Dropout probability for attention weights.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim={dim} must be divisible by num_heads={num_heads}."
        )

        # --- Cross-attention ---
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- Self-attention ---
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- MLP ---
        hidden_dim = int(dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tgt : (B, N_tgt, dim) -- query tokens (masked target patches).
            ctx : (B, N_ctx, dim) -- key/value tokens (context-frame encoder output).

        Returns:
            tgt : (B, N_tgt, dim) -- updated target tokens.
        """
        # 1. Cross-attention
        q  = self.norm_q(tgt)
        kv = self.norm_kv(ctx)
        ca_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        tgt = tgt + self.drop(ca_out)

        # 2. Self-attention
        sa_in = self.norm_sa(tgt)
        sa_out, _ = self.self_attn(query=sa_in, key=sa_in, value=sa_in)
        tgt = tgt + self.drop(sa_out)

        # 3. MLP
        tgt = tgt + self.drop(self.mlp(self.norm_mlp(tgt)))
        return tgt


# ---------------------------------------------------------------------------
# CrossFrameDecoder
# ---------------------------------------------------------------------------

class CrossFrameDecoder(nn.Module):
    """
    SiamMAE-style cross-frame reconstruction decoder.

    Given:
      - ctx_tokens : encoder output for Frame-i  (B, N_ctx, embed_dim)
                     (all encoder tokens, optionally including CLS).
      - mask       : binary mask (B, L), 1 = masked, 0 = visible.
      - delta_t    : scalar time gap per sample (B, 1), normalised to [0, 1].

    Returns:
      - pred : (B, L, patch_size^2 * in_chans) -- predictions for ALL patches.
               Callers compute MSE loss only on masked indices.

    Internal flow:
      1. Project ctx_tokens: embed_dim -> decoder_dim.
      2. Build query tensor: mask_token (at masked positions) + pos_embed + Delta-t.
      3. Run ``depth`` CrossAttentionBlocks (Q=queries, K/V=ctx).
      4. LayerNorm + linear head -> pixel predictions.

    Args:
        embed_dim    : Encoder output dimension (ViC-MAE backbone embed_dim).
        decoder_dim  : Internal decoder token dimension.
        depth        : Number of CrossAttentionBlocks.
        num_heads    : Attention heads per block.
        patch_size   : Spatial patch size in pixels.
        in_chans     : Number of image channels.
        img_size     : Input image spatial size (square assumed).
        mlp_ratio    : MLP ratio inside each CrossAttentionBlock.
        dropout      : Dropout for attention layers.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        decoder_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        patch_size: int = 16,
        in_chans: int = 3,
        img_size: int = 224,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.decoder_dim  = decoder_dim
        self.patch_size   = patch_size
        self.in_chans     = in_chans
        self.num_patches  = (img_size // patch_size) ** 2
        self.grid_size    = img_size // patch_size

        # --- Context projection ---
        self.ctx_proj = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.proj = self.ctx_proj  # alias for convenience

        # --- Learnable mask token ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # --- Spatial positional embedding (fixed 2-D sin-cos, covers all L patches) ---
        pos = _get_2d_sincos_pos_embed(decoder_dim, self.grid_size)  # (L, D)
        self.register_buffer("pos_embed", pos.unsqueeze(0))           # (1, L, D)

        # --- Continuous time-gap embedding Delta-t ---
        # 2-layer MLP: R^1 -> decoder_dim, using SiLU activation.
        self.time_embed = nn.Sequential(
            nn.Linear(1, decoder_dim),
            nn.SiLU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.time_mlp = self.time_embed  # alias for backward-compatibility

        # --- Cross-attention blocks ---
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=decoder_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # --- Output head ---
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(
            decoder_dim,
            patch_size * patch_size * in_chans,
            bias=True,
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise Linear and LayerNorm weights following MAE conventions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        # Mask token re-init (already in __init__, kept for explicit visibility)
        nn.init.normal_(self.mask_token, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ctx_tokens: torch.Tensor,
        mask_or_delta: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            ctx_tokens    : (B, N_ctx, embed_dim) -- Frame-i encoder tokens.
            mask_or_delta : (B, L) binary mask, OR (B, 1) delta_t if mask omitted.
            delta_t       : (B, 1) normalised time gap |t_j - t_i| in [0, 1].

        Returns:
            pred : (B, L, patch_size^2 * in_chans).
        """
        B = ctx_tokens.shape[0]

        # Handle 2-arg calling: forward(ctx_tokens, delta_t)
        if delta_t is None and mask_or_delta is not None:
            if mask_or_delta.ndim == 2 and mask_or_delta.shape[-1] == 1:
                delta_t = mask_or_delta
                mask = torch.ones(B, self.num_patches, device=ctx_tokens.device, dtype=torch.float32)
            else:
                mask = mask_or_delta
                delta_t = torch.zeros(B, 1, device=ctx_tokens.device, dtype=torch.float32)
        elif mask_or_delta is None and delta_t is None:
            mask = torch.ones(B, self.num_patches, device=ctx_tokens.device, dtype=torch.float32)
            delta_t = torch.zeros(B, 1, device=ctx_tokens.device, dtype=torch.float32)
        else:
            mask = mask_or_delta
            if delta_t is None:
                delta_t = torch.zeros(B, 1, device=ctx_tokens.device, dtype=torch.float32)

        B, L = mask.shape
        assert L == self.num_patches, (
            f"Expected {self.num_patches} patches, got {L}."
        )

        # 1. Project context tokens into decoder dim.
        ctx = self.ctx_proj(ctx_tokens)                    # (B, N_ctx, D)

        # 2. Time-gap embedding: (B, 1) -> (B, D) -> (B, 1, D)
        dt_embed = self.time_mlp(delta_t.float())          # (B, D)
        dt_embed = dt_embed.unsqueeze(1)                   # (B, 1, D)

        # 3. Build query tokens for ALL L patch positions.
        #    SiamMAE style: decoder does NOT see target visible patches.
        #    Masked positions  -> mask_token + pos + Delta-t
        #    Visible positions -> zeros      + pos + Delta-t
        pos = self.pos_embed.expand(B, -1, -1)             # (B, L, D)

        queries = torch.zeros(
            B, L, self.decoder_dim,
            device=ctx_tokens.device,
            dtype=ctx_tokens.dtype,
        )

        # Inject learnable mask token at masked positions
        mask_expanded = mask.bool().unsqueeze(-1)          # (B, L, 1)
        mask_tok      = self.mask_token.expand(B, L, -1)   # (B, L, D)
        queries = torch.where(mask_expanded, mask_tok, queries)

        # Add positional and temporal embeddings to ALL query positions
        queries = queries + pos + dt_embed                 # (B, L, D)

        # 4. Run cross-attention stack.
        for blk in self.blocks:
            queries = blk(tgt=queries, ctx=ctx)

        # 5. Output head.
        queries = self.norm(queries)                       # (B, L, D)
        pred    = self.pred(queries)                       # (B, L, p*p*C)

        return pred
