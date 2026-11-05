"""
engine_extended_pretrain.py
----------------------------
Training engine for the ViC-MAE temporal extension (Stage 1 & Stage 2).

Implements the full mixed-loss training loop over alternating video and
image batches:

  Video batch loss:
      loss = L_mae(frame_i) + L_mae(frame_j)
           + lambda_nce  * L_nce(pooled_i, pooled_j)
           + lambda_cross * L_cross(ctx_i, frame_j, delta_t)

  Image batch loss:
      loss = L_mae(image)
           + lambda_pres * L_pres(tokens, frozen_tokens)

The frozen_teacher is a deepcopy of the initial ViCMAE backbone kept in
eval mode with all gradients disabled.
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.extended_vicmae import ExtendedViCMAE


# ---------------------------------------------------------------------------
# Loss weights (can be overridden via args)
# ---------------------------------------------------------------------------

DEFAULT_LAMBDA_NCE   = 0.1
DEFAULT_LAMBDA_CROSS = 1.0
DEFAULT_LAMBDA_PRES  = 0.5


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(
    model: ExtendedViCMAE,
    frozen_teacher: nn.Module,
    batch: Dict,
    device: torch.device,
    lambda_nce:   float = DEFAULT_LAMBDA_NCE,
    lambda_cross: float = DEFAULT_LAMBDA_CROSS,
    lambda_pres:  float = DEFAULT_LAMBDA_PRES,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the combined loss for one batch.

    Args:
        model          : Active ExtendedViCMAE instance (training mode).
        frozen_teacher : Frozen deepcopy of ViCMAE (eval, no grad).
        batch          : Dict from mixed_collate_fn (video or image schema).
        device         : Compute device.
        lambda_nce     : Weight for InfoNCE contrastive loss.
        lambda_cross   : Weight for cross-frame reconstruction loss.
        lambda_pres    : Weight for feature preservation loss.

    Returns:
        loss       : Scalar combined loss.
        loss_dict  : Dict of individual loss values for logging.
    """
    is_video = batch["is_video"]
    loss_dict: Dict[str, float] = {}

    if is_video:
        frame_i = batch["frame_i"].to(device, non_blocking=True)
        frame_j = batch["frame_j"].to(device, non_blocking=True)
        delta_t = batch["delta_t"].to(device, non_blocking=True)

        # --- L_mae for frame_i and frame_j (via base ViCMAE) ---
        mae_i, _, _ = model.vicmae([frame_i, frame_i])   # symmetric pair
        mae_j, _, _ = model.vicmae([frame_j, frame_j])
        l_mae = 0.5 * (mae_i + mae_j)

        # --- L_nce: encode both frames with base mask_ratio ---
        mask_ratio = model.vicmae.mask_ratio
        enc_i, _, _ = model.vicmae.forward_encoder(frame_i, mask_ratio)
        enc_j, _, _ = model.vicmae.forward_encoder(frame_j, mask_ratio)

        pool_i = model.vicmae.projector(enc_i[:, 1:, :].mean(dim=1))
        pool_j = model.vicmae.projector(enc_j[:, 1:, :].mean(dim=1))
        z      = torch.cat([pool_i, pool_j], dim=0)        # (2B, proj_dim)
        l_nce  = model.vicmae.info_nce_loss(z, model.vicmae.temperature)

        # --- L_cross: SiamMAE reconstruction of frame_j ---
        # Use FULL (unmasked) Frame-i encoder tokens as context
        ctx_i  = model.encode_full(frame_i)                # (B, N+1, D)
        l_cross_res = model.forward_cross_recon(
            ctx_tokens=ctx_i,
            target_imgs=frame_j,
            delta_t=delta_t,
            mask_ratio=model.cross_mask_ratio,
        )
        l_cross = l_cross_res[0] if isinstance(l_cross_res, (tuple, list)) else l_cross_res

        loss = l_mae + lambda_nce * l_nce + lambda_cross * l_cross
        loss_dict = {
            "loss_mae"  : l_mae.item(),
            "loss_nce"  : l_nce.item(),
            "loss_cross": l_cross.item(),
        }

    else:
        image = batch["image"].to(device, non_blocking=True)

        # --- L_mae for image (single view) ---
        mae_loss, _, _ = model.vicmae([image, image])
        l_mae = mae_loss

        # --- L_pres: cosine distance to frozen teacher ---
        with torch.no_grad():
            frozen_tokens = frozen_teacher.forward_encoder(
                image, mask_ratio=0.0
            )[0].detach()

        current_tokens = model.encode_full(image)           # (B, N+1, D)
        l_pres = ExtendedViCMAE.compute_preservation_loss(current_tokens, frozen_tokens)
        del frozen_tokens

        # Cross-decoder reconstruction on image with delta_t = 0 (trains decoder in Stage 1)
        delta_zero = torch.zeros(image.size(0), 1, device=device)
        l_cross_res = model.forward_cross_recon(
            ctx_tokens=current_tokens[:, 1:, :],
            target_imgs=image,
            delta_t=delta_zero,
            mask_ratio=model.cross_mask_ratio,
        )
        l_cross = l_cross_res[0] if isinstance(l_cross_res, (tuple, list)) else l_cross_res

        loss = l_mae + lambda_pres * l_pres + lambda_cross * l_cross
        loss_dict = {
            "loss_mae"  : l_mae.item(),
            "loss_pres" : l_pres.item(),
            "loss_cross": l_cross.item(),
        }

    loss_dict["loss_total"] = loss.item()
    return loss, loss_dict


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_one_epoch_extended(
    model: ExtendedViCMAE,
    frozen_teacher: nn.Module,
    video_loader: Optional[Iterable],
    image_loader: Optional[Iterable],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accum_iter: int = 1,
    lambda_nce:   float = DEFAULT_LAMBDA_NCE,
    lambda_cross: float = DEFAULT_LAMBDA_CROSS,
    lambda_pres:  float = DEFAULT_LAMBDA_PRES,
    log_freq: int = 20,
) -> Dict[str, float]:
    """
    Train for one epoch over video and/or image DataLoaders.

    Alternates between video and image batches within each step.
    """
    model.train()
    frozen_teacher.eval()

    # Build a unified iterator: interleave video and image batches
    loaders = [l for l in (video_loader, image_loader) if l is not None]
    if not loaders:
        raise ValueError("At least one of video_loader or image_loader must be provided.")

    batches     = [iter(l) for l in loaders]
    total_steps = max(len(l) for l in loaders)

    running: Dict[str, float] = {}
    counts:  Dict[str, int]   = {}

    optimizer.zero_grad()

    for step in range(total_steps):
        # Alternate between loaders; cycle shorter one
        loader_idx = step % len(loaders)
        try:
            batch = next(batches[loader_idx])
        except StopIteration:
            batches[loader_idx] = iter(loaders[loader_idx])
            batch = next(batches[loader_idx])

        use_amp = scaler is not None
        device_type = "cuda" if device.type == "cuda" else "cpu"
        try:
            amp_ctx = torch.amp.autocast(device_type=device_type, enabled=use_amp)
        except (AttributeError, TypeError):
            amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)

        with amp_ctx:
            loss, loss_dict = training_step(
                model, frozen_teacher, batch, device,
                lambda_nce=lambda_nce,
                lambda_cross=lambda_cross,
                lambda_pres=lambda_pres,
            )

        if not math.isfinite(loss.item()):
            print(f"[Epoch {epoch}] Step {step}: Loss is {loss.item()}, stopping.")
            sys.exit(1)

        loss = loss / accum_iter
        if loss.requires_grad:
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum_iter == 0 or (step + 1) == total_steps:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        # Accumulate for logging
        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v
            counts[k]  = counts.get(k, 0) + 1

        if step % log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            parts = " | ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
            print(f"[Epoch {epoch}] Step {step}/{total_steps} | lr: {lr:.2e} | {parts}")

    return {k: running[k] / max(counts[k], 1) for k in running}


# ---------------------------------------------------------------------------
# Optimizer builders
# ---------------------------------------------------------------------------

def build_stage1_optimizer(
    model: ExtendedViCMAE,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """
    Stage 1: only the cross_decoder trains.
    Backbone must already be frozen via model.freeze_backbone().
    """
    params = [p for p in model.cross_decoder.parameters() if p.requires_grad]
    assert params, "No trainable parameters in cross_decoder. Did you call freeze_backbone()?"
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)


def build_stage2_optimizer(
    model: ExtendedViCMAE,
    backbone_lr: float = 1e-5,
    decoder_lr:  float = 5e-5,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """
    Stage 2: backbone (unfrozen portions) and cross_decoder train at different lrs.
    """
    decoder_ids = set(id(p) for p in model.cross_decoder.parameters())
    backbone_params = [
        p for p in model.vicmae.parameters()
        if p.requires_grad and id(p) not in decoder_ids
    ]
    decoder_params = [p for p in model.cross_decoder.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": decoder_lr})

    assert param_groups, "No trainable parameters found for Stage 2 optimizer."
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=weight_decay)
