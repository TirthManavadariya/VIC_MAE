"""Two-stage continued training for the extended ViC-MAE.

Stage 1 (warm-up): freeze the whole base model (``vicmae.requires_grad_(False)``)
and train only the new cross-attention decoder on the cross-frame
reconstruction loss (``L_cross``) with lr=1e-4 for 5-10 epochs.

Stage 2 (joint): unfreeze either LoRA adapters (r=16, alpha=32) on the ViT
backbone or the top-4 transformer blocks, and train with all four losses::

    video batch: loss = L_mae(i) + L_mae(j) + 0.1 * L_nce + 1.0 * L_cross
    image batch: loss = L_mae + 0.5 * L_pres

with lr=1e-5 for the backbone (and MAE decoder / projector) and lr=5e-5 for
the cross-frame decoder.

Usage:
    python train.py \
        --checkpoint /path/to/vicmae_pretrain_vit_base_in1k_k400.pth \
        --video_root /path/to/video_frames \
        --image_path /path/to/tiny-imagenet-200/train \
        --output_dir output/extended
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

import data as data_mod
from models import ExtendedViCMAE, inject_lora, mark_only_lora_trainable
from models_vicmae import ViCMAE, vicmae_vit_base_patch16, vicmae_vit_large_patch16
from util.pos_embed import interpolate_pos_embed

# Loss weights required by the task specification.
LAMBDA_NCE = 0.1
LAMBDA_CROSS = 1.0
LAMBDA_PRES = 0.5


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Extended ViC-MAE continued training", add_help=False)

    # Data
    p.add_argument("--video_manifest", default=None, type=str,
                   help="JSONL/CSV manifest of frame pairs (frame_i, frame_j, delta_t)")
    p.add_argument("--video_root", default=None, type=str,
                   help="root of per-video frame folders root/<video_id>/<frame>.jpg")
    p.add_argument("--image_path", default=None, type=str,
                   help="image folder for the preservation stage")
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--img_size", default=224, type=int)

    # Model
    p.add_argument("--model", default="vicmae_vit_base_patch16", type=str,
                   choices=["vicmae_vit_base_patch16", "vicmae_vit_large_patch16"])
    p.add_argument("--checkpoint", default="", type=str,
                   help="pretrained ViC-MAE checkpoint to continue from")
    p.add_argument("--mask_ratio", default=0.5, type=float,
                   help="masking ratio for the MAE losses")
    p.add_argument("--cross_mask_ratio", default=0.95, type=float,
                   help="masking ratio for cross-frame reconstruction target")

    # Stage 1 (warm-up)
    p.add_argument("--stage1_epochs", default=8, type=int,
                   help="epochs training only the cross-attention decoder (5-10)")
    p.add_argument("--stage1_lr", default=1e-4, type=float)

    # Stage 2 (joint)
    p.add_argument("--stage2_epochs", default=20, type=int)
    p.add_argument("--unfreeze_mode", default="lora", type=str, choices=["lora", "top4"],
                   help="unfreeze LoRA adapters (r=16, alpha=32) or the top 4 blocks")
    p.add_argument("--lora_r", default=16, type=int)
    p.add_argument("--lora_alpha", default=32.0, type=float)
    p.add_argument("--backbone_lr", default=1e-5, type=float)
    p.add_argument("--decoder_lr", default=5e-5, type=float)
    p.add_argument("--warmup_epochs", default=2, type=int)
    p.add_argument("--min_lr", default=0.0, type=float)

    # Optimizer
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.95))

    # Misc
    p.add_argument("--output_dir", default="output/extended", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--log_interval", default=20, type=int,
                   help="iterations between loss logs")
    p.add_argument("--save_every", default=2, type=int,
                   help="checkpoint every N epochs (per stage)")
    return p


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------
def build_base_model(args) -> ViCMAE:
    """Instantiate the base ViC-MAE model and load the pretrained checkpoint."""
    ctor = {
        "vicmae_vit_base_patch16": vicmae_vit_base_patch16,
        "vicmae_vit_large_patch16": vicmae_vit_large_patch16,
    }[args.model]
    model = ctor(norm_pix_loss=True)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        interpolate_pos_embed(model, state_dict)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint {args.checkpoint}")
        print(f"  missing keys: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)}")
    return model


def build_models(args) -> Tuple[ExtendedViCMAE, ExtendedViCMAE]:
    """Build the trainable model and the frozen teacher (deepcopy of the checkpoint)."""
    base = build_base_model(args)
    model = ExtendedViCMAE(base)

    # Frozen teacher: a deepcopy of the pretrained checkpoint, never updated.
    teacher_base = build_base_model(args) if args.checkpoint else deepcopy(base)
    teacher = ExtendedViCMAE(teacher_base)
    teacher.requires_grad_(False)
    teacher.eval()
    return model, teacher


# ----------------------------------------------------------------------
# Parameter groups / freezing helpers
# ----------------------------------------------------------------------
def _split_decay_no_decay(params: Iterable[nn.Parameter], weight_decay: float):
    """Split parameters into decay / no-decay groups (timm convention)."""
    decay, no_decay = [], []
    for p in params:
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or p.bias is None and p.ndim == 1:
            no_decay.append(p)
        elif p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(model: ExtendedViCMAE, args, stage: int) -> torch.optim.Optimizer:
    """Build the AdamW optimizer for the requested training stage."""
    if stage == 1:
        # Only the cross-attention decoder is trainable.
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.cross_decoder.parameters():
            p.requires_grad_(True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        decay, no_decay = _split_decay_no_decay(trainable, args.weight_decay)
        param_groups = [
            {"params": decay, "weight_decay": args.weight_decay, "lr": args.stage1_lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": args.stage1_lr},
        ]
        return torch.optim.AdamW(param_groups, lr=args.stage1_lr, betas=tuple(args.betas))

    # ---- Stage 2 ----
    for p in model.parameters():
        p.requires_grad_(False)

    backbone_names: List[str] = []
    if args.unfreeze_mode == "lora":
        wrapped = inject_lora(model.vicmae.blocks, r=args.lora_r, alpha=args.lora_alpha)
        n_lora = mark_only_lora_trainable(model.vicmae.blocks)
        print(f"Injected LoRA (r={args.lora_r}, alpha={args.lora_alpha}) into {len(wrapped)} layers")
        print(f"  LoRA trainable params: {n_lora / 1e6:.3f}M")
        backbone_names = wrapped
    else:
        # Unfreeze the top-4 transformer blocks of the ViT backbone.
        for blk in model.vicmae.blocks[-4:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        backbone_names = [f"blocks.{i}" for i in range(len(model.vicmae.blocks) - 4,
                                                        len(model.vicmae.blocks))]

    # MAE decoder + projector must train for L_mae / L_nce.
    for name, module in model.vicmae.named_children():
        if name.startswith("decoder") or name == "projector" or name == "mask_token":
            for p in module.parameters():
                p.requires_grad_(True)

    # The new cross-frame decoder is always trainable.
    for p in model.cross_decoder.parameters():
        p.requires_grad_(True)

    backbone_ids = {id(p) for name, p in model.vicmae.named_parameters()
                    if any(name.startswith(bn) for bn in backbone_names)}
    head_ids = {id(p) for name, p in model.vicmae.named_parameters()
                if name.startswith("decoder") or name.startswith("projector")
                or name == "mask_token"}

    backbone_params = [p for p in model.vicmae.parameters()
                       if p.requires_grad and id(p) in backbone_ids]
    head_params = [p for p in model.vicmae.parameters()
                   if p.requires_grad and id(p) in head_ids]
    decoder_params = [p for p in model.cross_decoder.parameters() if p.requires_grad]

    def mk_group(params, lr):
        decay, no_decay = _split_decay_no_decay(params, args.weight_decay)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": args.weight_decay, "lr": lr})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0, "lr": lr})
        return groups

    param_groups = (
        mk_group(backbone_params, args.backbone_lr)
        + mk_group(head_params, args.backbone_lr)
        + mk_group(decoder_params, args.decoder_lr)
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2 trainable params: {n_train / 1e6:.3f}M "
          f"(backbone {len(backbone_params)}, heads {len(head_params)}, "
          f"cross_decoder {len(decoder_params)})")
    return torch.optim.AdamW(param_groups, betas=tuple(args.betas))


# ----------------------------------------------------------------------
# LR schedule (warmup + cosine)
# ----------------------------------------------------------------------
def adjust_lr(optimizer: torch.optim.Optimizer, base_lrs: List[float], epoch: int,
              args, total_epochs: int) -> None:
    """Apply linear warmup then cosine decay to each param group."""
    if epoch < args.warmup_epochs:
        scale = (epoch + 1) / max(1, args.warmup_epochs)
    else:
        progress = (epoch - args.warmup_epochs) / max(1, total_epochs - args.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


# ----------------------------------------------------------------------
# Training loops
# ----------------------------------------------------------------------
def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def train_stage1(model: ExtendedViCMAE, video_loader, optimizer, device,
                 args, epoch: int) -> Dict[str, float]:
    """Stage 1: train only the cross-attention decoder on L_cross."""
    model.train()
    model.vicmae.requires_grad_(False)  # encoder (and everything but decoder) frozen
    for p in model.cross_decoder.parameters():
        p.requires_grad_(True)

    running = 0.0
    n = 0
    for step, batch in enumerate(video_loader):
        batch = _to_device(batch, device)
        frame_i, frame_j = batch["frame_i"], batch["frame_j"]
        delta_t = batch["delta_t"]

        # Context = visible encoder tokens of frame_i (encoder frozen).
        with torch.no_grad():
            tok_i, _, _ = model.encode_frame(frame_i, args.mask_ratio)
        ctx_tokens = tok_i[:, 1:, :]  # strip cls token

        loss, _, _ = model.forward_cross_recon(
            ctx_tokens, frame_j, delta_t, mask_ratio=args.cross_mask_ratio
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
        n += 1
        if (step + 1) % args.log_interval == 0:
            print(f"  [stage1] epoch {epoch} step {step + 1}/{len(video_loader)} "
                  f"loss_cross {running / n:.4f}")
    return {"loss_cross": running / max(n, 1)}


def train_stage2(model: ExtendedViCMAE, teacher: ExtendedViCMAE,
                 video_loader, image_loader, optimizer, device,
                 args, epoch: int) -> Dict[str, float]:
    """Stage 2: joint training with all four losses."""
    model.train()
    teacher.eval()

    totals: Dict[str, float] = {}
    n = 0

    def accumulate(metrics: Dict[str, torch.Tensor]) -> None:
        nonlocal n
        n += 1
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach())

    # -- video batches --
    if video_loader is not None:
        for step, batch in enumerate(video_loader):
            batch = _to_device(batch, device)
            batch["mask_ratio"] = args.mask_ratio
            batch["lambda_nce"] = LAMBDA_NCE
            batch["lambda_cross"] = LAMBDA_CROSS
            metrics = model(batch)  # full video loss
            loss = metrics["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accumulate(metrics)

    # -- image batches --
    if image_loader is not None:
        for step, batch in enumerate(image_loader):
            batch = _to_device(batch, device)
            batch["mask_ratio"] = args.mask_ratio
            batch["lambda_pres"] = LAMBDA_PRES
            with torch.no_grad():
                batch["frozen_tokens"] = teacher.extract_patch_tokens(batch["image"])
            metrics = model(batch)  # image loss (L_mae + lambda_pres * L_pres)
            loss = metrics["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accumulate(metrics)

    return {k: v / max(n, 1) for k, v in totals.items()}


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def save_checkpoint(path: Path, model: ExtendedViCMAE, optimizer, epoch: int,
                    stage: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "args": vars(args),
    }, path)
    print(f"Saved checkpoint: {path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(args) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    video_loader, image_loader = data_mod.build_dataloaders(
        video_manifest=args.video_manifest,
        video_root=args.video_root,
        image_path=args.image_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
    )
    print(f"video batches/epoch: {len(video_loader) if video_loader else 0}, "
          f"image batches/epoch: {len(image_loader) if image_loader else 0}")

    # Models
    model, teacher = build_models(args)
    model.to(device)
    teacher.to(device)
    print(f"Model: {args.model}, cross_decoder dim={model.cross_decoder.decoder_dim}, "
          f"depth={len(model.cross_decoder.blocks)}")

    log_file = out_dir / "log.txt"

    # ---------------- Stage 1: warm up the cross-attention decoder ----------
    if args.stage1_epochs > 0:
        if video_loader is None:
            raise ValueError("Stage 1 requires video data (video_manifest or video_root)")
        optimizer = build_optimizer(model, args, stage=1)
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        print(f"\n=== Stage 1: training cross_decoder only, "
              f"{args.stage1_epochs} epochs, lr={args.stage1_lr} ===")
        for epoch in range(args.stage1_epochs):
            t0 = time.time()
            adjust_lr(optimizer, base_lrs, epoch, args, args.stage1_epochs)
            stats = train_stage1(model, video_loader, optimizer, device, args, epoch)
            print(f"[stage1] epoch {epoch}: {stats} ({time.time() - t0:.1f}s)")
            with open(log_file, "a") as f:
                f.write(json.dumps({"stage": 1, "epoch": epoch, **stats}) + "\n")
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.stage1_epochs:
                save_checkpoint(out_dir / f"checkpoint-stage1-{epoch:03d}.pth",
                                model, optimizer, epoch, 1, args)

    # ---------------- Stage 2: joint training --------------------------------
    if args.stage2_epochs > 0:
        optimizer = build_optimizer(model, args, stage=2)
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        print(f"\n=== Stage 2: joint training ({args.unfreeze_mode}), "
              f"{args.stage2_epochs} epochs ===")
        for epoch in range(args.stage2_epochs):
            t0 = time.time()
            adjust_lr(optimizer, base_lrs, epoch, args, args.stage2_epochs)
            stats = train_stage2(model, teacher, video_loader, image_loader,
                                 optimizer, device, args, epoch)
            print(f"[stage2] epoch {epoch}: {stats} ({time.time() - t0:.1f}s)")
            with open(log_file, "a") as f:
                f.write(json.dumps({"stage": 2, "epoch": epoch, **stats}) + "\n")
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.stage2_epochs:
                save_checkpoint(out_dir / f"checkpoint-stage2-{epoch:03d}.pth",
                                model, optimizer, epoch, 2, args)

    print("\nDone.")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
