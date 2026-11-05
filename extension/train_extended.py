"""
train_extended.py
-----------------
Top-level training script for the ViC-MAE temporal extension.

Training protocol:
  Main Training (30 epochs by default, backbone_lr=1e-5, decoder_lr=5e-5):
    - Encoder adapted (LoRA adapters or top-4 blocks) + CrossFrameDecoder.
    - Full 4-loss training with feature preservation.

  Optional Stage 1 Warmup (stage1_epochs > 0, default: 0):
    - Backbone frozen.
    - Only CrossFrameDecoder trained.

Usage:
    python train_extended.py \\
        --checkpoint path/to/vicmae_pretrained.pth \\
        --video_root  path/to/video_frames \\
        --image_root  path/to/tiny_imagenet/train \\
        --stage1_epochs 5 \\
        --stage2_epochs 20 \\
        --use_lora \\
        --output_dir output/extended
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.backends.cudnn as cudnn

# Allow running from extension directory or root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in [_HERE, os.path.join(_HERE, "ViC-MAE"), os.path.join(_PARENT, "ViC-MAE")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import models_vicmae
from models.extended_vicmae import ExtendedViCMAE
from engine_extended_pretrain import (
    train_one_epoch_extended,
    build_stage1_optimizer,
    build_stage2_optimizer,
)
from datasets.video_image_dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ViC-MAE Temporal Extension Training", add_help=True)

    # Checkpoint
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to pretrained ViC-MAE checkpoint (.pth).")
    p.add_argument("--vicmae_arch", type=str, default="vicmae_vit_base_patch16",
                   help="ViCMAE architecture name (must match checkpoint).")

    # Data
    p.add_argument("--video_root", type=str, default=None,
                   help="Root directory of extracted video frames.")
    p.add_argument("--image_root", type=str, default=None,
                   help="Root directory of static images (e.g., TinyImageNet).")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_frame_gap", type=int, default=48,
                   help="Maximum frame-index gap for delta_t normalisation.")

    # Training
    p.add_argument("--epochs",       type=int, default=30,
                   help="Total training epochs across Stage 1 + Stage 2 (default: 30).")
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--stage1_epochs",type=int, default=0,
                   help="Epochs for Stage 1 (decoder warm-up, backbone frozen; default: 0).")
    p.add_argument("--stage2_epochs",type=int, default=30,
                   help="Epochs for Stage 2 / main training (joint fine-tuning; default: 30).")
    p.add_argument("--stage1_lr",    type=float, default=1e-4)
    p.add_argument("--backbone_lr",  type=float, default=1e-5)
    p.add_argument("--decoder_lr",   type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--accum_iter",   type=int,   default=4)
    p.add_argument("--use_amp",      action="store_true", default=False,
                   help="Enable AMP (automatic mixed precision).")

    # Stage 2 mode
    p.add_argument("--use_lora",     action="store_true", default=False,
                   help="Use LoRA adapters for Stage 2 (default: top-4 blocks).")
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=float, default=32.0)
    p.add_argument("--top_blocks",   type=int,   default=4,
                   help="Number of top ViT blocks to unfreeze in Stage 2.")

    # Loss weights
    p.add_argument("--lambda_nce",   type=float, default=0.1)
    p.add_argument("--lambda_cross", type=float, default=1.0)
    p.add_argument("--lambda_pres",  type=float, default=0.5)
    # Convenience aliases
    p.add_argument("--lr",                type=float, default=None, help="Global lr alias.")
    p.add_argument("--min_lr",            type=float, default=1e-6)
    p.add_argument("--warmup_epochs",     type=int,   default=2)
    p.add_argument("--log_freq",          type=int,   default=20)
    p.add_argument("--loss_weight_mae",   type=float, default=1.0)
    p.add_argument("--loss_weight_cross", type=float, default=None)
    p.add_argument("--loss_weight_pres",  type=float, default=None)

    # CrossDecoder config
    p.add_argument("--decoder_dim",   type=int, default=512)
    p.add_argument("--decoder_depth", type=int, default=4)
    p.add_argument("--decoder_heads", type=int, default=8)

    # Output
    p.add_argument("--output_dir", type=str, default="output/extended")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--resume",     type=str, default="",
                   help="Path to resume checkpoint (.pth).")
    p.add_argument("--resume_expected_epoch", type=int, default=None,
                   help="Require this saved zero-based epoch index before resuming.")
    p.add_argument("--start_epoch", type=int, default=None,
                   help="Explicit start epoch index (e.g. 10 to continue after 0-9).")
    p.add_argument("--save_freq",   type=int, default=1,
                   help="Save checkpoint every N epochs (default: 1).")

    args = p.parse_args()
    if args.lr is not None:
        args.stage1_lr = args.lr
        args.decoder_lr = args.lr
    if args.loss_weight_cross is not None:
        args.lambda_cross = args.loss_weight_cross
    if args.loss_weight_pres is not None:
        args.lambda_pres = args.loss_weight_pres
    return args


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: ExtendedViCMAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: str,
    tag: str = "",
    scaler=None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"checkpoint_epoch{epoch:04d}{tag}.pth"
    state = {
        "epoch"     : epoch,
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    # Publish completed file via temporary file to prevent corruption
    destination = os.path.join(output_dir, fname)
    temporary = destination + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, destination)

    # Also update checkpoint_latest.pth
    latest_path = os.path.join(output_dir, "checkpoint_latest.pth")
    temp_latest = latest_path + ".tmp"
    torch.save(state, temp_latest)
    os.replace(temp_latest, latest_path)
    print(f"  Saved checkpoint: {fname} (and updated checkpoint_latest.pth)")


def restore_training_checkpoint(
    model: ExtendedViCMAE,
    optimizer: torch.optim.Optimizer,
    scaler,
    checkpoint_path: str,
    expected_epoch: Optional[int] = None,
    end_epoch: int = 30,
    start_epoch_override: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Restore complete training state, or stop with clear error."""
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint_path}. "
            "Please ensure the checkpoint dataset is attached and path is correct."
        )
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError("Resume checkpoint must contain a training-state dictionary.")
    missing = {"model", "optimizer", "epoch"} - ckpt.keys()
    if missing:
        raise ValueError(f"Incomplete resume checkpoint; missing keys: {sorted(missing)}")
    saved_epoch = ckpt["epoch"]
    if isinstance(saved_epoch, bool) or not isinstance(saved_epoch, int):
        raise ValueError(f"Saved epoch must be an integer index, got {saved_epoch!r}")
    if expected_epoch is not None and saved_epoch != expected_epoch:
        raise ValueError(f"Expected saved epoch index {expected_epoch}, got {saved_epoch}")

    if start_epoch_override is not None:
        start_epoch = start_epoch_override
    else:
        start_epoch = saved_epoch + 1

    if not 0 <= start_epoch < end_epoch:
        raise ValueError(f"No valid remaining epochs: start={start_epoch}, stop={end_epoch}")
    if not isinstance(ckpt["model"], dict) or not ckpt["model"]:
        raise ValueError("Resume checkpoint has no model weights.")
    optimizer_state = ckpt["optimizer"]
    if (not isinstance(optimizer_state, dict)
            or not optimizer_state.get("state")
            or not optimizer_state.get("param_groups")):
        raise ValueError("Resume checkpoint has no trained optimizer state.")

    # Strict loading ensures 100% parameter alignment
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(optimizer_state)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if scaler is not None:
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
            print("  Restored AMP scaler state.")
        else:
            print("  NOTE: Checkpoint has no AMP scaler state; using fresh GradScaler.")

    print(f"  Restored complete model, optimizer, and scaler from {checkpoint_path}")
    print(f"  Saved epoch index: {saved_epoch} (completed {saved_epoch + 1} epochs); resuming indices "
          f"{start_epoch}..{end_epoch - 1} ({end_epoch - start_epoch} epochs to train).")
    print(f"  First displayed epoch: {start_epoch + 1}/{end_epoch}; "
          f"optimizer LRs: {[group['lr'] for group in optimizer.param_groups]}")
    return start_epoch


def load_vicmae_weights(model: ExtendedViCMAE, ckpt_path: str) -> None:
    """Load pretrained ViCMAE weights into the model's vicmae sub-module."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    # Support common checkpoint formats
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    # Strip 'module.' prefix from DDP checkpoints
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.vicmae.load_state_dict(state, strict=False)
    print(f"  Loaded ViCMAE weights from {ckpt_path}")
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    print("\n==> Building ExtendedViCMAE ...")
    # Resolve ViCMAE constructor kwargs from architecture name
    arch_defaults = {
        "vicmae_vit_base_patch16" : dict(embed_dim=768,  depth=12, num_heads=12,
                                         decoder_embed_dim=512, decoder_depth=8,
                                         decoder_num_heads=16),
        "vicmae_vit_large_patch16": dict(embed_dim=1024, depth=24, num_heads=16,
                                         decoder_embed_dim=512, decoder_depth=8,
                                         decoder_num_heads=16),
        "vicmae_vit_small_patch16": dict(embed_dim=384,  depth=12, num_heads=6,
                                         decoder_embed_dim=256, decoder_depth=8,
                                         decoder_num_heads=8),
    }
    vicmae_cfg = arch_defaults.get(args.vicmae_arch, {})

    model = ExtendedViCMAE(
        vicmae_cfg=vicmae_cfg,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        img_size=args.img_size,
        cross_mask_ratio=0.95,
    ).to(device)

    # Load pretrained ViCMAE backbone
    if args.checkpoint:
        load_vicmae_weights(model, args.checkpoint)

    # Frozen teacher: deepcopy of the pretrained ViCMAE, frozen forever
    frozen_teacher = deepcopy(model.vicmae).to(device)
    frozen_teacher.eval()
    frozen_teacher.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Build data loaders
    # -----------------------------------------------------------------------
    print("\n==> Building data loaders ...")
    video_loader, image_loader = build_dataloaders(
        video_root=args.video_root,
        image_root=args.image_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        max_frame_gap=args.max_frame_gap,
    )

    if args.use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    total_epochs = args.stage1_epochs + args.stage2_epochs

    # -----------------------------------------------------------------------
    # Stage 1: Decoder warm-up (backbone frozen)
    # -----------------------------------------------------------------------
    if args.stage1_epochs > 0:
        print(f"\n{'='*60}")
        print(f"  STAGE 1: Decoder warm-up for {args.stage1_epochs} epochs (Epochs 1-{args.stage1_epochs} of {total_epochs})")
        print(f"{'='*60}")

        model.freeze_backbone()
        optimizer = build_stage1_optimizer(
            model, lr=args.stage1_lr, weight_decay=args.weight_decay
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        log_history = []
        for epoch in range(args.stage1_epochs):
            t0 = time.time()
            metrics = train_one_epoch_extended(
                model=model,
                frozen_teacher=frozen_teacher,
                video_loader=video_loader,
                image_loader=image_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                scaler=scaler,
                accum_iter=args.accum_iter,
                lambda_nce=args.lambda_nce,
                lambda_cross=args.lambda_cross,
                lambda_pres=args.lambda_pres,
            )
            elapsed = time.time() - t0
            global_epoch = epoch + 1
            log = {"stage": 1, "epoch": global_epoch, "stage_epoch": epoch, "time_s": elapsed, **metrics}
            log_history.append(log)
            with open(os.path.join(args.output_dir, "train_extended_log.json"), "a") as f_log:
                f_log.write(json.dumps(log) + "\n")
            print(f"  [Epoch {global_epoch}/{total_epochs} (Stage 1)] {elapsed:.1f}s | " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.stage1_epochs:
                save_checkpoint(model, optimizer, epoch, args.output_dir, tag="_s1", scaler=scaler)

        # Save Stage 1 log
        with open(os.path.join(args.output_dir, "log_stage1.json"), "w") as f:
            json.dump(log_history, f, indent=2)

    # -----------------------------------------------------------------------
    # Stage 2: Joint fine-tuning
    # -----------------------------------------------------------------------
    if args.stage2_epochs > 0:
        if args.stage1_epochs > 0:
            s2_start_disp = args.stage1_epochs + 1
            s2_end_disp = total_epochs
            print(f"\n{'='*60}")
            print(f"  STAGE 2: Joint fine-tuning for {args.stage2_epochs} epochs (Epochs {s2_start_disp}-{s2_end_disp} of {total_epochs})")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"  MAIN TRAINING: Joint fine-tuning for {args.stage2_epochs} epochs (Epochs 1-{total_epochs})")
            print(f"{'='*60}")

        # Unfreeze backbone portion
        if args.use_lora:
            print(f"  Strategy: LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
            model.inject_lora_adapters(r=args.lora_r, alpha=args.lora_alpha)
        else:
            print(f"  Strategy: Unfreeze top {args.top_blocks} ViT blocks")
            model.unfreeze_top_blocks(n=args.top_blocks)

        # Cross decoder should always be trainable in Stage 2
        model.cross_decoder.requires_grad_(True)
        model.to(device)

        optimizer = build_stage2_optimizer(
            model,
            backbone_lr=args.backbone_lr,
            decoder_lr=args.decoder_lr,
            weight_decay=args.weight_decay,
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        # Resume if requested
        start_epoch = 0
        if args.resume:
            start_epoch = restore_training_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                checkpoint_path=args.resume,
                expected_epoch=args.resume_expected_epoch,
                end_epoch=args.stage2_epochs,
                start_epoch_override=args.start_epoch,
                device=device,
            )

        log_history = []
        for epoch in range(start_epoch, args.stage2_epochs):
            t0 = time.time()
            metrics = train_one_epoch_extended(
                model=model,
                frozen_teacher=frozen_teacher,
                video_loader=video_loader,
                image_loader=image_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                scaler=scaler,
                accum_iter=args.accum_iter,
                lambda_nce=args.lambda_nce,
                lambda_cross=args.lambda_cross,
                lambda_pres=args.lambda_pres,
            )
            elapsed = time.time() - t0
            global_epoch = args.stage1_epochs + epoch + 1
            log = {"stage": 2, "epoch": global_epoch, "stage_epoch": epoch, "time_s": elapsed, **metrics}
            log_history.append(log)
            with open(os.path.join(args.output_dir, "train_extended_log.json"), "a") as f_log:
                f_log.write(json.dumps(log) + "\n")
            stage_label = "Stage 2" if args.stage1_epochs > 0 else "Main"
            print(f"  [Epoch {global_epoch}/{total_epochs} ({stage_label})] {elapsed:.1f}s | " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

            if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.stage2_epochs:
                save_checkpoint(model, optimizer, epoch, args.output_dir, tag="_s2", scaler=scaler)

        with open(os.path.join(args.output_dir, "log_stage2.json"), "w") as f:
            json.dump(log_history, f, indent=2)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
