"""Verification & smoke tests for the extended ViC-MAE.

Runs dummy forward/backward passes on random synthetic tensors and checks:

* Cross-frame decoder output shapes and the ~95% binary mask.
* Video-batch loss composition (L_mae + L_nce + L_cross) and backward.
* Image-batch preservation loss (L_mae + L_pres) and backward.
* Stage 1 gradient isolation: only ``cross_decoder`` receives gradients.
* Stage 2 gradient propagation: both ``cross_decoder`` and the backbone
  (LoRA) receive gradients.
* No NaNs or dimension mismatches in the cross-attention queries.

Usage:
    python test_extension.py            # standalone runner
    pytest test_extension.py -v        # pytest
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Allow running from extension directory or root
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_REPO = _HERE / "ViC-MAE" if (_HERE / "ViC-MAE").exists() else _ROOT / "ViC-MAE"
for p in [_HERE, _REPO, _ROOT]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn as nn

from models import (
    CrossFrameDecoder,
    ExtendedViCMAE,
    inject_lora,
    mark_only_lora_trainable,
)
from models_vicmae import ViCMAE

B = 2
IMG = 224
EMBED_DIM = 384
DECODER_DIM = 256
PATCH_SIZE = 16
NUM_PATCHES = (IMG // PATCH_SIZE) ** 2
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3


def _tiny_base() -> ViCMAE:
    """A small ViCMAE (random weights) so the smoke test runs fast on CPU."""
    return ViCMAE(
        embed_dim=EMBED_DIM,
        depth=4,
        num_heads=6,
        decoder_embed_dim=DECODER_DIM,
        decoder_depth=2,
        decoder_num_heads=8,
        patch_size=PATCH_SIZE,
        norm_pix_loss=True,
    )


def _video_batch(b: int = B, device: str = "cpu") -> dict:
    return {
        "frame_i": torch.randn(b, 3, IMG, IMG, device=device),
        "frame_j": torch.randn(b, 3, IMG, IMG, device=device),
        "delta_t": torch.rand(b, 1, device=device),
        "is_video": True,
        "mask_ratio": 0.5,
        "lambda_nce": 0.1,
        "lambda_cross": 1.0,
    }


def _image_batch(b: int = B, device: str = "cpu") -> dict:
    img = torch.randn(b, 3, IMG, IMG, device=device)
    return {
        "image": img,
        "is_video": False,
        "mask_ratio": 0.5,
        "lambda_pres": 0.5,
        "frozen_tokens": torch.randn(b, NUM_PATCHES, EMBED_DIM, device=device),
    }


def _trainable_grads(module: nn.Module):
    return [(n, p) for n, p in module.named_parameters()
            if p.requires_grad and p.grad is not None]


def _any_nonzero_grad(module: nn.Module) -> bool:
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_cross_decoder_shapes_and_mask():
    dec = CrossFrameDecoder(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
                            depth=4, num_heads=8, patch_size=PATCH_SIZE)
    ctx = torch.randn(B, 10, EMBED_DIM)  # e.g. 10 visible tokens of frame_i
    delta = torch.rand(B, 1)

    pred = dec(ctx, delta)
    assert pred.shape == (B, NUM_PATCHES, PATCH_DIM), f"pred shape {pred.shape}"

    # Query construction shape check: mask_token + pos_embed + time_embed.
    assert dec.mask_token.shape == (1, 1, DECODER_DIM)
    assert dec.pos_embed.shape == (1, NUM_PATCHES, DECODER_DIM)
    assert dec.time_embed[0].in_features == 1 and dec.time_embed[0].out_features == DECODER_DIM
    assert torch.isfinite(pred).all(), "NaN/Inf in decoder output"
    print("PASS test_cross_decoder_shapes_and_mask")


def test_cross_recon_mask_ratio_and_loss():
    model = ExtendedViCMAE(_tiny_base())
    model.train()

    ctx = torch.randn(B, 10, EMBED_DIM)
    target = torch.randn(B, 3, IMG, IMG)
    delta = torch.rand(B, 1)

    loss, pred, mask = model.forward_cross_recon(ctx, target, delta, mask_ratio=0.95)
    assert loss.ndim == 0 and torch.isfinite(loss), f"loss {loss}"
    assert pred.shape == (B, NUM_PATCHES, PATCH_DIM)
    assert mask.shape == (B, NUM_PATCHES)
    masked_frac = mask.mean().item()
    assert 0.90 < masked_frac < 1.0, f"masked fraction {masked_frac}"
    # Loss is zero only if nothing masked; with 95% masked it must be > 0.
    assert loss.item() > 0
    print("PASS test_cross_recon_mask_ratio_and_loss")


def test_video_batch_losses_and_backward():
    model = ExtendedViCMAE(_tiny_base())
    model.train()
    batch = _video_batch()

    out = model(batch)
    expected_keys = {"loss", "loss_mae_i", "loss_mae_j", "loss_nce", "loss_cross"}
    assert expected_keys.issubset(out.keys()), f"missing keys: {expected_keys - set(out)}"
    assert all(torch.isfinite(v) for v in out.values()), "NaN/Inf in video losses"
    assert out["loss"].item() > 0

    out["loss"].backward()
    assert _any_nonzero_grad(model.cross_decoder), "cross_decoder got no gradient"
    print("PASS test_video_batch_losses_and_backward")


def test_image_batch_preservation_loss():
    model = ExtendedViCMAE(_tiny_base())
    model.train()
    batch = _image_batch()

    out = model(batch)
    assert {"loss", "loss_mae", "loss_pres"}.issubset(out.keys())
    assert 0.0 <= out["loss_pres"].item() <= 2.0, f"pres loss {out['loss_pres']}"
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    assert _any_nonzero_grad(model.cross_decoder) or _any_nonzero_grad(model.vicmae)
    print("PASS test_image_batch_preservation_loss")


def test_preservation_loss_math():
    model = ExtendedViCMAE(_tiny_base())
    tok = torch.randn(B, NUM_PATCHES, EMBED_DIM)
    # Identical tokens -> cosine similarity 1 -> loss 0.
    assert model.compute_preservation_loss(tok, tok).item() == 0.0
    # Opposed tokens -> cosine similarity -1 -> loss 2.
    assert abs(model.compute_preservation_loss(tok, -tok).item() - 2.0) < 1e-5
    print("PASS test_preservation_loss_math")


def test_stage1_gradient_isolation():
    """Stage 1: frozen encoder -> only cross_decoder receives gradients."""
    model = ExtendedViCMAE(_tiny_base())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.cross_decoder.parameters():
        p.requires_grad_(True)

    batch = _video_batch()
    tok_i, _, _ = model.encode_frame(batch["frame_i"], 0.5)
    loss, _, _ = model.forward_cross_recon(
        tok_i[:, 1:, :], batch["frame_j"], batch["delta_t"], mask_ratio=0.95
    )
    loss.backward()

    assert _any_nonzero_grad(model.cross_decoder), "cross_decoder got no gradient"
    vicmae_grads = _trainable_grads(model.vicmae)
    assert len(vicmae_grads) == 0, f"frozen encoder got gradients: {vicmae_grads[:3]}"
    print("PASS test_stage1_gradient_isolation")


def test_stage2_lora_gradient_propagation():
    """Stage 2 (LoRA): both cross_decoder and backbone LoRA get gradients."""
    model = ExtendedViCMAE(_tiny_base())
    inject_lora(model.vicmae.blocks, r=16, alpha=32)
    mark_only_lora_trainable(model.vicmae.blocks)
    for p in model.cross_decoder.parameters():
        p.requires_grad_(True)

    batch = _video_batch()
    out = model(batch)
    out["loss"].backward()

    assert _any_nonzero_grad(model.cross_decoder), "cross_decoder got no gradient"
    lora_grads = [(n, p) for n, p in model.vicmae.named_parameters()
                  if "lora_" in n and p.grad is not None]
    assert len(lora_grads) > 0, "LoRA params got no gradient"
    assert all(torch.isfinite(p.grad).all() for _, p in lora_grads), "NaN in LoRA grads"
    print("PASS test_stage2_lora_gradient_propagation")


def test_cross_attention_query_no_nan():
    """Explicitly exercise the query path and assert finite, correctly shaped queries."""
    dec = CrossFrameDecoder(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
                            depth=4, num_heads=8, patch_size=PATCH_SIZE)
    ctx = torch.randn(B, 10, EMBED_DIM)
    delta = torch.rand(B, 1)

    # Reconstruct the exact query tensor the decoder builds.
    time_emb = dec.time_embed(delta.float())
    query = dec.mask_token + dec.pos_embed + time_emb.unsqueeze(1)
    query = query.expand(B, NUM_PATCHES, DECODER_DIM)
    assert query.shape == (B, NUM_PATCHES, DECODER_DIM), f"query shape {query.shape}"
    assert torch.isfinite(query).all(), "NaN in cross-attention queries"

    # Run a full block and check the attention output shape.
    from models import CrossAttentionBlock
    block = CrossAttentionBlock(DECODER_DIM, 8)
    context = dec.proj(ctx)
    out = block(query, context)
    assert out.shape == (B, NUM_PATCHES, DECODER_DIM)
    assert torch.isfinite(out).all(), "NaN after cross-attention block"
    print("PASS test_cross_attention_query_no_nan")


def test_video_dataset_raw_video_loading(tmp_path=None):
    """Verify that VideoFramePairDataset can load directly from a raw video file using cv2."""
    import tempfile
    import numpy as np
    import cv2
    from pathlib import Path
    from datasets.video_image_dataset import VideoFramePairDataset

    tmp_dir = Path(tempfile.mkdtemp())
    video_file = tmp_dir / "test_clip.mp4"

    # Create a small dummy MP4 video with 10 synthetic frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 10.0, (64, 64))
    for i in range(10):
        frame = np.full((64, 64, 3), fill_value=i * 25, dtype=np.uint8)
        out.write(frame)
    out.release()

    # Load dataset pointing directly to the video file
    ds = VideoFramePairDataset(root=str(tmp_dir), img_size=224, max_frame_gap=5)
    assert len(ds) == 1
    sample = ds[0]

    assert sample["is_video"] is True
    assert sample["frame_i"].shape == (3, 224, 224)
    assert sample["frame_j"].shape == (3, 224, 224)
    assert sample["delta_t"].shape == (1,)
    assert 0.0 <= sample["delta_t"].item() <= 1.0
    print("PASS test_video_dataset_raw_video_loading")


# ----------------------------------------------------------------------
# Standalone runner
# ----------------------------------------------------------------------
def _run_all():
    tests = [
        test_cross_decoder_shapes_and_mask,
        test_cross_recon_mask_ratio_and_loss,
        test_video_batch_losses_and_backward,
        test_image_batch_preservation_loss,
        test_preservation_loss_math,
        test_stage1_gradient_isolation,
        test_stage2_lora_gradient_propagation,
        test_cross_attention_query_no_nan,
        test_video_dataset_raw_video_loading,
    ]
    for t in tests:
        t()
    print("\n[ALL SMOKE TESTS PASSED]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=None, help="run a single test by name")
    args = parser.parse_args()
    if args.test:
        globals()[args.test]()
        print(f"\n[PASSED] {args.test}")
    else:
        _run_all()
