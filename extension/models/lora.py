"""Minimal LoRA (Low-Rank Adaptation) utilities for the ViT backbone.

Self-contained implementation (no external LoRA dependency): wraps selected
``nn.Linear`` layers with a frozen base weight plus a trainable low-rank
update ``B @ A`` scaled by ``alpha / r``. Used by Stage 2 of the
continued-training protocol to unfreeze the backbone with minimal
parameter overhead (r=16, alpha=32 by default).
"""
from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank residual.

    ``forward(x) = base(x) + (x @ A^T @ B^T) * (alpha / r)``

    ``B`` is zero-initialized so the wrapped layer is exactly equivalent to
    the base linear at initialization.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        dev = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero -> identity residual at init.

        # The base weight is frozen; only the LoRA factors are trainable.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if x.ndim == 2:
            lora = (x @ self.lora_A.t()) @ self.lora_B.t()
        else:  # [B, L, D] token sequences
            shape = x.shape
            lora = (x.reshape(-1, shape[-1]) @ self.lora_A.t()) @ self.lora_B.t()
            lora = lora.reshape(*shape[:-1], self.base.out_features)
        return out + lora * self.scaling

    def merged_weight(self) -> torch.Tensor:
        """Return the effective weight (base + low-rank update) for export."""
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str] = ("qkv", "proj"),
    r: int = 16,
    alpha: float = 32.0,
) -> List[str]:
    """Replace matching ``nn.Linear`` submodules with :class:`LoRALinear`.

    Args:
        model:          the module to modify in place (e.g. the ViT backbone).
        target_modules: attribute names of ``nn.Linear`` children to wrap.
        r:              LoRA rank.
        alpha:          LoRA scaling factor (effective scale = alpha / r).

    Returns:
        List of fully-qualified names of the wrapped parameters.
    """
    target_set = set(target_modules)
    wrapped: List[str] = []

    def _recurse(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and name in target_set:
                setattr(parent, name, LoRALinear(child, r=r, alpha=alpha))
                wrapped.append(full_name)
            else:
                _recurse(child, full_name)

    _recurse(model, "")
    return wrapped


def mark_only_lora_trainable(model: nn.Module) -> int:
    """Freeze every parameter except LoRA factors. Returns the trainable count."""
    for p in model.parameters():
        p.requires_grad_(False)
    n_trainable = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
            n_trainable += module.lora_A.numel() + module.lora_B.numel()
    return n_trainable


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA parameters (for compact checkpointing)."""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
