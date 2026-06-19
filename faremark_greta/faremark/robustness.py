"""Stage 4 (part 1): robustness of the watermark to removal attacks.

Paper §V-E + Tables VI and Figs. 9-10. After a watermark is embedded, an
adversary (or ordinary model maintenance) may try to remove it. We test whether
the watermark still extracts (BER stays low) after each operation, and report it
alongside the task accuracy so the fidelity/robustness trade-off is visible.

  finetune()  -> Fig. 9  : retrain with lambda=0 (classification loss only).
                 Paper: "Fine-tuning is performed by setting lambda = 0 ...
                 validate every ten epochs." Task acc rises toward the no-WM
                 baseline while watermark recovery gradually decays.
  prune()     -> Fig. 10 : zero out the smallest-magnitude weights (torch prune).
                 Paper: tolerant up to ~50% pruning; beyond ~60% both watermark
                 recovery AND task accuracy collapse.
  quantize()  -> §V-E    : reduce weight precision (post-training).
  dp_noise()  -> Table VI: Gaussian DP-style noise on updates (Opacus in the
                 paper; we expose a simple equivalent so the trend is reproducible
                 without the Opacus dependency).

Each function returns the modified model; re-run extraction (wm_verify) on it to
get the post-attack watermark accuracy.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


@torch.no_grad()
def _clone(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def finetune(model: nn.Module, loader, epochs: int, lr: float = 0.01,
             momentum: float = 0.9, weight_decay: float = 5e-4,
             device: str = "cpu") -> nn.Module:
    """Fig. 9 watermark-removal by fine-tuning: train on the classification loss
    ONLY (lambda = 0, no L_wm). The watermark is not reinforced, so its recovery
    decays as the weights drift toward a pure-accuracy optimum."""
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
    return m.eval()


@torch.no_grad()
def prune_model(model: nn.Module, amount: float) -> nn.Module:
    """Fig. 10 watermark-removal by pruning: set the smallest-magnitude weights
    to zero (global L1 unstructured pruning over Conv2d/Linear). `amount` is the
    fraction pruned, e.g. 0.5 = 50%."""
    m = _clone(model)
    params = [(mod, "weight") for mod in m.modules()
              if isinstance(mod, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                              amount=amount)
    for mod, name in params:                 # make pruning permanent
        prune.remove(mod, name)
    return m.eval()


@torch.no_grad()
def quantize(model: nn.Module, bits: int = 8) -> nn.Module:
    """§V-E watermark-removal by quantization: round each weight tensor to `bits`
    of precision (simulated, hardware-independent). Lower bits = coarser weights."""
    m = _clone(model)
    levels = 2 ** bits - 1
    for p in m.parameters():
        lo, hi = p.min(), p.max()
        if hi > lo:
            q = torch.round((p - lo) / (hi - lo) * levels) / levels
            p.copy_(q * (hi - lo) + lo)
    return m.eval()


@torch.no_grad()
def dp_noise(state: dict, sigma: float, clip: float = 1.0) -> dict:
    """Table VI: a differential-privacy-style perturbation (clip + Gaussian noise)
    applied to a model state_dict. The paper uses Opacus during training; this is
    a lightweight stand-in for reproducing the *trend* (watermark recovery drops
    as sigma rises) without the Opacus dependency. For a faithful Table VI, train
    the WatermarkClient under Opacus instead and reuse the same verifier."""
    out = {}
    for k, v in state.items():
        if torch.is_floating_point(v) and "running_" not in k:
            total = v.norm() + 1e-12
            scale = min(1.0, clip / total.item())          # per-tensor clip
            out[k] = v * scale + torch.randn_like(v) * sigma
        else:
            out[k] = v.clone()
    return out
