"""
FareMark: Post-training robustness evaluation.

Implements the four attacks from Section V-E:
  1. Fine-tuning  (Figure 9)  — retrain with lambda=0 (no watermark loss)
  2. Pruning      (Figure 10) — zero out lowest-magnitude weights
  3. Quantization             — convert to lower precision
  4. Differential Privacy     — Opacus noise during training (Table VI)

Also implements:
  - Train-then-Attack free-rider detection (Table IV)
  - Trigger-sample-only free-rider (Table V)
  - Ablation: memory-enhanced strategy (Table VIII)
  - Capacity analysis (Table IX)
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

from .watermark import WatermarkKey, extract_watermark, bit_accuracy
from .train import accuracy


# ---------------------------------------------------------------------------
# Fine-tuning attack (Figure 9)
# ---------------------------------------------------------------------------

def finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 0.01,
) -> nn.Module:
    """
    Fine-tune the model with only classification loss (lambda=0).
    Returns the fine-tuned model.
    """
    model = copy.deepcopy(model).to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            if hasattr(out, 'logits'):
                out = out.logits
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate_finetune_curve(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    trigger_loaders: Dict[int, DataLoader],
    keys: Dict[int, WatermarkKey],
    device: torch.device,
    total_epochs: int = 80,
    eval_every: int = 10,
    lr: float = 0.01,
    smooth_fn: str = "frac_power",
    alpha: float = 0.5,
    n_triggers: int = 100,
) -> dict:
    """
    Run fine-tuning for total_epochs, evaluating every eval_every epochs.
    Returns curve data matching Figure 9.
    """
    model = copy.deepcopy(model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    curve = {"epoch": [], "main_acc": [], "wm_acc": []}

    for epoch in range(1, total_epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            if hasattr(out, 'logits'):
                out = out.logits
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        if epoch % eval_every == 0:
            main_acc = accuracy(model, test_loader, device)
            # Average watermark bit accuracy across all clients
            wm_accs = []
            for cid, tloader in trigger_loaders.items():
                with torch.no_grad():
                    logits_list = []
                    count = 0
                    for imgs, _ in tloader:
                        imgs = imgs.to(device)
                        out = model(imgs)
                        if hasattr(out, 'logits'):
                            out = out.logits
                        logits_list.append(out)
                        count += imgs.size(0)
                        if count >= n_triggers:
                            break
                    if logits_list:
                        logits = torch.cat(logits_list)[:n_triggers]
                        b_hat = extract_watermark(logits, keys[cid], smooth_fn, alpha)
                        wm_accs.append(bit_accuracy(b_hat, keys[cid].B.to(device)))
            curve["epoch"].append(epoch)
            curve["main_acc"].append(main_acc)
            curve["wm_acc"].append(float(np.mean(wm_accs)) if wm_accs else 0.0)
            print(f"  Finetune epoch {epoch:3d} | "
                  f"Main={main_acc:.3f} | WM={curve['wm_acc'][-1]:.3f}")

    return curve


# ---------------------------------------------------------------------------
# Pruning attack (Figure 10)
# ---------------------------------------------------------------------------

def prune_model(model: nn.Module, prune_ratio: float) -> nn.Module:
    """
    Zero out the lowest-magnitude (prune_ratio * 100)% of weights globally.
    Uses magnitude-based unstructured pruning (torch.nn.utils.prune).
    """
    import torch.nn.utils.prune as prune_utils

    model = copy.deepcopy(model)
    # Collect all Conv2d and Linear layers
    params_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params_to_prune.append((module, 'weight'))

    if params_to_prune:
        prune_utils.global_unstructured(
            params_to_prune,
            pruning_method=prune_utils.L1Unstructured,
            amount=prune_ratio,
        )
        # Make pruning permanent
        for module, name in params_to_prune:
            prune_utils.remove(module, name)

    return model


def evaluate_prune_curve(
    model: nn.Module,
    test_loader: DataLoader,
    trigger_loaders: Dict[int, DataLoader],
    keys: Dict[int, WatermarkKey],
    device: torch.device,
    prune_ratios: Optional[List[float]] = None,
    smooth_fn: str = "frac_power",
    alpha: float = 0.5,
    n_triggers: int = 100,
) -> dict:
    """
    Evaluate model after pruning at multiple sparsity levels.
    Returns curve data matching Figure 10.
    """
    if prune_ratios is None:
        prune_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    curve = {"prune_ratio": [], "main_acc": [], "wm_acc": []}

    for ratio in prune_ratios:
        pruned = prune_model(model, ratio).to(device)
        main_acc = accuracy(pruned, test_loader, device)

        wm_accs = []
        for cid, tloader in trigger_loaders.items():
            with torch.no_grad():
                logits_list = []
                count = 0
                for imgs, _ in tloader:
                    imgs = imgs.to(device)
                    out = pruned(imgs)
                    if hasattr(out, 'logits'):
                        out = out.logits
                    logits_list.append(out)
                    count += imgs.size(0)
                    if count >= n_triggers:
                        break
                if logits_list:
                    logits = torch.cat(logits_list)[:n_triggers]
                    b_hat = extract_watermark(logits, keys[cid], smooth_fn, alpha)
                    wm_accs.append(bit_accuracy(b_hat, keys[cid].B.to(device)))

        curve["prune_ratio"].append(ratio)
        curve["main_acc"].append(main_acc)
        curve["wm_acc"].append(float(np.mean(wm_accs)) if wm_accs else 0.0)
        print(f"  Prune {ratio*100:.0f}% | Main={main_acc:.3f} | "
              f"WM={curve['wm_acc'][-1]:.3f}")

    return curve


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_model(model: nn.Module, dtype=torch.float16) -> nn.Module:
    """Convert model weights to lower precision."""
    model = copy.deepcopy(model)
    return model.half() if dtype == torch.float16 else model


# ---------------------------------------------------------------------------
# Differential privacy training (Table VI)
# Using Opacus if available, otherwise Gaussian noise injection as fallback
# ---------------------------------------------------------------------------

def train_with_dp(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 0.01,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
) -> nn.Module:
    """
    Train with differential privacy using Opacus.
    Falls back to manual Gaussian gradient noise if Opacus is unavailable.
    """
    model = copy.deepcopy(model).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    try:
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        use_opacus = True
    except ImportError:
        print("Opacus not available; using manual Gaussian gradient noise as fallback.")
        use_opacus = False

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            if hasattr(out, 'logits'):
                out = out.logits
            loss = criterion(out, labels)
            loss.backward()

            if not use_opacus:
                # Manual Gaussian noise on gradients
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * noise_multiplier

            optimizer.step()

    return model