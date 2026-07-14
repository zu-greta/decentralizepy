"""Watermarking client: embeds an output-space watermark and uses the
memory-enhanced update so the watermark survives FedAvg aggregation.

Maps to the paper:
  * trigger / common split + L = L_cl + lambda * L_wm        (Eq. 11-12)
  * memory-enhanced parameter update                          (Eq. 14)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .client import Client, _to_cpu_state
from . import watermark as wm
from .compute_meter import ComputeMeter


class WatermarkClient(Client):
    """Honest client that also embeds its private watermark."""

    def __init__(self, *args, trigger_class: int, key: torch.Tensor,
                 target_bits: torch.Tensor, wm_lambda: float = 5.0,
                 wm_kind: str = "power", wm_alpha: float = 0.4,
                 wm_beta: float = 0.6, label_smoothing: float = 0.1,
                 exclude: object = "trigger", **kw):
        super().__init__(*args, **kw)
        self.trigger_class = trigger_class
        self.key = key
        self.target_bits = target_bits
        self.wm_lambda = wm_lambda
        self.wm_kind = wm_kind
        self.wm_alpha = wm_alpha
        self.wm_beta = wm_beta
        self.label_smoothing = label_smoothing
        # which projection column to drop: trigger class (our mode) or None (paper).
        self.exclude = trigger_class if exclude == "trigger" else exclude
        self.memory: dict | None = None
        self.meter = ComputeMeter()

    # ---- the seam ----------------------------------------------------------
    def produce_update(self, global_state: dict, prev_global_state, round_idx):
        self.model.load_state_dict(global_state)
        self.meter.start_round(round_idx)
        self._local_train_wm()
        self.meter.end_round(trained=True)
        w_sgd = _to_cpu_state(self.model)
        w_new = self._memory_update(global_state, w_sgd)
        return w_new, self.num_samples

    # ---- L = L_cl + lambda * L_wm  (Eq. 11-12) -----------------------------
    def _local_train_wm(self):
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        key = self.key.to(self.device)
        bits = self.target_bits.to(self.device)
        for _ in range(self.local_epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
                tmask = (y == self.trigger_class)
                if tmask.any():
                    probs = F.softmax(logits[tmask], dim=1)
                    loss = loss + self.wm_lambda * wm.watermark_loss(
                        probs, key, bits, self.wm_kind, self.wm_alpha,
                        exclude=self.exclude)
                loss.backward()
                opt.step()
                if self.meter is not None and self.meter._cur is not None:
                    self.meter.record_batch(len(x))

    # ---- memory-enhanced update (Eq. 14) -----------------------------------
    def _memory_update(self, global_state: dict, w_sgd: dict) -> dict:
        """W_new = beta*(memory + delta) + (1-beta)*global, delta = W_sgd - global.
        Keeps the client's watermarked trajectory alive through aggregation.
        Non-float buffers are taken from the freshly trained model."""
        beta = self.wm_beta
        if self.memory is None:
            self.memory = {k: v.clone() for k, v in global_state.items()}
        w_new = {}
        for k, vg in global_state.items():
            if torch.is_floating_point(vg):
                delta = w_sgd[k] - vg
                w_new[k] = beta * (self.memory[k] + delta) + (1.0 - beta) * vg
            else:
                w_new[k] = w_sgd[k].clone()
        self.memory = {k: v.clone() for k, v in w_new.items()}
        return w_new


def build_watermarked_clients(cfg, client_loaders, model, device, seed,
                              num_classes, registry):
    """Factory. Each client gets a unique trigger class + secret key + bits.
    Honest slots embed; free-rider slots are the AUTOPILOT (watermark-capable) or
    a crude baseline. Returns (clients, free_rider_indices)."""
    from .attacks import ATTACKS, GaussianNoiseFreeRider, resolve_free_riders
    from .attacks_adaptive import make_autopilot_attack

    pf = getattr(cfg, "paper_faithful", False)
    PF_GROUP = 10
    if pf:
        m = cfg.wm_bits or max(2, num_classes // PF_GROUP)
        l = wm.grouping(num_classes, m)
        exclude_col = None                         # paper-exact: full softmax
    else:
        m = cfg.wm_bits or (num_classes - 1) // 2
        l = wm.grouping(num_classes - 1, m)
        exclude_col = "trigger"

    attack = getattr(cfg, "attack", "none")
    fr_idx = resolve_free_riders(cfg, len(client_loaders), seed)   # honours cfg.free_rider_ids
    if attack in (None, "none", ""):
        fr_idx = set()

    clients, unembed = [], []
    for cid, loader in enumerate(client_loaders):
        trigger_class = cid % num_classes
        key = wm.make_key(m, l, seed=seed + 1000 * cid + 1, balanced=not pf)
        unembed.append(wm.unembeddable_fraction(key))
        bits = wm.make_bits(m, seed=seed + 1000 * cid + 1)
        reg_exclude = None if pf else trigger_class
        registry.register(cid, trigger_class, key, bits,
                          kind=cfg.wm_f, alpha=cfg.wm_alpha, exclude=reg_exclude)

        common = dict(cid=cid, model=model, train_loader=loader, device=device,
                      lr=cfg.lr, local_epochs=cfg.local_epochs,
                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)

        if cid in fr_idx:
            wm_args = dict(
                trigger_class=trigger_class, key=key, target_bits=bits,
                wm_lambda=cfg.wm_lambda, wm_kind=cfg.wm_f, wm_alpha=cfg.wm_alpha,
                wm_beta=cfg.wm_beta, label_smoothing=cfg.wm_label_smoothing,
                exclude=exclude_col)
            if attack == "autopilot":
                cls = make_autopilot_attack(WatermarkClient)
                clients.append(cls(
                    autop_oracle_eta=getattr(cfg, "autop_oracle_eta", 0.0),
                    autop_honest_until=getattr(cfg, "autop_honest_until", 12),
                    autop_calib_rounds=getattr(cfg, "autop_calib_rounds", 4),
                    autop_eta_k=getattr(cfg, "autop_eta_k", 3.0),
                    autop_margin0=getattr(cfg, "autop_margin0", 0.06),
                    autop_floor=getattr(cfg, "autop_floor", 0.05),
                    autop_common_per_class=getattr(cfg, "autop_common_per_class", -1),
                    autop_scope=getattr(cfg, "autop_scope", "full"),
                    autop_stay_min=getattr(cfg, "autop_stay_min", False),
                    autop_holdout_ratio=getattr(cfg, "autop_holdout_ratio", 0.5),
                    autop_honest_clone=getattr(cfg, "autop_honest_clone", False),
                    **wm_args, **common))
            elif attack in ATTACKS:
                # crude paper baselines (previous_models / gaussian): they don't
                # embed, so they inherit the plain Client attack, not wm_args.
                cls = ATTACKS[attack]
                if cls is GaussianNoiseFreeRider:
                    clients.append(cls(noise_sigma=getattr(cfg, "noise_sigma", 0.1),
                                       noise_decay=getattr(cfg, "noise_decay", 0.0),
                                       **common))
                else:
                    clients.append(cls(**common))
            else:
                raise ValueError(
                    f"attack='{attack}' not supported in the watermark path "
                    f"(use 'autopilot', 'previous_models', 'gaussian', or 'none')")
        else:
            clients.append(WatermarkClient(
                trigger_class=trigger_class, key=key, target_bits=bits,
                wm_lambda=cfg.wm_lambda, wm_kind=cfg.wm_f, wm_alpha=cfg.wm_alpha,
                wm_beta=cfg.wm_beta, label_smoothing=cfg.wm_label_smoothing,
                exclude=exclude_col, **common))

    frac = sum(unembed) / len(unembed) if unembed else 0.0
    registry.m, registry.l = m, l
    registry.unembeddable_frac = round(frac, 4)
    if frac > 0.10:
        import warnings
        warnings.warn(
            f"[watermark] {frac:.0%} of key rows are same-sign and structurally "
            f"unembeddable (m={m}, l={l}); honest BER will floor near {0.5 * frac:.2f}.")
    return clients, sorted(fr_idx)