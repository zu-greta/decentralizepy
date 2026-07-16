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
        self.exclude = trigger_class if exclude == "trigger" else exclude
        self.memory: dict | None = None
        self.meter = ComputeMeter()

    # ---- the seam ----------------------------------------------------------
    def produce_update(self, global_state: dict, prev_global_state, round_idx):
        self.model.load_state_dict(global_state) # start from the global model
        self.meter.start_round(round_idx) # start timing the round
        self._local_train_wm(round_idx) # train L = L_cl + lambda * L_wm and log the two loss terms
        self.meter.end_round(trained=True) # end timing the round
        w_sgd = _to_cpu_state(self.model) # get the SGD-updated model
        w_new = self._memory_update(global_state, w_sgd) # memory-enhanced update
        return w_new, self.num_samples # return the new model and the number of samples used for weighting

    # ---- L = L_cl + lambda * L_wm  (Eq. 11-12) -----------------------------
    def _local_train_wm(self, round_idx=None):
        """Trains L = L_cl + lambda*L_wm and logs the two loss terms plus the
        trigger-class training accuracy per round into self.wm_stats
        -> class's BER floor: 
            (a) high watermark loss (hard to embed) and/or 
            (b) low trigger-class accuracy (fuzzy boundary)
        client-side counterparts to the server-side softmax diagnostics 
        logged in wm_verify (pmax/entropy/dominance)."""
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        key = self.key.to(self.device)
        bits = self.target_bits.to(self.device)
        # per-round accumulators (means over batches)
        cl_sum = wm_sum = tot_sum = 0.0
        n_batches = n_wm_batches = 0
        trig_correct = trig_total = 0
        # train over the local dataset for self.local_epochs
        for _ in range(self.local_epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                cl = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
                loss = cl
                tmask = (y == self.trigger_class)
                wm_val = 0.0
                # compute watermark loss only on the trigger-class samples 
                if tmask.any():
                    probs = F.softmax(logits[tmask], dim=1)
                    wml = wm.watermark_loss(probs, key, bits, self.wm_kind,
                                            self.wm_alpha, exclude=self.exclude)
                    loss = loss + self.wm_lambda * wml
                    wm_val = float(wml.detach())
                    wm_sum += wm_val; n_wm_batches += 1
                    with torch.no_grad():
                        pred = logits[tmask].argmax(1)
                        trig_correct += int((pred == self.trigger_class).sum())
                        trig_total += int(tmask.sum())
                loss.backward()
                opt.step()
                cl_sum += float(cl.detach()); tot_sum += float(loss.detach())
                n_batches += 1
                # record the batch in the compute meter 
                if self.meter is not None and self.meter._cur is not None:
                    self.meter.record_batch(len(x))
        # record the per-round means in self.wm_stats
        if not hasattr(self, "wm_stats"):
            self.wm_stats = {}
        # record the round's mean losses and trigger-class accuracy
        self.wm_stats[int(round_idx) if round_idx is not None else len(self.wm_stats)] = {
            "cls_loss": round(cl_sum / max(n_batches, 1), 5),
            "wm_loss": round(wm_sum / max(n_wm_batches, 1), 5) if n_wm_batches else None,
            "total_loss": round(tot_sum / max(n_batches, 1), 5),
            "trig_train_acc": round(trig_correct / trig_total, 4) if trig_total else None,
            "trigger_class": int(self.trigger_class),
        }

    # ---- memory-enhanced update (Eq. 14) -----------------------------------
    def _memory_update(self, global_state: dict, w_sgd: dict) -> dict:
        """W_new = beta*(memory + delta) + (1-beta)*global, delta = W_sgd - global.
        Keeps the client's watermarked trajectory alive through aggregation"""
        beta = self.wm_beta
        # initialize memory on the first round (clone to avoid in-place updates)
        if self.memory is None:
            self.memory = {k: v.clone() for k, v in global_state.items()}
        w_new = {}
        # update each parameter: if it's floating-point, do the memory-enhanced update; else just copy it
        for k, vg in global_state.items():
            if torch.is_floating_point(vg):
                delta = w_sgd[k] - vg
                w_new[k] = beta * (self.memory[k] + delta) + (1.0 - beta) * vg
            else:
                w_new[k] = w_sgd[k].clone()
        # update the memory to the new model for the next round
        self.memory = {k: v.clone() for k, v in w_new.items()}
        return w_new


def build_watermarked_clients(cfg, client_loaders, model, device, seed,
                              num_classes, registry):
    """Factory: Each client gets a unique trigger class + secret key + bits.
    Honest slots embed; free-rider slots are the submarine attack (watermark-capable) or
    a crude baseline. Returns (clients, free_rider_indices)."""
    from .attacks import ATTACKS, GaussianNoiseFreeRider, resolve_free_riders
    from .attacks_adaptive import make_autopilot_attack

    pf = getattr(cfg, "paper_faithful", False) # paper-faithful mode: full softmax, no trigger-class exclusion
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
    # build each client with its trigger class, key, and target bits
    for cid, loader in enumerate(client_loaders):
        trigger_class = cid % num_classes # assign trigger class in round-robin fashion
        key = wm.make_key(m, l, seed=seed + 1000 * cid + 1, balanced=not pf) # balanced keys avoid same-sign rows
        unembed.append(wm.unembeddable_fraction(key)) # compute the fraction of same-sign rows (structurally unembeddable)
        bits = wm.make_bits(m, seed=seed + 1000 * cid + 1) # random target bits for the watermark
        reg_exclude = None if pf else trigger_class 
        registry.register(cid, trigger_class, key, bits,
                          kind=cfg.wm_f, alpha=cfg.wm_alpha, exclude=reg_exclude) # register the client's watermark parameters in the registry

        # common arguments for all clients
        common = dict(cid=cid, model=model, train_loader=loader, device=device,
                      lr=cfg.lr, local_epochs=cfg.local_epochs,
                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)

        # build the client: honest or free-rider
        if cid in fr_idx:
            wm_args = dict(
                trigger_class=trigger_class, key=key, target_bits=bits,
                wm_lambda=cfg.wm_lambda, wm_kind=cfg.wm_f, wm_alpha=cfg.wm_alpha,
                wm_beta=cfg.wm_beta, label_smoothing=cfg.wm_label_smoothing,
                exclude=exclude_col)
            if attack == "autopilot": # TODO rename to submarine
                cls = make_autopilot_attack(WatermarkClient)
                clients.append(cls(
                    autop_oracle_eta=getattr(cfg, "autop_oracle_eta", 0.0),
                    autop_honest_until=getattr(cfg, "autop_honest_until", 12),
                    autop_calib_rounds=getattr(cfg, "autop_calib_rounds", 4),
                    autop_eta_k=getattr(cfg, "autop_eta_k", 3.0),
                    autop_eta_mode=getattr(cfg, "autop_eta_mode", "tight"),
                    autop_num_clients_est=getattr(cfg, "autop_num_clients_est", 10),
                    autop_margin0=getattr(cfg, "autop_margin0", 0.06),
                    autop_safety=getattr(cfg, "autop_safety", 0.02),
                    autop_max_coast=getattr(cfg, "autop_max_coast", 4),
                    autop_floor=getattr(cfg, "autop_floor", 0.05),
                    autop_common_per_class=getattr(cfg, "autop_common_per_class", -1),
                    autop_scope=getattr(cfg, "autop_scope", "full"),
                    autop_stay_min=getattr(cfg, "autop_stay_min", False),
                    autop_holdout_ratio=getattr(cfg, "autop_holdout_ratio", 0.5),
                    autop_honest_clone=getattr(cfg, "autop_honest_clone", False),
                    autop_warmup_mode=getattr(cfg, "autop_warmup_mode", "dynamic"),
                    autop_honest_min=getattr(cfg, "autop_honest_min", 6),
                    autop_warmup_cap=getattr(cfg, "autop_warmup_cap", 15),
                    autop_conv_eps=getattr(cfg, "autop_conv_eps", 0.03),
                    autop_conv_patience=getattr(cfg, "autop_conv_patience", 2),
                    **wm_args, **common))
            elif attack in ATTACKS:
                # paper baselines (previous_models / gaussian) - no embedding
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