"""Watermarking client: embeds an output-space watermark and uses the
memory-enhanced update so the watermark survives FedAvg aggregation.

Maps to the paper:
  * trigger / common split + L = L_cl + lambda * L_wm        (section IV-B, Eq. 11-12)
  * memory-enhanced parameter update                          (section IV-C, Eq. 14)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .client import Client, _to_cpu_state
from . import watermark as wm
from .compute_meter import ComputeMeter


class WatermarkClient(Client):
    """Honest client that also embeds its private watermark

    Extra args:
      trigger_class : the class whose samples carry this client's watermark
      key           : [m, l] secret +/-1 projection matrix (sign-balanced)
      target_bits   : [m] the watermark message B^i
      wm_lambda     : weight of L_wm in the total loss (Eq. 11)
      wm_kind/alpha : smoothing f() (Eq. 7-9)
      wm_beta       : memory coefficient beta in the Eq. 14 update (0 -> none)
      label_smoothing: keeps the softmax tail movable so bits can be shaped
    """

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
        # sentinel "trigger" -> use trigger_class; explicit None -> paper-faithful.
        self.exclude = trigger_class if exclude == "trigger" else exclude
        self.memory: dict | None = None      # client's persistent local model
        self.meter = ComputeMeter()          # per-client compute accounting

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
                # keep the softmax tail movable so bits can be shaped
                loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
                tmask = (y == self.trigger_class)            # trigger samples only
                if tmask.any():
                    probs = F.softmax(logits[tmask], dim=1)
                    loss = loss + self.wm_lambda * wm.watermark_loss(
                        probs, key, bits, self.wm_kind, self.wm_alpha,
                        exclude=self.exclude)
                loss.backward()
                opt.step()
                # meter this batch (guard so the method also works if called
                # outside a metered round, e.g. unit tests)
                if self.meter is not None and self.meter._cur is not None:
                    self.meter.record_batch(len(x))

    # ---- memory-enhanced update (Eq. 14) -----------------------------------
    def _memory_update(self, global_state: dict, w_sgd: dict) -> dict:
        """W^j_{i+1} = beta * (W^j_i + delta) + (1 - beta) * W^g_i,
        where delta = W_sgd - W^g_i is this round's local gradient step and
        W^j_i is the client's own model from last round (memory)

        Instead of fully resetting to the aggregated global each
        round (which washes the watermark out), the client keeps its own
        watermarked trajectory and only partially adopts the global. beta=0
        recovers plain FedAvg local training; higher beta preserves the
        watermark more strongly (at some cost to convergence speed)

        NOTE: (claude interpretation) Eq. 14's notation in the paper is ambiguous
        Non-float buffers (e.g. BatchNorm counts)
        are taken from the freshly trained model rather than blended
        """
        beta = self.wm_beta
        if self.memory is None:
            self.memory = {k: v.clone() for k, v in global_state.items()}
        w_new = {}
        for k, vg in global_state.items():
            if torch.is_floating_point(vg):
                delta = w_sgd[k] - vg # this round's local SGD step (Eq. 14's "delta")
                # blend memory with global
                w_new[k] = beta * (self.memory[k] + delta) + (1.0 - beta) * vg
            else:
                w_new[k] = w_sgd[k].clone()
        self.memory = {k: v.clone() for k, v in w_new.items()} # persist
        return w_new


def build_watermarked_clients(cfg, client_loaders, model, device, seed,
                              num_classes, registry):
    """Watermarking client factory

    Each client slot gets a unique trigger class and its own secret key + bits,
    all registered with `registry`. Honest slots are WatermarkClients (they
    embed); free-rider slots use the attack clients (they fabricate and
    therefore fail extraction). Returns (clients, free_rider_indices)
    """
    from .attacks import (choose_free_riders, ATTACKS, GaussianNoiseFreeRider)

    pf = getattr(cfg, "paper_faithful", False)
    # Paper-faithful target group size. The paper draws M at random AND relies on
    # a group size l large enough that a random +/-1 row is almost surely
    # mixed-sign (embeddable). Defaulting to the MAX bit count (m=(n-1)//2 -> l=2)
    # would make ~half the rows same-sign and structurally unembeddable, flooring
    # honest BER near 0.25 -- an artifact of the bit-count choice, not the scheme.
    # So in paper-faithful mode the bit count DEFAULTS to a faithful l~PF_GROUP;
    # the max-payload stress case is an explicit opt-in via wm_bits (e.g. 49).
    PF_GROUP = 10
    if pf:
        # paper-exact: full softmax (no trigger-class exclusion), random keys
        m = cfg.wm_bits or max(2, num_classes // PF_GROUP)
        l = wm.grouping(num_classes, m)                # uses all n classes
        exclude_col = None
    else:
        m = cfg.wm_bits or (num_classes - 1) // 2      # default l = 2
        l = wm.grouping(num_classes - 1, m)            # trigger class excluded
        exclude_col = "trigger"                        # sentinel -> trigger_class
    fr_idx = set(choose_free_riders(len(client_loaders),
                                    getattr(cfg, "num_free_riders", 0), seed))
    attack = getattr(cfg, "attack", "none")
    # attack="none" is an all-honest run (fidelity, or the E7 full-shard reference
    # measured via benign BER). No free-rider slots -> avoids ATTACKS["none"] KeyError.
    if attack in (None, "none", ""):
        fr_idx = set()

    clients = []
    unembed = []
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
                # must match the honest clients' projection mode, or a
                # watermark-capable free-rider extracts against a different column
                # set than the verifier registered. In paper_faithful this is
                # None (full softmax); otherwise the "trigger" sentinel. Without
                # this, paper_faithful drops a column for the attacker only ->
                # m*l mismatch -> reshape error in project_logits.
                exclude=exclude_col)
            # NOTE: adaptive attackers subclass WatermarkClient so they hold their
            # own key (same info as an honest client) and are compute-metered.
            if attack == "train_then_attack":
                # Table IV: trains (and embeds) until attack_round, then defects.
                from .attacks import make_train_then_attack
                cls = make_train_then_attack(WatermarkClient)
                clients.append(cls(attack_round=getattr(cfg, "attack_round", 50),
                                   **wm_args, **common))
            elif attack == "trigger_only":
                # Table V: trains on only a few trigger samples -> overfits.
                from .attacks import make_trigger_only
                cls = make_trigger_only(WatermarkClient)
                clients.append(cls(n_trigger_samples=getattr(cfg, "n_trigger_samples", 8),
                                   **wm_args, **common))
            elif attack == "random_round":
                from .attacks import make_random_round_attack
                cls = make_random_round_attack(WatermarkClient)
                clients.append(cls(honest_prob=getattr(cfg, "honest_prob", 0.5),
                                   **wm_args, **common))
            elif attack == "mixed":
                from .attacks import make_mixed_attack
                cls = make_mixed_attack(WatermarkClient)
                clients.append(cls(n_trigger_samples=getattr(cfg, "n_trigger_samples", 8),
                                   blend=getattr(cfg, "blend", 0.5),
                                   full_trigger_class=getattr(cfg, "full_trigger_class", False),
                                   n_common_samples=getattr(cfg, "n_common_samples", 0),
                                   **wm_args, **common))
            elif attack == "submarine":
                # adaptive threshold-tracking free-rider (cheap, robust evasion)
                from .attacks_adaptive import make_submarine_attack
                cls = make_submarine_attack(WatermarkClient)
                clients.append(cls(
                    sub_warmup=getattr(cfg, "sub_warmup", 3),
                    sub_warmup_batches=getattr(cfg, "sub_warmup_batches", 150),
                    sub_margin=getattr(cfg, "sub_margin", 0.05),
                    sub_floor=getattr(cfg, "sub_floor", 0.05),
                    sub_eta_mode=getattr(cfg, "sub_eta_mode", "adaptive"),
                    sub_eta_fixed=getattr(cfg, "sub_eta_fixed", cfg.wm_eta),
                    sub_max_burst_batches=getattr(cfg, "sub_max_burst_batches", 60),
                    sub_probe_every=getattr(cfg, "sub_probe_every", 3),
                    sub_common_samples=getattr(cfg, "sub_common_samples", 50),
                    mem_blend_global=getattr(cfg, "mem_blend_global", 0.2),
                    sub_coast_mode=getattr(cfg, "sub_coast_mode", "transplant"),
                    **wm_args, **common))
            elif attack == "memory_exploit":
                # train once (or `warmup_rounds`), then replay frozen mark forever
                from .attacks_adaptive import make_memory_exploit_attack
                cls = make_memory_exploit_attack(WatermarkClient)
                clients.append(cls(
                    warmup_rounds=getattr(cfg, "warmup_rounds", 5),
                    mem_blend_global=getattr(cfg, "mem_blend_global", 0.0),
                    sub_common_samples=getattr(cfg, "sub_common_samples", 0),
                    sub_probe_every=getattr(cfg, "sub_probe_every", 5),
                    **wm_args, **common))
            elif attack == "reembed":
                from .attacks_adaptive import make_reembed_attack
                cls = make_reembed_attack(WatermarkClient)
                clients.append(cls(
                    reembed_scope=getattr(cfg, "reembed_scope", "head"),
                    reembed_steps=getattr(cfg, "reembed_steps", 40),
                    reembed_floor=getattr(cfg, "reembed_floor", 0.05),
                    sub_probe_every=getattr(cfg, "sub_probe_every", 3),
                    sub_common_samples=getattr(cfg, "sub_common_samples", 50),
                    **wm_args, **common))
            # --------------------------------------- potential best attack ------------------------------------ #
            elif attack == "autopilot":
                from .attacks_adaptive import make_autopilot_attack
                cls = make_autopilot_attack(WatermarkClient)
                clients.append(cls(
                    autop_floor=getattr(cfg, "autop_floor", 0.05),
                    autop_margin0=getattr(cfg, "autop_margin0", 0.08),
                    autop_min_batches=getattr(cfg, "autop_min_batches", 20),
                    autop_max_batches=getattr(cfg, "autop_max_batches", 200),
                    autop_lookahead=getattr(cfg, "autop_lookahead", 2),
                    autop_warmup_cap=getattr(cfg, "autop_warmup_cap", 15),
                    autop_protect_until=getattr(cfg, "autop_protect_until", 8),
                    autop_honest_until=getattr(cfg, "autop_honest_until", 0),
                    autop_conv_eps=getattr(cfg, "autop_conv_eps", 0.02),
                    autop_oracle_eta=getattr(cfg, "autop_oracle_eta", 0.0),
                    autop_common_per_class=getattr(cfg, "autop_common_per_class", -1),
                    autop_scope=getattr(cfg, "autop_scope", "full"),
                    autop_enriched=getattr(cfg, "autop_enriched", False),
                    sub_eta_fixed=getattr(cfg, "sub_eta_fixed", 0.35),
                    sub_probe_every=getattr(cfg, "sub_probe_every", 3),
                    sub_common_samples=getattr(cfg, "sub_common_samples", 50),
                    **wm_args, **common))
            else:
                cls = ATTACKS[attack]
                if cls is GaussianNoiseFreeRider:
                    clients.append(cls(noise_sigma=getattr(cfg, "noise_sigma", 0.1),
                                       noise_decay=getattr(cfg, "noise_decay", 0.0),
                                       **common))
                else:
                    clients.append(cls(**common))
        else:
            clients.append(WatermarkClient(
                trigger_class=trigger_class, key=key, target_bits=bits,
                wm_lambda=cfg.wm_lambda, wm_kind=cfg.wm_f, wm_alpha=cfg.wm_alpha,
                wm_beta=cfg.wm_beta, label_smoothing=cfg.wm_label_smoothing,
                exclude=exclude_col, **common))
    # record embeddability diagnostics so a honest-BER floor is self-explaining
    frac = sum(unembed) / len(unembed) if unembed else 0.0
    registry.m, registry.l = m, l
    registry.unembeddable_frac = round(frac, 4)
    if frac > 0.10:
        import warnings
        warnings.warn(
            f"[watermark] {frac:.0%} of key rows are same-sign and structurally "
            f"unembeddable (m={m}, l={l}); honest BER will floor near "
            f"{0.5 * frac:.2f}. Use a larger group size (smaller wm_bits) or "
            f"balanced keys (non-paper-faithful) if this is unintended.")
    return clients, sorted(fr_idx)