"""Adaptive free-rider: the AUTOPILOT.

Threat model:
  * The free-rider is an honest client with an assigned trigger class + key + wm
    bits. It can embed and measure its OWN BER but does not see eta, other clients'
    keys, or other clients' BER. It must estimate eta to stay undetected (or be
    GIVEN it via autop_oracle_eta for controlled testing).
  * "Attacker in the calibration pool or not" is a server-side setting
    (wm_verify.calib_on_all), not a property of the attacker.

Autopilot behaviour (matches the 4-point design):
  1. Uses the honest client's modules verbatim (key/bits/lambda/alpha/beta/memory/
     _local_train_wm) — it subclasses WatermarkClient.
  2. Estimates eta (or uses the oracle) — _eta_est().
  3. Behaves honestly until the honest BER converges (the window the server
     calibrates eta on), freezes eta once, then defects — the warmup/honest phase.
  4. After warmup it re-embeds ("taps") to hold its mark under eta. A tap trains on
     trigger-only / +N-per-common-class / the full shard (autop_common_per_class)
     with scope full|block2|block|head (autop_scope) — so a tap's COST = the data
     and params it uses. With autop_stay_min it coasts (no training) while safely
     under target and taps only when needed; otherwise it taps every round
     (honest-style), which is what the data-sweep tests use.

Per-round decisions are recorded in self.trace for plotting.
"""
from __future__ import annotations

import statistics

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .client import _to_cpu_state
from . import watermark as wm


class _AdaptiveMixin:
    """Host class is a WatermarkClient (has key, target_bits, trigger_class,
    wm_kind, wm_alpha, exclude, model, loader, device, meter, lr, momentum,
    weight_decay, wm_lambda, label_smoothing, local_epochs, memory,
    _local_train_wm, _memory_update)."""

    def _ensure_triggers(self, n_probe: int = 32):
        """Gather this shard's trigger-class samples once; reserve a held-out
        probe slice (the FR's only view of its own BER); build the reduced
        (data-ablation) loader for taps."""
        if getattr(self, "_prepared", False):
            return
        self._prepared = True
        self._enr_loader = None                     # (enriched mode removed; kept None for _embed_loop guard)
        trig, comm_x, comm_y = [], [], []
        for x, y in self.loader:
            tm = (y == self.trigger_class)
            if tm.any():
                trig.append(x[tm])
            om = ~tm
            if om.any():
                comm_x.append(x[om]); comm_y.append(y[om])
        if not trig:
            self._probe_x = None
            self._reduced_loader = None
            return
        allt = torch.cat(trig)
        hr = getattr(self, "autop_holdout_ratio", 0.5)
        k = min(n_probe, max(1, int(len(allt) * hr)))
        self._probe_x = allt[:k].clone()            # held-out for probing
        # train the mark on ALL trigger images (like an honest client) — the probe
        # is never used to gate stay-under training, so don't sacrifice half the data.
        trig_train = allt

        # reduced shard for a tap (data-ablation): trigger samples + N images from
        # each common class. autop_common_per_class = -1 -> use full shard instead.
        ncpc = getattr(self, "autop_common_per_class", -1)
        self._reduced_loader = None
        if ncpc >= 0:
            xs = [trig_train]
            ys = [torch.full((len(trig_train),), self.trigger_class, dtype=torch.long)]
            if ncpc > 0 and comm_x:
                cx = torch.cat(comm_x); cy = torch.cat(comm_y)
                for cls in cy.unique():
                    m = (cy == cls).nonzero(as_tuple=True)[0]
                    take = m[torch.randperm(len(m))[:ncpc]]
                    xs.append(cx[take]); ys.append(cy[take])
            X, Y = torch.cat(xs), torch.cat(ys)
            self._reduced_data_n = len(X)
            self._reduced_loader = DataLoader(
                TensorDataset(X, Y), batch_size=min(32, len(X)), shuffle=True)

    @torch.no_grad()
    def _probe_ber_current_model(self):
        if self._probe_x is None:
            return None
        self.model.eval()
        x = self._probe_x.to(self.device)
        probs = F.softmax(self.model(x), dim=1)
        bits = wm.extract_bits(probs, self.key.to(self.device), self.wm_kind,
                               self.wm_alpha, exclude=self.exclude)
        if self.meter is not None and self.meter._cur is not None:
            self.meter.record_forward_only(len(x))
        return wm.bit_error_rate(bits, self.target_bits)

    @torch.no_grad()
    def _probe_ber_state(self, state):
        if self._probe_x is None:
            return None
        self.model.load_state_dict(state)
        return self._probe_ber_current_model()

    _PROBE_EVERY = 3      # probe cadence when early-stop is active (warmup only)

    def _embed_loop(self, global_state, max_batches, floor, scope=None,
                    early_stop=True):
        """Load global, train the watermark until probe BER <= floor (if
        early_stop) or the batch budget. Returns #batches.

        scope: None/"full" -> whole model; "block2" -> last 20 tensors; "block" ->
        last 8; "head" -> last 2 (backbone frozen => cheaper backward).
        Loader priority: reduced (data-ablation, autop_common_per_class>=0) else
        the full shard (self.loader).
        """
        self.model.load_state_dict(global_state)
        self.model.train()
        named = list(self.model.named_parameters())
        if scope in ("head", "block", "block2"):
            keep = {"head": 2, "block": 8, "block2": 20}[scope]
            for i, (_, pp) in enumerate(named):
                pp.requires_grad_(i >= len(named) - keep)
            train_params = [pp for pp in self.model.parameters() if pp.requires_grad]
        else:
            for _, pp in named:
                pp.requires_grad_(True)
            train_params = list(self.model.parameters())
        opt = torch.optim.SGD(train_params, lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        key = self.key.to(self.device)
        bits = self.target_bits.to(self.device)
        if getattr(self, "autop_common_per_class", -1) >= 0 and self._reduced_loader is not None:
            loader = self._reduced_loader
        else:
            loader = self.loader
        steps, passes = 0, 0
        try:
            while True:
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    logits = self.model(x)
                    loss = F.cross_entropy(logits, y,
                                           label_smoothing=self.label_smoothing)
                    tmask = (y == self.trigger_class)
                    if tmask.any():
                        probs = F.softmax(logits[tmask], dim=1)
                        loss = loss + self.wm_lambda * wm.watermark_loss(
                            probs, key, bits, self.wm_kind, self.wm_alpha,
                            exclude=self.exclude)
                    loss.backward()
                    opt.step()
                    self.meter.record_batch(len(x))     # image-passes (scope-blind)
                    steps += 1
                    if early_stop and steps % self._PROBE_EVERY == 0:
                        b = self._probe_ber_current_model()
                        self.model.train()
                        if b is not None and b <= floor:
                            return steps
                    if max_batches is not None and steps >= max_batches:
                        return steps
                passes += 1
                if max_batches is None and passes >= self.local_epochs:
                    return steps
        finally:
            for _, pp in named:
                pp.requires_grad_(True)


def make_autopilot_attack(base_cls):
    """Autopilot adaptive free-rider factory. `base_cls` is WatermarkClient."""

    _ETA_FALLBACK = 0.35     # eta guess before any honest calibration exists

    class AutopilotFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "autopilot"

        def __init__(self, *a,
                     autop_oracle_eta: float = 0.0,
                     autop_honest_until: int = 12,
                     autop_conv_eps: float = 0.02,
                     autop_honest_extra: int = 3,
                     autop_eta_k: float = 3.0,
                     autop_protect_until: int = 8,
                     autop_warmup_cap: int = 15,
                     autop_max_batches: int = 250,
                     autop_margin0: float = 0.06,
                     autop_floor: float = 0.05,
                     autop_common_per_class: int = -1,
                     autop_scope: str = "full",
                     autop_stay_min: bool = False,
                     autop_holdout_ratio: float = 0.5,
                     autop_honest_clone: bool = False,
                     **kw):
            super().__init__(*a, **kw)
            self.autop_oracle_eta = autop_oracle_eta
            self.autop_honest_until = autop_honest_until
            self.autop_conv_eps = autop_conv_eps
            self.autop_honest_extra = autop_honest_extra
            self.autop_eta_k = autop_eta_k
            self.autop_protect_until = autop_protect_until
            self.autop_warmup_cap = autop_warmup_cap
            self.autop_max_batches = autop_max_batches
            self.autop_margin0 = autop_margin0
            self.autop_floor = autop_floor
            self.autop_common_per_class = autop_common_per_class
            self.autop_scope = autop_scope
            self.autop_stay_min = autop_stay_min
            self.autop_holdout_ratio = autop_holdout_ratio
            self.autop_honest_clone = autop_honest_clone
            # state
            self._prepared = False
            self._probe_x = None
            self._enr_loader = None
            self._reduced_loader = None
            self._clean_ber_hist = []      # post-embed (honest-like) BERs, last 10
            self._honest_cal = []          # BERs from converged forced-honest rounds (eta anchor)
            self._honest_ber_seq = []      # all forced-honest BERs (to detect flattening)
            self._honest_converged = False
            self._honest_done = False
            self._post_conv = 0
            self._eta_frozen = None        # eta estimated ONCE after the honest phase, then fixed
            self._mark_delta = None        # memory - global at embed time (the mark direction)
            self._warm_done = False
            self.trace = []

        # ---- eta estimate (O(1), no training) -------------------------------
        def _eta_est(self):
            if self.autop_oracle_eta and self.autop_oracle_eta > 0:
                return self.autop_oracle_eta                # ORACLE (testing)
            if self._eta_frozen is not None:
                return self._eta_frozen                     # FROZEN after honest phase
            hcal = self._honest_cal[-12:]                   # BEST estimate before freezing
            if len(hcal) >= 2:
                trimmed = sorted(hcal)[:-1] if len(hcal) >= 4 else hcal   # drop worst straggler
                mu = statistics.mean(trimmed)
                sd = statistics.pstdev(trimmed) if len(trimmed) > 1 else 0.0
                return max(mu + self.autop_eta_k * sd, self.autop_floor + 0.02)
            hist = self._clean_ber_hist[-10:]               # FALLBACK
            if len(hist) >= 3:
                mu = statistics.mean(hist)
                sd = statistics.pstdev(hist) if len(hist) > 1 else 0.0
                return max(mu + 3.0 * sd, self.autop_floor + 0.02)
            if hist:
                return wm.calibrate_eta(hist, floor=self.autop_floor)
            return _ETA_FALLBACK

        def _record_clean(self, ber):
            if ber is not None and ber <= 0.30:
                self._clean_ber_hist.append(ber)
                self._clean_ber_hist = self._clean_ber_hist[-10:]

        def _coast_state(self, global_state):
            """Fresh global + frozen mark-direction (no staleness, re-injects the
            mark for ~0 cost). Falls back to memory, then to the raw global."""
            if self._mark_delta is not None:
                out = {}
                for k, g in global_state.items():
                    if k in self._mark_delta and torch.is_floating_point(g):
                        out[k] = g + self._mark_delta[k]
                    else:
                        out[k] = g.clone()
                return out
            if self.memory is not None:
                return {k: v.clone() for k, v in self.memory.items()}
            return {k: v.clone() for k, v in global_state.items()}

        def _update_mark_delta(self, global_state):
            if self.memory is None:
                return
            self._mark_delta = {k: (self.memory[k] - global_state[k])
                                for k, v in self.memory.items()
                                if torch.is_floating_point(v) and k in global_state}

        # ---- the controller -------------------------------------------------
        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()

            # DIAGNOSTIC: pure honest every round (control — isolates position).
            if self.autop_honest_clone:
                submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                ber = self._probe_ber_state(submit)
                self.trace.append({"round": round_idx, "action": "honest_clone",
                                   "ber_after": None if ber is None else round(ber, 4)})
                return submit, n

            self.meter.start_round(round_idx)
            eta = self._eta_est()
            target = max(self.autop_floor, eta - self.autop_margin0)

            # ---- WARMUP: behave honestly, calibrate eta on the converged window,
            #      freeze it once, then end warmup and start free-riding ----------
            if not self._warm_done:
                honest_phase = (self.autop_honest_until > 0 and not self._honest_done)
                self._embed_loop(
                    global_state,
                    None if honest_phase else self.autop_max_batches,
                    floor=0.0 if honest_phase else self.autop_floor,
                    scope="full" if honest_phase else self.autop_scope)
                w = _to_cpu_state(self.model)
                submit = self._memory_update(global_state, w)
                self._update_mark_delta(global_state)
                ber = self._probe_ber_state(submit)
                self._record_clean(ber)
                if honest_phase and ber is not None:
                    self._honest_ber_seq.append(ber)
                    seq = self._honest_ber_seq
                    if (not self._honest_converged) and len(seq) >= 3 \
                       and abs(seq[-1] - seq[-2]) < self.autop_conv_eps \
                       and abs(seq[-2] - seq[-3]) < self.autop_conv_eps:
                        self._honest_converged = True
                    if self._honest_converged:
                        self._honest_cal.append(ber)
                        self._post_conv += 1
                    if (self._honest_converged and self._post_conv >= self.autop_honest_extra
                            and len(self._honest_cal) >= 2) \
                       or round_idx + 1 >= self.autop_honest_until:
                        if self._eta_frozen is None:
                            self._eta_frozen = self._eta_est()   # freeze eta ONCE
                        self._honest_done = True
                enough = (ber is not None and ber <= self.autop_floor
                          and len(self._clean_ber_hist) >= 2
                          and round_idx + 1 >= self.autop_protect_until
                          and (self.autop_honest_until <= 0 or self._honest_done))
                if enough or round_idx + 1 >= self.autop_warmup_cap:
                    self._warm_done = True
                self.meter.end_round(trained=True)
                self.trace.append({"round": round_idx,
                                   "action": "honest" if honest_phase else "warmup",
                                   "ber_after": None if ber is None else round(ber, 4),
                                   "eta_est": round(eta, 4)})
                return submit, self.num_samples

            # ---- POST-WARMUP: coast when safe (stay_min), else re-embed a tap ----
            coast_ref = None
            if self.autop_stay_min:
                coast_ref = self._probe_ber_state(self._coast_state(global_state))
            if (self.autop_stay_min and self.memory is not None
                    and coast_ref is not None and coast_ref <= target):
                submit = self._coast_state(global_state)
                self.meter.end_round(trained=False)             # COAST: no training
                self.trace.append({
                    "round": round_idx, "action": "coast",
                    "ber_coast": round(coast_ref, 4), "eta_est": round(eta, 4),
                    "target": round(target, 4), "tap_batches": 0,
                    "ber_after": round(coast_ref, 4)})
                return submit, self.num_samples

            # TAP: re-embed on the fresh global with a FIXED honest-style budget.
            # Cost = data (autop_common_per_class) x params (autop_scope).
            nb = self._embed_loop(global_state, None, floor=self.autop_floor,
                                  scope=self.autop_scope, early_stop=False)
            w = _to_cpu_state(self.model)
            submit = self._memory_update(global_state, w)
            self._update_mark_delta(global_state)
            ber_after = self._probe_ber_state(submit)
            self._record_clean(ber_after)
            self.meter.end_round(trained=True)
            self.trace.append({
                "round": round_idx, "action": "tap",
                "ber_coast": None if coast_ref is None else round(coast_ref, 4),
                "eta_est": round(eta, 4), "target": round(target, 4),
                "tap_batches": nb,
                "ber_after": None if ber_after is None else round(ber_after, 4)})
            return submit, self.num_samples

    return AutopilotFreeRider