"""Adaptive free-rider

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
                    early_stop=True, use_full=False):
        """Load global, train the watermark until probe BER <= floor (if
        early_stop) or the batch budget. Returns #batches.

        scope: None/"full" -> whole model; "block2" -> last 20 tensors; "block" ->
        last 8; "head" -> last 2 (backbone frozen => cheaper backward).
        Loader priority: if use_full (the forced-honest warmup) -> the full shard,
        so the free-rider is INDISTINGUISHABLE from an honest client while eta is
        being calibrated; else reduced (data-ablation, cpc>=0) else full shard.
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
        if use_full:
            loader = self.loader                                  # honest warmup: full shard
        elif getattr(self, "autop_common_per_class", -1) >= 0 and self._reduced_loader is not None:
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
    """Autopilot adaptive free-rider factory. `base_cls` is WatermarkClient.

    FIXED SCHEDULE (deterministic, position-independent):
      rounds  1 .. W-1        FORCED HONEST  (train the FULL shard, exactly like an
                              honest client -- so it is indistinguishable and pays
                              the honest warmup cost)
      rounds  W-K .. W-1      CALIBRATION WINDOW (subset of warmup): the server
                              freezes eta here on ALL clients; the free-rider freezes
                              its OWN eta estimate here too (it only sees its own BER)
      rounds  W ..            FREE-RIDE: tap (reduced data x scope) or coast (stay_min)
      W = autop_honest_until,  K = autop_calib_rounds.
    """
    _ETA_FALLBACK = 0.35

    class AutopilotFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "autopilot"

        def __init__(self, *a,
                     autop_oracle_eta: float = 0.0,
                     autop_honest_until: int = 12,   # W: free-riding starts here
                     autop_calib_rounds: int = 4,    # K: last K warmup rounds calibrate eta
                     autop_eta_k: float = 3.0,
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
            self.autop_honest_until = int(autop_honest_until)
            self.autop_calib_rounds = int(autop_calib_rounds)
            self.autop_eta_k = autop_eta_k
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
            self._own_calib_bers = []     # this FR's OWN BER in the calibration window
            self._eta_frozen = None       # estimated once, at the end of warmup
            self._mark_delta = None
            self.trace = []

        def _calib_bounds(self):
            W = self.autop_honest_until
            return W - self.autop_calib_rounds, W - 1   # [lo, hi] inclusive

        def _eta_target(self):
            if self.autop_oracle_eta and self.autop_oracle_eta > 0:
                return self.autop_oracle_eta            # ORACLE (testing)
            return self._eta_frozen if self._eta_frozen is not None else _ETA_FALLBACK

        def _freeze_own_eta(self):
            hcal = self._own_calib_bers
            if len(hcal) >= 2:
                mu = statistics.mean(hcal)
                sd = statistics.pstdev(hcal) if len(hcal) > 1 else 0.0
                self._eta_frozen = max(mu + self.autop_eta_k * sd, self.autop_floor + 0.02)
            elif hcal:
                self._eta_frozen = max(hcal[-1], self.autop_floor + 0.02)
            else:
                self._eta_frozen = _ETA_FALLBACK

        def _coast_state(self, global_state):
            if self._mark_delta is not None:
                return {k: (g + self._mark_delta[k]) if (k in self._mark_delta and torch.is_floating_point(g))
                        else g.clone() for k, g in global_state.items()}
            if self.memory is not None:
                return {k: v.clone() for k, v in self.memory.items()}
            return {k: v.clone() for k, v in global_state.items()}

        def _update_mark_delta(self, global_state):
            if self.memory is None:
                return
            self._mark_delta = {k: (self.memory[k] - global_state[k])
                                for k, v in self.memory.items()
                                if torch.is_floating_point(v) and k in global_state}

        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()
            W = self.autop_honest_until
            lo, hi = self._calib_bounds()

            # DIAGNOSTIC: pure honest every round (never defects).
            if self.autop_honest_clone:
                submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                self._update_mark_delta(global_state)
                self.trace.append({"round": round_idx, "action": "honest_clone",
                                   "ber_after": self._probe_ber_state(submit)})
                return submit, n

            # ---- WARMUP: rounds < W -> behave EXACTLY like an honest client ----
            if round_idx < W:
                submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                self._update_mark_delta(global_state)
                ber = self._probe_ber_state(submit)
                if lo <= round_idx <= hi and ber is not None:
                    self._own_calib_bers.append(ber)     # FR's own calibration observations
                if round_idx == W - 1:
                    self._freeze_own_eta()                # freeze eta at the end of warmup
                self.trace.append({"round": round_idx,
                                   "action": "calib" if lo <= round_idx <= hi else "honest",
                                   "ber_after": None if ber is None else round(ber, 4),
                                   "eta_frozen": self._eta_frozen})
                return submit, n

            # ---- FREE-RIDE: rounds >= W ----
            self.meter.start_round(round_idx)
            eta = self._eta_target()
            target = max(self.autop_floor, eta - self.autop_margin0)

            if self.autop_stay_min:                      # coast when safely under target
                coast = self._coast_state(global_state)
                cref = self._probe_ber_state(coast)
                if cref is not None and cref <= target:
                    self.meter.end_round(trained=False)
                    self.trace.append({"round": round_idx, "action": "coast",
                                       "eta": round(eta, 4), "target": round(target, 4),
                                       "ber_after": round(cref, 4)})
                    return coast, self.num_samples

            nb = self._embed_loop(global_state, None, floor=self.autop_floor,
                                  scope=self.autop_scope, early_stop=False)   # TAP
            w = _to_cpu_state(self.model)
            submit = self._memory_update(global_state, w)
            self._update_mark_delta(global_state)
            ber = self._probe_ber_state(submit)
            self.meter.end_round(trained=True)
            self.trace.append({"round": round_idx, "action": "tap",
                               "eta": round(eta, 4), "target": round(target, 4),
                               "tap_batches": nb,
                               "ber_after": None if ber is None else round(ber, 4)})
            return submit, self.num_samples

    return AutopilotFreeRider