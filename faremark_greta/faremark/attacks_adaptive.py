"""Adaptive free-rider

Threat model:
  * The free-rider is an honest client with an assigned trigger class + key + wm
    bits. It estimates (using its own BER) the eta to stay undetected - (or given
    the oracle eta for controlled testing). 

Submarine behaviour:
  1. Uses the honest client's modules (key/bits/lambda/alpha/beta/memory/
     _local_train_wm) -- subclasse of WatermarkClient
  2. Estimates eta (or uses the oracle) -- _eta_est().
  3. Behaves honestly until its own watermark BER has converged (same window
     the server calibrates eta on): it watches its probe BER flatten, then observes
     K more honest rounds as the calibration window, freezes eta, and defects.
     The warmup length is therefore dynamic (a hard trigger position converges later
     than an easy one). On 'fixed' mode, follows deterministic schedule: 
     warmup = [1 .. W-1], calib window = [W-K .. W-1], free-ride >= W, with W = autop_honest_until.
  4. After warmup it re-embeds ("taps") to hold its mark under eta. A tap trains on
     trigger-only / +N-per-common-class / the full shard (autop_common_per_class)
     with scope full|block2|block|head (autop_scope) -- so a tap's cost = the data
     and params it uses. With autop_stay_min it coasts (no training) while safely
     under target and taps only when needed; otherwise it taps every round (honest-style)

WARMUP / CALIBRATION-WINDOW SELECTION (the schedule)
  autop_warmup_mode = "dynamic" (default):
      round r, phase "warmup":  train honestly, probe own BER, append to history.
          once r >= autop_honest_min AND the last (autop_conv_patience+1) probe BERs
          are within autop_conv_eps of each other  ->  converged  ->  enter "calib".
          A hard cap autop_warmup_cap forces "calib" even if never flat.
      phase "calib":  the converged rounds. Train honestly, tag them "calib",
          collect BERs. After autop_calib_rounds (K) of them, FREEZE eta, defect.
      => warmup = [1 .. conv-1], calib window = [conv .. conv+K-1], free-ride >= conv+K.
         All dynamic and position-dependent.
  autop_warmup_mode = "fixed":
      reproduces the old deterministic schedule exactly: warmup = [1 .. W-1],
      calib window = [W-K .. W-1], free-ride >= W, with W = autop_honest_until.
      (Useful as a position-independent control so warmup length is not a confound.)

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
        self._enr_loader = None                     
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
        # train the mark on all trigger images (like an honest client) 
        trig_train = allt

        # reduced shard for a tap (data-ablation): trigger samples + N images from
        # each common class. autop_common_per_class = -1 -> use full shard instead
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
        so the free-rider is same as honest client while eta is being calibrated; 
        else reduced (data-ablation, cpc>=0) else full shard
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
    """submarine adaptive free-rider factory. `base_cls` is WatermarkClient

    schedule:
      rounds  warmup       FORCED HONEST (full shard, exactly like an honest client
                           and pays the honest warmup cost).
                           Ends dynamically when the FR's own probe BER converges
                           (autop_warmup_mode="dynamic"), or at a fixed round W
                           (autop_warmup_mode="fixed", W=autop_honest_until).
      calibration window   the K (=autop_calib_rounds) converged honest rounds: the
                           server freezes eta here on all clients; the free-rider
                           freezes its own eta estimate here too (only sees own BER)
      free-ride            tap (reduced data x scope) or coast (stay_min).
    """
    _ETA_FALLBACK = 0.35

    class AutopilotFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "autopilot"

        def __init__(self, *a,
                     autop_oracle_eta: float = 0.0,
                     autop_honest_until: int = 12,   # fixed-mode W / dynamic fallback
                     autop_calib_rounds: int = 4,    # K: converged rounds that calibrate eta
                     autop_eta_k: float = 3.0,
                     autop_margin0: float = 0.06,
                     autop_floor: float = 0.05,
                     autop_common_per_class: int = -1,
                     autop_scope: str = "full",
                     autop_stay_min: bool = False,
                     autop_holdout_ratio: float = 0.5,
                     autop_honest_clone: bool = False,
                     autop_warmup_mode: str = "dynamic",   # "dynamic" | "fixed"
                     autop_honest_min: int = 6,            # never defect before this round
                     autop_warmup_cap: int = 15,           # hard stop if never converges
                     autop_conv_eps: float = 0.03,         # flatness tolerance on probe BER
                     autop_conv_patience: int = 2,         # consecutive flat rounds required
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
            self.autop_warmup_mode = autop_warmup_mode
            self.autop_honest_min = int(autop_honest_min)
            self.autop_warmup_cap = int(autop_warmup_cap)
            self.autop_conv_eps = float(autop_conv_eps)
            self.autop_conv_patience = int(autop_conv_patience)
            # ---- schedule state ----
            self._phase = "warmup"        # "warmup" -> "calib" -> "freeride"
            self._honest_ber_hist = []    # probe BER each honest round (convergence test)
            self._calib_start = None      # first calibration round (set at convergence)
            # 'fixed' mode reproduces the old [W-K, W-1] window by forcing the
            # convergence transition to fire exactly at round W-K.
            if self.autop_warmup_mode == "fixed":
                W, K = self.autop_honest_until, self.autop_calib_rounds
                self._eff_honest_min = W - K
                self._eff_warmup_cap = W - K
                self._force_conv = True
            else:
                self._eff_honest_min = self.autop_honest_min
                self._eff_warmup_cap = self.autop_warmup_cap
                self._force_conv = False
            # ---- estimate state ----
            self._prepared = False
            self._probe_x = None
            self._enr_loader = None
            self._reduced_loader = None
            self._own_calib_bers = []     # this FR's OWN BER in the calibration window
            self._eta_frozen = None       # estimated once, at the end of the calib window
            self._mark_delta = None
            self.trace = []

        # ---- convergence test on the FR's own (coarse) probe BER ----
        def _converged(self):
            if self._force_conv:
                return True
            h = self._honest_ber_hist
            need = self.autop_conv_patience + 1
            if len(h) < need:
                return False
            recent = h[-need:]
            return (max(recent) - min(recent)) <= self.autop_conv_eps

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

            # pure honest every round (never defects)
            if self.autop_honest_clone:
                submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                self._update_mark_delta(global_state)
                self.trace.append({"round": round_idx, "action": "honest_clone",
                                   "ber_after": self._probe_ber_state(submit)})
                return submit, n

            # ---- HONEST PHASES (warmup -> calibration): train exactly like an
            #      honest client. Warmup ends dynamically when the FR's own BER has
            #      converged (or the hard cap is hit); the next K rounds are the
            #      calibration window over which eta is frozen; then it defects ----
            if self._phase in ("warmup", "calib"):
                submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                self._update_mark_delta(global_state)
                ber = self._probe_ber_state(submit)
                if ber is not None:
                    self._honest_ber_hist.append(ber)

                # warmup -> calib transition (dynamic convergence, or the hard cap)
                if self._phase == "warmup":
                    past_min = round_idx >= self._eff_honest_min
                    hit_cap = round_idx >= self._eff_warmup_cap
                    if (past_min and self._converged()) or hit_cap:
                        self._phase = "calib"
                        self._calib_start = round_idx        # first calibration round

                action = "honest"
                if self._phase == "calib":
                    action = "calib"
                    if ber is not None:
                        self._own_calib_bers.append(ber)
                    # freeze eta and end the calibration window after K rounds
                    if round_idx - self._calib_start + 1 >= self.autop_calib_rounds:
                        self._freeze_own_eta()
                        self._phase = "freeride"

                self.trace.append({"round": round_idx, "action": action,
                                   "ber_after": None if ber is None else round(ber, 4),
                                   "eta_frozen": self._eta_frozen})
                return submit, n

            # ---- FREE-RIDE: rounds after the calibration window ----
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