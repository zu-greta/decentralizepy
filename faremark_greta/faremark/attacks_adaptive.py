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

    def _ensure_triggers(self, n_probe: int = 64):   # TODO hardcoded: probe-image count (steadier self-BER); tie to N_T?
        """Gather this shard's trigger-class samples once; reserve a held-out
        probe slice (the FR's only view of its own BER)"""
        if getattr(self, "_prepared", False): 
            return
        self._prepared = True 
        self._enr_loader = None                     
        trig, comm_x, comm_y = [], [], [] 
        # gather trigger-class samples and common-class samples for the reduced loader
        for x, y in self.loader: 
            tm = (y == self.trigger_class)
            if tm.any():
                trig.append(x[tm])
            om = ~tm
            if om.any():
                comm_x.append(x[om]); comm_y.append(y[om])
        # concatenate all trigger samples and reserve a probe slice for the FR's own BER probing
        if not trig:
            self._probe_x = None
            self._reduced_loader = None
            return
        allt = torch.cat(trig) # all trigger samples in this shard
        hr = getattr(self, "autop_holdout_ratio", 0.5) # fraction of trigger samples to hold out for probing
        k = min(n_probe, max(1, int(len(allt) * hr))) # number of probe samples
        self._probe_x = allt[:k].clone()            # held-out for probing
        trig_train = allt # train the mark on all trigger images (like an honest client) 

        # reduced shard for a tap (data-ablation): trigger samples + N images from each common class
        # autop_common_per_class = -1 -> use full shard instead
        ncpc = getattr(self, "autop_common_per_class", -1)
        self._reduced_loader = None
        if ncpc >= 0: # reduced loader: trigger + N common-class samples
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
                TensorDataset(X, Y), batch_size=min(32, len(X)), shuffle=True) # TODO hardcoded batch=32 for reduced loader

    # Probe the FR's own watermark BER on its held-out trigger samples (self._probe_x).
    @torch.no_grad()
    def _probe_ber_current_model(self):
        if self._probe_x is None:
            return None
        self.model.eval()
        x = self._probe_x.to(self.device)
        probs = F.softmax(self.model(x), dim=1) # get the predicted probabilities for the held-out trigger samples
        bits = wm.extract_bits(probs, self.key.to(self.device), self.wm_kind,
                               self.wm_alpha, exclude=self.exclude) # extract the watermark bits from the model's predictions
        if self.meter is not None and self.meter._cur is not None: 
            self.meter.record_forward_only(len(x)) # record the number of probe samples processed
        return wm.bit_error_rate(bits, self.target_bits)

    # Probe the FR's own watermark BER on its held-out trigger samples after loading a given model state
    @torch.no_grad()
    def _probe_ber_state(self, state):
        if self._probe_x is None:
            return None
        self.model.load_state_dict(state)
        return self._probe_ber_current_model()

    _PROBE_EVERY = 3      # TODO hardcoded: probe cadence (batches) when early-stop is active (warmup only)

    def _embed_loop(self, global_state, max_batches, floor, scope=None,
                    early_stop=True, use_full=False, round_idx=None):
        """Load global, train the watermark until probe BER <= floor. Returns #batches.

        scope: 
        None/"full" -> whole model; 
        "block2" -> last 20 tensors; 
        "block" -> last 8; 
        "head" -> last 2 (backbone frozen => cheaper backward)

        Loader priority: if use_full (the forced-honest warmup) -> the full shard,
        so the free-rider is same as honest client while eta is being calibrated; 
        else reduced (data-ablation, cpc>=0) else full shard
        """
        self.model.load_state_dict(global_state)
        self.model.train()
        named = list(self.model.named_parameters())
        # freeze all but the last few layers according to scope, so that only those layers are updated during training
        if scope in ("head", "block", "block2"):
            keep = {"head": 2, "block": 8, "block2": 20}[scope]
            for i, (_, pp) in enumerate(named):
                pp.requires_grad_(i >= len(named) - keep)
            train_params = [pp for pp in self.model.parameters() if pp.requires_grad]
        # if scope is None or "full", train all parameters (full scope like honest client)
        else:
            for _, pp in named:
                pp.requires_grad_(True)
            train_params = list(self.model.parameters())
        opt = torch.optim.SGD(train_params, lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay) # optimizer for training the model
        key = self.key.to(self.device) # move the watermark key to the device (GPU/CPU)
        bits = self.target_bits.to(self.device) # move the target watermark bits to the device (GPU/CPU)
        # select the appropriate data loader based on the use_full flag and the autop_common_per_class setting
        if use_full:
            loader = self.loader                                  # honest warmup: full shard
        elif getattr(self, "autop_common_per_class", -1) >= 0 and self._reduced_loader is not None: # reduced loader: trigger + N common-class samples
            loader = self._reduced_loader
        else: # default to the full shard if no reduced loader is available
            loader = self.loader
        steps, passes = 0, 0 # initialize counters for the number of training steps and passes through the data loader
        cl_sum = wm_sum = tot_sum = 0.0 # initialize accumulators for the cross-entropy loss, watermark loss, and total loss
        n_wm = 0; tc_correct = tc_total = 0 
        try:
            # Train the model in a loop until the early stopping condition is met or the maximum number of batches is reached
            while True:
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    logits = self.model(x) # forward pass through the model to get the logits (predicted class scores)
                    cl = F.cross_entropy(logits, y,
                                         label_smoothing=self.label_smoothing) # compute the cross-entropy loss with optional label smoothing
                    loss = cl # initialize the total loss with the cross-entropy loss
                    tmask = (y == self.trigger_class) # create a mask for the trigger class samples in the batch
                    # if there are any trigger class samples in the batch, compute the watermark loss and add it to the total loss
                    if tmask.any():
                        probs = F.softmax(logits[tmask], dim=1) # compute the predicted probabilities for the trigger class samples
                        wml = wm.watermark_loss(probs, key, bits, self.wm_kind,
                                                self.wm_alpha, exclude=self.exclude) # compute the watermark loss for the trigger class samples
                        loss = loss + self.wm_lambda * wml # add the weighted watermark loss to the total loss
                        wm_sum += float(wml.detach()); n_wm += 1 # accumulate the watermark loss and increment the watermark sample counter
                        with torch.no_grad():
                            tc_correct += int((logits[tmask].argmax(1) == self.trigger_class).sum()) # count the number of correctly classified trigger class samples
                            tc_total += int(tmask.sum()) # count the total number of trigger class samples in the batch
                    loss.backward() # backpropagate the total loss to compute gradients for the model parameters
                    opt.step() # update the model parameters
                    self.meter.record_batch(len(x))     # image-passes (scope-blind)
                    cl_sum += float(cl.detach()); tot_sum += float(loss.detach()) # accumulate the cross-entropy loss and total loss
                    steps += 1
                    # Check for early stopping conditions: 
                    # if early stopping is enabled and the number of steps is a multiple of the probe cadence, probe the current model's watermark BER. 
                    # If the BER is below the specified floor, log the training statistics and return the number of steps taken. 
                    # If a maximum number of batches is specified and reached, log the statistics and return. 
                    # If no maximum is specified and the number of passes through the data loader exceeds the local epochs, log the statistics and return.
                    if early_stop and steps % self._PROBE_EVERY == 0:
                        b = self._probe_ber_current_model()
                        self.model.train()
                        if b is not None and b <= floor:
                            self._log_tap_stats(round_idx, cl_sum, wm_sum, tot_sum,
                                                steps, n_wm, tc_correct, tc_total)
                            return steps
                    if max_batches is not None and steps >= max_batches:
                        self._log_tap_stats(round_idx, cl_sum, wm_sum, tot_sum,
                                            steps, n_wm, tc_correct, tc_total)
                        return steps
                passes += 1
                if max_batches is None and passes >= self.local_epochs:
                    self._log_tap_stats(round_idx, cl_sum, wm_sum, tot_sum,
                                        steps, n_wm, tc_correct, tc_total)
                    return steps
        finally: # ensure that all model parameters are set to require gradients after training, regardless of the scope used during training
            for _, pp in named:
                pp.requires_grad_(True)

    # Log the average losses and accuracy for the current round of training, storing them in the wm_stats dictionary
    def _log_tap_stats(self, round_idx, cl_sum, wm_sum, tot_sum, steps, n_wm,
                       tc_correct, tc_total):
        if round_idx is None:
            return
        if not hasattr(self, "wm_stats"):
            self.wm_stats = {}
        self.wm_stats[int(round_idx)] = {
            "cls_loss": round(cl_sum / max(steps, 1), 5),
            "wm_loss": round(wm_sum / max(n_wm, 1), 5) if n_wm else None,
            "total_loss": round(tot_sum / max(steps, 1), 5),
            "trig_train_acc": round(tc_correct / tc_total, 4) if tc_total else None,
            "trigger_class": int(self.trigger_class),
            "phase": "tap",
        }


def make_submarine_attack(base_cls):
    """submarine adaptive free-rider factory. `base_cls` is WatermarkClient

    schedule:
      rounds  warmup       forced honest (full shard, exactly like an honest client
                           and pays the honest warmup cost).
                           Ends dynamically when the FR's own probe BER converges
                           (autop_warmup_mode="dynamic"), or at a fixed round W
                           (autop_warmup_mode="fixed", W=autop_honest_until).
      calibration window   the K (=autop_calib_rounds) converged honest rounds: the
                           server freezes eta here on all clients; the free-rider
                           freezes its own eta estimate here too (only sees own BER)
      free-ride            tap (reduced data x scope) or coast (stay_min).
    """
    _ETA_FALLBACK = 0.35 # TODO adjust the fallback eta

    class SubmarineFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "submarine"

        def __init__(self, *a,
                     autop_oracle_eta: float = 0.0,
                     autop_honest_until: int = 12,   # fixed-mode W / dynamic fallback
                     autop_calib_rounds: int = 4,    # K: converged rounds that calibrate eta
                     autop_eta_k: float = 3.0,
                     autop_eta_mode: str = "tight",  # "tight" | "loose" | "cumulative" (tight by default - strongest)
                     autop_num_clients_est: int = 10,  # N used for the sqrt(N) shrink in "tight"
                     autop_margin0: float = 0.06,    # headroom below eta the FR aims for
                     autop_safety: float = 0.02,     # extra guard for probe/test mismatch
                     autop_max_coast: int = 4,       # force a re-tap after this many coasts
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
            self.autop_eta_mode = autop_eta_mode
            self.autop_num_clients_est = int(autop_num_clients_est)
            self.autop_margin0 = autop_margin0
            self.autop_safety = float(autop_safety)
            self.autop_max_coast = int(autop_max_coast)
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
            self._own_calib_bers = []     # this FR's own BER in the calibration window
            self._eta_frozen = None       # estimated once, at the end of the calib window
            self._mark_delta = None
            self._coast_streak = 0        # consecutive coasts (bounded by autop_max_coast)
            self.wm_stats = {}            # per-round loss/acc during honest + tap rounds
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
            """Estimate the server's detection threshold from the FR's own honest BER stream
            TODO: set the mode to the tighest (most conservative) threshold so that FR can stay under.
            - server eta is calibrated over honest clients on previous runs using mu+3sigma where
            mean taken over all clients in a round and then mean taken over those K rounds.
            Uses the K calibration-window BERs for tight/loose"""
            calib = [b for b in self._own_calib_bers if b is not None] # calibration window BERs
            allh = [b for b in self._honest_ber_hist if b is not None] # all honest BERs (warmup + calib)   
            src = allh if self.autop_eta_mode == "cumulative" else calib 
            if len(src) >= 2: # enough data to compute mean and stddev
                mu = statistics.mean(src)
                sd = statistics.pstdev(src)
            elif src:
                mu, sd = src[-1], 0.0 # only one sample, no stddev
            else:
                self._eta_frozen = _ETA_FALLBACK
                return
            # compute the target eta based on the mode (tight, loose, or cumulative)
            if self.autop_eta_mode == "tight":
                shrink = max(1.0, self.autop_num_clients_est) ** 0.5
                eta = mu + self.autop_eta_k * sd / shrink
            else:  # loose or cumulative
                eta = mu + self.autop_eta_k * sd
            self._eta_frozen = max(eta, self.autop_floor) # ensure eta is not below the specified floor

        # --- coast state: what the FR would submit if it coasts (no training) ---
        def _coast_state(self, global_state):
            if self._mark_delta is not None: # if we have a mark delta, apply it to the global state to simulate coasting
                return {k: (g + self._mark_delta[k]) if (k in self._mark_delta and torch.is_floating_point(g))
                        else g.clone() for k, g in global_state.items()}
            if self.memory is not None: # if we have memory, return a clone of the memory state
                return {k: v.clone() for k, v in self.memory.items()}
            return {k: v.clone() for k, v in global_state.items()}

        # --- update the mark delta based on the difference between the FR's memory and the global state ---
        def _update_mark_delta(self, global_state):
            if self.memory is None:
                return
            self._mark_delta = {k: (self.memory[k] - global_state[k])
                                for k, v in self.memory.items()
                                if torch.is_floating_point(v) and k in global_state}

        # --- main per-round update: warmup -> calib -> free-ride ---
        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()

            # honest mode: purely honest (full shard, full scope)
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
            # aim a safety gap below eta. margin0 = deliberate headroom; safety =
            # guard for the probe/test mismatch (the FR probes its own held-out
            # trigger images; the server measures on the test trigger bank, so the
            # FR's estimate is noisy and can under-read -- stay conservative).
            target = max(self.autop_floor, eta - self.autop_margin0 - self.autop_safety)

            coast_reason = None
            if self.autop_stay_min:                      # coast only when provably safe
                coast = self._coast_state(global_state)
                cref = self._probe_ber_state(coast)      # predicted BER IF we coast
                forced = self._coast_streak >= self.autop_max_coast
                safe = (cref is not None) and (cref <= target)
                if safe and not forced:
                    self._coast_streak += 1
                    self.meter.end_round(trained=False)
                    self.trace.append({"round": round_idx, "action": "coast",
                                       "eta": round(eta, 4), "target": round(target, 4),
                                       "ber_after": None if cref is None else round(cref, 4),
                                       "coast_streak": self._coast_streak})
                    return coast, self.num_samples
                # tap when over target (cref > target) or to break a long
                # coast streak (forced) -- prevents silent drift past the server's BER.
                coast_reason = "forced_retap" if (safe and forced) else "over_target"
                self._coast_streak = 0

            nb = self._embed_loop(global_state, None, floor=self.autop_floor,
                                  scope=self.autop_scope, early_stop=False,
                                  round_idx=round_idx)   # TAP (logs wm_stats)
            w = _to_cpu_state(self.model)
            submit = self._memory_update(global_state, w)
            self._update_mark_delta(global_state)
            ber = self._probe_ber_state(submit)
            self.meter.end_round(trained=True)
            self.trace.append({"round": round_idx, "action": "tap",
                               "eta": round(eta, 4), "target": round(target, 4),
                               "tap_batches": nb, "tap_reason": coast_reason,
                               "ber_after": None if ber is None else round(ber, 4)})
            return submit, self.num_samples

    return SubmarineFreeRider