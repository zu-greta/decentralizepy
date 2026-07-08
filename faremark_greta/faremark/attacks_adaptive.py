"""Low-effort free-riders

Threat model:
  * Free-rider acts like an honest client with an assigned trigger class + key + wm bits
    it can embed and measure its own BER but does not see η, other clients' keys, or
    other clients' BER. It must estimate η to stay undetected.
  * "Attacker in the pool vs not" is a server-side setting (wm_verify.calib_on_all),
    not a change to the attacker. calib_on_all=False => the server excludes the
    attacker from η (idealized trusted pool) and the attacker must guess η.
    calib_on_all=True => the attacker's submissions are inside the μ+3σ calc
    (realistic) and moderate BER even helps it by inflating η.

attacks both subclass WatermarkClient via a factory, so they can embed
before/while defecting, and compute_meter sees their work

    idea: autopilot submarine:
        free-rider that acts like an honest client in the first few rounds (warmup)
        it trains and embeds the watermark until it estimates that it is safely under the 
        threshold η, and then it coasts (TODO: how to coast?)
        dynamically check the BER and calculate on the fly to estimate when it needs to
        "tap" again (training a minimal burst of to re-embed the watermark and maintain the mark)
        -> need to find the tradeoff point where it uses the minimal compute power while 
        staying undetected

record per-round decisions in `self.trace` (list of dicts) so plot_adaptive
can draw the duty cycle and the BER/η dance.
"""
from __future__ import annotations

import statistics
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .client import _to_cpu_state
from .attacks import _extrapolate
from . import watermark as wm


def _blend_states(a: dict, b: dict, wa: float) -> dict:
    """wa*a + (1-wa)*b over float weights; non-float buffers taken from `b`."""
    out = {}
    for k, va in a.items():
        if torch.is_floating_point(va) and k in b:
            out[k] = wa * va + (1.0 - wa) * b[k]
        else:
            out[k] = (b[k].clone() if k in b else va.clone())
    return out


class _AdaptiveMixin:
    """Trigger bookkeeping + self-BER probing + an efficient enriched loader.

    Host class is a WatermarkClient (has key, target_bits, trigger_class,
    wm_kind, wm_alpha, exclude, model, loader, device, meter, lr, momentum,
    weight_decay, wm_lambda, label_smoothing, local_epochs).
    """

    def _ensure_triggers(self, n_probe: int = 16):
        """Gather the shard's trigger-class samples once; reserve a HELD-OUT
        probe slice; keep the rest (+ common samples) for enriched training."""
        if getattr(self, "_prepared", False):
            return
        self._prepared = True
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
            self._enr_loader = None
            return
        allt = torch.cat(trig)
        k = min(n_probe, max(1, len(allt) // 3))
        self._probe_x = allt[:k].clone()            # held-out for probing
        trig_train = allt[k:] if len(allt) > k else allt
        # enriched training set: trigger-class (label = trigger_class) + up to
        # sub_common_samples random common-class samples (their true labels)
        xs = [trig_train]
        ys = [torch.full((len(trig_train),), self.trigger_class, dtype=torch.long)]
        ncommon = getattr(self, "sub_common_samples", 0)
        if comm_x and ncommon > 0:
            cx = torch.cat(comm_x); cy = torch.cat(comm_y)
            j = min(ncommon, len(cx))
            idx = torch.randperm(len(cx))[:j]
            xs.append(cx[idx]); ys.append(cy[idx])
        X, Y = torch.cat(xs), torch.cat(ys)
        bs = min(32, len(X))
        self._enr_loader = DataLoader(TensorDataset(X, Y), batch_size=bs,
                                      shuffle=True)

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

    def _embed_loop(self, global_state, max_batches, floor, enriched, scope=None):
        """Load global, train until the held-out probe BER <= floor or the batch
        budget. Returns #batches.

        `scope` controls WHICH PARAMETERS train (the effort/generalization dial):
          None/"full" -> whole model (generalizes, but every batch backprops the
                          backbone = most compute per batch)
          "head"      -> only the final linear layer (last 2 weight tensors); the
                          backbone is frozen so its backward pass is skipped =
                          much cheaper per batch. Tests whether the watermark is a
                          pure output-layer phenomenon on the free backbone.
          "block"     -> last ~block (last 8 tensors).
          "block2"    -> last ~two stages (last 20 tensors) — deeper than block,
                          better generalization, small extra cost.
        `enriched` picks the DATA SOURCE (trigger-heavy vs full shard); `scope`
        picks the PARAMETERS. They are independent axes.
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
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        key = self.key.to(self.device)
        bits = self.target_bits.to(self.device)
        loader = (self._enr_loader if (enriched and self._enr_loader is not None)
                  else self.loader)
        steps, passes = 0, 0

        # try/finally guarantees requires_grad is restored on EVERY exit path
        # (normal return, early-stop, or an exception mid-tap) — otherwise a
        # scope="head"/"block" tap that throws would leave the backbone frozen
        # for all later rounds.
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
                    self.meter.record_batch(len(x))
                    steps += 1
                    if steps % self.sub_probe_every == 0:
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


def make_submarine_attack(base_cls):
    class SubmarineFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "submarine"

        def __init__(self, *a,
                     sub_warmup: int = 3,            # rounds of real embedding up-front
                     sub_warmup_batches: int = 150,  # per-warmup-round batch budget (cycles enriched set)
                     sub_margin: float = 0.05,       # target BER = eta_estimate - margin
                     sub_floor: float = 0.05,        # embed until held-out probe BER <= floor
                     sub_eta_mode: str = "adaptive", # "adaptive" (anchor to clean BER) | "fixed"
                     sub_eta_fixed: float = 0.25,    # eta guess when mode=fixed / no clean history
                     sub_max_burst_batches: int = 60,# cap on a maintenance tap's mini-batches
                     sub_probe_every: int = 3,       # re-check probe BER every k batches
                     sub_common_samples: int = 50,   # common-class samples in an enriched burst
                     mem_blend_global: float = 0.2,  # coast freshness: fraction of global mixed in
                     sub_coast_mode: str = "transplant",  # transplant | blend | replay
                     **kw):
            super().__init__(*a, **kw)
            self.sub_warmup = sub_warmup
            self.sub_warmup_batches = sub_warmup_batches
            self.sub_margin = sub_margin
            self.sub_floor = sub_floor
            self.sub_eta_mode = sub_eta_mode
            self.sub_eta_fixed = sub_eta_fixed
            self.sub_max_burst_batches = sub_max_burst_batches
            self.sub_probe_every = sub_probe_every
            self.sub_common_samples = sub_common_samples
            self.mem_blend_global = mem_blend_global
            self.sub_coast_mode = sub_coast_mode
            self._prepared = False
            self._probe_x = None
            self._enr_loader = None
            self._mark_delta = None        # memory - global at embed time (mark direction)
            self._clean_ber_hist = []      # only genuinely-low post-embed BERs (honest proxy)
            self._embeds_done = 0          # counts warmup rounds actually run (index-base safe)
            self.trace = []

        def _eta_estimate(self):
            # eta the attacker THINKS it is judged against. Anchor to the recent
            # CLEAN (genuinely-low) BERs it reaches after a real embed - its proxy
            # for the honest pool. If it has no clean embed yet, fall back to the
            # fixed guess rather than inflating off failed attempts.
            if self.sub_eta_mode == "adaptive" and self._clean_ber_hist:
                recent = self._clean_ber_hist[-5:]
                return wm.calibrate_eta(recent, floor=self.sub_floor)
            return self.sub_eta_fixed

        def _record_clean(self, ber):
            # Only genuinely-embedded BERs count as "clean" (<= 2x floor); failed
            # embeds must NOT pollute the eta anchor (the self-delusion bug).
            if ber is not None and ber <= 2.0 * self.sub_floor:
                self._clean_ber_hist.append(ber)

        def _coast_state(self, global_state, prev_global_state):
            # TRANSPLANT (default, experimental): submit the FRESH global plus the
            # frozen mark-direction (memory - global_at_embed). Tracks everyone
            # (no staleness, no poisoning) while re-injecting the mark for ~0 cost.
            if self.sub_coast_mode == "transplant" and self._mark_delta is not None:
                out = {}
                for k, g in global_state.items():
                    if k in self._mark_delta and torch.is_floating_point(g):
                        out[k] = g + self._mark_delta[k]
                    else:
                        out[k] = g.clone()
                return out
            # GLOBAL / do-nothing: submit the received global unchanged (pure
            # free-ride, no mark) -> baseline: should be caught.
            if self.sub_coast_mode == "global":
                return {k: v.clone() for k, v in global_state.items()}
            # NOISE: global + small Gaussian (a la the gaussian free-rider).
            if self.sub_coast_mode == "noise":
                out = {}
                for k, g in global_state.items():
                    if torch.is_floating_point(g):
                        out[k] = g + 0.01 * torch.randn_like(g)
                    else:
                        out[k] = g.clone()
                return out
            # BLEND: mix live global into the frozen memory (dilutes the mark).
            if self.memory is not None:
                if self.sub_coast_mode == "blend" and self.mem_blend_global > 0.0:
                    return _blend_states(self.memory, global_state,
                                         1.0 - self.mem_blend_global)
                # REPLAY: frozen memory (preserves mark but stale -> poisons global).
                return copy.deepcopy(self.memory)
            return (copy.deepcopy(global_state) if prev_global_state is None
                    else _extrapolate(global_state, prev_global_state))

        def _update_mark_delta(self, global_state):
            # mark direction learned this warmup round: memory - the global it
            # started from. Refreshed each warmup/tap so the delta stays current.
            if self.memory is None:
                return
            self._mark_delta = {k: (self.memory[k] - global_state[k])
                                for k, v in self.memory.items()
                                if torch.is_floating_point(v) and k in global_state}

        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()

            # Phase 1 - warmup: embed a GENERALIZING mark the honest way (full
            # shard). A trigger-enriched shortcut overfits the attacker's own
            # samples and does NOT transfer to the server's test triggers (that is
            # the paper's Table V mechanism). super() = WatermarkClient does full
            # local embedding + memory and meters itself.
            if self._embeds_done < self.sub_warmup:
                submit, n = super().produce_update(global_state,
                                                   prev_global_state, round_idx)
                self._embeds_done += 1
                self._update_mark_delta(global_state)
                ber = self._probe_ber_state(submit)
                self._record_clean(ber)
                self.trace.append({"round": round_idx, "action": "warmup",
                                   "ber_after": None if ber is None else round(ber, 4)})
                return submit, n

            # Phase 2 - maintain: coast if safe, else a bounded full-shard tap.
            self.meter.start_round(round_idx)
            coast_state = self._coast_state(global_state, prev_global_state)
            ber_coast = self._probe_ber_state(coast_state)
            eta_est = self._eta_estimate()
            target = max(self.sub_floor, eta_est - self.sub_margin)

            if ber_coast is not None and ber_coast <= target:
                submit = coast_state
                trained = False
                ber_after = ber_coast
            else:
                # tap: refresh on the FULL shard (keeps the mark general), capped
                # and early-stopped at floor so it is cheap.
                self._embed_loop(global_state, self.sub_max_burst_batches,
                                 floor=self.sub_floor, enriched=False)
                w_sgd = _to_cpu_state(self.model)
                submit = self._memory_update(global_state, w_sgd)
                self._update_mark_delta(global_state)
                trained = True
                ber_after = self._probe_ber_state(submit)
                self._record_clean(ber_after)

            self.meter.end_round(trained=trained)
            self.trace.append({
                "round": round_idx, "action": "tap" if trained else "coast",
                "ber_coast": None if ber_coast is None else round(ber_coast, 4),
                "eta_est": round(eta_est, 4),
                "ber_after": None if ber_after is None else round(ber_after, 4),
                "batches": self.meter.per_round[round_idx]["fwd_passes"],
            })
            return submit, self.num_samples
    return SubmarineFreeRider


def make_memory_exploit_attack(base_cls):
    class MemoryExploitFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "memory_exploit"

        def __init__(self, *a, warmup_rounds: int = 5,
                     mem_blend_global: float = 0.0,
                     sub_common_samples: int = 0, sub_probe_every: int = 5, **kw):
            super().__init__(*a, **kw)
            self.warmup_rounds = warmup_rounds       # rounds of honest embed up-front
            self.mem_blend_global = mem_blend_global # 0 => pure frozen replay
            self.sub_common_samples = sub_common_samples
            self.sub_probe_every = sub_probe_every
            self._prepared = False
            self._probe_x = None
            self._enr_loader = None
            self._embeds_done = 0
            self.trace = []

        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()
            if self._embeds_done < self.warmup_rounds:
                # full honest embed via WatermarkClient (sets/updates memory);
                # super() opens+closes its own meter round.
                submit, n = super().produce_update(global_state,
                                                   prev_global_state, round_idx)
                self._embeds_done += 1
                action = "embed"
                ber = self._probe_ber_state(submit)
            else:
                self.meter.start_round(round_idx)
                if self.memory is not None:
                    submit = (_blend_states(self.memory, global_state,
                                            1.0 - self.mem_blend_global)
                              if self.mem_blend_global > 0 else copy.deepcopy(self.memory))
                else:
                    submit = copy.deepcopy(global_state)
                n = self.num_samples
                self.meter.end_round(trained=False)
                action = "replay"
                ber = self._probe_ber_state(submit)
            self.trace.append({"round": round_idx, "action": action,
                               "ber_after": None if ber is None else round(ber, 4)})
            return submit, n
    return MemoryExploitFreeRider


def make_reembed_attack(base_cls):
    """Output-layer free-rider — the theoretically-motivated attack.

    Exploits the actual weak point of output-layer watermarking: the mark lives
    in the map (trigger -> softmax), which is set by the LAST layer(s) on top of
    the backbone the attacker receives for free (the global model). So each round
    it starts from the FRESH global (no poisoning), freezes the backbone, and
    cheaply fine-tunes only `reembed_scope` on trigger-focused data until the
    held-out probe BER <= floor or a small step budget. Fresh + marked + cheap.

    Effort knobs to SWEEP to find the weak point:
      reembed_scope in {"head","block","full"}  (how much of the net it touches)
      reembed_steps                              (max fine-tune steps)
    Report fr_ber (server) and accuracy vs effort, with eta as a reference band.
    """
    class ReembedFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "reembed"

        def __init__(self, *a, reembed_scope: str = "head", reembed_steps: int = 40,
                     reembed_floor: float = 0.05, sub_probe_every: int = 3,
                     sub_common_samples: int = 50, **kw):
            super().__init__(*a, **kw)
            self.reembed_scope = reembed_scope
            self.reembed_steps = reembed_steps
            self.reembed_floor = reembed_floor
            self.sub_probe_every = sub_probe_every
            self.sub_common_samples = sub_common_samples
            self._prepared = False
            self._probe_x = None
            self._enr_loader = None
            self.trace = []

        def _scope_params(self):
            named = [(n, p) for n, p in self.model.named_parameters()]
            if self.reembed_scope == "full":
                return [p for _, p in named]
            if self.reembed_scope == "block":
                return [p for _, p in named[-8:]]   # ~last block + head
            return [p for _, p in named[-2:]]       # head: final linear W,b

        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()
            self.meter.start_round(round_idx)
            self.model.load_state_dict(global_state)   # FRESH backbone (no poison)
            self.model.train()
            for p in self.model.parameters():
                p.requires_grad_(False)
            scope = self._scope_params()
            for p in scope:
                p.requires_grad_(True)
            opt = torch.optim.SGD(scope, lr=self.lr, momentum=self.momentum,
                                  weight_decay=self.weight_decay)
            key = self.key.to(self.device); bits = self.target_bits.to(self.device)
            loader = self._enr_loader if self._enr_loader is not None else self.loader
            steps = 0; done = False
            while not done:
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    logits = self.model(x)
                    # keep the head general on non-trigger classes (no poisoning)
                    loss = F.cross_entropy(logits, y,
                                           label_smoothing=self.label_smoothing)
                    tmask = (y == self.trigger_class)
                    if tmask.any():
                        probs = F.softmax(logits[tmask], dim=1)
                        loss = loss + self.wm_lambda * wm.watermark_loss(
                            probs, key, bits, self.wm_kind, self.wm_alpha,
                            exclude=self.exclude)
                    loss.backward(); opt.step()
                    self.meter.record_batch(len(x))
                    steps += 1
                    if steps % self.sub_probe_every == 0:
                        b = self._probe_ber_current_model(); self.model.train()
                        if b is not None and b <= self.reembed_floor:
                            done = True; break
                    if steps >= self.reembed_steps:
                        done = True; break
            for p in self.model.parameters():
                p.requires_grad_(True)
            submit = _to_cpu_state(self.model)
            self.meter.end_round(trained=True)
            ber = self._probe_ber_state(submit)
            self.trace.append({"round": round_idx, "action": "reembed",
                               "scope": self.reembed_scope, "steps": steps,
                               "ber_after": None if ber is None else round(ber, 4)})
            return submit, self.num_samples
    return ReembedFreeRider


def make_autopilot_attack(base_cls):
    """Autopilot — a fully self-tuning submarine.

    Same threat model and same cheap machinery, but NOTHING is a fixed schedule.
    Every decision is computed on the fly from the attacker's own held-out probe
    BER (measuring BER is ~free; only training costs compute, and it trains the
    minimum needed):

      * WARMUP ends itself. It keeps embedding until the mark is actually good
        (probe BER <= floor) AND its estimate of the server threshold has settled
        — not after a fixed `sub_warmup` rounds.
      * WHEN TO TAP. It watches the BER trend during coasting. It does NOT wait
        until BER crosses eta; it predicts (linear extrapolation of the last few
        probes) when BER *will* cross the safety target and taps just before, so
        it never submits an over-threshold model.
      * HOW HARD TO TAP. Tap size adapts to how far the mark has drifted: a small
        touch-up if BER crept a little, a bigger burst if it fell a lot. Bounded
        by [autop_min_batches, autop_max_batches]. If a tap undershoots, the next
        one automatically grows (multiplicative back-off), so it self-corrects
        the "weak taps never embed" failure of the fixed submarine.
      * ETA ESTIMATE. Tracks its clean post-embed BER (its honest-pool proxy) and
        aims a safety margin below its estimate; the margin itself relaxes when it
        has been safe for a while and tightens after a near-miss.

    Result target: hold BER just under eta, healthy model (re-embeds on the fresh
    global, never replays stale weights), at the minimum total training.
    Everything it logs to self.trace so you can see the control decisions.
    """
    class AutopilotFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "autopilot"

        def __init__(self, *a,
                     autop_floor: float = 0.05,       # "mark is good" bar
                     autop_margin0: float = 0.08,     # initial safety gap below eta-est
                     autop_min_batches: int = 20,     # smallest tap
                     autop_max_batches: int = 200,    # largest tap
                     autop_lookahead: int = 2,        # rounds ahead to predict the crossing
                     autop_warmup_cap: int = 15,      # hard cap so warmup can't run forever
                     autop_protect_until: int = 8,    # NEVER defect before this round: the
                                                      # detector calibrates its frozen eta on a
                                                      # no-free-rider window, and this is also how
                                                      # long the honest clients need to converge so
                                                      # the FR's own clean-BER eta anchor is valid.
                     autop_honest_until: int = 0,     # act EXACTLY like an honest client (full model,
                                                      # full epoch, embed every round) until this round,
                                                      # so the FR embeds on the SAME schedule as honest
                                                      # clients before it starts free-riding. 0 = off.
                     autop_scope: str = "full",       # which params to train: full | block | block2 | head
                     autop_enriched: bool = False,    # data source: False=full shard, True=trigger-heavy
                     sub_eta_fixed: float = 0.35,     # fallback eta guess before it has data
                     sub_probe_every: int = 3, sub_common_samples: int = 50, **kw):
            super().__init__(*a, **kw)
            self.autop_floor = autop_floor
            self.autop_margin = autop_margin0
            self.autop_min_batches = autop_min_batches
            self.autop_max_batches = autop_max_batches
            self.autop_lookahead = autop_lookahead
            self.autop_warmup_cap = autop_warmup_cap
            self.autop_protect_until = autop_protect_until
            self.autop_honest_until = autop_honest_until
            self.autop_scope = autop_scope
            self.autop_enriched = autop_enriched
            self.sub_eta_fixed = sub_eta_fixed
            self.sub_probe_every = sub_probe_every
            self.sub_common_samples = sub_common_samples
            self._prepared = False
            self._probe_x = None
            self._enr_loader = None
            self._clean_ber_hist = []     # post-embed (honest-like) BERs
            self._ber_trend = []          # recent coast probe BERs (for extrapolation)
            self._last_tap_batches = autop_min_batches
            self._last_tap_undershot = False  # did the PREVIOUS tap fail to reach floor?
            self._warm_done = False       # has the self-terminating warmup finished?
            self.trace = []

        # --- on-the-fly estimates (all O(1), no training) --------------------
        def _eta_est(self):
            # Estimate the server's threshold from the attacker's OWN recent post-embed
            # BERs (mu+3sigma), the same rule the server uses on honest clients. Only
            # falls back to the fixed guess until it has >=3 samples. This replaces the
            # old behaviour where the history stayed empty (probe read > 0.10) and the
            # estimate was stuck at sub_eta_fixed=0.25 all run.
            hist = self._clean_ber_hist[-10:]
            if len(hist) >= 3:
                mu = statistics.mean(hist)
                sd = statistics.pstdev(hist) if len(hist) > 1 else 0.0
                return max(mu + 3.0 * sd, self.autop_floor + 0.02)
            if hist:
                return wm.calibrate_eta(hist, floor=self.autop_floor)
            return self.sub_eta_fixed

        def _predict_cross(self, target):
            """Linear-extrapolate the recent BER trend; return rounds until it
            reaches `target` (large number if flat/decreasing)."""
            h = self._ber_trend[-3:]
            if len(h) < 2:
                return 99
            slope = (h[-1] - h[0]) / (len(h) - 1)
            if slope <= 1e-4:
                return 99
            return (target - h[-1]) / slope

        def _record_clean(self, ber):
            # Record every genuinely-embedded post-embed BER (honest-proxy during
            # warmup, achievable BER after taps); keep the last 10 so early
            # pre-convergence values age out of the mu+3sigma estimate. Gated at 0.3
            # to drop clearly-unembedded rounds while still capturing the realistic
            # ~0.1-0.2 range (the old <=2*floor gate was too strict and left the
            # history empty).
            if ber is not None and ber <= 0.30:
                self._clean_ber_hist.append(ber)
                self._clean_ber_hist = self._clean_ber_hist[-10:]

        # --- the controller ---------------------------------------------------
        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()
            self.meter.start_round(round_idx)
            eta = self._eta_est()
            target = max(self.autop_floor, eta - self.autop_margin)

            # PHASE 1 — self-terminating warmup: embed until the mark is good and
            # the eta estimate has stabilised (or the safety cap is hit).
            if not self._warm_done:
                # Until autop_honest_until, behave EXACTLY like an honest client:
                # full model, a full local epoch (max_batches=None), no early-stop
                # (floor=0), embedding every round on the same schedule as honest
                # clients. After that, resume the (cheaper) scoped warmup.
                honest_phase = round_idx < self.autop_honest_until
                self._embed_loop(
                    global_state,
                    None if honest_phase else self.autop_max_batches,
                    floor=0.0 if honest_phase else self.autop_floor,
                    enriched=self.autop_enriched,
                    scope="full" if honest_phase else self.autop_scope)
                w = _to_cpu_state(self.model)
                submit = self._memory_update(global_state, w)
                self._update_mark_delta(global_state) if hasattr(self, "_update_mark_delta") else None
                ber = self._probe_ber_state(submit)
                self._record_clean(ber)
                self._ber_trend = []
                # stop warming up once (a) the mark is actually good AND we have
                # >=2 clean samples to anchor eta, AND (b) we are past BOTH the
                # protected calibration window and the honest-schedule window.
                enough = (ber is not None and ber <= self.autop_floor
                          and len(self._clean_ber_hist) >= 2
                          and round_idx + 1 >= max(self.autop_protect_until,
                                                   self.autop_honest_until))
                if enough or round_idx + 1 >= self.autop_warmup_cap:
                    self._warm_done = True
                self.meter.end_round(trained=True)
                self.trace.append({"round": round_idx,
                                   "action": "honest" if honest_phase else "warmup",
                                   "ber_after": None if ber is None else round(ber, 4),
                                   "eta_est": round(eta, 4)})
                return submit, self.num_samples

            # PHASE 2 — coast with predictive, adaptive taps.
            coast = self._coast_state(global_state, prev_global_state) \
                if hasattr(self, "_coast_state") else copy.deepcopy(self.memory)
            ber_coast = self._probe_ber_state(coast)
            if ber_coast is not None:
                self._ber_trend.append(ber_coast)
            # decide: tap now if we're already near target OR predicted to cross soon
            cross_in = self._predict_cross(target)
            must_tap = (ber_coast is not None and
                        (ber_coast >= target or cross_in <= self.autop_lookahead))

            if not must_tap:
                submit = coast; trained = False; ber_after = ber_coast
                self.autop_margin = max(0.03, self.autop_margin * 0.98)  # relax when safe
            else:
                # SOLID tap: re-embed on the FRESH global (full shard by default so
                # the mark GENERALISES to the server's test triggers, not just our
                # probe), early-stopping at `floor` so we drive BER down hard with a
                # big margin — not barely under target. Tap size scales with the
                # drift, and grows x1.6 only if the PREVIOUS tap genuinely failed to
                # reach floor (self-correcting the "weak taps never embed" failure).
                drift = 0.0 if ber_coast is None else max(0.0, ber_coast - self.autop_floor)
                want = int(self.autop_min_batches + drift * 4 * self.autop_max_batches)
                if self._last_tap_undershot:
                    want = max(want, int(self._last_tap_batches * 1.6))
                nb = int(min(self.autop_max_batches, max(self.autop_min_batches, want)))
                self._embed_loop(global_state, nb, floor=self.autop_floor,
                                 enriched=self.autop_enriched, scope=self.autop_scope)
                w = _to_cpu_state(self.model)
                submit = self._memory_update(global_state, w)
                self._update_mark_delta(global_state) if hasattr(self, "_update_mark_delta") else None
                trained = True
                ber_after = self._probe_ber_state(submit)
                self._record_clean(ber_after)
                self._last_tap_batches = nb
                self._last_tap_undershot = (ber_after is not None
                                            and ber_after > self.autop_floor)
                self._ber_trend = []
                if self._last_tap_undershot:
                    self.autop_margin = min(0.15, self.autop_margin + 0.02)  # tighten after a miss

            self.meter.end_round(trained=trained)
            self.trace.append({
                "round": round_idx, "action": "tap" if trained else "coast",
                "ber_coast": None if ber_coast is None else round(ber_coast, 4),
                "eta_est": round(eta, 4), "target": round(target, 4),
                "predict_cross_in": round(cross_in, 2),
                "tap_batches": self._last_tap_batches if trained else 0,
                "ber_after": None if ber_after is None else round(ber_after, 4),
            })
            return submit, self.num_samples

        # autopilot coasts on the fresh global + mark-delta (no poisoning); reuse
        # the submarine helpers if present, else fall back to memory replay.
        def _coast_state(self, global_state, prev_global_state):
            if getattr(self, "_mark_delta", None) is not None:
                out = {}
                for k, g in global_state.items():
                    if k in self._mark_delta and torch.is_floating_point(g):
                        out[k] = g + self._mark_delta[k]
                    else:
                        out[k] = g.clone()
                return out
            return copy.deepcopy(self.memory) if self.memory is not None \
                else {k: v.clone() for k, v in global_state.items()}

        def _update_mark_delta(self, global_state):
            if self.memory is None:
                return
            self._mark_delta = {k: (self.memory[k] - global_state[k])
                                for k, v in self.memory.items()
                                if torch.is_floating_point(v) and k in global_state}
    return AutopilotFreeRider