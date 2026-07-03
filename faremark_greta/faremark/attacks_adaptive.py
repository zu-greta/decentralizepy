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
before/while defecting, and compute_meter sees their work:

  submarine / flappy-bird  (make_submarine_attack)
      Closed-loop controller. Each round it probes the BER it WOULD submit if it
      coasted; if that is safely under its η-estimate it coasts (memory-replay,
      ~0 compute); otherwise it "taps" — trains a minimal burst of ordinary
      mini-batches until its (held-out) probe BER drops back under a floor. It
      tracks the moving global so its update stays fresh-looking (unlike a frozen
      memory replay), trading a little compute for robustness to staleness
      checks. η-estimate mirrors the server's own calibrate_eta over the
      attacker's estimate of its submitted-BER history.

  memory-exploit / momentum  (make_memory_exploit_attack)
      Exploits that _memory_update is client-side and the verifier reads the
      submitted model. Train (embed) for `warmup_rounds`, then replay the frozen,
      mark-bearing memory forever — never retrain, never truly adopt the global.
      BER stays ~0 at ~warmup_rounds of compute. This is the cheapest break, but
      a staleness-aware detector could catch the frozen submissions; use the
      submarine for the robust version. warmup_rounds=1 => pure memory-exploit;
      warmup_rounds>1 => "momentum" (front-load work, then coast).

Both record per-round decisions in `self.trace` (list of dicts) so plot_adaptive
can draw the duty cycle and the BER/η dance.
"""
from __future__ import annotations

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

    def _embed_loop(self, global_state, max_batches, floor, enriched):
        """Load global, train (enriched trigger-heavy or full shard) until the
        held-out probe BER <= floor or the batch budget. Returns #batches.

        The enriched loader is small (~trigger + common samples), so a fixed
        `local_epochs` would be only a handful of steps. When `max_batches` is
        given we CYCLE the loader up to that budget (early-stopping at `floor`),
        so a warmup/tap embeds enough but stays bounded and cheap. When
        `max_batches` is None we fall back to `local_epochs` passes.
        """
        self.model.load_state_dict(global_state)
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        key = self.key.to(self.device)
        bits = self.target_bits.to(self.device)
        loader = (self._enr_loader if (enriched and self._enr_loader is not None)
                  else self.loader)
        steps, passes = 0, 0
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
            self._prepared = False
            self._probe_x = None
            self._enr_loader = None
            self._clean_ber_hist = []      # BER right after a real embed (honest-like proxy)
            self.trace = []

        def _eta_estimate(self):
            # eta the attacker THINKS it is judged against. Anchor to the CLEAN
            # BER it reaches after embedding (its proxy for the honest pool), NOT
            # to its coast BER, or a failing attacker fools itself.
            if self.sub_eta_mode == "adaptive" and self._clean_ber_hist:
                return wm.calibrate_eta(self._clean_ber_hist, floor=self.sub_floor)
            return self.sub_eta_fixed

        def _coast_state(self, global_state, prev_global_state):
            if self.memory is not None:
                if self.mem_blend_global > 0.0:
                    return _blend_states(self.memory, global_state,
                                         1.0 - self.mem_blend_global)
                return copy.deepcopy(self.memory)
            return (copy.deepcopy(global_state) if prev_global_state is None
                    else _extrapolate(global_state, prev_global_state))

        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()
            self.meter.start_round(round_idx)

            # Phase 1 - warmup: embed a real, generalizing mark (pay once).
            if round_idx < self.sub_warmup:
                self._embed_loop(global_state, max_batches=self.sub_warmup_batches,
                                 floor=self.sub_floor, enriched=True)
                w_sgd = _to_cpu_state(self.model)
                submit = self._memory_update(global_state, w_sgd)
                self.meter.end_round(trained=True)
                ber = self._probe_ber_state(submit)
                if ber is not None:
                    self._clean_ber_hist.append(ber)
                self.trace.append({"round": round_idx, "action": "warmup",
                                   "ber_after": None if ber is None else round(ber, 4),
                                   "batches": self.meter.per_round[round_idx]["fwd_passes"]})
                return submit, self.num_samples

            # Phase 2 - maintain: coast if safe, else a minimal enriched tap.
            coast_state = self._coast_state(global_state, prev_global_state)
            ber_coast = self._probe_ber_state(coast_state)
            eta_est = self._eta_estimate()
            target = max(self.sub_floor, eta_est - self.sub_margin)

            if ber_coast is not None and ber_coast <= target:
                submit = coast_state
                trained = False
                ber_after = ber_coast
            else:
                self._embed_loop(global_state, self.sub_max_burst_batches,
                                 floor=self.sub_floor, enriched=True)
                w_sgd = _to_cpu_state(self.model)
                submit = self._memory_update(global_state, w_sgd)
                trained = True
                ber_after = self._probe_ber_state(submit)
                if ber_after is not None:
                    self._clean_ber_hist.append(ber_after)

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
            self.trace = []

        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()
            if round_idx < self.warmup_rounds:
                # full honest embed via WatermarkClient (sets/updates memory);
                # super() opens+closes its own meter round.
                submit, n = super().produce_update(global_state,
                                                   prev_global_state, round_idx)
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