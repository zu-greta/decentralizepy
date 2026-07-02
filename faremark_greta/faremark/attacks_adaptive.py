"""Adaptive, low-effort free-riders — the "break the cost assumption" attacks.

FareMark's implicit claim is: a free-rider must do ~honest work to reproduce the
mark, so rational free-riders won't bother. These attackers falsify that by
keeping detection BER under the threshold while spending a small fraction of an
honest client's compute (measured by compute_meter).

Threat model (matches the project's locked model + the user's clarification):
  * The attacker is an ordinary REGISTERED client. It holds its OWN assigned
    trigger class + key + bits (same information as any honest client) and can
    embed and measure its own BER. It does NOT see η, other clients' keys, or
    other clients' BER. It therefore must ESTIMATE η.
  * "Attacker in the pool vs not" is a SERVER-side setting (wm_verify.calib_on_all),
    not a change to the attacker. calib_on_all=False => the server excludes the
    attacker from η (idealized trusted pool) and the attacker must guess η.
    calib_on_all=True => the attacker's submissions are inside the μ+3σ calc
    (realistic) and moderate BER even helps it by inflating η.

Two families here (both subclass WatermarkClient via a factory, so they can embed
before/while defecting, and so compute_meter sees their work):

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
      SUBMITTED model. Train (embed) for `warmup_rounds`, then replay the frozen,
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

from .client import _to_cpu_state
from .attacks import _extrapolate
from . import watermark as wm


# ---------------------------------------------------------------------------
# shared helpers for watermark-capable adaptive attackers
# ---------------------------------------------------------------------------
def _blend_states(a: dict, b: dict, wa: float) -> dict:
    """wa*a + (1-wa)*b over float weights; non-float buffers taken from `b`
    (the fresher/global source) to keep normalization stats valid."""
    out = {}
    for k, va in a.items():
        if torch.is_floating_point(va) and k in b:
            out[k] = wa * va + (1.0 - wa) * b[k]
        else:
            out[k] = (b[k].clone() if k in b else va.clone())
    return out


class _AdaptiveMixin:
    """Trigger-sample bookkeeping + self-BER probing shared by the attacks.

    Assumes the host class is a WatermarkClient (has key, target_bits,
    trigger_class, wm_kind, wm_alpha, exclude, model, loader, device, meter).
    """

    # gather up to `n` trigger-class samples from the attacker's OWN shard, split
    # into a training slice and a HELD-OUT probe slice (probing on held-out
    # triggers avoids the trigger_only overfitting trap, so self-BER predicts the
    # server's test-trigger BER instead of flattering the attacker).
    def _ensure_triggers(self, n_probe: int = 32, n_train_cap: int = 256):
        if getattr(self, "_probe_x", None) is not None:
            return
        xs = []
        for x, y in self.loader:
            m = (y == self.trigger_class)
            if m.any():
                xs.append(x[m])
            if sum(len(t) for t in xs) >= n_probe + 8:
                break
        if not xs:
            self._probe_x = None
            return
        allx = torch.cat(xs)
        k = min(n_probe, max(1, len(allx) // 2))
        self._probe_x = allx[:k].clone()             # held-out for probing
        # (training still uses the full shard loader; probe slice only reserved
        #  conceptually — we do not remove it from the loader to keep the honest
        #  shard intact, but we never *probe* on batches we optimized this step.)

    @torch.no_grad()
    def _probe_ber_current_model(self) -> float | None:
        """BER of self.model (as-is) on the held-out probe triggers."""
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
    def _probe_ber_state(self, state: dict) -> float | None:
        """BER a given state dict WOULD show (loads it into self.model first)."""
        if self._probe_x is None:
            return None
        self.model.load_state_dict(state)
        return self._probe_ber_current_model()


# ---------------------------------------------------------------------------
# submarine / flappy-bird
# ---------------------------------------------------------------------------
def make_submarine_attack(base_cls):
    class SubmarineFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "submarine"

        def __init__(self, *a,
                     sub_margin: float = 0.03,       # η-estimate minus this = target
                     sub_floor: float = 0.05,        # burst-train until probe BER <= floor
                     sub_eta_mode: str = "adaptive", # "adaptive" (mirror server μ+3σ) | "fixed"
                     sub_eta_fixed: float = 0.25,    # η guess when mode=fixed / no history
                     sub_max_burst_batches: int = 40,# hard cap on a tap's mini-batches
                     sub_probe_every: int = 5,       # re-check probe BER every k batches
                     mem_blend_global: float = 0.3,  # coast freshness: fraction of global mixed in
                     **kw):
            super().__init__(*a, **kw)
            self.sub_margin = sub_margin
            self.sub_floor = sub_floor
            self.sub_eta_mode = sub_eta_mode
            self.sub_eta_fixed = sub_eta_fixed
            self.sub_max_burst_batches = sub_max_burst_batches
            self.sub_probe_every = sub_probe_every
            self.mem_blend_global = mem_blend_global
            self._probe_x = None
            self._submitted_ber_hist = []            # attacker's estimate of what server saw
            self.trace = []

        # η the attacker THINKS it is being judged against.
        def _eta_estimate(self) -> float:
            if self.sub_eta_mode == "adaptive" and self._submitted_ber_hist:
                # mirror the server's own rule (wm_verify.calibrate_eta) on the
                # attacker's estimate of its submitted-BER series.
                return wm.calibrate_eta(self._submitted_ber_hist, floor=self.sub_floor)
            return self.sub_eta_fixed

        def _coast_state(self, global_state, prev_global_state) -> dict:
            if self.memory is not None:
                if self.mem_blend_global > 0.0:
                    return _blend_states(self.memory, global_state,
                                         1.0 - self.mem_blend_global)
                return copy.deepcopy(self.memory)
            # never trained yet: nothing to replay -> extrapolated global (BER~0.5)
            return (copy.deepcopy(global_state) if prev_global_state is None
                    else _extrapolate(global_state, prev_global_state))

        def _burst_train(self, global_state, max_batches) -> int:
            """A minimal honest-style tap: ordinary mini-batches (trigger rows
            carry L_wm) until probe BER <= floor or the cap. Returns #batches."""
            self.model.load_state_dict(global_state)
            self.model.train()
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                  momentum=self.momentum,
                                  weight_decay=self.weight_decay)
            key = self.key.to(self.device)
            bits = self.target_bits.to(self.device)
            steps = 0
            for _ in range(self.local_epochs):        # capped by max_batches anyway
                for x, y in self.loader:
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
                        if b is not None and b <= self.sub_floor:
                            return steps
                    if steps >= max_batches:
                        return steps
            return steps

        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()
            self.meter.start_round(round_idx)

            coast_state = self._coast_state(global_state, prev_global_state)
            ber_coast = self._probe_ber_state(coast_state)     # what a coast would show
            eta_est = self._eta_estimate()
            target = max(self.sub_floor, eta_est - self.sub_margin)

            if ber_coast is not None and ber_coast <= target:
                # COAST: submit the (fresh-ish) memory replay, no training
                submit = coast_state
                trained = False
                ber_after = ber_coast
            else:
                # TAP: minimal burst, then memory-enhance and submit
                self._burst_train(global_state, self.sub_max_burst_batches)
                w_sgd = _to_cpu_state(self.model)
                submit = self._memory_update(global_state, w_sgd)
                trained = True
                ber_after = self._probe_ber_state(submit)

            self.meter.end_round(trained=trained)
            if ber_after is not None:
                self._submitted_ber_hist.append(ber_after)
            self.trace.append({
                "round": round_idx, "action": "tap" if trained else "coast",
                "ber_coast": None if ber_coast is None else round(ber_coast, 4),
                "eta_est": round(eta_est, 4),
                "ber_after": None if ber_after is None else round(ber_after, 4),
                "batches": self.meter.per_round[round_idx]["fwd_passes"],
            })
            return submit, self.num_samples
    return SubmarineFreeRider


# ---------------------------------------------------------------------------
# memory-exploit / momentum
# ---------------------------------------------------------------------------
def make_memory_exploit_attack(base_cls):
    class MemoryExploitFreeRider(_AdaptiveMixin, base_cls):
        is_free_rider = True
        attack_name = "memory_exploit"

        def __init__(self, *a, warmup_rounds: int = 1,
                     mem_blend_global: float = 0.0, **kw):
            super().__init__(*a, **kw)
            self.warmup_rounds = warmup_rounds       # rounds of honest embed up-front
            self.mem_blend_global = mem_blend_global # 0 => pure frozen replay
            self._probe_x = None
            self.trace = []

        def produce_update(self, global_state, prev_global_state, round_idx):
            self._ensure_triggers()
            if round_idx < self.warmup_rounds:
                # honest embed (sets/updates self.memory). super().produce_update
                # opens+closes its OWN meter round, so we do not open one here.
                submit, n = super().produce_update(global_state,
                                                   prev_global_state, round_idx)
                action = "embed"
                ber = self._probe_ber_state(submit)
            else:
                # coast forever on the frozen (optionally freshened) memory
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
            self.trace.append({
                "round": round_idx, "action": action,
                "ber_after": None if ber is None else round(ber, 4),
            })
            return submit, n
    return MemoryExploitFreeRider
