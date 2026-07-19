"""Minimal free-riders

Design goals:
  * one training path -- the honest client's own produce_update / _local_train_wm
    / memory update. During warmup these clients are honest clients, on the original loader.

Schedule:
    rounds 1 .. W-K-1     honest, full shard            (trace action "honest")
    rounds W-K .. W-1      honest, full shard            (trace action "calib")
    rounds >= W            defect                        (trace action "tap"/"coast")
  where W = honest_rounds, K = calib_rounds (K only tags the window for plots).

Attacker A -- ReducedDataFreeRider ("+N"):
    After W, keep training exactly like an honest client but on far less data:
    every trigger-class image in the shard + N images from each common class.

Attacker B -- OracleTapFreeRider:
    After W, it is handed the true threshold eta (oracle). Each round it checks
    whether the watermark is still present in the model it just received; if the
    mark is safely under eta it COASTS (submits the global unchanged, zero
    compute); otherwise it TAPS (one honest-style training pass on the reduced
    loader) to refresh the mark. That is the whole attack.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .client import _to_cpu_state
from . import watermark as wm


# --------------------------------------------------------------------------- #
#  shared helpers (data prep + self-probe)                                     #
# --------------------------------------------------------------------------- #
class _SimpleFRMixin:
    """Host is a WatermarkClient. Adds a reduced (trigger + N/common) loader and
    a self-BER probe on held-out trigger images. Nothing here touches training."""

    def _prepare(self, common_per_class: int, n_probe_holdout: int = 0):
        """Build the reduced loader once. Optionally hold out a few trigger
        images (never trained on) so the probe measures generalisation, matching
        how the server tests on a separate trigger bank."""
        if getattr(self, "_prepared", False):
            return
        self._prepared = True
        bs = getattr(self.loader, "batch_size", 16) or 16

        trig, comm_x, comm_y = [], [], []
        for x, y in self.loader:                      # original shard, once
            tm = (y == self.trigger_class)
            if tm.any():
                trig.append(x[tm])
            if (~tm).any():
                comm_x.append(x[~tm]); comm_y.append(y[~tm])

        allt = torch.cat(trig) if trig else torch.empty(0)
        # hold out a slice of trigger images for the self-probe (attacker B only)
        k = min(n_probe_holdout, max(0, len(allt) - 1)) if n_probe_holdout else 0
        self._probe_x = allt[:k].clone() if k > 0 else None
        trig_train = allt[k:] if k > 0 else allt

        xs = [trig_train]
        ys = [torch.full((len(trig_train),), self.trigger_class, dtype=torch.long)]
        if common_per_class > 0 and comm_x:
            cx = torch.cat(comm_x); cy = torch.cat(comm_y)
            for cls in cy.unique():
                idx = (cy == cls).nonzero(as_tuple=True)[0]
                take = idx[torch.randperm(len(idx))[:common_per_class]]
                xs.append(cx[take]); ys.append(cy[take])
        X, Y = torch.cat(xs), torch.cat(ys)
        self._reduced_n = len(X)
        self._reduced_loader = DataLoader(TensorDataset(X, Y),
                                          batch_size=min(bs, max(1, len(X))),
                                          shuffle=True)

    @torch.no_grad()
    def _probe_ber(self, state) -> float | None:
        """BER of this client's mark in `state`, on held-out trigger images."""
        if getattr(self, "_probe_x", None) is None:
            return None
        self.model.load_state_dict(state)
        self.model.eval()
        probs = F.softmax(self.model(self._probe_x.to(self.device)), dim=1)
        bits = wm.extract_bits(probs, self.key.to(self.device),
                               self.wm_kind, self.wm_alpha, exclude=self.exclude)
        return wm.bit_error_rate(bits, self.target_bits)

    # window bookkeeping shared by both -------------------------------------
    def _phase_action(self, round_idx: int) -> str:
        """honest | calib (last K warmup rounds) | freeride."""
        W, K = self.honest_rounds, self.calib_rounds
        if round_idx >= W:
            return "freeride"
        return "calib" if round_idx >= (W - K) else "honest"


# --------------------------------------------------------------------------- #
#  Attacker A: honest, then honest-on-less-data                                #
# --------------------------------------------------------------------------- #
def make_reduced_attack(base_cls):

    class ReducedDataFreeRider(_SimpleFRMixin, base_cls):
        is_free_rider = True
        attack_name = "reduced"

        def __init__(self, *a, common_per_class: int = 5, honest_rounds: int = 12,
                     calib_rounds: int = 4, **kw):
            super().__init__(*a, **kw)
            self.common_per_class = int(common_per_class)
            self.honest_rounds = int(honest_rounds)
            self.calib_rounds = int(calib_rounds)
            self._prepared = False
            self._orig_loader = self.loader
            self.trace = []

        def produce_update(self, global_state, prev_global_state, round_idx):
            phase = self._phase_action(round_idx)
            if phase == "freeride":
                # switch to the reduced shard and keep training like an honest client
                self._prepare(self.common_per_class)
                self.loader = self._reduced_loader
                submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                self.trace.append({"round": round_idx, "action": "tap",   # re-embeds every round
                                   "eta_frozen": None, "reduced_n": self._reduced_n})
                return submit, n
            # warmup / calibration window: pure honest client on the original shard
            submit, n = super().produce_update(global_state, prev_global_state, round_idx)
            self.trace.append({"round": round_idx, "action": phase, "eta_frozen": None})
            return submit, n

    return ReducedDataFreeRider


# --------------------------------------------------------------------------- #
#  Attacker B: honest, then oracle-threshold tap/coast                         #
# --------------------------------------------------------------------------- #
def make_tap_attack(base_cls):

    class OracleTapFreeRider(_SimpleFRMixin, base_cls):
        is_free_rider = True
        attack_name = "tap_oracle"

        def __init__(self, *a, oracle_eta: float, honest_rounds: int = 12,
                     calib_rounds: int = 4, common_per_class: int = 5,
                     margin: float = 0.02, **kw):
            super().__init__(*a, **kw)
            self.oracle_eta = float(oracle_eta)          # the true server threshold
            self.honest_rounds = int(honest_rounds)
            self.calib_rounds = int(calib_rounds)
            self.common_per_class = int(common_per_class)
            self.margin = float(margin)                  # stay this far under eta
            self._prepared = False
            self._orig_loader = self.loader
            self.trace = []

        def produce_update(self, global_state, prev_global_state, round_idx):
            phase = self._phase_action(round_idx)

            if phase != "freeride":
                # honest warmup / calibration on the original shard
                submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                # expose the oracle on the last calib round so the timeline can draw it
                eta = self.oracle_eta if phase == "calib" else None
                self.trace.append({"round": round_idx, "action": phase, "eta_frozen": eta})
                return submit, n

            # ---- free-ride: coast if the mark is safely present, else tap ----
            self._prepare(self.common_per_class, n_probe_holdout=64)
            target = max(0.0, self.oracle_eta - self.margin)
            ber_now = self._probe_ber(global_state)      # is my mark still in the model?

            if ber_now is not None and ber_now <= target:
                # COAST: submit the global unchanged -> zero training compute
                self.meter.start_round(round_idx); self.meter.end_round(trained=False)
                self.trace.append({"round": round_idx, "action": "coast",
                                   "eta_frozen": self.oracle_eta,
                                   "ber_after": round(ber_now, 4)})
                return {k: v.clone() for k, v in global_state.items()}, self.num_samples

            # TAP: one honest-style pass on the reduced shard to refresh the mark
            self.loader = self._reduced_loader
            submit, n = super().produce_update(global_state, prev_global_state, round_idx)
            self.loader = self._orig_loader
            self.trace.append({"round": round_idx, "action": "tap",
                               "eta_frozen": self.oracle_eta,
                               "ber_after": None if self._probe_ber(submit) is None
                                            else round(self._probe_ber(submit), 4)})
            return submit, n

    return OracleTapFreeRider
