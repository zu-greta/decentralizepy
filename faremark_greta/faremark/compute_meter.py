"""Per-client compute accounting 

Measuring effort for attacks vs. honest clients
This module gives every training client a meter that accumulates, per round and in total:

  * fwd_passes / bwd_passes : # forward / backward calls (batch granularity)
  * samples                 : # training examples processed (sum of batch sizes)
  * opt_steps               : # optimizer.step() calls
  * gpu_ms                  : GPU wall-time of the metered region (CUDA events)
  * wall_ms                 : CPU wall-clock of the metered region
  * flops                   : estimated fwd+bwd FLOPs = samples * fps * FWD_BWD_MULT
  * trained                 : bool - if client trained this round (or coasted / free-rode)

Cluster note: on the RunAI A100 the meaningful cost unit is GPU-seconds, so
`gpu_ms` is measured with CUDA events (accurate for the GPU stream), not just
Python wall time. `samples`/`passes`/`flops` are device-independent and
deterministic, so they are what you should put on the x-axis of an
effort-vs-detection plot when comparing across machines. Use `gpu_ms` when you
specifically want "what did this cost on the cluster".
CUDA-event timing is used only when torch+CUDA are present, otherwise it falls back to perf_counter.
"""
from __future__ import annotations

import time

try:
    import torch
    _HAS_TORCH = True
except Exception:                                   # torch-free environments (tests)
    _HAS_TORCH = False


_ACCUM_KEYS = ("fwd_passes", "bwd_passes", "opt_steps", "samples",
               "gpu_ms", "wall_ms", "flops")


def _zero_bucket() -> dict:
    b = {k: 0.0 for k in _ACCUM_KEYS}
    b["trained"] = False
    return b


class ComputeMeter:
    """Accumulate training work for one client across rounds.

    Typical use inside a client's produce_update:

        self.meter.start_round(round_idx)
        ... training loop, calling self.meter.record_batch(n) per batch ...
        self.meter.end_round(trained=True)      # or trained=False for a coast

    A no-training free-rider still calls start/end with trained=False so its
    (near-zero) fabrication cost is recorded and comparable.
    """

    FWD_BWD_MULT = 3.0            # backward ~= 2x forward => fwd+bwd ~= 3x forward FLOPs

    def __init__(self, flops_per_sample_fwd: float | None = None):
        self.flops_per_sample_fwd = flops_per_sample_fwd
        self.total = _zero_bucket()
        self.total["rounds_trained"] = 0
        self.total["rounds_total"] = 0
        self.per_round: dict[int, dict] = {}
        self._cur: dict | None = None
        self._idx: int | None = None
        self._t0: float | None = None
        self._evt_start = None
        self._evt_end = None

    # ---- round lifecycle ---------------------------------------------------
    def start_round(self, round_idx: int):
        self._cur = _zero_bucket()
        self._idx = round_idx
        self._t0 = time.perf_counter()
        if _HAS_TORCH and torch.cuda.is_available():
            self._evt_start = torch.cuda.Event(enable_timing=True)
            self._evt_end = torch.cuda.Event(enable_timing=True)
            self._evt_start.record()
        else:
            self._evt_start = None

    def record_batch(self, n_samples: int, fwd: int = 1, bwd: int = 1, opt: int = 1):
        b = self._cur
        b["fwd_passes"] += fwd
        b["bwd_passes"] += bwd
        b["opt_steps"] += opt
        b["samples"] += n_samples

    def record_forward_only(self, n_samples: int, fwd: int = 1):
        """For probe/extraction forward passes that carry no backward"""
        b = self._cur
        b["fwd_passes"] += fwd
        b["samples"] += n_samples

    def end_round(self, trained: bool):
        wall_ms = (time.perf_counter() - self._t0) * 1000.0
        if self._evt_start is not None:
            self._evt_end.record()
            torch.cuda.synchronize()
            gpu_ms = float(self._evt_start.elapsed_time(self._evt_end))
        else:
            gpu_ms = wall_ms
        b = self._cur
        b["wall_ms"] = round(wall_ms, 3)
        b["gpu_ms"] = round(gpu_ms, 3)
        b["trained"] = bool(trained)
        if self.flops_per_sample_fwd is not None:
            b["flops"] = b["samples"] * self.flops_per_sample_fwd * self.FWD_BWD_MULT
        self.per_round[self._idx] = b
        for k in _ACCUM_KEYS:
            self.total[k] += b[k]
        self.total["rounds_total"] += 1
        self.total["rounds_trained"] += int(trained)
        self._cur = None

    # ---- readout -----------------------------------------------------------
    def summary(self, attack_name: str = "honest", is_free_rider: bool = False) -> dict:
        t = dict(self.total)
        t["gpu_ms"] = round(t["gpu_ms"], 3)
        t["wall_ms"] = round(t["wall_ms"], 3)
        t["duty_cycle"] = (round(t["rounds_trained"] / t["rounds_total"], 4)
                           if t["rounds_total"] else 0.0)
        return {
            "attack_name": attack_name,
            "is_free_rider": is_free_rider,
            "total": t,
            "per_round": self.per_round,
        }


def estimate_flops_per_sample_fwd(model, input_shape, device="cpu") -> float | None:
    """Best-effort forward-FLOPs-per-sample estimate for one (model, input).

    Tries fvcore, then thop, then ptflops. Returns None if none is installed
    (the meter then reports passes/samples/time but no FLOPs). Call ONCE at
    startup where the model exists (e.g. run_experiment) and hand the result to
    every client meter. input_shape is a single-sample shape, e.g. (3, 32, 32).
    """
    if not _HAS_TORCH:
        return None
    dummy = torch.zeros((1, *input_shape), device=device)
    was_training = model.training
    model.eval()
    try:
        try:
            from fvcore.nn import FlopCountAnalysis
            flops = float(FlopCountAnalysis(model, dummy).total())
            return flops
        except Exception:
            pass
        try:
            from thop import profile
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            return float(macs) * 2.0                 # MACs -> FLOPs
        except Exception:
            pass
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(
                model, input_shape, as_strings=False,
                print_per_layer_stat=False, verbose=False)
            return float(macs) * 2.0
        except Exception:
            pass
        return None
    finally:
        if was_training:
            model.train()
