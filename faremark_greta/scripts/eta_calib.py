"""Shared threshold (eta) calibration

FAIR / FROZEN eta: during the forced-honest warmup every client behaves honestly. 
server calibrates eta once, on all clients, over the converged rounds at the end of
warmup — before any free-riding starts — then freezes it

  frozen_eta(runs) -> (eta_tight, eta_loose)
     eta_tight = mu + 3*sigma over the per-ROUND-MEAN BER of all clients in the
                 calibration window   (what the live detector approximates)
     eta_loose = mu + 3*sigma over PER-CLIENT BER of all clients in the window
                 (the looser reading; sigma inflated by hard positions)

The window = the last K rounds that are converged and still all-honest,
i.e. ending just before the first free-rider defects 
"""
import numpy as np


def mu3s(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(np.mean(xs)) + 3.0 * (float(np.std(xs)) if len(xs) > 1 else 0.0)


def calib_window(run, k=5, default_end=12):
    """[lo, hi] rounds to calibrate on: the last k all-honest converged rounds."""
    first_defect = None
    pc = (run.get("compute", {}) or {}).get("per_client", {}) or {}
    for c in pc.values():
        if c.get("is_free_rider"):
            ds = [t["round"] for t in c.get("trace", [])
                  if t.get("action") in ("tap", "coast")]
            if ds:
                first_defect = min(ds) if first_defect is None else min(first_defect, min(ds))
    if first_defect is not None:
        end = first_defect - 1                      # last all-honest round
    else:
        end = (run.get("config", {}) or {}).get("autop_honest_until") or default_end
    end = max(k, int(end))
    return end - k + 1, end


def frozen_eta(runs, k=5):
    """(eta_tight, eta_loose) pooled over ALL clients in each run's calib window."""
    rm, pc = [], []
    for r in runs:
        lo, hi = calib_window(r, k)
        for h in r.get("history", []):
            rd = h.get("round")
            if rd is None or rd < lo or rd > hi:
                continue
            vals = [p["ber"] for p in (h.get("wm_per_client") or [])]   # ALL clients
            if vals:
                rm.append(float(np.mean(vals)))
                pc.extend(vals)
    return mu3s(rm), mu3s(pc)
