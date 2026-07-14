"""threshold (eta)

Protocol:
  rounds 1 .. W-1      forced-honest warmup (every client trains full-shard honestly)
  rounds W-K .. W-1    calibration window: eta computed here once and frozen (every client honest);
                       free-rider estimates its own eta from this same window.
  rounds W .. end      free-riding (tap or coast)
  W = autop_honest_until,  K = autop_calib_rounds.

The attack tags calibration rounds with trace action "calib"; we read the window
from the trace, else fall back to config [W-K, W-1].
"""
import numpy as np


def mu3s(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(np.mean(xs)) + 3.0 * (float(np.std(xs)) if len(xs) > 1 else 0.0)


def _cfg(run, key, default):
    v = (run.get("config", {}) or {}).get(key)
    return default if v is None else v


def _calib_tagged_rounds(run):
    """Rounds the free-rider tagged 'calib' in its trace (the true calibration
    window; DYNAMIC under autop_warmup_mode='dynamic')."""
    tagged = set()
    for c in ((run.get("compute", {}) or {}).get("per_client", {}) or {}).values():
        if c.get("is_free_rider"):
            for t in c.get("trace", []):
                if t.get("action") == "calib":
                    tagged.add(t["round"])
    return tagged


def calib_window(run):
    """[lo, hi] calibration rounds — from 'calib' trace tags (dynamic), else the
    config window [W-K, W-1] (e.g. the all-honest run has no free-rider to tag)."""
    tagged = _calib_tagged_rounds(run)
    if tagged:
        return min(tagged), max(tagged)
    W = int(_cfg(run, "autop_honest_until", 12))
    K = int(_cfg(run, "autop_calib_rounds", 4))
    return W - K, W - 1


def freeride_start(run):
    """W = first free-riding round = (last calibration round) + 1. Prefers the
    run's actual (dynamic) calib window; falls back to config autop_honest_until."""
    tagged = _calib_tagged_rounds(run)
    if tagged:
        return max(tagged) + 1
    return int(_cfg(run, "autop_honest_until", 12))


def window_bounds(runs):
    return calib_window(runs[0]) if runs else (8, 11)


def last_round(runs):
    return max((h.get("round", 0) for r in runs for h in r.get("history", [])), default=50)


def _pool(runs, lo, hi, honest_only=False):
    """(round_means, individual_bers) over rounds in [lo, hi]."""
    rm, pc = [], []
    for r in runs:
        for h in r.get("history", []):
            rd = h.get("round")
            if rd is None or rd < lo or rd > hi:
                continue
            vals = [p["ber"] for p in (h.get("wm_per_client") or [])
                    if not (honest_only and p.get("is_free_rider"))]
            if vals:
                rm.append(float(np.mean(vals)))
                pc.extend(vals)
    return rm, pc


def frozen_eta(runs):
    """(tight, loose) on ALL clients in the calibration window — the FAIR eta."""
    tight_rm, loose_pc = [], []
    for r in runs:
        lo, hi = calib_window(r)
        rm, pc = _pool([r], lo, hi, honest_only=False)
        tight_rm += rm
        loose_pc += pc
    return mu3s(tight_rm), mu3s(loose_pc)


def _cumulative(runs):
    """μ+3σ over ALL honest round-means across the whole run (the swingy live one)."""
    end = last_round(runs)
    rm, _ = _pool(runs, 1, end, honest_only=True)
    return mu3s(rm)


def _easy_hard(runs, lo, hi, split=0.03):
    """Per-client honest BER in the calib window, split into easy vs hard positions
    by per-trigger-class mean BER."""
    byclass = {}
    for r in runs:
        for h in r.get("history", []):
            rd = h.get("round")
            if rd is None or rd < lo or rd > hi:
                continue
            for p in (h.get("wm_per_client") or []):
                if not p.get("is_free_rider"):
                    byclass.setdefault(p["trigger_class"], []).append(p["ber"])
    cmean = {c: float(np.mean(v)) for c, v in byclass.items()}
    easy = [b for c, v in byclass.items() if cmean[c] < split for b in v]
    hard = [b for c, v in byclass.items() if cmean[c] >= split for b in v]
    return mu3s(easy), mu3s(hard)


def all_thresholds(attack, honest):
    """The seven threshold definitions (bar chart source). `attack` = a data-sweep
    family (all clients honest during warmup); `honest` = the all-honest run."""
    ref = attack or honest
    lo, hi = window_bounds(ref)
    W = freeride_start(ref[0])
    end = last_round(ref)
    T = {}
    rm, _ = _pool(attack or honest, lo, hi);                       T["1. SPEC\n(calib win,\nround-mean)"] = mu3s(rm)
    rm, _ = _pool(honest or attack, end - 19, end, honest_only=True); T["2. longer\nhonest window"] = mu3s(rm)
    _, pc = _pool(attack or honest, lo, hi);                       T["3. per-client\n(calib win)"] = mu3s(pc)
    rm, _ = _pool(attack or honest, 1, W - 1);                     T["4. incl. full\nwarmup"] = mu3s(rm)
    T["5. cumulative\n(live)"] = _cumulative(honest or attack)
    rm, _ = _pool(attack, W, end);                                T["6. FR-inflated\n(post-warmup)"] = mu3s(rm)
    e, h = _easy_hard(honest or attack, lo, hi)
    T["7a. all-honest\nEASY pos"] = e
    T["7b. all-honest\nHARD pos"] = h
    return T