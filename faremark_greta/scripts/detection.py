#!/usr/bin/env python
"""detection -- the detection threshold eta


SECTION 1  ETA          mu3s, round_means, eta_from_round_means, frozen_eta,
                        adaptive_clip_eta, calibrate, verify, load_fixed, find_fixed
SECTION 2  SEPARABILITY threshold rules (coded/loose/MAD/trimmed/clip/p95/p99/EER/
                        Youden), overlap_coefficient, summarise, print_block

CLI:
    # freeze eta on honest-only runs -> feed back as WM_ETA_FIXED
    python scripts/detection.py calibrate --in 'results/*/result.json' \
        --honest-family honest_c100_bdef_iid --tail 20 --out results/eta_c100.json

    # confirm every attack run actually used the frozen constant
    python scripts/detection.py verify --in 'results/*/result.json' \
        --honest-family honest_c100_bdef_iid --eta-file results/eta_c100.json

    # rule-independent non-separability table (the headline evidence)
    python scripts/detection.py separability \
        --honest-in 'results/*/result.json' --honest-family honest_c100_bdef_iid \
        --attack-in 'results/*/result.json' --attack-family reduced_c100_bdef_iid_c36 \
        --tail 20 --per-class --emit results/sep_c36.json
"""

from __future__ import annotations
import os, sys, glob, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resultio import (  # noqa: E402
    load, fam, is_honest_run, round_means, calib_window, freeride_start,
    last_round, DEFAULT_TAIL,
)
from resultio import _calib_tagged_rounds  # noqa: E402,F401  (kept: used by plots)


# ============================================================================
# SECTION 1 -- THRESHOLD ETA
# ============================================================================

# ----------------------------------------------------------------- primitives
def mu3s(xs):
    xs = [x for x in xs if x is not None] # ignore None values (e.g. from empty rounds)
    if not xs: 
        return None
    return float(np.mean(xs)) + 3.0 * (float(np.std(xs)) if len(xs) > 1 else 0.0) # mu + 3*sigma


# ------------------------------------------------------ eta
def eta_from_round_means(ms):
    """(eta, mu, sigma) = (mu+3sigma, mean, std) over the per-round means"""
    if not ms:
        return None, None, 0.0
    if len(ms) < 2:
        return float(ms[0]), float(ms[0]), 0.0
    mu = float(np.mean(ms)); sigma = float(np.std(ms))
    return mu + 3.0 * sigma, mu, sigma


def frozen_eta(runs, tail=20):
    """Canonical eta recomputed from `runs`: AVERAGE of per-seed etas
    (each run/seed -> eta_s = mu_s + 3*sigma_s over its round-means)."""
    etas = []
    for r in runs:
        e, _, _ = eta_from_round_means(round_means([r], tail=tail, honest_only=True))
        if e is not None:
            etas.append(e)
    return float(np.mean(etas)) if etas else None


def adaptive_clip_eta(bers, k=3.0, max_iter=10):
    """TODO: test different thresholds
    Iterative sigma-clip (astronomy-style) robust threshold on honest BER.

    This is the "clip-and-adapt during the calibration rounds" idea: start from the
    full honest BER set, drop every point above mu+k*sigma, recompute mu/sigma on the
    survivors, and repeat until the surviving set stops changing. Each pass discards
    the heavy UPPER tail (the hard-class honest outliers) and re-estimates, so eta
    converges onto the BULK of honest clients instead of being dragged up by a few
    hard classes.

    Returns (eta, kept_fraction). IMPORTANT: the honest clients that get clipped out
    then sit ABOVE this eta -> they are exactly the ones the deployed detector would
    false-positive on. So a tighter, "better-behaved" eta on the bulk buys itself a
    guaranteed set of honest false positives -- which is the separability point, not a
    bug. Pass per-client BER (full variance) or round-means, as you like.
    """
    x = np.asarray([b for b in bers if b is not None], float)
    if x.size == 0:
        return None, 0.0
    keep = np.ones(x.size, dtype=bool)
    for _ in range(max_iter):
        cur = x[keep]
        if cur.size < 2:
            break
        mu = float(cur.mean()); sd = float(cur.std())
        eta = mu + k * sd
        new_keep = x <= eta
        if int(new_keep.sum()) == int(keep.sum()):   # inlier set stable -> converged
            keep = new_keep
            break
        keep = new_keep
    cur = x[keep] if keep.any() else x
    eta = float(cur.mean() + k * (cur.std() if cur.size > 1 else 0.0))
    return eta, float(keep.mean())


def all_thresholds(runs, tail=20):
    # return a dict of all thresholds (for plotting)
    return {"eta = mean-over-clients,\nthen mu+3sigma over rounds": frozen_eta(runs, tail)}


# --------------------------------------------------------- freeze / load
def load_fixed(path):
    """Read the pre-calibrated constant written by `calibrate`"""
    try:
        return float(json.load(open(path))["eta"])
    except Exception:
        return None


def find_fixed(near_dir):
    """Look for eta_calibrated.json in near_dir or its parent; return its eta"""
    for cand in (os.path.join(near_dir, "eta_calibrated.json"),
                 os.path.join(os.path.dirname(near_dir.rstrip("/")), "eta_calibrated.json")):
        if os.path.exists(cand):
            v = load_fixed(cand)
            if v is not None:
                return v, cand
    return None, None


def calibrate(inp, honest_family=None, tail=20, out=None):
    """Calibrate the canonical eta on honest-only multi-seed runs and freeze it
    to eta_calibrated.json. Returns the result dict."""
    runs = [(f, r) for f, r in load(inp) if is_honest_run(r)]
    if honest_family:
        runs = [(f, r) for f, r in runs if fam(r) == honest_family]
    if not runs:
        raise SystemExit("no honest-only runs found (check --in / --honest-family).")

    # PER-SEED first. For seed s: mu_s = mean over its last `tail` round-means,
    # sigma_s = std of those round-means, eta_s = mu_s + 3*sigma_s.
    pooled, per_seed, etas, mus, sds = [], [], [], [], []
    for f, r in runs:
        ms = round_means([r], tail=tail, honest_only=True)     # this seed's round-means
        pooled += ms
        e, mu_s, sd_s = eta_from_round_means(ms)
        if e is not None:
            etas.append(e); mus.append(mu_s); sds.append(sd_s)
        per_seed.append({"file": os.path.basename(os.path.dirname(f)), "seed": r.get("seed"),
                         "n_rounds": len(ms), "eta": None if e is None else round(e, 5),
                         "mu": None if mu_s is None else round(mu_s, 5), "sigma": round(sd_s, 5)})

    # FINAL eta = AVERAGE of the per-seed etas (== mean(mu_s) + 3*mean(sigma_s)).
    eta = float(np.mean(etas)) if etas else None
    eta_std = float(np.std(etas)) if len(etas) > 1 else 0.0     # seed-to-seed variability
    grand_mean = float(np.mean(mus)) if mus else None           # mean over seeds of mu_s
    grand_std = float(np.mean(sds)) if sds else None            # mean over seeds of sigma_s
    eta_pooled, _, _ = eta_from_round_means(pooled)             # reference: pooled mu+3sigma
    eta_all, _, _ = eta_from_round_means(round_means([r for _, r in runs], tail=0))

    result = {
        "eta": round(eta, 5),
        "definition": "mean over seeds of (mu_s + 3*sigma_s); "
                      "mu_s,sigma_s over per-round mean-over-clients benign BER",
        "window": f"tail:{tail}" if tail else "all_rounds",
        "eta_std_across_seeds": round(eta_std, 5),
        "grand_mean": round(grand_mean, 5), "grand_std": round(grand_std, 5),
        "n_seeds": len(runs), "n_round_means_pooled": len(pooled),
        "honest_family": honest_family, "per_seed": per_seed,
        "eta_pooled_for_reference": round(eta_pooled, 5) if eta_pooled is not None else None,
        "eta_all_rounds_for_reference": round(eta_all, 5) if eta_all is not None else None,
    }
    if out is None:
        base = os.path.dirname(os.path.commonpath([f for f, _ in runs]))
        out = os.path.join(base, "eta_calibrated.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    result["_out"] = out
    return result


def verify(inp, honest_family=None, tail=20, eta_file=None):
    """Double-check the pipeline:
      1. recompute eta from the honest runs and compare to eta_calibrated.json.
      2. Confirm every non-honest run actually used the frozen constant, i.e. its
         wm_eta_round is flat and == the frozen eta (post-warmup rounds).
    Prints a PASS/FAIL report. Returns True iff all checks pass."""
    runs = load(inp)
    honest = [(f, r) for f, r in runs if is_honest_run(r)
              and (honest_family is None or fam(r) == honest_family)]
    frozen = load_fixed(eta_file) if eta_file else None
    ok = True

    print("== 1. recompute eta from honest runs (avg of per-seed etas) ==")
    if not honest:
        print("  (no honest runs found)"); ok = False
    else:
        eta_re = frozen_eta([r for _, r in honest], tail=tail)
        per = [eta_from_round_means(round_means([r], tail=tail, honest_only=True))[0]
               for _, r in honest]
        print(f"  recomputed eta = {eta_re:.5f}  (avg over {len(honest)} seeds; "
              f"per-seed etas = {[round(p,4) for p in per if p is not None]})")
        if frozen is not None:
            match = abs(eta_re - frozen) < 1e-4
            print(f"  eta_calibrated.json eta = {frozen:.5f}  -> "
                  f"{'MATCH' if match else 'MISMATCH'}")
            ok &= match

    print("== 2. attack runs used the frozen constant (flat wm_eta_round) ==")
    attack = [(f, r) for f, r in runs if not is_honest_run(r)]
    if not attack:
        print("  (no attack runs found yet)")
    for f, r in attack:
        etas = [h.get("wm_eta_round") for h in r.get("history", [])
                if h.get("wm_eta_round") is not None]
        tag = os.path.basename(os.path.dirname(f))
        if not etas:
            print(f"  {tag}: no wm_eta_round logged"); continue
        flat = (max(etas) - min(etas)) < 1e-6
        val = etas[-1]
        near = frozen is None or abs(val - frozen) < 1e-4
        status = "OK" if (flat and near) else "CHECK"
        print(f"  {tag}: wm_eta_round={val:.4f} flat={flat} "
              f"matches_frozen={near} -> {status}")
        ok &= (flat and near)
    print("== RESULT:", "PASS" if ok else "FAIL", "==")
    return ok




# ============================================================================
# SECTION 2 -- SEPARABILITY
# ============================================================================

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# Canonical eta (mean-over-clients per round, then mu+3sigma, averaged over seeds)
try:
    import threshold as _th  # type: ignore
    _HAVE_TH = True
except Exception:
    _HAVE_TH = False


# --------------------------------------------------------------------------- io
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resultio as _rio

fam = _rio.fam
is_honest = _rio.is_honest_run
round_means = _rio.round_means


def load_runs(globs):
    """Bare-run list -- the shape the separability half wants"""
    return _rio.load(globs, with_path=False)


def select(runs, family, honest):
    return _rio.select(runs, family=family, honest=honest)


def per_client_bers(runs, tail, free_rider):
    """Converged-tail (trigger_class, ber) pairs for honest or free-rider clients."""
    return _rio.per_client_ber_pairs(runs, tail=tail, free_rider=free_rider)


# --------------------------------------------------------------- threshold rules
def _mu_k_sigma(xs, k=3.0):
    xs = np.asarray(xs, float)
    if xs.size == 0:
        return None
    mu = float(xs.mean())
    sd = float(xs.std()) if xs.size > 1 else 0.0
    return mu + k * sd


def _median_k_mad(xs, k=3.0):
    xs = np.asarray(xs, float)
    if xs.size == 0:
        return None
    med = float(np.median(xs))
    mad = float(np.median(np.abs(xs - med))) * 1.4826  # ~sigma for normal
    return med + k * mad


def _trimmed_mu_sigma(xs, trim=0.10, k=3.0):
    xs = np.sort(np.asarray(xs, float))
    if xs.size == 0:
        return None
    c = int(np.floor(trim * xs.size))
    core = xs[c: xs.size - c] if xs.size - 2 * c > 0 else xs
    return _mu_k_sigma(core, k)


def _percentile(xs, q):
    xs = np.asarray(xs, float)
    return float(np.percentile(xs, q)) if xs.size else None


def coded_eta(honest_runs, tail):
    """The repo's canonical eta (single source of truth = threshold.frozen_eta)"""
    if _HAVE_TH:
        e = _th.frozen_eta(honest_runs, tail=tail)
        if e is not None:
            return float(e)
    ms = round_means(honest_runs, tail, honest_only=True)
    return _mu_k_sigma(ms, 3.0)


# ---------------------------------------------------------- metrics for one eta
def _rates(H, F, eta):
    """flag iff ber >= eta. FPR on honest H, recall (TPR) on free-riders F"""
    H = np.asarray(H, float)
    F = np.asarray(F, float)
    fpr = float(np.mean(H >= eta)) if H.size else float("nan")
    tpr = float(np.mean(F >= eta)) if F.size else float("nan")
    return fpr, tpr


def best_threshold(H, F):
    """Sweep every candidate eta (midpoints of the pooled sorted BER values) and
    return the one that minimises balanced error (FPR+FNR)/2 -- i.e. the best any
    scalar threshold can do. Returns (eta*, balanced_err*, fpr*, tpr*)."""
    H = np.asarray(H, float)
    F = np.asarray(F, float)
    if H.size == 0 or F.size == 0:
        return None, float("nan"), float("nan"), float("nan")
    vals = np.unique(np.concatenate([H, F]))
    # candidate thresholds: just below the smallest, midpoints, just above largest
    cands = np.concatenate([[vals[0] - 1e-6],
                            (vals[:-1] + vals[1:]) / 2.0,
                            [vals[-1] + 1e-6]])
    best = (None, np.inf, np.nan, np.nan)
    for eta in cands:
        fpr, tpr = _rates(H, F, eta)
        bal_err = 0.5 * (fpr + (1.0 - tpr))
        if bal_err < best[1]:
            best = (float(eta), float(bal_err), float(fpr), float(tpr))
    return best


def eer_threshold(H, F):
    """Equal-error-rate threshold: where FPR crosses FNR."""
    H = np.asarray(H, float)
    F = np.asarray(F, float)
    if H.size == 0 or F.size == 0:
        return None
    vals = np.unique(np.concatenate([H, F]))
    cands = np.concatenate([[vals[0] - 1e-6],
                            (vals[:-1] + vals[1:]) / 2.0,
                            [vals[-1] + 1e-6]])
    best_eta, best_gap = None, np.inf
    for eta in cands:
        fpr, tpr = _rates(H, F, eta)
        gap = abs(fpr - (1.0 - tpr))
        if gap < best_gap:
            best_gap, best_eta = gap, float(eta)
    return best_eta


def overlap_coefficient(H, F, bins=40):
    """Histogram overlap (OVL) of two samples on a shared BER grid in [0,1].
    OVL = sum_b min(p_H[b], p_F[b]) with p normalised to sum 1. 1.0 = identical."""
    H = np.asarray(H, float)
    F = np.asarray(F, float)
    if H.size == 0 or F.size == 0:
        return float("nan")
    lo = float(min(H.min(), F.min()))
    hi = float(max(H.max(), F.max()))
    if hi <= lo:
        return 1.0  # both degenerate at the same value
    edges = np.linspace(lo, hi, bins + 1)
    ph, _ = np.histogram(H, bins=edges, density=False)
    pf, _ = np.histogram(F, bins=edges, density=False)
    ph = ph / ph.sum()
    pf = pf / pf.sum()
    return float(np.minimum(ph, pf).sum())


# ------------------------------------------------------------------- reporting
def summarise(H, F, honest_runs, tail, label=""):
    """Return a dict of {rule -> (eta, fpr, recall, bal_acc)} plus the
    rule-independent bound (best-threshold balanced error + overlap)."""
    H = np.asarray(H, float)
    F = np.asarray(F, float)

    rules = {}

    def add(name, eta):
        if eta is None:
            rules[name] = None
            return
        fpr, tpr = _rates(H, F, eta)
        rules[name] = {
            "eta": round(float(eta), 4),
            "fpr": None if fpr != fpr else round(fpr, 4),
            "recall": None if tpr != tpr else round(tpr, 4),
            "bal_acc": None if (fpr != fpr or tpr != tpr)
            else round(0.5 * ((1 - fpr) + tpr), 4),
        }

    add("coded (mu+3s round-mean)", coded_eta(honest_runs, tail) if honest_runs else _mu_k_sigma(round_means(honest_runs, tail)) )
    add("loose (mu+3s per-client)", _mu_k_sigma(H, 3.0))
    add("median+3*MAD", _median_k_mad(H, 3.0))
    add("trimmed10 mu+3s", _trimmed_mu_sigma(H, 0.10, 3.0))
    # adaptive clipping: iterative sigma-clip on honest BER 
    if _HAVE_TH:
        _ace, _kept = _th.adaptive_clip_eta(list(H), k=3.0)
        add(f"adaptive-clip (kept {round(_kept,2)})", _ace)
    add("honest p95", _percentile(H, 95))
    add("honest p99", _percentile(H, 99))
    add("equal-error-rate", eer_threshold(H, F))
    eta_star, bal_err_star, fpr_star, tpr_star = best_threshold(H, F)
    add("Youden-optimal (best)", eta_star)

    ovl = overlap_coefficient(H, F)

    return {
        "label": label,
        "n_honest": int(H.size),
        "n_free_rider": int(F.size),
        "honest_ber_mean": round(float(H.mean()), 4) if H.size else None,
        "honest_ber_std": round(float(H.std()), 4) if H.size else None,
        "fr_ber_mean": round(float(F.mean()), 4) if F.size else None,
        "fr_ber_std": round(float(F.std()), 4) if F.size else None,
        "rules": rules,
        "overlap_coefficient": None if ovl != ovl else round(ovl, 4),
        "best_threshold_balanced_error": None if bal_err_star != bal_err_star
        else round(bal_err_star, 4),
        "best_threshold_eta": None if eta_star is None else round(eta_star, 4),
    }


def print_block(res):
    print(f"\n=== {res['label']} ===")
    print(f"  honest: n={res['n_honest']:4d}  BER {res['honest_ber_mean']} "
          f"+/- {res['honest_ber_std']}")
    print(f"  free  : n={res['n_free_rider']:4d}  BER {res['fr_ber_mean']} "
          f"+/- {res['fr_ber_std']}")
    print(f"  {'rule':>26}  {'eta':>7}  {'FPR':>6}  {'recall':>7}  {'bal_acc':>7}")
    for name, d in res["rules"].items():
        if d is None:
            print(f"  {name:>26}  {'--':>7}")
            continue
        def fmt(v):
            return "  --  " if v is None else f"{v:.3f}"
        print(f"  {name:>26}  {d['eta']:>7.3f}  {fmt(d['fpr']):>6}  "
              f"{fmt(d['recall']):>7}  {fmt(d['bal_acc']):>7}")
    print(f"  --> overlap coefficient (OVL)          : {res['overlap_coefficient']}  "
          f"(1.0 = honest & FR BER identical)")
    print(f"  --> best-possible balanced error       : {res['best_threshold_balanced_error']}  "
          f"(0 = some eta separates perfectly; 0.5 = no eta helps at all)")
    if res["overlap_coefficient"] is not None and res["overlap_coefficient"] >= 0.5:
        print("      READING: high overlap -> NO scalar threshold separates these "
              "populations (not a tuning problem).")



# ============================================================================
# CLI dispatcher 
# ============================================================================
#   python scripts/detection.py calibrate ...
#   python scripts/detection.py verify ...
#   python scripts/detection.py separability <args>


def _cmd_calibrate(a):
    res = calibrate(a.inp, a.honest_family, a.tail, a.out)
    print(f"CANONICAL eta = {res['eta']:.5f} +/- {res['eta_std_across_seeds']:.5f}  "
          f"(avg over {res['n_seeds']} seeds of mu_s+3*sigma_s; "
          f"mean mu={res['grand_mean']:.4f}, mean sigma={res['grand_std']:.4f})")
    print(f"  window={res['window']}")
    print(f"  per-seed etas: {[s['eta'] for s in res['per_seed']]}")
    print(f"  reference: pooled eta={res['eta_pooled_for_reference']}, "
          f"all-rounds eta={res['eta_all_rounds_for_reference']} (warmup-inflated)")
    print(f"wrote {res['_out']}")
    print(f"\nUse it downstream:  WM_ETA_FIXED={res['eta']:.5f}  (or --wm_eta_fixed)")


def _cmd_verify(a):
    sys.exit(0 if verify(a.inp, a.honest_family, a.tail, a.eta_file) else 1)


def _cmd_separability(a):
    honest_runs = select(load_runs(a.honest_in), a.honest_family, honest=True)
    attack_src = load_runs(a.attack_in) if a.attack_in else load_runs(a.honest_in)
    attack_runs = select(attack_src, a.attack_family, honest=False)

    if not honest_runs:
        raise SystemExit("no honest runs matched (check --honest-in / --honest-family).")
    if not attack_runs:
        raise SystemExit("no attack runs matched (check --attack-in / --attack-family).")

    print(f"loaded {len(honest_runs)} honest run(s)"
          + (f" [{a.honest_family}]" if a.honest_family else "")
          + f" and {len(attack_runs)} attack run(s)"
          + (f" [{a.attack_family}]" if a.attack_family else "")
          + f"; converged tail = last {a.tail} rounds")

    H = [b for _c, b in per_client_bers(honest_runs, a.tail, free_rider=False)]
    F = [b for _c, b in per_client_bers(attack_runs, a.tail, free_rider=True)]

    out = {"honest_family": a.honest_family, "attack_family": a.attack_family,
           "tail": a.tail, "global": None, "per_class": {}}

    glob_res = summarise(H, F, honest_runs, a.tail,
                         label="GLOBAL  (server view: honest-all vs free-rider-all)")
    print_block(glob_res)
    out["global"] = glob_res

    if a.per_class:
        Hc = defaultdict(list)
        for c, b in per_client_bers(honest_runs, a.tail, free_rider=False):
            Hc[c].append(b)
        Fc = defaultdict(list)
        for c, b in per_client_bers(attack_runs, a.tail, free_rider=True):
            Fc[c].append(b)
        shared = sorted(set(Hc) & set(Fc))
        if not shared:
            print("\n(no trigger class is shared by honest & free-rider runs; "
                  "the per-class slice needs a same-class run -- see "
                  "run_experiment.py --trigger_class_map)")
        for c in shared:
            res = summarise(Hc[c], Fc[c], honest_runs, a.tail,
                            label=f"trigger class {c}  (honest vs free-rider, SAME class)")
            print_block(res)
            out["per_class"][c] = res

    if a.emit:
        os.makedirs(os.path.dirname(a.emit) or ".", exist_ok=True)
        json.dump(out, open(a.emit, "w"), indent=2)
        print(f"\nwrote {a.emit}")


def main():
    ap = argparse.ArgumentParser(
        description="detection threshold eta + non-separability analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="freeze eta on honest-only runs")
    c.add_argument("--in", dest="inp", nargs="+", required=True)
    c.add_argument("--honest-family", default=None,
                   help="restrict to this manifest family (e.g. honest_iid).")
    c.add_argument("--tail", type=int, default=DEFAULT_TAIL,
                   help="last N rounds (0 = ALL rounds).")
    c.add_argument("--out", default=None)
    c.set_defaults(fn=_cmd_calibrate)

    v = sub.add_parser("verify", help="confirm attack runs used the frozen eta")
    v.add_argument("--in", dest="inp", nargs="+", required=True)
    v.add_argument("--honest-family", default=None)
    v.add_argument("--tail", type=int, default=DEFAULT_TAIL)
    v.add_argument("--eta-file", default=None,
                   help="eta_calibrated.json to compare against")
    v.set_defaults(fn=_cmd_verify)

    s = sub.add_parser("separability", help="does ANY threshold separate the two?")
    s.add_argument("--honest-in", dest="honest_in", nargs="+", required=True)
    s.add_argument("--honest-family", dest="honest_family", default=None)
    s.add_argument("--attack-in", dest="attack_in", nargs="+", default=None,
                   help="attack result.json glob(s). Omit to reuse --honest-in "
                        "(the free-rider slices are read from the same files).")
    s.add_argument("--attack-family", dest="attack_family", default=None)
    s.add_argument("--tail", type=int, default=DEFAULT_TAIL,
                   help="converged window: last N rounds per run (default 20).")
    s.add_argument("--per-class", action="store_true",
                   help="also break down separability per trigger class shared by "
                        "honest & free-rider (the airtight same-class slice).")
    s.add_argument("--emit", default=None, help="write all numbers to this .json.")
    s.set_defaults(fn=_cmd_separability)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()