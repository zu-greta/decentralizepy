#!/usr/bin/env python
"""
"does ANY scalar threshold separate free-riders from honest?"

FareMark flags client i as a free-rider iff its bit-error-rate BER_i >= eta, for a
single scalar threshold eta. The whole detector is that one comparison. This script
takes honest runs and free-rider (attack) runs and asks, on the smae converged BER
distributions, two things:

  1. REGIME OF THRESHOLDS -- for a battery of eta rules (the paper's mu+3sigma over
     round-means, a per-client/loose variant, robust median+k*MAD, a trimmed-mean
     variant, honest percentiles, and the Youden-optimal and
     equal-error-rate thresholds) report honest FPR, free-rider recall, and balanced
     accuracy. This shows every named rule trades FPR against recall.

  2. RULE-INDEPENDENT BOUND -- the number that settles it regardless of rule:
       * overlap coefficient (OVL) of the honest vs free-rider BER histograms
         (1.0 = identical distributions, 0 = disjoint), and
       * min over ALL thresholds of the balanced error (FPR+FNR)/2 -- the best a
         scalar-threshold detector could ever do on these two samples.
     If OVL is high and the best-threshold balanced error is far from 0, then NO
     eta separates the two populations: the detector is not merely mis-tuned, it is
     information-theoretically incapable of the task on this distribution.

WHY THIS IS THE HEADLINE
------------------------
"Tune eta better" is answered by rule 1 (every rule has a bad corner). "Pick the
perfect eta" is answered by rule 2 (even the oracle-optimal eta leaves a large
irreducible error because the two BER clouds overlap). The strongest slice is
--per-class on a run where a free-rider and an honest client share a trigger class:
their BER clouds are drawn from the same class floor, so per-class OVL -> ~1 and the
best-threshold error -> ~0.5. That is the clean impossibility result.

INPUT  : honest result.json (for the honest BER cloud + the coded eta) and attack
         result.json (for the free-rider BER cloud). Globs, same as threshold.py.
OUTPUT : a table on stdout; optional --emit <path>.json with every number.

USAGE
-----
  # global (server's real view: honest-all vs FR-all), + the per-class breakdown
  python separability.py \
      --honest-in 'results/*/result.json' --honest-family honest_c100_bdef_iid \
      --attack-in 'results/*/result.json' --attack-family reduced_c100_bdef_iid_c36 \
      --tail 20 --per-class --emit results/separability_c36.json

  # the airtight same-class slice (needs a run where a FR shares a class with honest;
  # see run_experiment.py --trigger_class_map)
  python separability.py --honest-in '...' --attack-in '...' \
      --attack-family sameclass_c100_c6 --per-class
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# Canonical eta (mean-over-clients per round, then mu+3sigma, averaged over seeds)
# lives in threshold.py. Import it so there is ONE definition of the coded rule;
# fall back to a local copy if this script is run outside the package dir.
try:
    import threshold as _th  # type: ignore
    _HAVE_TH = True
except Exception:
    _HAVE_TH = False


# --------------------------------------------------------------------------- io
def load(globs):
    out = []
    for g in (globs if isinstance(globs, (list, tuple)) else [globs]):
        for f in sorted(glob.glob(g)):
            try:
                out.append(json.load(open(f)))
            except Exception as e:
                print(f"  (skip {f} -> {e})")
    return out


def fam(run):
    return (run.get("manifest", {}) or {}).get("family")


def is_honest(run):
    if run.get("free_rider_indices"):
        return False
    for h in run.get("history", []):
        for p in (h.get("wm_per_client") or []):
            if p.get("is_free_rider"):
                return False
    return True


def select(runs, family, honest):
    out = []
    for r in runs:
        if honest and not is_honest(r):
            continue
        if (not honest) and is_honest(r):
            continue
        if family and fam(r) != family:
            continue
        out.append(r)
    return out


# ------------------------------------------------------------------ extraction
def per_client_bers(runs, tail, free_rider):
    """Converged-tail (trigger_class, ber) pairs for honest or free-rider clients."""
    pairs = []
    for r in runs:
        for h in r.get("history", [])[-tail:] if tail else r.get("history", []):
            for p in (h.get("wm_per_client") or []):
                if bool(p.get("is_free_rider")) == free_rider:
                    pairs.append((int(p["trigger_class"]), float(p["ber"])))
    return pairs


def round_means(runs, tail, honest_only=True):
    """m_r = mean BER over clients per round, over the converged tail (for the
    coded mu+3sigma rule). Mirrors threshold.round_means."""
    ms = []
    for r in runs:
        hist = r.get("history", [])
        if tail:
            hist = hist[-tail:]
        for h in hist:
            vals = [p["ber"] for p in (h.get("wm_per_client") or [])
                    if not (honest_only and p.get("is_free_rider"))]
            if vals:
                ms.append(float(np.mean(vals)))
    return ms


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
    """The repo's canonical eta (single source of truth = threshold.frozen_eta)."""
    if _HAVE_TH:
        e = _th.frozen_eta(honest_runs, tail=tail)
        if e is not None:
            return float(e)
    ms = round_means(honest_runs, tail, honest_only=True)
    return _mu_k_sigma(ms, 3.0)


# ---------------------------------------------------------- metrics for one eta
def _rates(H, F, eta):
    """flag iff ber >= eta. FPR on honest H, recall (TPR) on free-riders F."""
    H = np.asarray(H, float)
    F = np.asarray(F, float)
    fpr = float(np.mean(H >= eta)) if H.size else float("nan")
    tpr = float(np.mean(F >= eta)) if F.size else float("nan")
    return fpr, tpr


def best_threshold(H, F):
    """Sweep every candidate eta (midpoints of the pooled sorted BER values) and
    return the one that MINIMISES balanced error (FPR+FNR)/2 -- i.e. the best any
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--honest-in", dest="honest_in", nargs="+", required=True)
    ap.add_argument("--honest-family", dest="honest_family", default=None)
    ap.add_argument("--attack-in", dest="attack_in", nargs="+", default=None,
                    help="attack result.json glob(s). Omit to reuse --honest-in "
                         "(the free-rider slices are read from the same files).")
    ap.add_argument("--attack-family", dest="attack_family", default=None)
    ap.add_argument("--tail", type=int, default=20,
                    help="converged window: last N rounds per run (default 20).")
    ap.add_argument("--per-class", action="store_true",
                    help="also break down separability per trigger class shared by "
                         "honest & free-rider (the airtight same-class slice).")
    ap.add_argument("--emit", default=None, help="write all numbers to this .json.")
    a = ap.parse_args()

    honest_runs = select(load(a.honest_in), a.honest_family, honest=True)
    attack_src = load(a.attack_in) if a.attack_in else load(a.honest_in)
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


if __name__ == "__main__":
    main()