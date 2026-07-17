"""detection threshold eta = mu+3sigma over per-round (mean-over-clients) benign BER

Import it as a library, or run it as a CLI to calibrate:

    python threshold.py calibrate --in '/path/results/*/result.json' \
        --honest-family honest_iid --tail 20 --out /path/results/eta_calibrated.json

threshold definition:
    m_r   = mean BER over all honest clients in round r       (mean over clients)
    mu    = mean_r(m_r)                                       (mean over rounds)
    sigma = std_r(m_r)
    eta   = mu + 3*sigma
Calibrated once on honest-only multi-seed runs, frozen to a constant, reused for every experiment
"""
from __future__ import annotations
import os, sys, glob, json, argparse
import numpy as np


# ----------------------------------------------------------------- primitives
def mu3s(xs):
    xs = [x for x in xs if x is not None] # ignore None values (e.g. from empty rounds)
    if not xs: 
        return None
    return float(np.mean(xs)) + 3.0 * (float(np.std(xs)) if len(xs) > 1 else 0.0) # mu + 3*sigma


def _cfg(run, key, default):
    v = (run.get("config", {}) or {}).get(key) # get config value, or None if missing
    return default if v is None else v 


# ------------------------------------------------------------------- io/select
def load(globs):
    out = []
    # load all JSON files matching the given glob patterns
    for g in (globs if isinstance(globs, (list, tuple)) else [globs]):
        for f in sorted(glob.glob(g)):
            try:
                out.append((f, json.load(open(f))))
            except Exception as e:
                print("  (skip", f, "->", e, ")")
    return out


def fam(run):
    # return the manifest family of a run (or None if missing)
    return (run.get("manifest", {}) or {}).get("family")


def is_honest_run(run):
    """honest run to claibrate on (no free-riders)"""
    if run.get("free_rider_indices"):
        return False
    for h in run.get("history", []):
        for p in (h.get("wm_per_client") or []):
            if p.get("is_free_rider"):
                return False
    return True


# ------------------------------------------------------------- calib windows
def _calib_tagged_rounds(run):
    tagged = set()
    # find all rounds where a free-rider client performed a calibration action
    for c in ((run.get("compute", {}) or {}).get("per_client", {}) or {}).values():
        if c.get("is_free_rider"):
            for t in c.get("trace", []):
                if t.get("action") == "calib":
                    tagged.add(t["round"]) 
    return tagged


def calib_window(run):
    """[lo, hi] calibration rounds (for shading plots)"""
    tagged = _calib_tagged_rounds(run)
    if tagged:
        return min(tagged), max(tagged)
    W = int(_cfg(run, "autop_honest_until", 12))
    K = int(_cfg(run, "autop_calib_rounds", 4))
    return W - K, W - 1


def freeride_start(run):
    """W = first free-riding round = last calib round + 1 (else config W)."""
    tagged = _calib_tagged_rounds(run)
    if tagged:
        return max(tagged) + 1
    return int(_cfg(run, "autop_honest_until", 12))


def last_round(runs):
    # return the last round number across all runs (or 50 if no history)
    return max((h.get("round", 0) for r in runs for h in r.get("history", [])), default=50)


# ------------------------------------------------------ eta
def round_means(runs, tail=20, honest_only=True):   # TODO hardcoded tail=20 (~converged region of a 50-round run; paper Fig.8 saturates ~round 30)
    """m_r = mean BER over clients per round, pooled across runs (converged tail)
    tail>0 keeps the last N rounds; tail=0 uses all rounds."""
    ms = []
    for r in runs:
        hist = r.get("history", [])
        if tail and tail > 0:
            hist = hist[-tail:]
        for h in hist:
            vals = [p["ber"] for p in (h.get("wm_per_client") or [])
                    if not (honest_only and p.get("is_free_rider"))]
            if vals:
                ms.append(float(np.mean(vals)))
    return ms


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
      1. RECOMPUTE eta from the honest runs and compare to eta_calibrated.json.
      2. Confirm every NON-honest run actually USED the frozen constant, i.e. its
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


# --------------------------------------------------------------------- CLI
def _cli():
    ap = argparse.ArgumentParser(description="calibrate the canonical detection threshold eta")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--in", dest="inp", nargs="+", required=True)
    c.add_argument("--honest-family", default=None,
                   help="restrict to this manifest family (e.g. honest_iid).")
    c.add_argument("--tail", type=int, default=20, help="last N rounds (0 = ALL rounds).")
    c.add_argument("--out", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--in", dest="inp", nargs="+", required=True)
    v.add_argument("--honest-family", default=None)
    v.add_argument("--tail", type=int, default=20)
    v.add_argument("--eta-file", default=None, help="eta_calibrated.json to compare against")
    a = ap.parse_args()
    if a.cmd == "verify":
        import sys as _s
        _s.exit(0 if verify(a.inp, a.honest_family, a.tail, a.eta_file) else 1)
    if a.cmd == "calibrate":
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


if __name__ == "__main__":
    _cli()