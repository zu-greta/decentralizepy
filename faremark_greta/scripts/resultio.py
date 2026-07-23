#!/usr/bin/env python
"""resultio -- results.json

Everything that reads result.json imports from this file

THE CONTRACT (what analysis code is allowed to rely on)
-------------------------------------------------------
    schema_version                          int, 2 = post-cleanup runs
    manifest.family                         grouping key for plots/calibration
    manifest.{sweep_var, sweep_level}       sweep axis, if any
    seed, config{...}                       full ExpConfig snapshot
    free_rider_indices                      [cid, ...]
    summary{...}                            flat digest (NEW, v2; see _summary())
    per_class.by_class[c]  = {acc, loss, n} final-model per-class test metrics
    history[r] = {
        round, test_acc,
        wm_eta_round, wm_benign_ber, wm_fr_ber, wm_fpr, wm_fr_recall,
        wm_detect_acc, wm_flagged_cids,
        wm_per_client[i] = {cid, trigger_class, ber, is_free_rider, flagged,
                            pmax, entropy, dominance, trig_acc},
    }
    compute.summary{honest_mean_*, fr_mean_*, effort_ratio_*}
    compute.per_client[cid] = {attack_name, is_free_rider, total{...},
                               per_round{r: {samples, gpu_ms, trained}},
                               trace[{round, action, ...}], wm_stats{r: {...}}}

CLI (for eyeballing a run without opening the JSON):
    python scripts/resultio.py digest --in 'results/*/result.json'
    python scripts/resultio.py digest --in 'results/*/result.json' --family honest_c100_bdef_iid
    python scripts/resultio.py contract --in results/<run>/result.json   # key inventory
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

SCHEMA_VERSION = 2

# tail=20 -> last 20 of 50 rounds = the converged region (paper Fig. 8 saturates ~round 30)
DEFAULT_TAIL = 20


# --------------------------------------------------------------------------- io
def load(globs, with_path=True):
    """Load every result.json matching `globs`.

    with_path=True  -> [(path, run), ...]   (what threshold.py expects)
    with_path=False -> [run, ...]           (what separability.py expects)
    Unreadable files are skipped with a note rather than killing the run.
    """
    out = []
    for g in (globs if isinstance(globs, (list, tuple)) else [globs]):
        for f in sorted(glob.glob(g)):
            try:
                r = json.load(open(f))
            except Exception as e:
                print(f"  (skip {f} -> {e})")
                continue
            out.append((f, r) if with_path else r)
    return out


def schema_version(run) -> int:
    """1 = pre-cleanup run (no schema_version key), 2 = current."""
    return int(run.get("schema_version", 1))


def family(run):
    return (run.get("manifest", {}) or {}).get("family")


fam = family # alias

def cfg(run, key, default=None):
    v = (run.get("config", {}) or {}).get(key)
    return default if v is None else v


def is_honest_run(run) -> bool:
    """True iff the run contains no free-riders at all (calibration source)."""
    if run.get("free_rider_indices"):
        return False
    for h in run.get("history", []):
        for p in (h.get("wm_per_client") or []):
            if p.get("is_free_rider"):
                return False
    return True


def select(runs, family=None, honest=None):
    """Filter loaded runs by manifest family and/or honest-ness.

    Accepts either the (path, run) or the bare-run list shape and returns the
    same shape it was given.
    """
    def _run(x):
        return x[1] if isinstance(x, tuple) else x

    out = runs
    if family is not None:
        out = [x for x in out if fam(_run(x)) == family]
    if honest is not None:
        out = [x for x in out if is_honest_run(_run(x)) == bool(honest)]
    return out


def runs_only(runs):
    """Drop paths: [(path, run), ...] or [run, ...] -> [run, ...]."""
    return [x[1] if isinstance(x, tuple) else x for x in runs]


# ------------------------------------------------------------------- extraction
def history(run, tail=0):
    """Rounds of a run; tail>0 keeps only the last N (converged tail)."""
    h = run.get("history", []) or []
    return h[-tail:] if (tail and tail > 0) else h


def per_client_bers(runs, tail=DEFAULT_TAIL, free_rider=False, trigger_class=None):
    """Flat list of per-client BERs over the tail rounds ("H" or "F" in the docs).

    free_rider=False -> honest clients only; True -> free-riders only.
    trigger_class=int restricts to one trigger class (the per-class slice).
    """
    out = []
    for r in runs_only(runs):
        for h in history(r, tail):
            for p in (h.get("wm_per_client") or []):
                if bool(p.get("is_free_rider")) != bool(free_rider):
                    continue
                if trigger_class is not None and int(p.get("trigger_class", -1)) != int(trigger_class):
                    continue
                if p.get("ber") is not None:
                    out.append(float(p["ber"]))
    return out


def per_client_ber_pairs(runs, tail=DEFAULT_TAIL, free_rider=False):
    """(trigger_class, ber) pairs -- the per-class flavour separability.py needs"""
    out = []
    for r in runs_only(runs):
        for h in history(r, tail):
            for p in (h.get("wm_per_client") or []):
                if bool(p.get("is_free_rider")) == bool(free_rider):
                    out.append((int(p["trigger_class"]), float(p["ber"])))
    return out


def round_means(runs, tail=DEFAULT_TAIL, honest_only=True):
    """m_r = mean BER over clients within a round, pooled across runs.

    This is the quantity the live eta is built on. Its spread is ~sigma/sqrt(N)
    because it averages over clients first
    """
    ms = []
    for r in runs_only(runs):
        for h in history(r, tail):
            vals = [p["ber"] for p in (h.get("wm_per_client") or [])
                    if p.get("ber") is not None
                    and not (honest_only and p.get("is_free_rider"))]
            if vals:
                ms.append(float(np.mean(vals)))
    return ms


def bers_by_class(runs, tail=DEFAULT_TAIL, free_rider=False):
    """{trigger_class: [ber, ...]} -- the per-class floor."""
    out = {}
    for r in runs_only(runs):
        for h in history(r, tail):
            for p in (h.get("wm_per_client") or []):
                if bool(p.get("is_free_rider")) != bool(free_rider):
                    continue
                if p.get("ber") is None:
                    continue
                out.setdefault(int(p.get("trigger_class", -1)), []).append(float(p["ber"]))
    return out


def trigger_classes(run):
    """Observed trigger classes, and how many clients share each one"""
    h = run.get("history", []) or []
    if not h:
        return {}
    counts = {}
    for p in (h[-1].get("wm_per_client") or []):
        c = int(p.get("trigger_class", -1))
        counts[c] = counts.get(c, 0) + 1
    return counts


# --------------------------------------------------------- attacker schedule
def _calib_tagged_rounds(run):
    """Rounds a free-rider tagged 'calib' in its trace (authoritative window)"""
    tagged = set()
    for c in ((run.get("compute", {}) or {}).get("per_client", {}) or {}).values():
        if c.get("is_free_rider"):
            for t in (c.get("trace") or []):
                if t.get("action") == "calib":
                    tagged.add(t["round"])
    return tagged


def calib_window(run):
    """[lo, hi] calibration rounds -- used to shade plots"""
    tagged = _calib_tagged_rounds(run)
    if tagged:
        return min(tagged), max(tagged)
    W = int(cfg(run, "autop_honest_until", 12))
    K = int(cfg(run, "autop_calib_rounds", 4))
    return W - K, W - 1


def freeride_start(run):
    """W = first free-riding round = last calib round + 1 (else the config W)"""
    tagged = _calib_tagged_rounds(run)
    return (max(tagged) + 1) if tagged else int(cfg(run, "autop_honest_until", 12))


def last_round(runs):
    return max((h.get("round", 0) for r in runs_only(runs)
                for h in r.get("history", [])), default=50)


# ------------------------------------------------------------------- summaries
def summary_of(run) -> dict:
    """The flat digest of a run"""
    s = run.get("summary")
    if isinstance(s, dict) and s:
        return s
    # ---- v1 fallback: reassemble from the old flat top level ----
    return {
        "family": family(run),
        "seed": run.get("seed"),
        "rounds": len(run.get("history", []) or []),
        "num_clients": cfg(run, "num_clients"),
        "attack": run.get("attack"),
        "n_free_riders": len(run.get("free_rider_indices") or []),
        "final_acc": run.get("final_acc"),
        "best_acc": run.get("best_acc"),
        "wm_benign_ber": run.get("wm_benign_ber"),
        "wm_fr_ber": run.get("wm_fr_ber"),
        "wm_fpr": run.get("wm_fpr"),
        "wm_fr_recall": run.get("wm_fr_recall"),
        "wm_detect_acc": run.get("wm_detect_acc"),
        "wm_eta_used": run.get("wm_eta_used"),
        "wm_bits_m": run.get("wm_bits_m"),
        "wm_group_size_l": run.get("wm_group_size_l"),
        "wm_unembeddable_frac": run.get("wm_unembeddable_frac"),
        "elapsed_min": (round(run["elapsed_sec"] / 60.0, 1)
                        if run.get("elapsed_sec") is not None else None),
    }


def compute_summary(run) -> dict:
    return ((run.get("compute", {}) or {}).get("summary", {}) or {})


def wm_accuracy(run, tail=1):
    """Watermark accuracy % = 100*(1 - mean honest BER). The paper's headline."""
    b = per_client_bers([run], tail=tail, free_rider=False)
    return 100.0 * (1.0 - float(np.mean(b))) if b else None


def test_acc(run, tail=1):
    xs = [h.get("test_acc") for h in history(run, tail) if h.get("test_acc") is not None]
    return float(np.mean(xs)) if xs else None


# --------------------------------------------------------------------- CLI
def _digest(args):
    runs = load(args.inp)
    if args.family:
        runs = select(runs, family=args.family)
    if not runs:
        raise SystemExit(f"no result.json matched (in={args.inp} family={args.family})")
    hdr = (f"{'run':<44}{'v':>2}{'seed':>6}{'acc%':>8}{'wm_ber':>9}"
           f"{'fr_ber':>8}{'fpr':>7}{'recall':>8}{'eta':>8}")
    print(hdr); print("-" * len(hdr))
    for f, r in runs:
        s = summary_of(r)
        tag = os.path.basename(os.path.dirname(f))[:43]

        def _n(v, w, p=3):
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
        print(f"{tag:<44}{schema_version(r):>2}{str(s.get('seed')):>6}"
              f"{_n(s.get('final_acc'), 8, 2)}{_n(s.get('wm_benign_ber'), 9)}"
              f"{_n(s.get('wm_fr_ber'), 8)}{_n(s.get('wm_fpr'), 7)}"
              f"{_n(s.get('wm_fr_recall'), 8)}{_n(s.get('wm_eta_used'), 8, 4)}")
    print(f"\n{len(runs)} run(s).  v=schema_version (1=pre-cleanup, 2=current)")


def _contract(args):
    """Print the key inventory of one run -- what a consumer may rely on."""
    runs = load(args.inp)
    if not runs:
        raise SystemExit("no result.json matched")
    f, r = runs[0]
    print(f"file          : {f}")
    print(f"schema_version: {schema_version(r)}")
    print(f"top-level keys: {sorted(r.keys())}")
    h = (r.get("history") or [{}])[-1]
    print(f"history[-1]   : {sorted(h.keys())}")
    pc = (h.get("wm_per_client") or [{}])[0]
    print(f"  per_client  : {sorted(pc.keys())}")
    comp = (r.get("compute", {}) or {}).get("per_client", {}) or {}
    if comp:
        k0 = sorted(comp)[0]
        print(f"compute[{k0}]   : {sorted(comp[k0].keys())}")
        pr = (comp[k0].get("per_round") or {})
        if pr:
            r0 = sorted(pr)[0]
            print(f"  per_round[{r0}]: {sorted(pr[r0].keys())}")
    print(f"trigger classes (cid count per class): {trigger_classes(r)}")


def _cli():
    ap = argparse.ArgumentParser(description="inspect result.json files")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("digest", "contract"):
        s = sub.add_parser(name)
        s.add_argument("--in", dest="inp", nargs="+", required=True)
        s.add_argument("--family", default=None)
    a = ap.parse_args()
    {"digest": _digest, "contract": _contract}[a.cmd](a)


if __name__ == "__main__":
    _cli()