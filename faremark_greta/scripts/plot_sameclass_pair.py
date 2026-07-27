#!/usr/bin/env python
"""plot_sameclass_pair.py -- the "same class, side by side" timeline.

For a same-trigger-class experiment (A4 / AK), plot the ONE free-rider and the
ONE (or few) honest client that live on the SAME trigger class, as individual
per-round BER lines -- NOT the pooled means. This is the plot that shows, on a
single class, that the free-rider's mark is at least as clean as the honest
client's, and that the frozen eta flags the wrong one.

    # one seed (uses the exact honest/FR clients found in the run):
    python plot_sameclass_pair.py --in result.json --out figs/A4_pair

    # aggregate several seeds of the same family (mean +/- band per client):
    python plot_sameclass_pair.py \
        --in 'results/A4_sameclass_c100_c6_rep*/result.json' \
        --family A4_sameclass_c100_c6 --out figs/A4_pair

    # force which class to show (default = the free-rider's trigger class):
    python plot_sameclass_pair.py --in result.json --class 6 --out figs/A4_pair

Reads result.json directly (history[*].wm_per_client[*] = {cid, trigger_class,
ber, is_free_rider, flagged}). Self-contained; no repo imports.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(patterns, family=None):
    runs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if family and (d.get("manifest") or {}).get("family") != family:
                continue
            runs.append(d)
    return runs


def series(run):
    """-> rounds[list[int]], eta[list[float]], and per-cid dict:
    cid -> {"tc": trigger_class, "fr": bool, "ber": {round: ber}, "flag": {round: bool}}"""
    rounds, eta = [], []
    cid_info = {}
    for h in run.get("history", []):
        r = h.get("round")
        if r is None:
            continue
        pcs = h.get("wm_per_client")
        if not pcs:
            continue
        rounds.append(r)
        eta.append(h.get("wm_eta_round"))
        for p in pcs:
            cid = p["cid"]
            d = cid_info.setdefault(cid, {"tc": p.get("trigger_class"),
                                          "fr": bool(p.get("is_free_rider")),
                                          "ber": {}, "flag": {}})
            d["ber"][r] = p.get("ber")
            d["flag"][r] = p.get("flagged")
    return rounds, eta, cid_info


def pick_class(cid_info, forced=None):
    """Default target class = the free-rider's trigger class."""
    if forced is not None:
        return int(forced)
    for cid, d in sorted(cid_info.items()):
        if d["fr"] and d["tc"] is not None:
            return int(d["tc"])
    # fallback: the most-shared class
    from collections import Counter
    c = Counter(d["tc"] for d in cid_info.values() if d["tc"] is not None)
    return c.most_common(1)[0][0] if c else None


def collect(runs, target_class):
    """Across runs, gather per-round BER arrays for every (role, cid) that sits
    on target_class. Returns fr_curves, honest_curves as dicts:
        label -> (rounds_sorted, mean_ber, std_ber, flag_frac)
    Curves are aggregated by CID across seeds so 'honest cid6' is one line."""
    # union of rounds
    all_rounds = sorted({r for run in runs for r in series(run)[0]})
    fr_acc, hon_acc = {}, {}
    eta_acc = {r: [] for r in all_rounds}
    for run in runs:
        rounds, eta, cid_info = series(run)
        for r, e in zip(rounds, eta):
            if e is not None:
                eta_acc[r].append(e)
        for cid, d in cid_info.items():
            if d["tc"] != target_class:
                continue
            bucket = fr_acc if d["fr"] else hon_acc
            slot = bucket.setdefault(cid, {r: [] for r in all_rounds})
            fslot_key = ("flag", cid)
            for r in all_rounds:
                if d["ber"].get(r) is not None:
                    slot[r].append(d["ber"][r])

    def finalize(bucket, role):
        out = {}
        for cid, per_round in sorted(bucket.items()):
            rr = [r for r in all_rounds if per_round[r]]
            mean = np.array([np.mean(per_round[r]) for r in rr])
            std = np.array([np.std(per_round[r]) for r in rr])
            out[cid] = (np.array(rr), mean, std)
        return out

    eta_line = np.array([np.mean(eta_acc[r]) if eta_acc[r] else np.nan
                         for r in all_rounds])
    return all_rounds, eta_line, finalize(fr_acc, "fr"), finalize(hon_acc, "hon")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--family", default=None,
                    help="if --in is a glob over many seeds, keep only this family")
    ap.add_argument("--class", dest="cls", type=int, default=None,
                    help="trigger class to show (default: the free-rider's)")
    ap.add_argument("--warmup", type=int, default=None,
                    help="round free-riding starts (draws the grey marker); "
                         "default = read from config.autop_honest_until")
    ap.add_argument("--out", default="sameclass_pair")
    a = ap.parse_args()

    runs = load(a.inp, a.family)
    if not runs:
        raise SystemExit(f"no runs matched {a.inp}"
                         + (f" family={a.family}" if a.family else ""))

    # class + warmup from the first run's config if not given
    cfg0 = runs[0].get("config", {})
    _, _, cid_info0 = series(runs[0])
    target = pick_class(cid_info0, a.cls)
    if target is None:
        raise SystemExit("could not determine a trigger class to plot")
    warmup = a.warmup if a.warmup is not None else cfg0.get("autop_honest_until")
    calib_k = cfg0.get("autop_calib_rounds")
    fam = (runs[0].get("manifest") or {}).get("family", "?")

    all_rounds, eta_line, fr_curves, hon_curves = collect(runs, target)
    x0, x1 = min(all_rounds), max(all_rounds)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # shading: warmup (forced-honest) and free-riding region
    if warmup:
        ax.axvspan(x0 - 0.5, warmup - 0.5, color="#f4a300", alpha=.10, zorder=0,
                   label="forced-honest warmup")
        if calib_k:
            ax.axvspan(warmup - calib_k - 0.5, warmup - 0.5, color="#2ca02c",
                       alpha=.12, zorder=0, label=f"calibration window (eta frozen)")
        ax.axvline(warmup - 0.5, color="0.4", ls="--", lw=1.2, zorder=1)
        ax.text(warmup - 0.3, ax.get_ylim()[1] * 0.96, "free-riding starts",
                color="0.35", fontsize=9, va="top")

    # eta line (frozen constant, but plot the per-round record in case it varied)
    if np.isfinite(np.nanmean(eta_line)):
        eta_val = float(np.nanmean(eta_line))
        ax.axhline(eta_val, color="k", ls="--", lw=2.0, zorder=6,
                   label=f"frozen detection threshold eta = {eta_val:.3f}")

    # honest client(s) on this class -- BLUE
    blues = plt.cm.Blues(np.linspace(0.55, 0.95, max(len(hon_curves), 1)))
    for (cid, (rr, mean, std)), col in zip(sorted(hon_curves.items()), blues):
        ax.plot(rr, mean, color=col, lw=2.4, zorder=5,
                label=f"HONEST cid{cid}  (class {target})")
        if len(runs) > 1:
            ax.fill_between(rr, mean - std, mean + std, color=col, alpha=.18, zorder=2)

    # free-rider(s) on this class -- ORANGE/RED
    reds = plt.cm.Oranges(np.linspace(0.6, 0.95, max(len(fr_curves), 1)))
    for (cid, (rr, mean, std)), col in zip(sorted(fr_curves.items()), reds):
        ax.plot(rr, mean, color=col, lw=2.6, zorder=5, marker="v", markersize=4,
                label=f"FREE-RIDER cid{cid}  (class {target})")
        if len(runs) > 1:
            ax.fill_between(rr, mean - std, mean + std, color=col, alpha=.18, zorder=2)

    seeds = len(runs)
    ax.set_xlabel("communication round")
    ax.set_ylabel("bit-error-rate  (0 = mark present · 0.5 = coin flip = no mark)")
    ax.set_title(f"Same trigger class, side by side  ·  {fam}  ·  class {target}  ·  "
                 f"{seeds} seed{'s' if seeds > 1 else ''}")
    ax.set_ylim(-0.03, 0.55)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=.95, ncol=1)
    cap = ("One line per client on this class. The free-rider (orange) and the honest "
           "client (blue) share the class, key differs only by draw.\n"
           "A free-rider whose BER stays below eta is NOT flagged; an honest client "
           "above eta IS (a false positive).")
    fig.text(0.01, -0.02, cap, fontsize=8, color="0.35", ha="left", va="top")
    fig.tight_layout()

    png = a.out if a.out.endswith(".png") else a.out + ".png"
    os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")
    # also print the tail numbers so you can quote them
    def tail_mean(curves, k=20):
        for cid, (rr, mean, std) in sorted(curves.items()):
            sel = mean[-k:] if len(mean) >= k else mean
            print(f"  {'FR' if curves is fr_curves else 'honest'} cid{cid}: "
                  f"tail-{k} mean BER = {np.mean(sel):.4f}")
    print(f"class {target} tail summary ({seeds} seed(s)):")
    tail_mean(fr_curves)
    tail_mean(hon_curves)


if __name__ == "__main__":
    main()
