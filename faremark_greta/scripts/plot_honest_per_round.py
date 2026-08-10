#!/usr/bin/env python
"""plot_honest_per_round.py -- per-ROUND honest BER and trigger-class ACCURACY.

The bar-chart `class_acc` plot shows ONE number per client (tail-mean), which hides
whether the trigger-class accuracy is zero for a data reason or a watermark reason,
and whether it is zero for the whole run or only early. This script draws the two
quantities as TIME SERIES, aggregated over seeds, one line per trigger class:

  top  panel : honest bit-error-rate per round      (lower = mark embedded)
  bottom panel: honest trigger-class TEST accuracy per round
               (argmax==trigger_class on the trigger bank -- read from
                wm_per_client[*].trig_acc, the server-side diagnostic)

Reading it:
  * BER descends toward each class's floor  -> the mark embeds.
  * trig_acc sits near 0 for the SAME rounds -> the watermark loss (Eq. 4-6/10)
    suppresses the trigger class below argmax to embed. This is EXPECTED, not a
    logging bug: it is a different model (each client's submitted watermarked
    local model) than the FedAvg global whose per-class TEST accuracy is high.
  * trig_acc==0 with BER ~0.5 (not ~0)   -> STARVATION: the client holds ~0
    images of its own trigger class (non-IID). Distinguished in the printed
    summary (suppression vs starvation split).

Self-contained; reads result.json directly (no repo imports), so it runs
identically on the cluster and locally. Mirrors plot_sameclass_pair.py's CLI.

    python plot_honest_per_round.py \
        --in 'results/A1_honest_c100_rep*/result.json' \
        --family A1_honest_c100 --out figs/A1_honest_per_round
    # non-IID (shows starvation classes flatten high):
    python plot_honest_per_round.py \
        --in 'results/E1_honest_niid_c100_rep*/result.json' \
        --family E1_honest_niid_c100 --out figs/E1_honest_per_round
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# eta reference lines (same frozen constants the other plots draw)
ETA_TIGHT_DEFAULT = 0.064
ETA_LOOSE_DEFAULT = 0.264
SUPPRESS_BER = 0.12     # trig_acc==0 AND BER < this  -> watermark suppression
STARVE_BER   = 0.30     # trig_acc==0 AND BER >= this -> data starvation


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


def collect(runs):
    """-> {cls: {round: [ber...]}}, {cls: {round: [trig_acc...]}}, sorted rounds.
    Honest clients only (is_free_rider False). Keyed by trigger_class so seeds
    aggregate per class."""
    ber = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(lambda: defaultdict(list))
    rounds = set()
    for d in runs:
        for h in d.get("history", []):
            r = h.get("round")
            if r is None:
                continue
            rounds.add(r)
            for p in h.get("wm_per_client") or []:
                if p.get("is_free_rider"):
                    continue
                c = p.get("trigger_class")
                if c is None:
                    continue
                if p.get("ber") is not None:
                    ber[c][r].append(p["ber"])
                if p.get("trig_acc") is not None:
                    acc[c][r].append(p["trig_acc"])
    return ber, acc, sorted(rounds)


def _series(bucket, cls, rounds):
    rr = [r for r in rounds if bucket[cls].get(r)]
    mean = np.array([np.mean(bucket[cls][r]) for r in rr])
    std = np.array([np.std(bucket[cls][r]) for r in rr])
    return np.array(rr), mean, std


def summarize(runs, tail=20):
    """Print the suppression-vs-starvation split so the 'zero accuracy' is explained."""
    n_cr = n_pos = supp = starv = mid = 0
    tail_acc, tail_ber = [], []
    max_round = max((h.get("round", 0) for d in runs for h in d.get("history", [])), default=0)
    lo = max(1, max_round - tail + 1)
    for d in runs:
        for h in d.get("history", []):
            for p in h.get("wm_per_client") or []:
                if p.get("is_free_rider"):
                    continue
                ta, be = p.get("trig_acc"), p.get("ber")
                if ta is None:
                    continue
                n_cr += 1
                n_pos += int(ta > 0)
                if ta == 0 and be is not None:
                    if be < SUPPRESS_BER:
                        supp += 1
                    elif be >= STARVE_BER:
                        starv += 1
                    else:
                        mid += 1
                if h.get("round", 0) >= lo:
                    if ta is not None:
                        tail_acc.append(ta)
                    if be is not None:
                        tail_ber.append(be)
    print(f"  honest client-rounds        : {n_cr}")
    print(f"  trig_acc > 0                : {n_pos} ({100*n_pos/max(n_cr,1):.1f}%)")
    print(f"  trig_acc == 0  -> suppression(BER<{SUPPRESS_BER}) = {supp}"
          f"  |  starvation(BER>={STARVE_BER}) = {starv}  |  mid = {mid}")
    if tail_acc:
        print(f"  tail-{tail} mean trig_acc        : {np.mean(tail_acc):.4f}")
        print(f"  tail-{tail} mean BER             : {np.mean(tail_ber):.4f}")
    return dict(n_cr=n_cr, n_pos=n_pos, supp=supp, starv=starv, mid=mid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--family", default=None)
    ap.add_argument("--eta_tight", type=float, default=ETA_TIGHT_DEFAULT)
    ap.add_argument("--eta_loose", type=float, default=ETA_LOOSE_DEFAULT)
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--out", default="honest_per_round")
    a = ap.parse_args()

    runs = load(a.inp, a.family)
    if not runs:
        raise SystemExit(f"no honest runs matched {a.inp}"
                         + (f" family={a.family}" if a.family else ""))
    fam = (runs[0].get("manifest") or {}).get("family", "honest")
    nseed = len(runs)

    ber, acc, rounds = collect(runs)
    classes = sorted(set(ber) | set(acc))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(classes), 1)))

    fig, (axB, axA) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)

    # ---- top: BER per round per class ----
    for c, col in zip(classes, cmap):
        rr, mean, std = _series(ber, c, rounds)
        if len(rr) == 0:
            continue
        floor = np.mean(mean[-a.tail:]) if len(mean) >= a.tail else np.mean(mean)
        axB.plot(rr, mean, color=col, lw=2.0, label=f"cls {c} (floor {floor:.3f})")
        if nseed > 1:
            axB.fill_between(rr, mean - std, mean + std, color=col, alpha=.12)
    axB.axhline(a.eta_tight, color="black", ls="--", lw=1.8,
                label=f"η tight (ref) = {a.eta_tight:.3f}")
    axB.axhline(a.eta_loose, color="#3B6FB5", ls=(0, (5, 2)), lw=1.8,
                label=f"η loose (ref) = {a.eta_loose:.3f}")
    axB.set_ylabel("honest BER  (lower = mark embedded)")
    axB.set_ylim(-0.03, 0.6)
    axB.grid(alpha=.3)
    axB.set_title(f"Honest BER & trigger-class accuracy per round  ·  {fam}  ·  {nseed} seed(s)")
    axB.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=.9)

    # ---- bottom: trigger-class accuracy per round per class ----
    for c, col in zip(classes, cmap):
        rr, mean, std = _series(acc, c, rounds)
        if len(rr) == 0:
            continue
        tailv = np.mean(mean[-a.tail:]) if len(mean) >= a.tail else np.mean(mean)
        axA.plot(rr, mean, color=col, lw=2.0, label=f"cls {c} (tail {tailv:.2f})")
        if nseed > 1:
            axA.fill_between(rr, mean - std, mean + std, color=col, alpha=.12)
    axA.set_xlabel("communication round")
    axA.set_ylabel("honest trigger-class test acc\n(argmax == trigger class)")
    axA.set_ylim(-0.03, 1.0)
    axA.grid(alpha=.3)
    axA.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=.9)
    cap = ("Top: honest BER descends to each class's floor (mark embeds). "
           "Bottom: trigger-class accuracy stays near 0 on the SAME rounds because "
           "the watermark loss (Eq. 4-6/10) suppresses the trigger class below argmax "
           "to embed the mark -- EXPECTED, measured on each client's submitted "
           "watermarked model (a different model than the FedAvg global whose per-class "
           "TEST accuracy is high). trig_acc==0 with BER~0.5 instead means data "
           "starvation (non-IID empty trigger shard).")
    fig.text(0.01, -0.03, cap, fontsize=8, color="0.35", ha="left", va="top", wrap=True)

    fig.tight_layout()
    png = a.out if a.out.endswith(".png") else a.out + ".png"
    os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"summary for {fam} ({nseed} seed(s)):")
    summarize(runs, tail=a.tail)


if __name__ == "__main__":
    main()
