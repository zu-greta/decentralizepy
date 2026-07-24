#!/usr/bin/env python
"""plot_all_thresholds.py -- every candidate threshold on ONE timeline, plus a
table that spells out exactly how each number was produced.

This is plot request (a). `plots.py thresholds` shows the derivation of the single
canonical eta; this shows ALL of them against the honest BER trace, so you can see
at a glance that no horizontal line separates anything.

    python scripts/plot_all_thresholds.py \
        --in 'results/*/result.json' --family R1_paper_c100_nc100 \
        --tail 20 --out figs/thresholds_all_R1

Writes <out>.png and <out>.md (the explanation table, paste-ready).

Self-contained: reads result.json directly, no resultio/detection import.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------- loading
def load(patterns, family):
    runs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if (d.get("manifest") or {}).get("family") == family:
                runs.append(d)
    return runs


def per_round(run):
    """[(round_index, [ber of each honest client])] over the whole run."""
    out = []
    for i, h in enumerate(run.get("history", [])):
        pcs = h.get("wm_per_client")
        if not pcs:
            continue
        bers = [p["ber"] for p in pcs
                if not p.get("is_free_rider") and p.get("ber") is not None]
        if bers:
            out.append((i + 1, bers))
    return out


# ---------------------------------------------------------------- rules
def _mu_k_sigma(xs, k=3.0):
    return st.mean(xs) + k * (st.pstdev(xs) if len(xs) > 1 else 0.0)


def _median_k_mad(xs, k=3.0):
    med = st.median(xs)
    mad = st.median([abs(x - med) for x in xs])
    return med + k * 1.4826 * mad


def _trimmed(xs, trim=0.10, k=3.0):
    s = sorted(xs)
    c = int(len(s) * trim)
    s = s[c:len(s) - c] or s
    return _mu_k_sigma(s, k)


def _adaptive_clip(xs, k=3.0, iters=10):
    s = list(xs)
    for _ in range(iters):
        if len(s) < 3:
            break
        mu, sd = st.mean(s), (st.pstdev(s) if len(s) > 1 else 0.0)
        keep = [x for x in s if x <= mu + k * sd]
        if len(keep) == len(s):
            break
        s = keep
    return _mu_k_sigma(s, k), len(s) / max(len(xs), 1)


def _pct(xs, q):
    return float(np.percentile(np.asarray(xs), q))


def build_rules(runs, tail):
    """All honest-only threshold rules. Returns (rules, H, round_means)."""
    H, round_means, per_seed_eta = [], [], []
    for r in runs:
        rounds = per_round(r)
        sel = rounds[-tail:] if tail else rounds
        rm = [st.mean(b) for _, b in sel]
        round_means += rm
        for _, b in sel:
            H += b
        if rm:
            per_seed_eta.append(_mu_k_sigma(rm, 3.0))

    ac, kept = _adaptive_clip(H)
    rules = {
        "coded (paper, mean-over-clients then mu+3s over rounds, avg over seeds)":
            (st.mean(per_seed_eta) if per_seed_eta else float("nan"),
             "for each seed: average BER over the N clients in each round -> one number per "
             "round; take mu+3*sigma of those; average across seeds. This is what the paper's "
             "text most plausibly means and what run_all.sh freezes."),
        "pooled (mu+3s over all seeds' round-means at once)":
            (_mu_k_sigma(round_means, 3.0),
             "same as above but pool every (seed, round) mean into one sample before mu+3*sigma. "
             "Looser, because between-seed spread is added to the sigma."),
        "loose (mu+3s over PER-CLIENT BER)":
            (_mu_k_sigma(H, 3.0),
             "mu and sigma of individual client-round BERs -- no averaging over clients. This is "
             "the ONLY variant whose sigma matches the population the test is applied to. "
             "Roughly sqrt(N) larger than 'coded'."),
        "median + 3*MAD (robust location/scale)":
            (_median_k_mad(H, 3.0),
             "median instead of mean, 1.4826*MAD instead of sigma. Immune to outliers, but "
             "collapses to 0 when more than half the honest clients sit at BER=0."),
        "trimmed-10% mu+3s":
            (_trimmed(H, 0.10, 3.0),
             "drop the top and bottom 10% of client-rounds, then mu+3*sigma on the rest."),
        f"adaptive sigma-clip (kept {kept:.2f})":
            (ac,
             "iteratively drop points above mu+3*sigma and recompute until stable, then mu+3*sigma "
             "on what survives. Excludes the hard-class tail from its own calibration."),
        "honest p95":
            (_pct(H, 95),
             "the 95th percentile of honest client-rounds. Fixes the false-positive rate at 5% by "
             "construction -- no distributional assumption at all."),
        "honest p99":
            (_pct(H, 99),
             "the 99th percentile. Targets 1% FPR."),
    }
    return rules, H, round_means


# ---------------------------------------------------------------- plot
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--out", default="thresholds_all")
    a = ap.parse_args()

    runs = load(a.inp, a.family)
    if not runs:
        raise SystemExit(f"no runs with family={a.family}")

    m = ((runs[0].get("summary") or {}).get("wm_bits_m")
         or (runs[0].get("manifest") or {}).get("wm_bits_m"))
    rules, H, round_means = build_rules(runs, a.tail)

    # honest mean BER per round, averaged over seeds
    by_round = {}
    for r in runs:
        for rnd, bers in per_round(r):
            by_round.setdefault(rnd, []).extend(bers)
    xs = sorted(by_round)
    mean_line = [st.mean(by_round[x]) for x in xs]
    p90_line = [_pct(by_round[x], 90) for x in xs]
    max_line = [max(by_round[x]) for x in xs]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(xs, mean_line, lw=2.5, color="#1f77b4", label="honest MEAN BER (what eta is built from)", zorder=5)
    ax.plot(xs, p90_line, lw=1.4, color="#1f77b4", ls="--", alpha=.75, label="honest p90 per round")
    ax.plot(xs, max_line, lw=1.0, color="#1f77b4", ls=":", alpha=.6, label="honest WORST client per round")
    ax.fill_between(xs, mean_line, max_line, color="#1f77b4", alpha=.10,
                    label="honest spread (mean -> worst): what the test is applied to")

    if xs:
        ax.axvspan(max(xs) - a.tail + 1, max(xs), color="0.85", alpha=.55, zorder=0,
                   label=f"converged tail (last {a.tail}) = calibration window")

    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (name, (eta, _)) in enumerate(sorted(rules.items(), key=lambda kv: kv[1][0])):
        if eta is None or not np.isfinite(eta):
            continue
        fpr = float(np.mean([h >= eta for h in H]))
        ax.axhline(eta, color=cols[i % 10], lw=1.8, ls="-", alpha=.9,
                   label=f"{name.split(' (')[0]}  eta={eta:.3f}  FPR={fpr:.0%}")

    if m:
        ax.axhline(1.0 / m, color="crimson", lw=2.2, ls="-.", zorder=6,
                   label=f"1/m = {1.0/m:.3f}  (smallest non-zero BER; any eta BELOW this "
                         f"means 'one bit wrong')")

    ax.set_xlabel("communication round")
    ax.set_ylabel("bit-error-rate  (0.5 = coin flip = no watermark)")
    ax.set_title(f"Every candidate threshold vs the honest BER trace  ·  {a.family}  ·  "
                 f"{len(runs)} seed(s)" + (f"  ·  m={m}" if m else ""))
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=.95)
    fig.tight_layout()
    png = a.out if a.out.endswith(".png") else a.out + ".png"
    os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    fig.savefig(png, dpi=150)

    # ------------------------------------------------------------ table
    mu, sd = st.mean(H), st.pstdev(H)
    lines = [f"# Threshold rules — `{a.family}`", "",
             f"- seeds: **{len(runs)}**, calibration window: last **{a.tail}** rounds",
             f"- honest client-rounds: **{len(H)}**, mean BER **{mu:.4f}**, per-client sd **{sd:.4f}**",
             f"- watermark bits m = **{m}**, so BER can only take values 0, {1.0/m if m else 0:.3f}, "
             f"{2.0/m if m else 0:.3f}, …" if m else "", "",
             "| rule | eta | how it is computed | honest FPR | headroom | degenerate? |",
             "|---|---|---|---|---|---|"]
    for name, (eta, how) in sorted(rules.items(), key=lambda kv: kv[1][0]):
        if eta is None or not np.isfinite(eta):
            continue
        fpr = float(np.mean([h >= eta for h in H]))
        k = (eta - mu) / sd if sd > 0 else float("nan")
        deg = ("**yes** — below 1/m, so this is exactly 'flag if ≥1 bit wrong'; "
               "the value of eta does nothing" if (m and eta < 1.0 / m) else "no")
        lines.append(f"| {name} | {eta:.4f} | {how} | {fpr:.1%} | {k:+.2f}σ | {deg} |")
    lines += ["",
              "**Headroom** is `(eta − mean) / per-client sd`. The paper specifies μ+3σ, i.e. "
              "3.00σ. Any value well below that means σ was measured on a *different, narrower* "
              "population (the mean over N clients, spread σ/√N) than the one the test is applied "
              "to (individual clients, spread σ).", "",
              "**Degenerate** means the threshold falls below `1/m`, the smallest non-zero BER "
              "attainable. Every eta in `(0, 1/m)` produces an identical detector, so calibration "
              "is doing no work at all."]
    md = png[:-4] + ".md"
    open(md, "w").write("\n".join(x for x in lines if x is not None))
    print(f"wrote {png}\nwrote {md}")


if __name__ == "__main__":
    main()
