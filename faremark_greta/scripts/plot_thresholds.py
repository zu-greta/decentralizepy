#!/usr/bin/env python3
"""Threshold overlay + worth/cheap multi-metric plots.

Style: colour-blind-safe (Okabe-Ito), NO dual y-axes (uses stacked panels),
clear legends/labels/grids. See scripts/plotstyle.py.

1) overlay (per run): fr_ber + benign_ber vs round with ALL eta variants:
     python scripts/plot_thresholds.py overlay --in RUN/result.json --out figs/thr
2) worth (across configs): stacked panels — effort metrics, then BER (vs eta),
   then accuracy — mean +/- std over seeds:
     python scripts/plot_thresholds.py worth --in "$RES/*/result.json" \
            --family autopilot_scope R_frontier S_samples --out figs/worth
"""
import argparse, glob, json, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import plotstyle as ps
ps.apply()
import matplotlib.pyplot as plt
import numpy as np
from importlib import util as _u
_spec = _u.spec_from_file_location("thresholds",
        os.path.join(os.path.dirname(__file__), "..", "faremark", "thresholds.py"))
thr = _u.module_from_spec(_spec); _spec.loader.exec_module(thr)


def _load(globs):
    out = []
    for g in globs:
        for f in glob.glob(g):
            try: out.append((f, json.load(open(f))))
            except Exception: pass
    return out


def _fr_trace(r):
    """The free-rider's per-round decision trace (warmup/coast/tap), if present."""
    for _, c in r.get("compute", {}).get("per_client", {}).items():
        if c.get("is_free_rider") and c.get("trace"):
            return c["trace"]
    return None


def overlay(a):
    _, r = _load(a.inp)[0]
    h = r.get("history", [])
    rounds = [x["round"] for x in h]
    benign = [x.get("wm_benign_ber") for x in h]
    frber = [x.get("wm_fr_ber") for x in h]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(rounds, benign, color=ps.C_HONEST, lw=2.4, marker="", label="benign BER (honest clients)")
    ax.plot(rounds, frber, color=ps.C_FR, lw=2.4, marker="", label="free-rider BER")
    # all eta variants, each in its own consistent colour/linestyle (thresholds.STYLE)
    for v in thr.ALL_VARIANTS:
        stl = thr.STYLE[v]
        et = thr.eta_series([b if b is not None else 0.5 for b in benign], v)
        ax.plot(rounds, et, color=stl["color"], lw=1.4, linestyle=stl["ls"],
                label=stl["label"])
    # mark the attacker's actions (warmup / tap) on its own BER curve, so the
    # warmup -> coast -> tap -> coast "sawtooth" is visible: proof of HOW it works.
    tr = _fr_trace(r)
    if tr:
        fr_at = {x["round"]: x.get("wm_fr_ber") for x in h}
        taps = [(t["round"], fr_at.get(t["round"])) for t in tr
                if t.get("action") == "tap" and fr_at.get(t["round"]) is not None]
        warm = [(t["round"], fr_at.get(t["round"])) for t in tr
                if t.get("action") in ("warmup", "embed") and fr_at.get(t["round"]) is not None]
        if warm:
            ax.scatter(*zip(*warm), s=55, marker="s", color=ps.C_FR,
                       edgecolor="k", zorder=5, label="warmup embed (honest)")
        if taps:
            ax.scatter(*zip(*taps), s=70, marker="^", color=ps.C_FR,
                       edgecolor="k", zorder=5, label="tap (re-embed)")
    ax.set_xlabel("communication round")
    ax.set_ylabel("bit-error-rate, BER  (lower = watermark present)")
    ax.set_ylim(0, 0.7)
    txt = "   ".join(f"{v}: {thr.evades_under(frber, benign, v):.0%} evade"
                     for v in thr.ALL_VARIANTS
                     if thr.evades_under(frber, benign, v) is not None)
    ax.set_title("Free-rider vs benign BER, under every threshold (eta) definition\n" + txt)
    ax.legend(ncol=2, loc="upper right", fontsize=8)
    ps.finish(fig, a.out + ".png")


def decay(a):
    """Watermark DECAY + RE-EMBED vs round — the mechanism plot.

    Panel 1: free-rider server BER vs round. During coast rounds the mark DECAYS
      (BER climbs from floor toward eta); at a tap it drops back. The frozen eta
      line shows the ceiling it must stay under. This is 'how long the watermark
      lasts' — the coasting budget.
    Panel 2: re-embed COST per tap (batches-to-floor from the trace). This is 'how
      long it takes to re-embed properly'. Coast-rounds-gained / tap-batches-spent
      is the effort frontier, read straight off the two panels.
    Everything is post-hoc from history + the FR trace (no re-runs).
    """
    runs = _load(a.inp)
    if a.family:
        runs = [(f, r) for f, r in runs
                if (r.get("manifest", {}) or {}).get("family") in a.family]
    if not runs:
        print("no matching runs"); return
    _, r = runs[0]                       # one representative run (the sawtooth)
    h = r.get("history", [])
    rounds = [x["round"] for x in h]
    frber = [x.get("wm_fr_ber") for x in h]
    benign = [x.get("wm_benign_ber") for x in h]
    eta_f = thr.eta_series([b if b is not None else 0.5 for b in benign], "frozen")
    tr = _fr_trace(r) or []
    fr_at = {x["round"]: x.get("wm_fr_ber") for x in h}
    taps = [(t["round"], t.get("tap_batches") or 0) for t in tr if t.get("action") == "tap"]
    warm = [t["round"] for t in tr if t.get("action") in ("warmup", "embed")]

    fig, (a1, a2) = ps.stacked_panels(2, figsize=(10, 6.4), height_ratios=[2, 1])
    a1.plot(rounds, frber, color=ps.C_FR, lw=2.2, label="free-rider BER (decays while coasting)")
    a1.plot(rounds, eta_f, color=thr.STYLE["frozen"]["color"],
            ls=thr.STYLE["frozen"]["ls"], lw=1.6, label="η frozen (fair ceiling)")
    if warm:
        a1.scatter(warm, [fr_at.get(w) for w in warm], s=55, marker="s",
                   color=ps.C_FR, edgecolor="k", zorder=5, label="warmup embed")
    if taps:
        a1.scatter([t for t, _ in taps], [fr_at.get(t) for t, _ in taps], s=75,
                   marker="^", color=ps.C_FR, edgecolor="k", zorder=5, label="tap (re-embed)")
    a1.set_ylabel("free-rider BER")
    a1.set_ylim(0, 0.7)
    a1.set_title("Watermark decay (coast) and recovery (tap) vs round")
    a1.legend(loc="upper right", fontsize=8)
    if taps:
        a2.bar([t for t, _ in taps], [b for _, b in taps], width=0.8,
               color=ps.OKABE["blue"], label="batches to re-embed to floor")
    a2.set_ylabel("tap cost\n(batches)")
    a2.set_xlabel("communication round")
    a2.set_title("Re-embed cost per tap (lower = mark re-forms fast)")
    a2.legend(loc="upper right", fontsize=8)
    ps.finish(fig, a.out + ".png")
    """PROVE-IT plot: for each attack config, the fraction of converged rounds it
    EVADES under every eta variant (mean +/- std over seeds). Reading it:
      * bars high ONLY under 'cumulative' -> the evasion is the threshold artifact
        (poisoning inflated the swingy eta), NOT a real break.
      * bars high under 'frozen'/'converged' (the fair thresholds) -> a genuine,
        credible evasion.
    """
    runs = _load(a.inp)
    groups = {}
    for _, r in runs:
        m = r.get("manifest", {}) or {}
        if a.family and m.get("family") not in a.family:
            continue
        lab = (m.get("note") or f"{r.get('attack')}").strip()
        benign = [hh.get("wm_benign_ber") for hh in r.get("history", [])]
        frb = [hh.get("wm_fr_ber") for hh in r.get("history", [])]
        if not benign:
            continue
        ev = thr.summary_evasion(frb, benign)
        groups.setdefault(lab, []).append(ev)
    labs = list(groups.keys())
    if not labs:
        print("no matching runs"); return
    x = np.arange(len(labs))
    nV = len(thr.ALL_VARIANTS)
    w = 0.8 / nV
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(labs)), 5.6))
    for i, v in enumerate(thr.ALL_VARIANTS):
        means = [np.mean([g[v] for g in groups[l] if g[v] is not None] or [0]) for l in labs]
        stds = [np.std([g[v] for g in groups[l] if g[v] is not None] or [0]) for l in labs]
        ax.bar(x + (i - (nV - 1) / 2) * w, means, w, yerr=stds, capsize=2,
               color=thr.STYLE[v]["color"], label=thr.STYLE[v]["label"])
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("fraction of converged rounds evaded\n(1.0 = always slips past)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color=ps.OKABE["grey"], ls=":", lw=1)
    ax.set_title("Does the attack evade the FAIR threshold, or only the swingy one?\n"
                 "(high under 'frozen'/'converged' = real break; high only under "
                 "'cumulative' = artifact)")
    ax.legend(ncol=2, loc="upper right", fontsize=8)
    ps.finish(fig, a.out + ".png")


def _ms(xs):
    xs = [x for x in xs if x is not None]
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (0.0, 0.0)


def _fr_duty(r):
    for _, c in r.get("compute", {}).get("per_client", {}).items():
        if c.get("is_free_rider"):
            return c.get("total", {}).get("duty_cycle")
    return None


def worth(a):
    runs = _load(a.inp)
    groups = {}
    for _, r in runs:
        m = r.get("manifest", {}) or {}
        if a.family and m.get("family") not in a.family:
            continue
        lab = (m.get("note") or f"{r.get('attack')}").strip()
        cs = r.get("compute", {}).get("summary", {}) or {}
        groups.setdefault(lab, []).append(dict(
            eff_s=cs.get("effort_ratio_samples"), eff_g=cs.get("effort_ratio_gpu"),
            duty=(cs.get("duty_cycle") if cs.get("duty_cycle") is not None else _fr_duty(r)),
            frber=r.get("wm_fr_ber"), acc=r.get("final_acc"),
            benign=[h.get("wm_benign_ber") for h in r.get("history", [])],
            frb_series=[h.get("wm_fr_ber") for h in r.get("history", [])]))
    labs = list(groups.keys())
    if not labs:
        print("no matching runs"); return
    x = np.arange(len(labs)); w = 0.26

    def col(k): return [_ms([g[k] for g in groups[l]]) for l in labs]
    eff_s, eff_g, duty, frb, acc = (col("eff_s"), col("eff_g"), col("duty"),
                                    col("frber"), col("acc"))
    # eta (converged, fair) per config, averaged
    eta_ref = []
    for l in labs:
        vals = [thr.eta_series([b or 0.5 for b in g["benign"]], "converged")[-1]
                for g in groups[l] if g["benign"]]
        eta_ref.append(np.mean(vals) if vals else 0.35)

    # THREE stacked panels sharing the x-axis (NO dual axis)
    fig, (a1, a2, a3) = ps.stacked_panels(3, figsize=(max(9, 1.3 * len(labs)), 8.4),
                                          height_ratios=[1.1, 1, 1])
    # panel 1: effort (three grouped bars)
    a1.bar(x - w, [v[0] for v in eff_s], w, yerr=[v[1] for v in eff_s], capsize=3,
           color=ps.OKABE["blue"], label="effort ratio (samples)")
    a1.bar(x, [v[0] for v in eff_g], w, yerr=[v[1] for v in eff_g], capsize=3,
           color=ps.OKABE["sky"], label="effort ratio (GPU-ms)")
    a1.bar(x + w, [v[0] for v in duty], w, yerr=[v[1] for v in duty], capsize=3,
           color=ps.OKABE["orange"], label="duty cycle")
    a1.axhline(1.0, color=ps.OKABE["grey"], ls=":", lw=1)
    a1.text(0, 1.02, "honest = 1.0", fontsize=8, color=ps.OKABE["grey"])
    a1.set_ylabel("fraction of\nan honest client")
    a1.set_title("How CHEAP  (lower = cheaper)")
    a1.legend(ncol=3, loc="upper right")
    # panel 2: free-rider BER with per-config eta marker
    a2.bar(x, [v[0] for v in frb], 0.5, yerr=[v[1] for v in frb], capsize=3,
           color=ps.C_FR, label="free-rider BER")
    a2.plot(x, eta_ref, color=ps.C_ETA, marker="D", ms=6, ls="none",
            label="eta (converged, fair)")
    for xi, e in zip(x, eta_ref):
        a2.hlines(e, xi - 0.35, xi + 0.35, color=ps.C_ETA, lw=1.4)
    a2.set_ylabel("free-rider BER")
    a2.set_ylim(0, 0.7)
    a2.set_title("Does it EVADE  (BER below the eta marker = evades)")
    a2.legend(loc="upper right")
    # panel 3: accuracy (model health)
    bars = a3.bar(x, [v[0] for v in acc], 0.5, yerr=[v[1] for v in acc], capsize=3,
                  color=ps.C_ACC, label="final accuracy")
    a3.axhline(72, color=ps.OKABE["grey"], ls=":", lw=1)
    a3.text(0, 73, "honest ~72%", fontsize=8, color=ps.OKABE["grey"])
    a3.set_ylabel("accuracy (%)")
    a3.set_ylim(20, 80)
    a3.set_title("Is the model HEALTHY  (near 72% = not poisoned)")
    a3.set_xticks(x); a3.set_xticklabels(labs, rotation=30, ha="right", fontsize=9)
    a3.legend(loc="upper right")
    fig.suptitle("Worth / cheap: effort vs evasion vs model health  (mean +/- std over seeds)",
                 fontsize=13, fontweight="bold")
    ps.finish(fig, a.out + ".png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("overlay", "worth", "evade_bars", "decay"):
        s = sub.add_parser(name)
        s.add_argument("--in", dest="inp", nargs="+", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--family", nargs="+", default=None)
    a = ap.parse_args()
    {"overlay": overlay, "worth": worth, "evade_bars": evade_bars,
     "decay": decay}[a.cmd](a)