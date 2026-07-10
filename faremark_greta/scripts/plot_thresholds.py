#!/usr/bin/env python3
"""Threshold overlay + worth/cheap multi-metric plots

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


# What the effort ratio is measured on, spelled out wherever it appears.
EFFORT_CAP = ("effort ratio = free-rider image-passes \u00f7 honest image-passes\n"
              "(image-passes = samples = \u03a3 batch_size over all trained batches; honest = 1.0)")


def _cpc_label(v):
    """Human-readable label for an autop_common_per_class level.
      -1  -> 'full shard'      (train on the whole local shard, like honest)
       0  -> 'triggers only'   (only trigger-class images; overfits, see paper Table V)
       N  -> '+N/class'        (trigger imgs + N random images per common class)"""
    try:
        iv = int(float(v))
    except (TypeError, ValueError):
        return str(v)
    if iv < 0:
        return "full shard"
    if iv == 0:
        return "triggers\nonly"
    return f"+{iv}/class"


def _scope_of(groups, levels):
    """The autop_scope (full|block2|...) shared by the runs in a knob sweep."""
    for l in levels:
        for g in groups.get(l, []):
            if g.get("scope") and g["scope"] != "?":
                return g["scope"]
    return "?"


def _knob_axis_label(sweep_var):
    if sweep_var == "autop_common_per_class":
        return "training data per tap  (triggers-only \u2192 +N/common-class \u2192 full shard)"
    if sweep_var == "autop_max_batches":
        return "tap budget (batches)  \u2014 samples/tap = batches \u00d7 16"
    return sweep_var or "knob"


def _pick(runs, family):
    """First run matching --family (so overlay/decay show the dataset you asked
    for, not just whatever glob returned first)."""
    if family:
        for f, r in runs:
            if (r.get("manifest", {}) or {}).get("family") in family:
                return f, r
    return runs[0]


def overlay(a):
    _, r = _pick(_load(a.inp), a.family)
    h = r.get("history", [])
    rounds = [x["round"] for x in h]
    benign = [x.get("wm_benign_ber") for x in h]
    frber = [x.get("wm_fr_ber") for x in h]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(rounds, benign, color=ps.C_HONEST, lw=2.4, marker="", label="benign BER (honest clients)")
    ax.plot(rounds, frber, color=ps.C_FR, lw=2.4, marker="", label="free-rider BER")
    # all eta variants, each in its own consistent colour/linestyle 
    for v in thr.ALL_VARIANTS:
        stl = thr.STYLE[v]
        et = thr.eta_series([b if b is not None else 0.5 for b in benign], v)
        ax.plot(rounds, et, color=stl["color"], lw=1.4, linestyle=stl["ls"],
                label=stl["label"])
    # mark the attacker's actions (warmup / tap) on its own BER curve, so the
    # warmup -> coast -> tap -> coast is visible
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
               color=ps.OKABE["blue"], label="training samples to re-embed")
    a2.set_ylabel("samples\nto re-embed")
    a2.set_xlabel("communication round")
    a2.set_title("Re-embed cost per tap (lower = mark re-forms fast)")
    a2.legend(loc="upper right", fontsize=8)
    ps.finish(fig, a.out + ".png")


def evade_bars(a):
    """PROVE-IT plot: for each attack config, the fraction of converged rounds it
    EVADES under every eta variant (mean +/- std over seeds). Reading it:
      * bars high only under 'cumulative' -> the evasion is the threshold artifact
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
    ax.set_title("attack evade the fair threshold?\n"
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
    a1.set_title("How cheap  (lower = cheaper)")
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
    a2.set_title("Evasion  (BER below the eta marker = evades)")
    a2.legend(loc="upper right")
    # panel 3: accuracy (model health)
    bars = a3.bar(x, [v[0] for v in acc], 0.5, yerr=[v[1] for v in acc], capsize=3,
                  color=ps.C_ACC, label="final accuracy")
    a3.axhline(72, color=ps.OKABE["grey"], ls=":", lw=1)
    a3.text(0, 73, "honest ~72%", fontsize=8, color=ps.OKABE["grey"])
    a3.set_ylabel("accuracy (%)")
    a3.set_ylim(20, 80)
    a3.set_title("Is the model accuracy ok  (near 72% = not poisoned)")
    a3.set_xticks(x); a3.set_xticklabels(labs, rotation=30, ha="right", fontsize=9)
    a3.legend(loc="upper right")
    fig.suptitle("Worth / cheap: effort vs evasion vs model health  (mean +/- std over seeds)",
                 fontsize=13, fontweight="bold")
    ps.finish(fig, a.out + ".png")


def _phases(tr):
    """From the FR trace: the FORCED rounds (honest warmup, where it trains like an
    honest client to calibrate/seed the mark) and the first coast/tap round (where
    the submarine phase begins). Returns (forced_rounds, phase_start_round)."""
    if not tr:
        return [], None
    forced = [t["round"] for t in tr if t.get("action") in ("honest", "warmup", "embed")]
    later = [t["round"] for t in tr if t.get("action") in ("coast", "tap")]
    return forced, (min(later) if later else None)


def _mark_phases(ax, tr, y=None):
    """Shade the forced warmup region and draw the 'submarine begins' divider."""
    forced, start = _phases(tr)
    if forced:
        ax.axvspan(min(forced) - 0.5, max(forced) + 0.5, color=ps.OKABE["yellow"],
                   alpha=0.16, zorder=0, label="forced honest warmup (calibrates η)")
    if start is not None:
        ax.axvline(start - 0.5, color=ps.OKABE["grey"], ls=(0, (2, 2)), lw=1.4, zorder=1)
        ymax = ax.get_ylim()[1] if y is None else y
        ax.text(start - 0.3, ymax * 0.96, "coast/tap begins", rotation=90,
                va="top", ha="left", fontsize=7.5, color=ps.OKABE["grey"])


def _fr_id(r):
    for cid, c in r.get("compute", {}).get("per_client", {}).items():
        if c.get("is_free_rider"):
            return cid
    return None


def _cum_effort(r):
    """Per-round cumulative attacker-effort ratio = cumsum(FR samples) /
    cumsum(honest samples), round by round. Shows effort flat while coasting,
    stepping up at each tap."""
    h = r.get("history", [])
    rounds = [x["round"] for x in h]
    pc = r.get("compute", {}).get("per_client", {})
    fid = _fr_id(r)
    honest = [cid for cid in pc if not pc[cid].get("is_free_rider")]
    def per_round(cid):
        pr = pc[cid].get("per_round", {})
        return {int(k): v.get("samples", 0) for k, v in pr.items()} if isinstance(pr, dict) else {}
    frpr = per_round(fid) if fid else {}
    hopr = [per_round(cid) for cid in honest]
    cfr = chon = 0.0
    out = []
    for rd in rounds:
        cfr += frpr.get(rd, 0.0)
        chon += (sum(d.get(rd, 0.0) for d in hopr) / len(hopr)) if hopr else 0.0
        out.append(cfr / chon if chon else 0.0)
    return rounds, out


def timeline(a):
    """interpretive per-run plot. Top: free-rider BER, honest BER, and ALL
    eta thresholds vs round (warmup/tap marked). Bottom (shared x): cumulative
    attacker effort as a fraction of honest, vs round. Read together: the red
    line dips under the fair (frozen) eta while the effort line stays far below
    1.0 = evading cheaply."""
    _, r = _pick(_load(a.inp), a.family)
    h = r.get("history", [])
    rounds = [x["round"] for x in h]
    benign = [x.get("wm_benign_ber") for x in h]
    frb = [x.get("wm_fr_ber") for x in h]
    fig, (a1, a2) = ps.stacked_panels(2, figsize=(10.5, 7), height_ratios=[2.1, 1])
    a1.plot(rounds, benign, color=ps.C_HONEST, lw=2.4, label="honest clients' BER")
    a1.plot(rounds, frb, color=ps.C_FR, lw=2.4, label="free-rider BER")
    for v in thr.ALL_VARIANTS:
        st_ = thr.STYLE[v]
        et = thr.eta_series([b if b is not None else 0.5 for b in benign], v)
        a1.plot(rounds, et, color=st_["color"], ls=st_["ls"], lw=1.4, label=st_["label"])
    tr = _fr_trace(r)
    if tr:
        at = {x["round"]: x.get("wm_fr_ber") for x in h}
        warm = [(t["round"], at.get(t["round"])) for t in tr if t.get("action") in ("warmup", "embed") and at.get(t["round"]) is not None]
        taps = [(t["round"], at.get(t["round"])) for t in tr if t.get("action") == "tap" and at.get(t["round"]) is not None]
        # the attacker's OWN threshold line (frozen estimate, or the oracle value)
        est = [(t["round"], t.get("eta_est")) for t in tr if t.get("eta_est") is not None]
        if est:
            is_oracle = ((r.get("config", {}) or {}).get("autop_oracle_eta") or 0) > 0
            a1.plot(*zip(*est), color="#555555", ls="--", lw=2.0,
                    label=("oracle η (given true)" if is_oracle else "attacker's estimated η"))
        if warm: a1.scatter(*zip(*warm), s=50, marker="s", color=ps.C_FR, edgecolor="k", zorder=5, label="warmup embed")
        if taps: a1.scatter(*zip(*taps), s=68, marker="^", color=ps.C_FR, edgecolor="k", zorder=5, label="tap (re-embed)")
    _mark_phases(a1, tr, y=0.7)
    a1.set_ylabel("bit-error-rate  (lower = watermark present)")
    a1.set_ylim(0, 0.7)
    a1.set_title("Free-rider vs honest BER, with every threshold, per round")
    a1.legend(ncol=2, loc="upper right", fontsize=7)
    rr, eff = _cum_effort(r)
    a2.plot(rr, eff, color=ps.OKABE["blue"], lw=2.4, label="free-rider ÷ honest image-passes (cumulative)")
    a2.axhline(1.0, color=ps.OKABE["grey"], ls=":", lw=1)
    a2.text(rounds[0], 1.02, "honest = 1.0", fontsize=8, color=ps.OKABE["grey"])
    _mark_phases(a2, tr)
    a2.set_ylabel("effort ratio\n(image-passes)")
    a2.set_xlabel("communication round")
    a2.set_ylim(0, max(1.1, max(eff) * 1.15 if eff else 1.1))
    a2.set_title("Cumulative cost: FR image-passes \u00f7 honest image-passes (samples), per round")
    a2.legend(loc="upper left", fontsize=8)
    ps.finish(fig, a.out + ".png")


def knob(a):
    """Per-knob sweep: filter by family and by which knob was actually
    swept (manifest.sweep_var), so the three autopilot knobs don't pool onto one
    axis. Two stacked panels vs the knob value: free-rider BER (with the fair
    converged-eta line) and attacker effort — mean +/- std over seeds."""
    runs = _load(a.inp)
    groups = {}
    for _, r in runs:
        m = r.get("manifest", {}) or {}
        if a.family and m.get("family") not in a.family:
            continue
        if a.sweep_var and m.get("sweep_var") != a.sweep_var:   # the fix: match the knob
            continue
        lvl = m.get("sweep_level")
        benign = [hh.get("wm_benign_ber") for hh in r.get("history", [])]
        eta = thr.eta_series([b or 0.5 for b in benign], "converged")[-1] if benign else None
        groups.setdefault(lvl, []).append(dict(
            ber=r.get("wm_fr_ber"), eff=r.get("compute", {}).get("summary", {}).get("effort_ratio_samples"),
            acc=r.get("final_acc"), eta=eta,
            scope=(r.get("config", {}) or {}).get("autop_scope", "?")))
    if not groups:
        print("no matching runs for", a.family, a.sweep_var); return
    # order levels numerically; for the data knob put full-shard (-1) LAST so the
    # x-axis reads triggers-only -> +N/class -> full shard (increasing data).
    def _order(v):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return (2, 0.0, str(v))
        return (0, 1e9, "") if fv < 0 else (1, fv, "")   # -1 (full shard) to the far right
    if a.sweep_var == "autop_common_per_class":
        levels = sorted(groups, key=_order)
        xs = [_cpc_label(l) for l in levels]
    else:
        try:
            levels = sorted(groups, key=lambda x: float(x))
        except (TypeError, ValueError):
            levels = sorted(groups, key=str)
        xs = [str(l) for l in levels]
    def ms(key):
        out = []
        for l in levels:
            vs = [g[key] for g in groups[l] if g[key] is not None]
            out.append((np.mean(vs) if vs else 0.0, np.std(vs) if len(vs) > 1 else 0.0))
        return out
    ber, eff, eta = ms("ber"), ms("eff"), ms("eta")
    scope = _scope_of(groups, levels)
    fig, (a1, a2) = ps.stacked_panels(2, figsize=(max(7.5, 1.5 * len(xs)), 6.9), height_ratios=[1, 1])
    xi = np.arange(len(xs))
    a1.errorbar(xi, [v[0] for v in ber], yerr=[v[1] for v in ber], marker="o", lw=2,
                color=ps.C_FR, capsize=3, label="free-rider BER (server-measured)")
    a1.plot(xi, [v[0] for v in eta], marker="D", ls="--", color=ps.C_ETA, lw=1.5,
            label="η fair (converged) — BER below = stays under")
    a1.fill_between(xi, 0, [v[0] for v in eta], color=ps.C_ACC, alpha=0.08)
    a1.set_ylabel("bit-error-rate (BER)")
    a1.set_title(f"Does it stay under the fair η?   (scope = {scope};  green band = SAFE, below η)")
    a1.legend(fontsize=8, loc="best")
    a2.errorbar(xi, [v[0] for v in eff], yerr=[v[1] for v in eff], marker="s", lw=2,
                color=ps.OKABE["blue"], capsize=3, label="free-rider effort \u00f7 honest (image-passes)")
    a2.axhline(1.0, color=ps.OKABE["grey"], ls=":", lw=1)
    a2.text(0, 1.02, "honest = 1.0", fontsize=8, color=ps.OKABE["grey"])
    a2.set_ylabel("effort ratio\n(image-passes)")
    a2.set_xlabel(_knob_axis_label(a.sweep_var))
    a2.set_ylim(0, max(1.1, max([v[0] for v in eff] + [0.0]) * 1.2))
    a2.set_title("Cost (lower = cheaper).  " + EFFORT_CAP.split("\n")[0])
    a2.legend(fontsize=8, loc="best")
    for ax in (a1, a2):
        ax.set_xticks(xi); ax.set_xticklabels(xs, fontsize=9)
    ps.finish(fig, a.out + ".png")


def submarine(a):
    """THE submarine plot: why the free-rider taps, round by round.
      Panel 1 — free-rider BER (the "submarine line") vs the fair frozen eta.
        Rounds where it TRAINS (warmup or tap) are shaded; coasting rounds are
        clear. You can see the sub let BER rise toward eta while coasting, then
        dive (tap) just before crossing — staying submerged under the line.
      Panel 2 — per-round training cost (tap batches) as bars, so each dive's
        effort is visible; annotated with the cumulative effort ratio.
    Reads straight off the trace: action in {warmup, tap} = training (a dive);
    action == coast = drifting up."""
    _, r = _pick(_load(a.inp), a.family)
    h = r.get("history", [])
    rounds = [x["round"] for x in h]
    frb = [x.get("wm_fr_ber") for x in h]
    benign = [x.get("wm_benign_ber") for x in h]
    eta_f = thr.eta_series([b if b is not None else 0.5 for b in benign], "frozen")
    eta_c = thr.eta_series([b if b is not None else 0.5 for b in benign], "converged")
    tr = _fr_trace(r) or []
    act = {t["round"]: t.get("action") for t in tr}
    tapb = {t["round"]: (t.get("tap_batches") or 0) * ((r.get("config", {}) or {}).get("batch_size", 16))
            for t in tr if t.get("action") == "tap"}
    warmr = [t["round"] for t in tr if t.get("action") in ("warmup", "embed")]
    tapr = [t["round"] for t in tr if t.get("action") == "tap"]
    fr_at = {x["round"]: x.get("wm_fr_ber") for x in h}
    eff = r.get("compute", {}).get("summary", {}).get("effort_ratio_samples")

    fig, (a1, a2) = ps.stacked_panels(2, figsize=(11, 6.8), height_ratios=[2.2, 1])
    # shade every training round (a "dive")
    lbl_used = False
    for rd, ac in act.items():
        if ac in ("warmup", "embed", "tap"):
            a1.axvspan(rd - 0.5, rd + 0.5, color=ps.OKABE["orange"], alpha=0.18,
                       label=("training round (dive)" if not lbl_used else None))
            lbl_used = True
    # the submarine line + the fair thresholds it hides under
    a1.plot(rounds, frb, color=ps.C_FR, lw=2.6, marker="o", ms=3, label="free-rider BER (the submarine)")
    a1.plot(rounds, benign, color=ps.C_HONEST, lw=2.0, label="honest clients BER (reference)")
    a1.plot(rounds, eta_f, color=thr.STYLE["frozen"]["color"], lw=1.8, label="η frozen (fair ceiling)")
    a1.plot(rounds, eta_c, color=thr.STYLE["converged"]["color"], ls="-.", lw=1.4, label="η converged (fair)")
    if warmr:
        a1.scatter(warmr, [fr_at.get(x) for x in warmr], s=55, marker="s",
                   color=ps.C_FR, edgecolor="k", zorder=6, label="warmup embed")
    if tapr:
        a1.scatter(tapr, [fr_at.get(x) for x in tapr], s=80, marker="v",
                   color=ps.C_FR, edgecolor="k", zorder=6, label="tap = dive (re-embed)")
    _mark_phases(a1, tr, y=a1.get_ylim()[1])
    a1.set_ylabel("bit-error-rate  (below η = hidden)")
    a1.set_ylim(0, max(0.7, (max([f for f in frb if f is not None] + [0.3])) * 1.1))
    a1.set_title("The submarine: coasting lets BER drift toward η; each tap (dive) re-embeds. "
                 "Below η = hidden")
    a1.legend(ncol=2, loc="upper right", fontsize=7.5)
    # per-round dive cost. samples/tap = tap_batches x batch_size. In STAY-UNDER mode
    # every post-warmup round is a fixed-size tap, so these bars are ~constant.
    if tapb:
        a2.bar(list(tapb), list(tapb.values()), width=0.8, color=ps.OKABE["blue"],
               label="samples per tap = tap_batches \u00d7 16 (image-passes)")
    a2.set_ylabel("samples\nper tap")
    a2.set_xlabel("communication round")
    a2.set_title(("Cost per tap.  TOTAL free-rider effort = {:.0%} of an honest client "
                  "(\u03a3 image-passes \u00f7 honest \u03a3)").format(eff)
                 if eff is not None else "Cost per tap (image-passes)")
    a2.legend(loc="upper right", fontsize=8)
    ps.finish(fig, a.out + ".png")


def estimate(a):
    """Attacker's BELIEVED threshold vs the ACTUAL fair one. The free-rider can't
    see the server's eta, so it ESTIMATES it (eta_est, from its own recent clean
    BERs, or a fallback) and aims a margin below that. This plot overlays:
      - free-rider BER (what it actually submits),
      - eta_est (what the attacker THINKS the ceiling is),
      - the actual fair eta (frozen) it is really judged against.
    Where eta_est sits ABOVE the actual eta, the attacker believes it is safe but
    is not — the visual explanation of why a config fails to stay under."""
    _, r = _pick(_load(a.inp), a.family)
    h = r.get("history", [])
    rounds = [x["round"] for x in h]
    frb = [x.get("wm_fr_ber") for x in h]
    benign = [x.get("wm_benign_ber") for x in h]
    eta_actual = thr.eta_series([b if b is not None else 0.5 for b in benign], "frozen")
    tr = _fr_trace(r) or []
    est = {t["round"]: t.get("eta_est") for t in tr if t.get("eta_est") is not None}
    tgt = {t["round"]: t.get("target") for t in tr if t.get("target") is not None}
    est_line = [est.get(rd) for rd in rounds]
    tgt_line = [tgt.get(rd) for rd in rounds]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(rounds, frb, color=ps.C_FR, lw=2.6, marker="o", ms=3, label="free-rider BER (submitted)")
    ax.plot(rounds, eta_actual, color=thr.STYLE["frozen"]["color"], lw=2.2,
            label="ACTUAL fair η (frozen) — judged against this")
    # attacker's believed ceiling + where it aims
    ex = [rd for rd in rounds if est.get(rd) is not None]
    ey = [est[rd] for rd in ex]
    if ex:
        ax.plot(ex, ey, color=ps.OKABE["grey"], ls="--", lw=2, label="attacker's BELIEVED η (its estimate)")
    tx = [rd for rd in rounds if tgt.get(rd) is not None]
    ty = [tgt[rd] for rd in tx]
    if tx:
        ax.plot(tx, ty, color=getattr(ps,"C_PURPLE","#CC79A7"), ls=":", lw=1.8, label="attacker's target (η_est − margin)")
    ax.fill_between(ex, ey, [eta_actual[rounds.index(rd)] for rd in ex],
                    where=[est[rd] > eta_actual[rounds.index(rd)] for rd in ex],
                    color="#D55E00", alpha=0.12,
                    label="danger gap (thinks safe, isn't)") if ex else None
    ax.set_xlabel("communication round")
    ax.set_ylabel("bit-error-rate / threshold")
    ax.set_ylim(0, 0.55)
    ax.set_title("Attacker's believed threshold vs the actual fair one\n"
                 "(estimate above actual = false confidence → why it can drift into being caught)")
    ax.legend(loc="upper right", fontsize=8.5)
    ps.finish(fig, a.out + ".png")


def thresholds_demo(a):
    """Slide-5 plot: the SAME run's free-rider + honest BER with ALL FIVE eta
    definitions overlaid vs round — to show how differently each 'threshold' behaves
    on identical data (cumulative swings up; frozen/converged sit low and fixed)."""
    _, r = _pick(_load(a.inp), a.family)
    h = r.get("history", [])
    rounds = [x["round"] for x in h]
    benign = [x.get("wm_benign_ber") for x in h]
    frb = [x.get("wm_fr_ber") for x in h]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(rounds, benign, color=ps.C_HONEST, lw=2.6, label="honest clients' BER")
    ax.plot(rounds, frb, color=ps.C_FR, lw=2.6, label="free-rider BER")
    for v in thr.ALL_VARIANTS:
        stl = thr.STYLE[v]
        et = thr.eta_series([b if b is not None else 0.5 for b in benign], v)
        ax.plot(rounds, et, color=stl["color"], ls=stl["ls"], lw=1.8, label=stl["label"])
    ax.set_xlabel("communication round")
    ax.set_ylabel("bit-error-rate / threshold η")
    ax.set_ylim(0, 0.7)
    ax.set_title("One run, five ways to set η: the choice decides who is flagged")
    ax.legend(ncol=2, loc="upper right", fontsize=8)
    ps.finish(fig, a.out + ".png")



def meters(a):
    """Compare ALL compute meters so you can see which one best captures effort.
    For each config, the free-rider's total is shown as a FRACTION of the honest
    client's, for every meter: samples (image-passes), gpu_ms (wall GPU time),
    fwd/bwd passes, opt steps. bwd_passes & gpu_ms reward scope attacks (block skips
    most backprop); samples does not. Grouped bars, mean over seeds."""
    runs = _load(a.inp)
    METERS = ["samples", "gpu_ms", "fwd_passes", "bwd_passes", "opt_steps"]
    groups = {}
    for _, r in runs:
        m = r.get("manifest", {}) or {}
        if a.family and m.get("family") not in a.family:
            continue
        lab = (m.get("note") or r.get("attack") or "?").strip()
        pc = r.get("compute", {}).get("per_client", {})
        fr = [c for c in pc.values() if c.get("is_free_rider")]
        ho = [c for c in pc.values() if not c.get("is_free_rider")]
        if not fr or not ho:
            continue
        def total(clients, key):
            tot = 0.0
            for c in clients:
                pr = c.get("per_round", {})
                vals = pr.values() if isinstance(pr, dict) else pr
                tot += sum((v.get(key, 0.0) or 0.0) for v in vals)
            return tot
        row = {}
        for k in METERS:
            fv = total(fr, k) / max(1, len(fr))
            hv = total(ho, k) / max(1, len(ho))
            row[k] = (fv / hv) if hv else 0.0
        groups.setdefault(lab, []).append(row)
    labs = list(groups.keys())
    if not labs:
        print("no matching runs"); return
    import numpy as _np
    x = _np.arange(len(labs)); w = 0.8 / len(METERS)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labs)), 5.6))
    for i, k in enumerate(METERS):
        means = [_np.mean([g[k] for g in groups[l]]) for l in labs]
        ax.bar(x + (i - (len(METERS) - 1) / 2) * w, means, w, label=k)
    ax.axhline(1.0, color="#999999", ls=":", lw=1)
    ax.text(0, 1.01, "honest = 1.0", fontsize=8, color="#999999")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("free-rider \u00f7 honest  (lower = cheaper)")
    ax.set_title("Effort under every compute meter — which one best captures the attack?\n"
                 "samples & fwd_passes are scope-BLIND (same forward work); "
                 "gpu_ms / bwd_passes / opt_steps DROP when block2 skips backbone backprop",
                 fontsize=11)
    ax.legend(ncol=5, fontsize=8, loc="upper right")
    ps.finish(fig, a.out + ".png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("overlay", "worth", "evade_bars", "decay", "timeline", "knob", "submarine", "estimate", "thresholds_demo", "meters"):
        s = sub.add_parser(name)
        s.add_argument("--in", dest="inp", nargs="+", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--family", nargs="+", default=None)
        s.add_argument("--sweep_var", default=None)
    a = ap.parse_args()
    {"overlay": overlay, "worth": worth, "evade_bars": evade_bars,
     "decay": decay, "timeline": timeline, "knob": knob, "submarine": submarine,
     "estimate": estimate, "thresholds_demo": thresholds_demo,
     "meters": meters}[a.cmd](a)