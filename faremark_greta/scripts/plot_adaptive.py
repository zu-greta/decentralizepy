#!/usr/bin/env python
"""Cross-condition plots for the adaptive-attack story.

Design rules baked in (per meeting notes):
  * EVERY curve/point carries standard deviation over seeds (repeats).
  * SINGLE y-axis only. Where two quantities matter (e.g. BER and accuracy) they
    go in stacked panels sharing the x-axis, never a twin y-axis.

Figures:
  squeezing   : per-round honest BER vs free-rider BER (mean +/- std bands) with
                the eta line, and the margin (mu_FR - mu_honest) shrinking. Shows
                the honest/FR distributions being squeezed together.
  effort      : attacker effort (normalized to honest = 1.0) vs a detection metric
                (recall or detect_acc), error bars over seeds; one point per
                condition. The thesis plane: cheap + low recall = broken.
  sweep       : one outcome metric vs a swept variable, std bands, single axis.
  duty        : submarine tap/coast timeline + BER-vs-eta dance (from trace).

The data layer (load/group/aggregate) is torch/matplotlib-free and unit-tested;
plotting is guarded so import failures degrade to a clear message.

Usage:
  python scripts/plot_adaptive.py squeezing --in $RES/*a7-sub* --out figs/a7_squeeze
  python scripts/plot_adaptive.py effort --in $RES/*a7-* $RES/*a5-* --out figs/effort_plane \
        --effort samples --metric wm_fr_recall
  python scripts/plot_adaptive.py sweep --in $RES/*a7-mbg* --out figs/a7_mbg \
        --sweep_var mem_blend_global --metric wm_fr_recall
  python scripts/plot_adaptive.py duty --in $RES/*a7-sub*rep0* --out figs/a7_duty
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict


# --------------------------------------------------------------------------
# data layer (no torch / no matplotlib)
# --------------------------------------------------------------------------
def load_results(patterns):
    """Expand globs / dirs to result.json paths and load them."""
    out = []
    for pat in patterns:
        cands = []
        if os.path.isdir(pat):
            cands = glob.glob(os.path.join(pat, "**", "result.json"), recursive=True)
        elif pat.endswith("result.json"):
            cands = glob.glob(pat) if any(ch in pat for ch in "*?[") else [pat]
        else:
            cands = glob.glob(os.path.join(pat, "result.json")) or \
                    glob.glob(os.path.join(pat, "**", "result.json"), recursive=True) or \
                    glob.glob(pat)
        for c in cands:
            if os.path.isdir(c):
                c = os.path.join(c, "result.json")
            if os.path.isfile(c):
                try:
                    with open(c) as f:
                        out.append(json.load(f))
                except json.JSONDecodeError:
                    print(f"  (skip unreadable {c})")
    return out


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def condition_key(r):
    """A stable label for a condition (everything but the seed)."""
    man = r.get("manifest", {})
    fam = man.get("family") or r.get("attack", "none")
    sv, sl = man.get("sweep_var"), man.get("sweep_level")
    if sv is not None:
        return f"{fam}[{sv}={sl}]"
    return fam


def group_by_condition(results):
    g = defaultdict(list)
    for r in results:
        g[condition_key(r)].append(r)
    return g


def per_round_series(runs, key):
    """Align a history key across seeds by round -> {round: (mean, std)}."""
    by_round = defaultdict(list)
    for r in runs:
        for h in r.get("history", []):
            v = h.get(key)
            if v is not None and "round" in h:
                by_round[h["round"]].append(v)
    return {rd: _ms(vs) for rd, vs in sorted(by_round.items())}


def converged_metric(runs, key, window=10):
    """Mean/std over seeds of a top-level metric (falls back to compute.summary,
    then to the last-`window` mean of the history key)."""
    vals = []
    for r in runs:
        if r.get(key) is not None:
            vals.append(r[key])
        elif (r.get("compute", {}).get("summary", {}) or {}).get(key) is not None:
            vals.append(r["compute"]["summary"][key])
        else:
            tail = [h.get(key) for h in r.get("history", [])[-window:]
                    if h.get(key) is not None]
            if tail:
                vals.append(st.mean(tail))
    return _ms(vals)


def effort_of(run, effort="samples"):
    """Free-rider effort normalized to honest = 1.0 for one run."""
    c = run.get("compute", {}).get("summary", {})
    if effort == "samples":
        return c.get("effort_ratio_samples")
    if effort in ("gpu", "gpu_ms"):
        return c.get("effort_ratio_gpu")
    raise ValueError(effort)


# --------------------------------------------------------------------------
# plotting layer (matplotlib)
# --------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os as _o, sys as _s
    _s.path.insert(0, _o.path.dirname(__file__))
    import plotstyle as _ps
    _ps.apply()
    return plt


def plot_squeezing(groups, out):
    plt = _mpl()
    for cond, runs in groups.items():
        honest = per_round_series(runs, "wm_benign_ber")
        fr = per_round_series(runs, "wm_fr_ber")
        eta = per_round_series(runs, "wm_eta_round")
        if not honest:
            continue
        rounds = sorted(honest)
        hb = [honest[r][0] for r in rounds]
        hs = [honest[r][1] for r in rounds]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot(rounds, hb, label="honest BER", color="tab:blue")
        ax1.fill_between(rounds, [m - s for m, s in zip(hb, hs)],
                         [m + s for m, s in zip(hb, hs)], alpha=0.2, color="tab:blue")
        if fr:
            fb = [fr[r][0] for r in rounds if r in fr]
            fs = [fr[r][1] for r in rounds if r in fr]
            fr_rounds = [r for r in rounds if r in fr]
            ax1.plot(fr_rounds, fb, label="free-rider BER", color="tab:red")
            ax1.fill_between(fr_rounds, [m - s for m, s in zip(fb, fs)],
                             [m + s for m, s in zip(fb, fs)], alpha=0.2, color="tab:red")
        if eta:
            er = [eta[r][0] for r in rounds if r in eta]
            ax1.plot([r for r in rounds if r in eta], er, "--", color="black",
                     label="eta (mu+3sigma)")
        ax1.set_ylabel("bit-error-rate")
        ax1.set_title(f"squeezing — {cond}")
        ax1.legend(fontsize=8)
        # margin panel (single axis): mu_FR - mu_honest
        if fr:
            margin = [(fr[r][0] - honest[r][0]) for r in rounds if r in fr and r in honest]
            ax2.plot([r for r in rounds if r in fr and r in honest], margin,
                     color="tab:purple")
            ax2.axhline(0, color="grey", lw=0.6)
        ax2.set_ylabel("BER margin\n(FR - honest)")
        ax2.set_xlabel("communication round")
        _save(plt, fig, out, f"squeezing__{_slug(cond)}")


def plot_effort(groups, out, effort="samples", metric="wm_fr_recall"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond, runs in sorted(groups.items()):
        xs = [effort_of(r, effort) for r in runs]
        xs = [x for x in xs if x is not None]
        xm, xsd = _ms(xs)
        ym, ysd = converged_metric(runs, metric)
        if xm is None or ym is None:
            continue
        ax.errorbar(xm, ym, xerr=xsd, yerr=ysd, marker="o", capsize=3, label=cond)
        ax.annotate(cond, (xm, ym), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_xlabel(f"free-rider effort / honest effort  ({effort})")
    ax.set_ylabel(metric)
    ax.set_title("attacker effort vs detection")
    ax.set_xscale("symlog", linthresh=0.01)
    _save(plt, fig, out, f"effort__{effort}__{metric}")


def plot_sweep(groups, out, sweep_var, metric="wm_fr_recall"):
    plt = _mpl()
    # collapse each condition to its sweep level
    pts = []
    for cond, runs in groups.items():
        lvl = runs[0].get("manifest", {}).get("sweep_level")
        if lvl is None:
            lvl = runs[0]["config"].get(sweep_var)
        m, s = converged_metric(runs, metric)
        try:
            lvl = float(lvl)
        except (TypeError, ValueError):
            pass
        if m is not None:
            pts.append((lvl, m, s))
    pts.sort(key=lambda p: (isinstance(p[0], str), p[0]))
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    es = [p[2] for p in pts]
    ax.errorbar(range(len(xs)), ys, yerr=es, marker="o", capsize=3)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(sweep_var)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {sweep_var}")
    _save(plt, fig, out, f"sweep__{sweep_var}__{metric}")


def plot_duty(groups, out):
    plt = _mpl()
    for cond, runs in groups.items():
        # use the free-rider with a trace from the first run that has one
        trace = None
        for r in runs:
            for cid, c in r.get("compute", {}).get("per_client", {}).items():
                if c.get("trace"):
                    trace = c["trace"]
                    break
            if trace:
                break
        if not trace:
            continue
        rounds = [t["round"] for t in trace]
        taps = [1 if t.get("action") in ("tap", "embed") else 0 for t in trace]
        ber = [t.get("ber_after") for t in trace]
        eta = [t.get("eta_est") for t in trace]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                                       gridspec_kw={"height_ratios": [1, 2]})
        ax1.bar(rounds, taps, color="tab:orange")
        ax1.set_ylabel("train?\n(1=tap)")
        ax1.set_yticks([0, 1])
        if any(b is not None for b in ber):
            ax2.plot(rounds, ber, marker=".", label="submitted BER", color="tab:red")
        if any(e is not None for e in eta):
            ax2.plot(rounds, eta, "--", label="eta estimate", color="black")
        ax2.set_ylabel("bit-error-rate")
        ax2.set_xlabel("communication round")
        ax2.legend(fontsize=8)
        ax1.set_title(f"submarine duty cycle — {cond}")
        _save(plt, fig, out, f"duty__{_slug(cond)}")


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:60]


def _save(plt, fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["squeezing", "effort", "sweep", "duty"])
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", default="samples", choices=["samples", "gpu", "gpu_ms"])
    ap.add_argument("--metric", default="wm_fr_recall")
    ap.add_argument("--sweep_var", default=None)
    ap.add_argument("--family", nargs="+", default=None,
                    help="keep only runs whose manifest.family is in this list "
                         "(use when the run TAG is not in the folder name)")
    args = ap.parse_args()

    results = load_results(args.inputs)
    if args.family:
        keep = set(args.family)
        results = [r for r in results
                   if (r.get("manifest", {}) or {}).get("family") in keep]
    if not results:
        print("no result.json found for the given --in globs / --family filter")
        return
    groups = group_by_condition(results)
    print(f"loaded {len(results)} runs in {len(groups)} conditions")

    if args.kind == "squeezing":
        plot_squeezing(groups, args.out)
    elif args.kind == "effort":
        plot_effort(groups, args.out, effort=args.effort, metric=args.metric)
    elif args.kind == "sweep":
        if not args.sweep_var:
            ap.error("sweep needs --sweep_var")
        plot_sweep(groups, args.out, args.sweep_var, metric=args.metric)
    elif args.kind == "duty":
        plot_duty(groups, args.out)


if __name__ == "__main__":
    main()