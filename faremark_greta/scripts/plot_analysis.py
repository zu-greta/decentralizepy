"""
Subcommands
  timeline    BER vs communication round for ONE run: per-FR & per-honest + means,
              warmup end, tap/coast markers, and the fair thresholds (annotated).
  frontier    BER vs EFFORT across a data sweep (and across scopes/positions):
              "how cheap can I get while staying hidden?"  Pareto view.
  scorecard   grid rows=data level x cols=family; each cell = BER + effort%, green if
              hidden (BER < fair eta) else red. "what settings worked, at a glance."
  thresholds  false-positive rate under every eta definition (bar), with how each is
              computed. "which threshold gives the fewest false positives."

THRESHOLDS (all use the converged tail = last N rounds of honest BER):
  eta_tight  = mu + 3*sigma over per-ROUND-MEAN honest BER   (~0.09; what the live
               detector approximates; catches the free-rider but false-positives hard
               honest positions)
  eta_loose  = mu + 3*sigma over PER-CLIENT honest BER        (~0.27; spares honest
               clients but lets the free-rider pass)
  eta_cumul  = the run's live wm_eta_round (paper-faithful cumulative mu+3sigma; swings
               high, ~0.3; NOT fair for judging evasion)
  eta_fixed  = 0.25 (a constant baseline the paper caps at)
"""
import json, glob, sys, argparse, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "faremark"); sys.path.insert(0, "scripts")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotstyle as ps
import eta_calib
ps.apply()

TAIL = 20
GREY = ps.OKABE.get("grey", "#888888") if hasattr(ps, "OKABE") else "#888888"
BLACK = ps.OKABE.get("black", "#000000") if hasattr(ps, "OKABE") else "#000000"


# ---------- io ----------
def load(globs):
    out = []
    for g in globs:
        for f in sorted(glob.glob(g)):
            try: out.append(json.load(open(f)))
            except Exception: pass
    return out

def fam(r): return (r.get("manifest", {}) or {}).get("family")
def lvl(r):
    v = (r.get("manifest", {}) or {}).get("sweep_level")
    if v is None: v = (r.get("config", {}) or {}).get("autop_common_per_class")
    try: return float(v)
    except (TypeError, ValueError): return None

def mu3s(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None
    return float(np.mean(xs)) + 3.0 * (float(np.std(xs)) if len(xs) > 1 else 0.0)

def lvl_label(v):
    if v is None: return "?"
    if v < 0: return "full\nshard"
    if v == 0: return "trig\nonly"
    return f"+{int(v)}/cls"


def eta_defs(runs, tail=TAIL):
    """Compute all four thresholds from the honest clients of the given runs."""
    cumul = []
    for r in runs:
        for h in r.get("history", [])[-tail:]:
            if h.get("wm_eta_round") is not None: cumul.append(h["wm_eta_round"])
    et, el = eta_calib.frozen_eta(runs)   # frozen on ALL clients during warmup window
    return {
        "eta_tight": et,
        "eta_loose": el,
        "eta_cumul": float(np.mean(cumul)) if cumul else None,
        "eta_fixed": 0.25,
    }


# ================================================================= TIMELINE
def timeline(a):
    runs = [r for r in load(a.inp) if (a.family is None or fam(r) == a.family)
            and (a.level is None or lvl(r) == float(a.level))
            and (a.seed is None or r.get("seed") == int(a.seed))]
    if not runs: print("no matching run"); return
    r = runs[0]
    hist = r.get("history", [])
    rounds = [h["round"] for h in hist]
    # per-client series
    honest, freer = {}, {}
    for h in hist:
        for p in (h.get("wm_per_client") or []):
            (freer if p.get("is_free_rider") else honest).setdefault(p["cid"], {})[h["round"]] = p["ber"]

    def series(d, cid): return [d[cid].get(rd, np.nan) for rd in rounds]
    def mean_series(d):
        return [np.nanmean([d[c].get(rd, np.nan) for c in d]) if d else np.nan for rd in rounds]

    # schedule from the fixed window: warmup=[1,W-1], calib=[W-K,W-1], free-ride>=W
    lo, hi = eta_calib.calib_window(r)      # [W-K, W-1]
    W = hi + 1
    taps, coasts = set(), set()
    pc = (r.get("compute", {}) or {}).get("per_client", {}) or {}
    for cid, c in pc.items():
        for t in c.get("trace", []):
            if t.get("action") == "tap": taps.add(t["round"])
            elif t.get("action") == "coast": coasts.add(t["round"])

    E = eta_defs([r])
    fig, ax = ps.stacked_panels(1, figsize=(12, 6.2))[0] if False else plt.subplots(figsize=(12, 6.2))

    for cid in honest:
        ax.plot(rounds, series(honest, cid), color=ps.C_HONEST, lw=0.8, alpha=0.25)
    for cid in freer:
        ax.plot(rounds, series(freer, cid), color=ps.C_FR, lw=0.9, alpha=0.5,
                label=f"free-rider cid {cid} (cls {cid%100})")
    ax.plot(rounds, mean_series(honest), color=ps.C_HONEST, lw=2.8, label="honest mean BER")
    ax.plot(rounds, mean_series(freer), color=ps.C_FR, lw=2.8, label="free-rider mean BER")

    ytop = ax.get_ylim()[1]
    # warmup band + calibration window + free-ride delimiter
    ax.axvspan(min(rounds), lo - 0.5, color="#FADFA6", alpha=0.30, lw=0, label="forced-honest warmup")
    ax.axvspan(lo - 0.5, hi + 0.5, color="#BFE3C6", alpha=0.55, lw=0, label=f"calibration window [{lo},{hi}] (η frozen here)")
    ax.axvline(lo - 0.5, color="#2C7A3F", ls="-", lw=1.4)
    ax.axvline(W - 0.5, color=GREY, ls="--", lw=1.6)
    ax.text(lo - 0.4, ytop*0.97, " converged → calibrate η", color="#2C7A3F", fontsize=8.5, va="top")
    ax.text(W - 0.4, ytop*0.90, " free-riding starts", color=GREY, fontsize=8.5, va="top")

    frm = mean_series(freer)
    tap_x = [rd for rd in rounds if rd in taps]
    ax.scatter(tap_x, [frm[rounds.index(rd)] for rd in tap_x], marker="v", s=34,
               color=ps.C_FR, edgecolor="white", linewidth=0.5, zorder=5, label="tap (re-embed)")
    if coasts:
        cx = [rd for rd in rounds if rd in coasts]
        ax.scatter(cx, [frm[rounds.index(rd)] for rd in cx], marker="s", s=30,
                   color="#FFFFFF", edgecolor=ps.C_FR, zorder=5, label="coast (no train)")

    # thresholds (calibrated ON the green window, all clients)
    if E["eta_tight"]: ax.axhline(E["eta_tight"], color=BLACK, ls="--", lw=2,
        label=f"fair η tight (round-mean) = {E['eta_tight']:.3f}")
    if E["eta_loose"]: ax.axhline(E["eta_loose"], color=GREY, ls=":", lw=1.8,
        label=f"loose η (per-client) = {E['eta_loose']:.3f}")

    ax.set_xlabel("communication round"); ax.set_ylabel("bit-error-rate (lower = mark present)")
    ax.set_title(a.title or f"BER vs round  ·  {fam(r)}  ·  cpc={lvl(r)}  ·  seed={r.get('seed')}")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    note = ("η frozen on the CALIBRATION window (green, all clients honest):  tight = μ+3σ of the "
            "per-round MEAN BER;  loose = μ+3σ of all per-client BERs.  Same window the free-rider uses.")
    ax.text(0.005, -0.16, note, transform=ax.transAxes, fontsize=8.5, color=GREY)
    ps.finish(fig, a.out + ".png")
    print(f"calib window [{lo},{hi}] | free-ride from {W} | n_taps={len(taps)} n_coasts={len(coasts)} | eta={E}")


# ================================================================= FRONTIER
def frontier(a):
    runs = load(a.inp)
    fams = a.families or sorted({fam(r) for r in runs if fam(r)})
    E = eta_defs([r for r in runs if fam(r) in fams])
    et = E["eta_tight"]

    fig, (axS, axG) = ps.stacked_panels(2, figsize=(11, 8.4), height_ratios=[1, 1])
    markers = ["o", "s", "^", "D", "v", "P"]
    colors = [ps.C_FR, ps.C_HONEST, "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    for fi, fm in enumerate(fams):
        rs = [r for r in runs if fam(r) == fm]
        levels = sorted({lvl(r) for r in rs}, key=lambda v: (v == -1, v if v is not None else 1e9))
        er_s, er_g, ber = [], [], []
        for lv in levels:
            sub = [r for r in rs if lvl(r) == lv]
            b, rs_, rg_ = [], [], []
            for r in sub:
                for h in r.get("history", [])[-TAIL:]:
                    for p in (h.get("wm_per_client") or []):
                        if p.get("is_free_rider"): b.append(p["ber"])
                cs = (r.get("compute", {}) or {}).get("summary", {}) or {}
                if cs.get("effort_ratio_samples") is not None: rs_.append(cs["effort_ratio_samples"])
                if cs.get("effort_ratio_gpu") is not None: rg_.append(cs["effort_ratio_gpu"])
            ber.append(np.mean(b) if b else np.nan)
            er_s.append(np.mean(rs_) if rs_ else np.nan)
            er_g.append(np.mean(rg_) if rg_ else np.nan)
        mk, cl = markers[fi % 6], colors[fi % 6]
        for ax, eff in ((axS, er_s), (axG, er_g)):
            ax.plot(eff, ber, "-", color=cl, lw=1.5, alpha=0.7)
            ax.scatter(eff, ber, marker=mk, s=70, color=cl, edgecolor="white", zorder=5, label=fm)
            for e, bb, lv in zip(eff, ber, levels):
                if not (np.isnan(e) or np.isnan(bb)):
                    ax.annotate(lvl_label(lv).replace("\n", ""), (e, bb), fontsize=7.5,
                                xytext=(3, 4), textcoords="offset points", color=cl)

    for ax, xl in ((axS, "free-rider effort ÷ honest  (image-passes = DATA cost)"),
                   (axG, "free-rider effort ÷ honest  (GPU-ms = COMPUTE cost)")):
        if et is not None:
            ax.axhspan(et, ax.get_ylim()[1] if ax.get_ylim()[1] > et else et + 0.3,
                       color="#F4C7C3", alpha=0.35, lw=0)
            ax.axhline(et, color=BLACK, ls="--", lw=1.8, label=f"fair η = {et:.3f} (above = CAUGHT)")
        ax.axvline(1.0, color=GREY, ls=":", lw=1.4)
        ax.text(1.0, ax.get_ylim()[1]*0.02, " honest = 1.0", color=GREY, fontsize=8)
        ax.set_xlabel(xl); ax.set_ylabel("free-rider BER\n(converged)")
        ax.legend(loc="upper right", fontsize=8)
    axS.set_title(a.title or "Effort frontier — cheap AND below η (green) is the sweet spot")
    ps.finish(fig, a.out + ".png")
    print("families:", fams, "| eta_tight:", et)


# ================================================================= SCORECARD
def scorecard(a):
    runs = load(a.inp)
    fams = a.families or sorted({fam(r) for r in runs if fam(r)})
    E = eta_defs([r for r in runs if fam(r) in fams]); et = E["eta_tight"] or 0.09
    all_lv = sorted({lvl(r) for r in runs if fam(r) in fams},
                    key=lambda v: (v == -1, v if v is not None else 1e9))

    ber = np.full((len(all_lv), len(fams)), np.nan)
    eff = np.full((len(all_lv), len(fams)), np.nan)
    for ci, fm in enumerate(fams):
        for ri, lv in enumerate(all_lv):
            sub = [r for r in runs if fam(r) == fm and lvl(r) == lv]
            b, e = [], []
            for r in sub:
                for h in r.get("history", [])[-TAIL:]:
                    for p in (h.get("wm_per_client") or []):
                        if p.get("is_free_rider"): b.append(p["ber"])
                cs = (r.get("compute", {}) or {}).get("summary", {}) or {}
                if cs.get("effort_ratio_samples") is not None: e.append(cs["effort_ratio_samples"])
            if b: ber[ri, ci] = np.mean(b)
            if e: eff[ri, ci] = np.mean(e)

    fig, ax = plt.subplots(figsize=(1.6 + 1.7*len(fams), 1.2 + 0.62*len(all_lv)))
    hidden = ber < et
    ax.imshow(np.where(np.isnan(ber), 0.5, hidden.astype(float)), cmap="RdYlGn",
              vmin=0, vmax=1, aspect="auto", alpha=0.55)
    for ri in range(len(all_lv)):
        for ci in range(len(fams)):
            if np.isnan(ber[ri, ci]): txt = "—"
            else:
                tag = "hidden" if hidden[ri, ci] else "CAUGHT"
                txt = f"BER {ber[ri,ci]:.2f}\n{eff[ri,ci]*100:.0f}% effort\n{tag}"
            ax.text(ci, ri, txt, ha="center", va="center", fontsize=8.5,
                    color=BLACK, fontweight="bold" if not np.isnan(ber[ri,ci]) and hidden[ri,ci] else "normal")
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_lv))); ax.set_yticklabels([lvl_label(v).replace("\n"," ") for v in all_lv])
    ax.set_title(a.title or f"Scorecard — green = below fair η ({et:.3f}) = hidden. Cheapest hidden cell wins.")
    ax.set_xlabel("setting (scope × position)"); ax.set_ylabel("training data / round")
    ps.finish(fig, a.out + ".png")
    print("eta_tight:", et)


# ================================================================= THRESHOLDS
def thresholds(a):
    runs = [r for r in load(a.inp) if fam(r) == a.family]
    if not runs: print("no runs for", a.family); return
    E = eta_defs(runs)
    indiv = [p["ber"] for r in runs for h in r.get("history", [])[-TAIL:]
             for p in (h.get("wm_per_client") or []) if not p.get("is_free_rider")]
    names = ["eta_tight\n(round-mean)", "eta_loose\n(per-client)", "eta_cumul\n(live)", "eta_fixed\n0.25"]
    keys = ["eta_tight", "eta_loose", "eta_cumul", "eta_fixed"]
    vals = [E[k] for k in keys]
    fpr = [100.0*np.mean([b >= v for b in indiv]) if v is not None else np.nan for v in vals]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    cols = [ps.C_FR, "#009E73", GREY, "#CC79A7"]
    bars = ax.bar(names, fpr, color=cols, edgecolor="white")
    for b, v, f in zip(bars, vals, fpr):
        ax.text(b.get_x()+b.get_width()/2, f+0.6, f"FPR {f:.0f}%\nη={v:.3f}" if v is not None else "n/a",
                ha="center", fontsize=9)
    ax.set_ylabel("honest false-positive rate  (% of honest client-rounds flagged)")
    ax.set_title(a.title or "Which threshold gives the fewest false positives?")
    ax.set_ylim(0, max([f for f in fpr if not np.isnan(f)] + [10]) * 1.25)
    ax.text(0.0, -0.17, "All η from honest clients' last %d rounds. tight/loose = μ+3σ over round-means / per-client; "
            "cumul = live paper-faithful; fixed = 0.25. Lower bar = fewer honest clients wrongly flagged."%TAIL,
            transform=ax.transAxes, fontsize=8.3, color=GREY)
    ps.finish(fig, a.out + ".png")
    print("eta:", E, "| n_honest_obs:", len(indiv))


def all_thresholds(a):
    """Bar chart of the seven threshold definitions + honest false-positive rate."""
    runs = load(a.inp)
    attack = [r for r in runs if fam(r) == a.family]            # a data-sweep family
    honest = [r for r in runs if fam(r) == (a.honest_family or "t1_all_honest")]
    if not attack and not honest:
        print("no runs for", a.family, "/", a.honest_family); return
    T = eta_calib.all_thresholds(attack, honest)
    # honest client-rounds (converged tail of the all-honest run) for FPR
    n_rounds = max((h.get("round", 0) for r in (honest or attack)
                    for h in r.get("history", [])), default=50)
    _, indiv = eta_calib._pool(honest or attack, n_rounds - 19, n_rounds, honest_only=True)

    names = list(T.keys()); vals = [T[k] for k in names]
    fpr = [100.0*np.mean([b >= v for b in indiv]) if (v is not None and indiv) else np.nan for v in vals]
    lo, hi = eta_calib.window_bounds(attack or honest)

    fig, ax = plt.subplots(figsize=(15, 6.6))
    cols = [ps.C_FR, "#0072B2", "#009E73", "#E69F00", GREY, "#CC79A7", "#7FB069", "#B23A2E"]
    bars = ax.bar(range(len(names)), vals, color=[cols[i % len(cols)] for i in range(len(names))],
                  edgecolor="white")
    for i, (b, v, f) in enumerate(zip(bars, vals, fpr)):
        txt = f"η={v:.3f}\nFPR {f:.0f}%" if (v is not None and not np.isnan(f)) else "n/a"
        ax.text(b.get_x()+b.get_width()/2, (v or 0)+0.008, txt, ha="center", fontsize=8.5)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("threshold η   (bar)   +   honest FPR (label)")
    ax.set_title(a.title or f"All threshold definitions  ·  calibration window = rounds [{lo},{hi}]")
    ax.text(0.0, -0.30,
            "Window [W-K,W-1] = converged warmup, all clients honest. 1=spec (round-mean over the window). "
            "2=longer honest window. 3=per-client (individual BERs -> bigger σ). 4=all warmup incl. non-converged. "
            "5=live cumulative μ+3σ. 6=all clients incl. cheating FRs (post-warmup). 7a/7b=all-honest, easy vs hard positions. "
            "Lower FPR = fewer honest clients wrongly flagged; but lower η catches more free-riders.",
            transform=ax.transAxes, fontsize=8.2, color=GREY, wrap=True)
    ps.finish(fig, a.out + ".png")
    print("thresholds:", {k.split(chr(10))[0]: (round(v,3) if v else None) for k,v in T.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("timeline", "frontier", "scorecard", "thresholds", "all_thresholds"):
        s = sub.add_parser(name)
        s.add_argument("--in", dest="inp", nargs="+", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--title", default="")
        s.add_argument("--family", default=None)
        s.add_argument("--families", nargs="+", default=None)
        s.add_argument("--honest_family", default=None)
        s.add_argument("--level", default=None)
        s.add_argument("--seed", default=None)
    a = ap.parse_args()
    {"timeline": timeline, "frontier": frontier, "scorecard": scorecard,
     "thresholds": thresholds, "all_thresholds": all_thresholds}[a.cmd](a)