#!/usr/bin/env python
"""plots.py -- ALL FareMark figures in one place.

Only the plots that appear in STORYLINE.md are kept. Everything is self-contained
(no repo imports): reads result.json directly, so it runs identically on the
cluster and locally.

Subcommands (one per figure family):

  honest_lines     honest BER per trigger class over rounds        -> A1/E1_class_floors
  honest_per_round honest BER + trigger-class ACCURACY per round    -> A1/E1/EA1_honest_per_round
                   (also the no-watermark control: trig_acc stays HIGH when lambda=0)
  class_acc        per-client trigger vs non-trigger vs global acc  -> A0_class_acc
  sweep            +N data-budget spectrum (reduced FR)             -> D1_spectrum
  timeline         BER vs round, taps/coasts, eta lines             -> A2/A3/E2/E3 timelines
  accuracy         global test acc, attack vs honest               -> accuracy_K4/K5
  dirichlet_dist   reference heatmap of the Dirichlet partition     -> dirichlet_dist
  gpu_savings      cumulative compute, FR vs honest                 -> gpu_savings_*
  iso_pair         isolated same-class BER, honest vs FR            -> iso_*  (cross-run)
  iso_acc          isolated same-class ACCURACY, honest vs FR       -> iso_acc_*

  --- submarine (K/J) tap-coast, three views of the SAME data ---
  tap_perfr        seed-band single graph (mean over seeds)         -> tap_perfr_* / tap_J4_*
  tap_perseed      one panel PER SEED (no marker collisions)        -> *_perseed
  tap_effort       BER + cumulative SAMPLES side by side (effort)   -> *_effort

Usage:
  python plots.py honest_lines --in 'results/*/result.json' --family A1_honest_c100 --out figs/A1_class_floors
  python plots.py tap_perseed  --in 'results/*/result.json' --family K4_alldyn_block2_c36 \
      --honest_in 'results/*/result.json' --honest_family A1_honest_c100 --out figs/tap_perfr_K4
"""
from __future__ import annotations
import argparse, glob, json, os, re
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ###########################################################################
# ##  EDITABLE CONSTANTS  --  change thresholds / windows here, ONE place.  ##
# ###########################################################################
TAIL = 20                    # "converged" window = last N rounds (calibration + tail-mean)

# Two reference detection thresholds, drawn on the timelines / tap plots.
# IID (Group A/D/K) values are the frozen references from STORYLINE.md 5.1.
ETA_TIGHT_IID  = 0.064       # aggressive (mu+3s over round-means)  -- STORYLINE 5.1
ETA_LOOSE_IID  = 0.264       # lenient    (mu+3s over per-client)   -- STORYLINE 5.1
# non-IID (Group E/EA) values -- pass with --eta_tight/--eta_loose (STORYLINE E1).
ETA_TIGHT_NIID = 0.161
ETA_LOOSE_NIID = 0.576

ETA_TIGHT_DEFAULT = ETA_TIGHT_IID   # used when neither CLI nor run config gives one
ETA_LOOSE_DEFAULT = ETA_LOOSE_IID

# honest_per_round: how to bucket a trig_acc==0 client-round (STORYLINE 6c).
SUPPRESS_BER = 0.12          # trig_acc==0 AND BER <  this -> watermark SUPPRESSION (expected)
STARVE_BER   = 0.30          # trig_acc==0 AND BER >= this -> data STARVATION (non-IID empty shard)

DIRICHLET_ALPHAS = [0.1, 0.5, 1.0]   # dirichlet_dist reference heatmaps
HARD_DRAW_GAP    = -10       # class_acc: trig-class acc this far below global => "HARD draw" flag


# ###########################################################################
# ##  EDITABLE TEXT  --  every axis label / band label / line label.       ##
# ##  Change the wording here; it applies across all figures.              ##
# ###########################################################################
LBL_ROUND      = "communication round"
LBL_BER        = "bit-error-rate  (0 = mark present · 0.5 = no mark)"
LBL_BER_SHORT  = "bit-error-rate"
LBL_BER_HONEST = "honest bit-error-rate (lower = mark embeds)"
LBL_TRIGACC    = "honest trigger-class test acc\n(argmax == trigger class)"
LBL_TESTACC    = "global test accuracy (%)"
LBL_SAMPLES    = "cumulative samples (contention-free)"
LBL_FRACTION   = "cum FR / cum honest"

LBL_ETA_TIGHT  = "η tight"            # + " = {val}" appended
LBL_ETA_LOOSE  = "η loose"
LBL_WARMUP     = "forced-honest warmup"
LBL_CALIB      = "calibration window"
LBL_TAP        = "TAP (trains)"
LBL_COAST      = "COAST (no train)"
LBL_HONEST_MEAN = "honest mean BER"
LBL_FR_MEAN     = "free-rider mean BER"
LBL_HONEST_TWIN = "honest same-class twin"      # + " (class {c})"
LBL_FR_SERVER   = "FR server-measured BER (what gets flagged)"
LBL_FR_PROBE    = "FR self-probe (drives tap/coast)"

# titles (each f-string is marked  # TEXT  so you can grep them)
TITLE_HONEST_LINES = "Honest BER per trigger class"                       # TEXT
TITLE_PER_ROUND    = "Honest BER & trigger-class accuracy per round"      # TEXT
TITLE_CLASS_ACC    = "Per-client trigger-class accuracy (all-honest)"     # TEXT
TITLE_SWEEP1       = "Free-rider BER over rounds, per data budget"        # TEXT
TITLE_SWEEP2       = "Converged free-rider BER vs data budget"            # TEXT
TITLE_TIMELINE     = "BER vs round"                                       # TEXT
TITLE_ACCURACY     = ("Global test accuracy: attack vs honest\n"
                      "(free-riders barely dent global accuracy -- they steal a good model)")  # TEXT
TITLE_DIRICHLET    = ("Dirichlet(α) label-skew partition (rows=clients, cols=classes)\n"
                      "small α = one client hogs each class; larger α = more even")            # TEXT
TITLE_GPU          = "Cumulative compute per round"                       # TEXT
TITLE_ISO          = "Isolated same-class comparison"                     # TEXT
TITLE_TAP          = "Submarine tap / coast"                              # TEXT


# ###########################################################################
# ##  STYLE  (Okabe-Ito colour-blind-safe palette; no dual y-axes).        ##
# ###########################################################################
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
         "yellow": "#F0E442", "black": "#000000", "grey": "#7F7F7F"}
CYCLE = [OKABE[k] for k in ("blue", "orange", "green", "red", "purple", "sky", "black")]
OK = OKABE
GREY, BLACK = OKABE["grey"], OKABE["black"]
C_HONEST = OKABE["blue"]
C_FR     = OKABE["red"]
C_ACC    = OKABE["green"]
C_GOOD   = OKABE["green"]
C_BAD    = OKABE["red"]
C_TWIN   = OKABE["purple"]


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 12, "font.family": "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 12,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC", "legend.fontsize": 8,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })


def finish(fig, path):
    path = path if str(path).endswith(".png") else str(path) + ".png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def stacked_panels(n, figsize=None, height_ratios=None):
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=figsize or (11, 2.8 * n),
                             gridspec_kw={"height_ratios": height_ratios})
    return fig, ([axes] if n == 1 else list(axes))


# a tiny namespace so ported bodies that say ps.C_HONEST / ps.finish still work
ps = SimpleNamespace(OKABE=OKABE, CYCLE=CYCLE, C_HONEST=C_HONEST, C_FR=C_FR,
                     C_GOOD=C_GOOD, C_BAD=C_BAD, finish=finish,
                     stacked_panels=stacked_panels)


# ###########################################################################
# ##  IO / analysis helpers (self-contained; no detection/resultio import) ##
# ###########################################################################
def load(globs):
    out = []
    for g in (globs if isinstance(globs, (list, tuple)) else [globs]):
        for f in sorted(glob.glob(g)):
            try:
                out.append(json.load(open(f)))
            except Exception as e:
                print("  (skip", f, "->", e, ")")
    return out


def fam(r):
    return (r.get("manifest", {}) or {}).get("family")


def pick(runs, family):
    return runs if not family else [r for r in runs if fam(r) == family]


def cfg(r, key, default=None):
    v = (r.get("config", {}) or {}).get(key)
    return default if v is None else v


def mu3s(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(np.mean(xs)) + 3.0 * (float(np.std(xs)) if len(xs) > 1 else 0.0)


def is_honest_run(r):
    """True iff the run has no free-riders (a calibration / honest-baseline run)."""
    if r.get("free_rider_indices"):
        return False
    for h in r.get("history", []):
        for p in (h.get("wm_per_client") or []):
            if p.get("is_free_rider"):
                return False
    return True


def honest_runs(runs, family=None):
    return [r for r in runs if is_honest_run(r) and (family is None or fam(r) == family)]


def _calib_tagged_rounds(r):
    tagged = set()
    for c in ((r.get("compute", {}) or {}).get("per_client", {}) or {}).values():
        if c.get("is_free_rider"):
            for t in (c.get("trace") or []):
                if t.get("action") == "calib":
                    tagged.add(t["round"])
    return tagged


def calib_window(r):
    """[lo, hi] calibration rounds (for shading). Falls back to the config W/K."""
    tagged = _calib_tagged_rounds(r)
    if tagged:
        return min(tagged), max(tagged)
    W = int(cfg(r, "autop_honest_until", 12) or 12)
    K = int(cfg(r, "autop_calib_rounds", 4) or 4)
    return W - K, W - 1


def data_lvl(r):
    """Data budget the attacker actually used, for titles."""
    atk = cfg(r, "attack") or r.get("attack")
    key = "tap_data_cpc" if atk == "adaptive_tap" else "autop_common_per_class"
    try:
        return float(cfg(r, key))
    except (TypeError, ValueError):
        return None


def converged_perclient(runs, tail=TAIL, free_rider=False):
    out = []
    for r in runs:
        for h in r.get("history", [])[-tail:]:
            for p in (h.get("wm_per_client") or []):
                if bool(p.get("is_free_rider")) == free_rider and p.get("ber") is not None:
                    out.append(float(p["ber"]))
    return out


def honest_ber_by_round(runs, tclass=None):
    """{round: [ber over honest clients]}, optionally restricted to one trigger class."""
    out = defaultdict(list)
    for r in runs:
        for h in r.get("history", []):
            rd = h.get("round")
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider") or p.get("ber") is None:
                    continue
                if tclass is not None and int(p.get("trigger_class", -1)) != int(tclass):
                    continue
                out[rd].append(float(p["ber"]))
    return out


def default_out(inp):
    paths = []
    for g in (inp or []):
        paths += glob.glob(g)
    if not paths:
        base = os.path.dirname((inp[0] if inp else "results").split("*")[0].rstrip("/")) or "."
    else:
        base = os.path.dirname(os.path.commonpath([os.path.dirname(p) for p in paths])) or "."
    return os.path.join(base, "figs")


def eta_pair(a, runs=None):
    """Resolve (eta_tight, eta_loose) for a figure: CLI overrides win, then the run's
    frozen WM_ETA_FIXED (tight only), then the IID defaults. Loose is recomputed from
    honest_in per-client BER when available."""
    et = a.eta_tight
    if et is None:
        cf = None
        if runs:
            live = [h.get("wm_eta_round") for r in runs for h in r.get("history", [])
                    if h.get("wm_eta_round") is not None]
            cf = float(np.median(live)) if live else (cfg(runs[0], "wm_eta_fixed") or None)
        et = cf if cf else ETA_TIGHT_DEFAULT
    el = a.eta_loose
    if el is None and getattr(a, "honest_in", None):
        indiv = converged_perclient(honest_runs(load(a.honest_in), a.honest_family),
                                    tail=getattr(a, "tail", TAIL))
        if indiv:
            el = mu3s(indiv)
    if el is None:
        el = ETA_LOOSE_DEFAULT
    return float(et), float(el)


# ###########################################################################
# ##  GROUP A -- honest baselines                                          ##
# ###########################################################################
def honest_lines(a):
    """Honest client BER over rounds, ONE line per trigger class. The right end of
    each line == that class's converged floor.  -> A1/E1_class_floors."""
    tail = a.tail or TAIL
    runs = honest_runs(load(a.inp), a.family)
    if not runs:
        print("no honest runs matched (check --in / --family)."); return
    only = set(int(c) for c in a.classes.split(",")) if getattr(a, "classes", None) else None

    by_cr = defaultdict(lambda: defaultdict(list))
    per_seed = defaultdict(lambda: defaultdict(dict))
    max_round = 0
    for si, r in enumerate(runs):
        for h in r.get("history", []):
            rd = h.get("round")
            if rd is None:
                continue
            max_round = max(max_round, rd)
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider"):
                    continue
                c = int(p["trigger_class"])
                if only and c not in only:
                    continue
                by_cr[c][rd].append(p["ber"])
                per_seed[c][si][rd] = p["ber"]
    classes = sorted(by_cr)
    if not classes:
        print("no matching trigger classes."); return
    rounds = list(range(1, max_round + 1))
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(11, 6.2))
    if tail and max_round > tail:
        ax.axvspan(max_round - tail + 0.5, max_round + 0.5, color="#DDDDDD", alpha=0.35,
                   lw=0, label=f"converged tail (last {tail})")   # TEXT
    floors = {}
    for i, c in enumerate(classes):
        col = cmap(i % cmap.N)
        mean = np.array([np.mean(by_cr[c][rd]) if by_cr[c].get(rd) else np.nan for rd in rounds])
        std = np.array([np.std(by_cr[c][rd]) if by_cr[c].get(rd) else np.nan for rd in rounds])
        if getattr(a, "per_seed", False):
            for si in per_seed[c]:
                ax.plot(rounds, [per_seed[c][si].get(rd, np.nan) for rd in rounds],
                        color=col, lw=0.6, alpha=0.20)
        else:
            ax.fill_between(rounds, mean - std, mean + std, color=col, alpha=0.12, lw=0)
        tailvals = [np.mean(by_cr[c][rd]) for rd in rounds[-tail:] if by_cr[c].get(rd)]
        floors[c] = float(np.mean(tailvals)) if tailvals else float("nan")
        ax.plot(rounds, mean, color=col, lw=2.2, label=f"cls {c}  (floor {floors[c]:.3f})")  # TEXT
    if a.eta is not None:
        ax.axhline(a.eta, color=BLACK, ls="--", lw=2, label=f"calibrated η = {a.eta:.3f}")
    ax.set_xlabel(LBL_ROUND); ax.set_ylabel(LBL_BER_HONEST)
    ttl = f"{TITLE_HONEST_LINES}  ·  {a.family or 'honest'}  ·  {len(runs)} seed(s)"   # TEXT
    if only:
        ttl += f"  ·  classes {sorted(only)}"
    ax.set_title(ttl)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    finish(fig, a.out or "honest_lines")
    print("converged floors:", {c: round(floors[c], 4) for c in classes})


def honest_per_round(a):
    """Per-ROUND honest BER (top) and trigger-class ACCURACY (bottom), one line per
    trigger class, aggregated over seeds. Reads wm_per_client[*].trig_acc.

    Reading it (STORYLINE 6c):
      * BER descends to each class floor  -> the mark embeds.
      * trig_acc stays ~0 on the SAME rounds -> the watermark loss suppresses the
        trigger class below argmax to embed (EXPECTED; measured on each client's
        submitted watermarked model).
      * trig_acc ~0 with BER ~0.5 (not ~0) -> data STARVATION (non-IID empty shard).
      * For the NO-WATERMARK control (lambda=0), trig_acc stays HIGH -> proves the
        ~0 accuracy in A1 is caused by the watermark embedding, not the class.
    -> A1/E1/EA1_honest_per_round  (and A0_nowm_per_round).
    """
    runs = load(a.inp)
    if a.family:
        runs = pick(runs, a.family)
    if not runs:
        raise SystemExit(f"no runs matched {a.inp}" + (f" family={a.family}" if a.family else ""))
    famname = fam(runs[0]) or "honest"
    nseed = len(runs)
    eta_t, eta_l = eta_pair(a, runs)

    ber = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(lambda: defaultdict(list))
    rounds = set()
    for d in runs:
        for h in d.get("history", []):
            rd = h.get("round")
            if rd is None:
                continue
            rounds.add(rd)
            for p in h.get("wm_per_client") or []:
                if p.get("is_free_rider"):
                    continue
                c = p.get("trigger_class")
                if c is None:
                    continue
                if p.get("ber") is not None:
                    ber[c][rd].append(p["ber"])
                if p.get("trig_acc") is not None:
                    acc[c][rd].append(p["trig_acc"])
    rounds = sorted(rounds)
    classes = sorted(set(ber) | set(acc))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(classes), 1)))

    def series(bucket, c):
        rr = [r for r in rounds if bucket[c].get(r)]
        return (np.array(rr),
                np.array([np.mean(bucket[c][r]) for r in rr]),
                np.array([np.std(bucket[c][r]) for r in rr]))

    fig, (axB, axA) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    for c, col in zip(classes, cmap):
        rr, mean, std = series(ber, c)
        if len(rr) == 0:
            continue
        floor = np.mean(mean[-a.tail:]) if len(mean) >= a.tail else np.mean(mean)
        axB.plot(rr, mean, color=col, lw=2.0, label=f"cls {c} (floor {floor:.3f})")   # TEXT
        if nseed > 1:
            axB.fill_between(rr, mean - std, mean + std, color=col, alpha=.12)
    axB.axhline(eta_t, color="black", ls="--", lw=1.8, label=f"{LBL_ETA_TIGHT} = {eta_t:.3f}")
    axB.axhline(eta_l, color=C_HONEST, ls=(0, (5, 2)), lw=1.8, label=f"{LBL_ETA_LOOSE} = {eta_l:.3f}")
    axB.set_ylabel(LBL_BER_HONEST); axB.set_ylim(-0.03, 0.6); axB.grid(alpha=.3)
    axB.set_title(f"{TITLE_PER_ROUND}  ·  {famname}  ·  {nseed} seed(s)")   # TEXT
    axB.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=.9)

    for c, col in zip(classes, cmap):
        rr, mean, std = series(acc, c)
        if len(rr) == 0:
            continue
        tailv = np.mean(mean[-a.tail:]) if len(mean) >= a.tail else np.mean(mean)
        axA.plot(rr, mean, color=col, lw=2.0, label=f"cls {c} (tail {tailv:.2f})")   # TEXT
        if nseed > 1:
            axA.fill_between(rr, mean - std, mean + std, color=col, alpha=.12)
    axA.set_xlabel(LBL_ROUND); axA.set_ylabel(LBL_TRIGACC)
    axA.set_ylim(-0.03, 1.0); axA.grid(alpha=.3)
    axA.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=.9)
    finish(fig, a.out or "honest_per_round")

    # suppression-vs-starvation split (printed; explains the zeros)
    n_cr = n_pos = supp = starv = mid = 0
    max_round = max(rounds) if rounds else 0
    lo = max(1, max_round - a.tail + 1)
    tail_acc, tail_ber = [], []
    for d in runs:
        for h in d.get("history", []):
            for p in h.get("wm_per_client") or []:
                if p.get("is_free_rider"):
                    continue
                ta, be = p.get("trig_acc"), p.get("ber")
                if ta is None:
                    continue
                n_cr += 1; n_pos += int(ta > 0)
                if ta == 0 and be is not None:
                    supp += be < SUPPRESS_BER
                    starv += be >= STARVE_BER
                    mid += SUPPRESS_BER <= be < STARVE_BER
                if h.get("round", 0) >= lo:
                    tail_acc.append(ta)
                    if be is not None:
                        tail_ber.append(be)
    print(f"summary {famname} ({nseed} seed(s)):")
    print(f"  honest client-rounds : {n_cr}")
    print(f"  trig_acc > 0         : {n_pos} ({100*n_pos/max(n_cr,1):.1f}%)")
    print(f"  trig_acc == 0 -> suppression(BER<{SUPPRESS_BER})={supp} | "
          f"starvation(BER>={STARVE_BER})={starv} | mid={mid}")
    if tail_acc:
        print(f"  tail-{a.tail} mean trig_acc={np.mean(tail_acc):.4f}  mean BER={np.mean(tail_ber):.4f}")


def class_acc(a):
    """PER-CLIENT trigger-class accuracy check (all-honest run). One panel per client:
    trigger-class vs mean-non-trigger vs global test accuracy of the shared global
    model. Isolates trigger-class DIFFICULTY from the watermark.  -> A0_class_acc."""
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return
    acc_by, overall = defaultdict(list), []
    for r in runs:
        pc = r.get("per_class")
        if pc and pc.get("by_class"):
            for c, d in pc["by_class"].items():
                acc_by[int(c)].append(d["acc"])
        if pc and pc.get("overall_acc") is not None:
            overall.append(float(pc["overall_acc"]))
    if not acc_by:
        print("  NOTE: result['per_class'] absent -> cannot draw class_acc."); return
    acc_mean = {c: float(np.mean(v)) for c, v in acc_by.items()}
    global_acc = float(np.mean(overall)) if overall else float(np.mean(list(acc_mean.values())))

    cid_tc = {}
    for r in runs:
        h = r.get("history") or []
        if h:
            for p in (h[-1].get("wm_per_client") or []):
                if p.get("trigger_class") is not None:
                    cid_tc[int(p["cid"])] = int(p["trigger_class"])
    if not cid_tc:
        print("  NOTE: no wm_per_client rows -> cannot map clients to trigger classes."); return
    cids = sorted(cid_tc)

    ncol = min(5, len(cids)); nrow = int(np.ceil(len(cids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 3.0 * nrow),
                             squeeze=False, sharey=True)
    for i, cid in enumerate(cids):
        ax = axes[i // ncol][i % ncol]
        tc = cid_tc[cid]
        trig = acc_mean.get(tc, float("nan"))
        others = [acc_mean[c] for c in acc_mean if c != tc]
        nontrig = float(np.mean(others)) if others else float("nan")
        vals = [trig, nontrig, global_acc]
        bars = ax.bar([0, 1, 2], vals, color=[C_FR, C_HONEST, GREY], width=0.7, edgecolor="white")
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8)
        ax.axhline(global_acc, color=GREY, ls="--", lw=1.0, zorder=0)
        flag = "  (HARD draw)" if (not np.isnan(trig) and trig - global_acc <= HARD_DRAW_GAP) else ""  # TEXT
        ax.set_title(f"cid {cid} · trig cls {tc}{flag}", fontsize=9.5,
                     color=(C_BAD if flag else "black"))
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["trig", "non-trig", "global"], fontsize=8)  # TEXT
        ax.set_ylim(0, 100)
        if i % ncol == 0:
            ax.set_ylabel("test acc (%)")
    for j in range(len(cids), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{TITLE_CLASS_ACC} · {a.family or 'all'}\n"     # TEXT
                 f"orange = client's trigger class · blue = mean of other classes · "
                 f"grey = global ({global_acc:.1f}%). A short orange bar = a hard trigger-class draw.",
                 fontsize=11, y=1.005)
    finish(fig, a.out or "class_acc")


# ###########################################################################
# ##  GROUP D -- reduced free-rider data-budget spectrum                   ##
# ###########################################################################
def sweep(a):
    """+N spectrum: FR BER over rounds per data budget (top) + converged BER vs
    budget (bottom). N=-1 = full shard.  -> D1_spectrum."""
    tail = a.tail or TAIL
    runs = load(a.inp)

    def _level(man):
        if man.get("sweep_var") == "common_per_class":
            try:
                return int(man.get("sweep_level"))
            except (TypeError, ValueError):
                pass
        m = re.search(r"_n(-?\d+)$", man.get("family") or "")
        return int(m.group(1)) if m else None

    want = set(a.families) if getattr(a, "families", None) else None
    prefix = a.family if (want is None and a.family) else None
    byN = defaultdict(list)
    for r in runs:
        man = r.get("manifest", {}) or {}
        f = man.get("family") or ""
        if want is not None and f not in want:
            continue
        if prefix is not None and not f.startswith(prefix):
            continue
        n = _level(man)
        if n is not None:
            byN[n].append(r)
    if not byN:
        print("no +N sweep runs found (need family `_n<val>` suffix or --families)."); return

    Ns = sorted(byN)
    order = [n for n in Ns if n >= 0] + [n for n in Ns if n < 0]
    lab = lambda n: ("full shard" if n < 0 else ("triggers only" if n == 0 else f"+{n}/class"))

    def fr_series(rs):
        acc = defaultdict(list)
        for r in rs:
            for h in r.get("history", []):
                rd = h.get("round")
                vals = [p["ber"] for p in (h.get("wm_per_client") or []) if p.get("is_free_rider")]
                if rd and vals:
                    acc[rd].append(float(np.mean(vals)))
        return {rd: float(np.mean(v)) for rd, v in acc.items()}

    def conv(rs):
        vals = []
        for r in rs:
            for h in r.get("history", [])[-tail:]:
                vals += [p["ber"] for p in (h.get("wm_per_client") or []) if p.get("is_free_rider")]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)

    eta_t, eta_l = eta_pair(a)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(order):
        ser = fr_series(byN[n])
        if not ser:
            continue
        rds = sorted(ser)
        col = BLACK if n < 0 else cmap(i / max(len(order) - 1, 1))
        ax1.plot(rds, [ser[r] for r in rds], lw=2, ls="--" if n < 0 else "-",
                 color=col, label=f"{lab(n)}  (n={len(byN[n])} seeds)")   # TEXT
    ax1.axhline(eta_t, color=BLACK, ls="--", lw=2, label=f"{LBL_ETA_TIGHT} = {eta_t:.3f}")
    ax1.axhline(eta_l, color=C_HONEST, ls=(0, (5, 2)), lw=2, label=f"{LBL_ETA_LOOSE} = {eta_l:.3f}")
    ax1.set_ylabel("free-rider BER"); ax1.set_title(TITLE_SWEEP1)   # TEXT
    ax1.legend(fontsize=8, ncol=2, loc="upper right")

    xs = list(range(len(order)))
    mus = [conv(byN[n])[0] for n in order]
    sds = [conv(byN[n])[1] for n in order]
    ax2.errorbar(xs, mus, yerr=sds, marker="o", lw=2, color=C_FR, capsize=3,
                 label="free-rider (converged)")   # TEXT
    ax2.axhline(eta_t, color=BLACK, ls="--", lw=2, label=f"{LBL_ETA_TIGHT} = {eta_t:.3f}")
    ax2.axhline(eta_l, color=C_HONEST, ls=(0, (5, 2)), lw=2, label=f"{LBL_ETA_LOOSE} = {eta_l:.3f}")
    ax2.set_xlim(-0.5, len(order) - 0.5); ax2.set_xticks(xs)
    ax2.set_xticklabels([lab(n) for n in order], fontsize=9, rotation=30, ha="right")
    ax2.set_ylabel("converged BER"); ax2.set_title(TITLE_SWEEP2)   # TEXT
    ax2.legend(fontsize=8)
    finish(fig, a.out or "sweep")
    for n, m in zip(order, mus):
        print(f"  {lab(n):>16}: converged FR BER = {m:.4f}")


# ###########################################################################
# ##  GROUP A/D/E -- BER-vs-round timeline (reduced free-riders)           ##
# ###########################################################################
def timeline(a):
    """BER over rounds: honest mean band, free-rider mean band, taps/coasts, the
    frozen eta lines, and (via --honest_in) the honest floor at the FR's own classes.
    -> A2/A3_timeline, E2/E3_timeline."""
    runs = [r for r in load(a.inp) if (a.family is None or fam(r) == a.family)
            and (a.seed is None or r.get("seed") == int(a.seed))]
    if not runs:
        print("no matching run"); return
    nseed = len(runs)
    agg = nseed > 1
    r_ref = runs[0]
    rounds = [h["round"] for h in r_ref.get("history", [])]

    taps, coasts = defaultdict(int), defaultdict(int)
    for r in runs:
        for c in ((r.get("compute", {}) or {}).get("per_client", {}) or {}).values():
            for t in c.get("trace", []):
                if t.get("action") == "tap":
                    taps[t["round"]] += 1
                elif t.get("action") == "coast":
                    coasts[t["round"]] += 1

    h_per_seed, f_per_seed, fr_indiv, hon_sameclass = [], [], {}, {}
    fr_classes = {p["trigger_class"] for r in runs for h in r.get("history", [])
                  for p in (h.get("wm_per_client") or [])
                  if p.get("is_free_rider") and p.get("trigger_class") is not None}
    for si, r in enumerate(runs):
        hm, fm = [], []
        for h in r.get("history", []):
            pcs = h.get("wm_per_client") or []
            hv = [p["ber"] for p in pcs if not p.get("is_free_rider")]
            fv = [p["ber"] for p in pcs if p.get("is_free_rider")]
            hm.append(np.mean(hv) if hv else np.nan)
            fm.append(np.mean(fv) if fv else np.nan)
            for p in pcs:
                if p.get("is_free_rider"):
                    fr_indiv.setdefault((si, p.get("cid")), []).append(p["ber"])
                elif p.get("trigger_class") in fr_classes:
                    hon_sameclass.setdefault((si, p.get("cid")), []).append(p["ber"])
        h_per_seed.append(hm); f_per_seed.append(fm)
    h_arr, f_arr = np.array(h_per_seed), np.array(f_per_seed)
    h_mean, h_std = np.nanmean(h_arr, 0), np.nanstd(h_arr, 0)
    f_mean, f_std = np.nanmean(f_arr, 0), np.nanstd(f_arr, 0)

    lo, hi = calib_window(r_ref); W = hi + 1
    eta_t, eta_l = eta_pair(a, runs)
    fig, ax = plt.subplots(figsize=(12, 6.2))

    if agg:
        ax.fill_between(rounds, h_mean - h_std, h_mean + h_std, color=C_HONEST, alpha=.2, lw=0,
                        label="honest mean ± std")
        ax.plot(rounds, h_mean, color=C_HONEST, lw=3, label=LBL_HONEST_MEAN)
        ax.fill_between(rounds, f_mean - f_std, f_mean + f_std, color=C_FR, alpha=.2, lw=0,
                        label="free-rider mean ± std")
        ax.plot(rounds, f_mean, color=C_FR, lw=3, label=LBL_FR_MEAN)
        done = False
        for (_, _), tr in sorted(fr_indiv.items()):
            n = min(len(tr), len(rounds))
            ax.plot(rounds[:n], tr[:n], color=C_FR, lw=0.7, alpha=.35, zorder=2,
                    label=("individual free-riders" if not done else None)); done = True
        doneh = False
        for (_, _), tr in sorted(hon_sameclass.items()):
            n = min(len(tr), len(rounds))
            ax.plot(rounds[:n], tr[:n], color=C_HONEST, lw=0.9, alpha=.5, ls=(0, (4, 2)), zorder=3,
                    label=("honest client(s) at FR's class" if not doneh else None)); doneh = True
        tapx = [rd for rd, c in taps.items() if c > nseed / 2]
        ax.scatter(tapx, [f_mean[rounds.index(rd)] for rd in tapx], marker="v", s=34,
                   color=C_FR, edgecolor="white", zorder=5, label=f"{LBL_TAP} [majority]")
        coax = [rd for rd, c in coasts.items() if c > nseed / 2]
        ax.scatter(coax, [f_mean[rounds.index(rd)] for rd in coax], marker="s", s=30,
                   color="white", edgecolor=C_FR, zorder=5, label=f"{LBL_COAST} [majority]")
    else:
        honest, freer = {}, {}
        for h in r_ref.get("history", []):
            for p in (h.get("wm_per_client") or []):
                (freer if p.get("is_free_rider") else honest).setdefault(p["cid"], {})[h["round"]] = p["ber"]
        for cid in honest:
            ax.plot(rounds, [honest[cid].get(rd, np.nan) for rd in rounds], color=C_HONEST, lw=0.8, alpha=.25)
        for cid in freer:
            ax.plot(rounds, [freer[cid].get(rd, np.nan) for rd in rounds], color=C_FR, lw=0.9, alpha=.5,
                    label=f"free-rider cid {cid}")
        ax.plot(rounds, h_mean, color=C_HONEST, lw=2.8, label=LBL_HONEST_MEAN)
        ax.plot(rounds, f_mean, color=C_FR, lw=2.8, label=LBL_FR_MEAN)
        tapx = [rd for rd in rounds if rd in taps]
        ax.scatter(tapx, [f_mean[rounds.index(rd)] for rd in tapx], marker="v", s=34,
                   color=C_FR, edgecolor="white", zorder=5, label=LBL_TAP)
        coax = [rd for rd in rounds if rd in coasts]
        ax.scatter(coax, [f_mean[rounds.index(rd)] for rd in coax], marker="s", s=30,
                   color="white", edgecolor=C_FR, zorder=5, label=LBL_COAST)

    if rounds:
        ax.axvspan(min(rounds), lo - 0.5, color="#FADFA6", alpha=.30, lw=0, label=LBL_WARMUP)
        ax.axvspan(lo - 0.5, hi + 0.5, color="#BFE3C6", alpha=.55, lw=0, label=f"{LBL_CALIB} [{lo},{hi}]")
        ax.axvline(W - 0.5, color=GREY, ls="--", lw=1.6)
    ax.axhline(eta_t, color=BLACK, ls="--", lw=2.2, label=f"{LBL_ETA_TIGHT} (frozen) = {eta_t:.3f}")
    ax.axhline(eta_l, color="#3B6FB5", ls=(0, (5, 2)), lw=2.0, label=f"{LBL_ETA_LOOSE} (ref) = {eta_l:.3f}")

    # honest floor at the FR's own trigger classes (fair band), via --honest_in
    fr_cls = sorted({int(p["trigger_class"]) for r in runs for h in r.get("history", [])
                     for p in (h.get("wm_per_client") or []) if p.get("is_free_rider")})
    if getattr(a, "honest_in", None) and fr_cls:
        floor = {}
        for r in honest_runs(load(a.honest_in), a.honest_family):
            for h in r.get("history", [])[-(a.tail or TAIL):]:
                for p in (h.get("wm_per_client") or []):
                    if not p.get("is_free_rider") and int(p["trigger_class"]) in fr_cls:
                        floor.setdefault(int(p["trigger_class"]), []).append(p["ber"])
        pcm = {c: float(np.mean(v)) for c, v in floor.items() if v}
        if pcm:
            vals = list(pcm.values())
            cls_str = (", ".join(f"cls {c} {pcm[c]:.2f}" for c in sorted(pcm)) if len(pcm) <= 4
                       else f"{len(pcm)} classes, floor {min(vals):.2f}-{max(vals):.2f}")
            ax.axhspan(min(vals), max(vals), color=C_HONEST, alpha=.12, lw=0,
                       label=f"honest floor @ FR classes ({cls_str})")   # TEXT

    ax.set_xlabel(LBL_ROUND); ax.set_ylabel(LBL_BER)
    seedstr = f"aggregated over {nseed} seeds" if agg else f"seed={r_ref.get('seed')}"
    ax.set_title(a.title or f"{TITLE_TIMELINE}  ·  {fam(r_ref)}  ·  cpc={data_lvl(r_ref)}  ·  {seedstr}")  # TEXT
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    finish(fig, a.out or "timeline")
    print(f"calib [{lo},{hi}] | free-ride from {W} | taps={len(taps)} coasts={len(coasts)} | seeds={nseed}")


# ###########################################################################
# ##  GROUP K -- global test accuracy (attack vs honest)                   ##
# ###########################################################################
def accuracy(a):
    """Global test accuracy over rounds: attack vs honest reference, + the FR's own
    trigger-class TEST accuracy from per_class of the final model.  -> accuracy_K4/K5."""
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family <fam> or --families f1 f2 ...")
    runs_all = load(a.inp)

    def _curve(rs):
        acc = defaultdict(list)
        for r in rs:
            for h in r.get("history", []):
                if h.get("test_acc") is not None:
                    acc[h["round"]].append(float(h["test_acc"]))
        xs = sorted(acc)
        return xs, [float(np.mean(acc[rd])) for rd in xs], [float(np.std(acc[rd])) for rd in xs]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    hon = honest_runs(load(a.honest_in), a.honest_family) if getattr(a, "honest_in", None) else []
    if not hon:
        hon = honest_runs(runs_all)
    hm = []
    if hon:
        hx, hm, hs = _curve(hon)
        ax.plot(hx, hm, color=C_HONEST, lw=2.6, label="honest run (global)")   # TEXT
        ax.fill_between(hx, np.array(hm) - np.array(hs), np.array(hm) + np.array(hs),
                        color=C_HONEST, alpha=.12)
    for i, f in enumerate(fams):
        rs = pick(runs_all, f)
        if not rs:
            print(f"  (skip {f} -- no runs)"); continue
        fx, fmean, _ = _curve(rs)
        ax.plot(fx, fmean, color=CYCLE[(i + 1) % len(CYCLE)], lw=2.2, ls="--",
                label=f"{f.split('_rep')[0]} (global)")   # TEXT
    ax.set_xlabel(LBL_ROUND); ax.set_ylabel(LBL_TESTACC)
    ax.set_title(TITLE_ACCURACY); ax.legend(fontsize=8, loc="lower right")   # TEXT
    finish(fig, a.out or "accuracy")


# ###########################################################################
# ##  GROUP E/EA -- reference Dirichlet partition heatmaps                 ##
# ###########################################################################
def dirichlet_dist(a):
    """Reference heatmaps of a Dirichlet(alpha) label-skew partition (no result files
    needed). rows=clients, cols=classes, colour=share of a class a client holds.
    -> dirichlet_dist."""
    n_classes = int(a.classes) if getattr(a, "classes", None) else 10
    n_clients = 10
    rng = np.random.default_rng(int(a.seed) if getattr(a, "seed", None) else 0)
    fig, axes = plt.subplots(1, len(DIRICHLET_ALPHAS), figsize=(4.2 * len(DIRICHLET_ALPHAS), 4.2))
    axes = np.atleast_1d(axes)
    im = None
    for ax, alpha in zip(axes, DIRICHLET_ALPHAS):
        mat = np.zeros((n_clients, n_classes))
        for c in range(n_classes):
            mat[:, c] = rng.dirichlet(alpha * np.ones(n_clients))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"α = {alpha}", fontsize=12)
        ax.set_xlabel("class")
        if ax is axes[0]:
            ax.set_ylabel("client")
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_clients))
    cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cb.set_label("fraction of the class held by the client")
    fig.suptitle(TITLE_DIRICHLET, fontsize=12, fontweight="bold")   # TEXT
    finish(fig, a.out or "dirichlet_dist")


# ###########################################################################
# ##  GROUP D/E/EA/K -- cumulative compute (effort) FR vs honest           ##
# ###########################################################################
def _cumulative_by_round(run, cid, field):
    c = ((run.get("compute", {}) or {}).get("per_client", {}) or {}).get(str(cid)) \
        or ((run.get("compute", {}) or {}).get("per_client", {}) or {}).get(cid) or {}
    pr = c.get("per_round") or {}
    cum, tot = {}, 0.0
    for rd in sorted(int(k) for k in pr):
        cell = pr.get(str(rd)) or pr.get(rd) or {}
        tot += float(cell.get(field, 0.0))
        cum[rd] = tot
    return cum


def _fr_cids(runs):
    return sorted({int(p["cid"]) for r in runs for h in r.get("history", [])
                   for p in (h.get("wm_per_client") or []) if p.get("is_free_rider")})


def gpu_savings(a):
    """Cumulative compute per round: honest mean vs each free-rider, gpu_ms + samples,
    plus the running FR/honest fraction.  -> gpu_savings_*."""
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family or --families")
    runs_all = load(a.inp)
    for f in fams:
        runs = pick(runs_all, f)
        if not runs:
            print(f"  (skip {f} -- no runs)"); continue
        nseed = len(runs)
        fr = _fr_cids(runs)
        all_cids = sorted({int(k) for r in runs
                           for k in ((r.get("compute", {}) or {}).get("per_client", {}) or {})})
        honest = [c for c in all_cids if c not in fr]
        if not fr:
            print(f"  (skip {f} -- no free-riders)"); continue

        def avg_cum(cids, fld):
            acc = defaultdict(list)
            for r in runs:
                for cid in cids:
                    for rd, v in _cumulative_by_round(r, cid, fld).items():
                        acc[rd].append(v)
            xs = sorted(acc)
            return xs, [float(np.mean(acc[rd])) for rd in xs]

        hx, hcum = avg_cum(honest, "gpu_ms"); hat = dict(zip(hx, hcum))
        hxs, hcums = avg_cum(honest, "samples"); hats = dict(zip(hxs, hcums))

        fig, (ax0, axS, ax1) = stacked_panels(3, figsize=(11, 9), height_ratios=[2, 2, 1])
        ax0.plot(hx, hcum, color=C_HONEST, lw=2.6, marker="o", ms=3, label="honest mean (cumulative)")  # TEXT
        for i, cid in enumerate(fr):
            fx, fcum = avg_cum([cid], "gpu_ms")
            col = CYCLE[(i + 1) % len(CYCLE)]
            ax0.plot(fx, fcum, lw=2.2, color=col, marker="s", ms=3, label=f"free-rider cid{cid}")
            ax1.plot(fx, [fcum[j] / hat[rd] if hat.get(rd) else np.nan for j, rd in enumerate(fx)],
                     lw=2.0, color=col, marker="s", ms=3, label=f"cid{cid}")
            fxs, fcums = avg_cum([cid], "samples")
            axS.plot(fxs, fcums, lw=2.0, color=col, marker="s", ms=3, label=f"cid{cid}")
        axS.plot(hxs, hcums, color=C_HONEST, lw=2.4, marker="o", ms=3, label="honest mean")
        ax0.set_ylabel("cumulative gpu_ms")
        ax0.set_title(f"{TITLE_GPU}  ·  {f.split('_rep')[0]}  ({nseed} seed(s))  ·  "  # TEXT
                      f"gap below honest = compute saved", fontsize=10)
        ax0.legend(fontsize=8, loc="upper left")
        axS.set_ylabel(LBL_SAMPLES); axS.legend(fontsize=8, loc="upper left")
        ax1.axhline(1.0, color=C_HONEST, ls="--", lw=1.2)
        ax1.set_ylabel(LBL_FRACTION); ax1.set_xlabel(LBL_ROUND); ax1.set_ylim(0, 1.15)
        ax1.legend(fontsize=8, loc="upper right")
        out = a.out or f"gpu_savings_{f}"
        finish(fig, out if len(fams) == 1 else f"{str(out).rstrip('.png')}_{f}")


# ###########################################################################
# ##  ISOLATED cross-run same-class comparisons                            ##
# ###########################################################################
def _iso_series(run):
    """rounds, eta, {cid: {tc, fr, ber:{r:ber}}} for one run."""
    rounds, eta, info = [], [], {}
    for h in run.get("history", []):
        r = h.get("round"); pcs = h.get("wm_per_client")
        if r is None or not pcs:
            continue
        rounds.append(r); eta.append(h.get("wm_eta_round"))
        for p in pcs:
            d = info.setdefault(p["cid"], {"tc": p.get("trigger_class"),
                                           "fr": bool(p.get("is_free_rider")), "ber": {}})
            d["ber"][r] = p.get("ber")
    return rounds, eta, info


def _iso_collect(runs, tclass):
    """Aggregate per-cid BER curves (mean over seeds) for one trigger class."""
    all_rounds = sorted({r for run in runs for r in _iso_series(run)[0]})
    fr_acc, hon_acc = {}, {}
    for run in runs:
        _, _, info = _iso_series(run)
        for cid, d in info.items():
            if d["tc"] != tclass:
                continue
            slot = (fr_acc if d["fr"] else hon_acc).setdefault(cid, {r: [] for r in all_rounds})
            for r in all_rounds:
                if d["ber"].get(r) is not None:
                    slot[r].append(d["ber"][r])

    def fin(bucket):
        out = {}
        for cid, per in sorted(bucket.items()):
            rr = [r for r in all_rounds if per[r]]
            out[cid] = (np.array(rr), np.array([np.mean(per[r]) for r in rr]),
                        np.array([np.std(per[r]) for r in rr]))
        return out
    return fin(fr_acc), fin(hon_acc)


def iso_pair(a):
    """Isolated same-class BER: the honest client on class X (from --honest_in, an
    all-honest run) vs the free-rider on class X (from --fr_in, an attack run where
    it is the only client on X). They never share the class inside one model -> no
    watermark conflict.  -> iso_*."""
    if not (a.honest_in and a.fr_in):
        raise SystemExit("iso_pair needs --honest_in and --fr_in")
    hruns = load(a.honest_in)
    fruns = pick(load(a.fr_in), a.family) if a.family else load(a.fr_in)
    if not hruns or not fruns:
        raise SystemExit("no honest and/or FR runs matched"); 
    _, _, fr_info = _iso_series(fruns[0])
    target = a.cls if a.cls is not None else next(
        (int(d["tc"]) for _, d in sorted(fr_info.items()) if d["fr"] and d["tc"] is not None), None)
    if target is None:
        raise SystemExit("could not determine a trigger class (pass --class)")

    warmup = a.warmup if a.warmup is not None else cfg(fruns[0], "autop_honest_until")
    eta_t, eta_l = eta_pair(a, fruns)
    fam_h = fam(hruns[0]) or "honest"; fam_f = fam(fruns[0]) or "attack"
    _, h_hon = _iso_collect(hruns, target)     # honest client on X (honest runs)
    f_fr, _ = _iso_collect(fruns, target)      # free-rider on X (attack runs)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    if warmup:
        ax.axvline(warmup - 0.5, color="0.4", ls="--", lw=1.2)
        ax.text(warmup - 0.3, eta_l + .03, "free-riding starts", color="0.35", fontsize=9, va="top")  # TEXT
    ax.axhline(eta_t, color="black", ls="--", lw=2.0, label=f"{LBL_ETA_TIGHT} (frozen) = {eta_t:.3f}")
    ax.axhline(eta_l, color="#3B6FB5", ls=(0, (5, 2)), lw=1.8, label=f"{LBL_ETA_LOOSE} = {eta_l:.3f}")

    blues = plt.cm.Blues(np.linspace(0.6, 0.95, max(len(h_hon), 1)))
    for (cid, (rr, mean, std)), col in zip(sorted(h_hon.items()), blues):
        ax.plot(rr, mean, color=col, lw=2.6, zorder=5,
                label=f"HONEST cid{cid} on class {target} [{fam_h}, {len(hruns)} seed(s)]")  # TEXT
        if len(hruns) > 1:
            ax.fill_between(rr, mean - std, mean + std, color=col, alpha=.18)
    reds = plt.cm.Oranges(np.linspace(0.6, 0.95, max(len(f_fr), 1)))
    for (cid, (rr, mean, std)), col in zip(sorted(f_fr.items()), reds):
        ax.plot(rr, mean, color=col, lw=2.8, marker="v", ms=4, zorder=5,
                label=f"FREE-RIDER cid{cid} on class {target} [{fam_f}, {len(fruns)} seed(s)]")  # TEXT
        if len(fruns) > 1:
            ax.fill_between(rr, mean - std, mean + std, color=col, alpha=.18)
    ax.set_xlabel(LBL_ROUND); ax.set_ylabel(LBL_BER); ax.set_ylim(-0.03, 0.55)
    ax.set_title(f"{TITLE_ISO}  ·  class {target}  ·  honest [{fam_h}] vs free-rider [{fam_f}]")  # TEXT
    ax.legend(fontsize=8.5, loc="upper right")
    finish(fig, a.out or f"iso_c{target}")

    def tail_mean(curves, role, k=TAIL):
        for cid, (rr, mean, _) in sorted(curves.items()):
            print(f"  {role} cid{cid}: tail-{k} mean BER = {np.mean(mean[-k:]) if len(mean) else float('nan'):.4f}")
    print(f"class {target} isolated tail:"); tail_mean(h_hon, "HONEST"); tail_mean(f_fr, "FR    ")


def iso_acc(a):
    """Isolated same-class ACCURACY companion to iso_pair. Left: trigger-sample
    accuracy over rounds, honest vs FR. Right: global test accuracy per run.
    -> iso_acc_*."""
    if not (a.honest_in and a.fr_in) or a.cls is None:
        raise SystemExit("iso_acc needs --honest_in, --fr_in and --class")
    honest = load(a.honest_in)
    fr = pick(load(a.fr_in), a.family) if a.family else load(a.fr_in)

    def trig(runs, is_fr):
        acc = defaultdict(list)
        for r in runs:
            for h in r.get("history", []):
                rd = h.get("round")
                for p in (h.get("wm_per_client") or []):
                    if p.get("trigger_class") == a.cls and bool(p.get("is_free_rider")) == is_fr \
                            and p.get("trig_acc") is not None and rd:
                        acc[rd].append(float(p["trig_acc"]))
        xs = sorted(acc)
        return xs, [float(np.mean(acc[x])) for x in xs]

    def test(runs):
        acc = defaultdict(list)
        for r in runs:
            for h in r.get("history", []):
                rd = h.get("round")
                if rd and h.get("test_acc") is not None:
                    acc[rd].append(float(h["test_acc"]))
        xs = sorted(acc)
        return xs, [float(np.mean(acc[x])) for x in xs]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    hx, hy = trig(honest, False); fx, fy = trig(fr, True)
    if hx:
        axA.plot(hx, hy, lw=2, color=C_HONEST, label=f"honest (cls {a.cls})")   # TEXT
    if fx:
        axA.plot(fx, fy, lw=2, color=C_FR, label=f"free-rider (cls {a.cls})")   # TEXT
    axA.set_title(f"accuracy on TRIGGER samples · class {a.cls}")   # TEXT
    axA.set_xlabel(LBL_ROUND); axA.set_ylabel("trigger-class accuracy"); axA.legend()

    hbx, hby = test(honest); fbx, fby = test(fr)
    if hbx:
        axB.plot(hbx, hby, lw=2, color=C_HONEST, label="honest run (global)")   # TEXT
    if fbx:
        axB.plot(fbx, fby, lw=2, color=C_FR, label="free-rider run (global)")   # TEXT
    axB.set_title("global test accuracy per run")   # TEXT
    axB.set_xlabel(LBL_ROUND); axB.set_ylabel(LBL_TESTACC); axB.legend()
    finish(fig, a.out or f"iso_acc_c{a.cls}")


# ###########################################################################
# ##  GROUP K / J -- SUBMARINE tap/coast (three views of the same data)    ##
# ###########################################################################
def _submarine(runs, honest_ext=None, honest_family=None):
    """Extract, per free-rider cid:
        tc                 : trigger class
        seeds[i]           : {rounds, srv{r:ber}, probe{r:ber}, act{r:'tap'/'coast'},
                              samp{r:cumulative samples}}
        twin{r:mean_ber}   : same-class honest twin (from the attack runs, else honest_ext)
        target             : the tap target line (if logged)
    """
    fr_cids, tclass = set(), {}
    for r in runs:
        for h in r.get("history", []):
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider"):
                    fr_cids.add(int(p["cid"]))
                    if p.get("trigger_class") is not None:
                        tclass[int(p["cid"])] = int(p["trigger_class"])
    fr_cids = sorted(fr_cids)

    data = {cid: {"tc": tclass.get(cid), "seeds": [], "target": None} for cid in fr_cids}
    for r in runs:
        srv = {cid: {} for cid in fr_cids}
        for h in r.get("history", []):
            rd = h["round"]
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider") and p.get("ber") is not None:
                    srv[int(p["cid"])][rd] = float(p["ber"])
        comp = (r.get("compute", {}) or {}).get("per_client", {}) or {}
        for cid in fr_cids:
            c = comp.get(str(cid)) or comp.get(cid) or {}
            probe, act = {}, {}
            for t in (c.get("trace") or []):
                rd, ac = t.get("round"), t.get("action")
                if t.get("ber_before") is not None:
                    probe[rd] = float(t["ber_before"])
                if ac in ("tap", "coast"):
                    act[rd] = ac
                if t.get("target") is not None:
                    data[cid]["target"] = float(t["target"])
            samp = _cumulative_by_round(r, cid, "samples")
            rounds = sorted(set(srv[cid]) | set(probe) | set(act) | set(samp))
            data[cid]["seeds"].append({"rounds": rounds, "srv": srv[cid], "probe": probe,
                                       "act": act, "samp": samp})

    # same-class honest twin: from the attack runs, else the external honest family
    twin = {}
    for cid in fr_cids:
        tc = tclass.get(cid)
        hb = honest_ber_by_round(runs, tclass=tc)
        if not any(hb.values()) and honest_ext:
            hb = honest_ber_by_round(honest_runs(honest_ext, honest_family), tclass=tc)
        twin[cid] = {rd: float(np.mean(v)) for rd, v in hb.items() if v}
    return fr_cids, tclass, data, twin


def _mean_over_seeds(seeds, key):
    acc = defaultdict(list)
    for s in seeds:
        for rd, v in s[key].items():
            acc[rd].append(v)
    xs = sorted(acc)
    return xs, [float(np.mean(acc[rd])) for rd in xs], [float(np.std(acc[rd])) for rd in xs]


def _tap_fraction(seeds):
    fr = [(rd, ac) for s in seeds for rd, ac in s["act"].items()]
    nt = sum(1 for _, ac in fr if ac == "tap")
    return (nt / len(fr)) if fr else float("nan")


def tap_perfr(a):
    """VIEW 1 (seed-band): ONE figure per free-rider, mean over seeds with a std band.
    tap/coast markers are drawn where the MAJORITY of seeds tapped/coasted.
    -> tap_perfr_* / tap_J4_*."""
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family or --families")
    runs_all = load(a.inp)
    hon_ext = load(a.honest_in) if getattr(a, "honest_in", None) else None
    for f in fams:
        runs = pick(runs_all, f)
        if not runs:
            print(f"  (skip {f} -- no runs)"); continue
        nseed = len(runs)
        eta_t, eta_l = eta_pair(a, runs)
        fr_cids, tclass, data, twin = _submarine(runs, hon_ext, a.honest_family)
        if not fr_cids:
            print(f"  (skip {f} -- no free-riders)"); continue
        lo, hi = calib_window(runs[0]); W = hi + 1
        base = str(a.out or "tap_perfr").rstrip(".png")

        for cid in fr_cids:
            tc = data[cid]["tc"]; seeds = data[cid]["seeds"]
            fig, ax = plt.subplots(figsize=(12, 5.2))
            ax.axvspan(0.5, W - 0.5, color=OKABE["yellow"], alpha=.12, lw=0, label=LBL_WARMUP)
            ax.axvspan(lo - 0.5, hi + 0.5, color=OKABE["green"], alpha=.16, lw=0, label=LBL_CALIB)
            ax.axvline(W - 0.5, color="0.5", ls="--", lw=1)

            tx = sorted(twin[cid])
            if tx:
                ax.plot(tx, [twin[cid][r] for r in tx], color=C_TWIN, lw=2.0, ls=(0, (1, 1)),
                        zorder=3, label=f"{LBL_HONEST_TWIN} (class {tc})")
            sx, smean, sstd = _mean_over_seeds(seeds, "srv")
            px, pmean, _ = _mean_over_seeds(seeds, "probe")
            ax.plot(sx, smean, color=C_FR, lw=2.4, marker="o", ms=3, zorder=4, label=LBL_FR_SERVER)
            if nseed > 1:
                ax.fill_between(sx, np.array(smean) - np.array(sstd), np.array(smean) + np.array(sstd),
                                color=C_FR, alpha=.15)
            ax.plot(px, pmean, color=OKABE["orange"], lw=1.3, ls=(0, (4, 2)), zorder=3, label=LBL_FR_PROBE)

            srv_at = dict(zip(sx, smean))
            tap_ct, coa_ct = defaultdict(int), defaultdict(int)
            for s in seeds:
                for rd, ac in s["act"].items():
                    (tap_ct if ac == "tap" else coa_ct)[rd] += 1
            tapx = [rd for rd in srv_at if tap_ct.get(rd, 0) > nseed / 2]
            coax = [rd for rd in srv_at if coa_ct.get(rd, 0) > nseed / 2]
            ax.scatter(tapx, [srv_at[rd] for rd in tapx], marker="v", s=72, color=OKABE["blue"],
                       edgecolor="white", zorder=6, label=f"{LBL_TAP} [majority]")
            ax.scatter(coax, [srv_at[rd] for rd in coax], marker="s", s=44, facecolor="white",
                       edgecolor=C_FR, zorder=6, label=f"{LBL_COAST} [majority]")

            ax.axhline(eta_l, color=C_HONEST, ls="--", lw=1.7, label=f"{LBL_ETA_LOOSE} = {eta_l:.3f}")
            ax.axhline(eta_t, color="black", ls=":", lw=1.3, label=f"{LBL_ETA_TIGHT} = {eta_t:.3f}")
            if data[cid]["target"] is not None:
                ax.axhline(data[cid]["target"], color="0.45", ls="-.", lw=1.0,
                           label=f"target {data[cid]['target']:.3f}")

            frac = _tap_fraction(seeds)
            tail = [smean[i] for i, rd in enumerate(sx) if rd >= W]
            srv_tail = float(np.mean(tail)) if tail else float("nan")
            htail = [twin[cid][r] for r in tx if r >= W]
            hon_tail = float(np.mean(htail)) if htail else float("nan")
            verdict = "UNDER η_loose (evades)" if srv_tail < eta_l else "OVER η_loose (caught)"
            ax.set_title(f"{f.split('_rep')[0]}  ·  cid{cid} · class {tc}  ({nseed} seed(s))\n"  # TEXT
                         f"tap-fraction {frac:.0%}  ·  tail FR-BER {srv_tail:.2f} vs "
                         f"same-class honest {hon_tail:.2f}  →  {verdict}", fontsize=10)
            ax.set_xlabel(LBL_ROUND); ax.set_ylabel(LBL_BER_SHORT)
            ax.set_ylim(-0.03, max(0.62, eta_l + 0.06))
            ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=.95)
            finish(fig, f"{base}_{f}_cid{cid}")


def tap_perseed(a):
    """VIEW 2 (per-seed panels): ONE figure per free-rider, with ONE PANEL PER SEED so
    the tap/coast markers of different seeds never collide.  -> *_cid<cid>_perseed."""
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family or --families")
    runs_all = load(a.inp)
    hon_ext = load(a.honest_in) if getattr(a, "honest_in", None) else None
    for f in fams:
        runs = pick(runs_all, f)
        if not runs:
            print(f"  (skip {f} -- no runs)"); continue
        eta_t, eta_l = eta_pair(a, runs)
        fr_cids, tclass, data, twin = _submarine(runs, hon_ext, a.honest_family)
        if not fr_cids:
            print(f"  (skip {f} -- no free-riders)"); continue
        lo, hi = calib_window(runs[0]); W = hi + 1
        base = str(a.out or "tap_perfr").rstrip(".png")

        for cid in fr_cids:
            tc = data[cid]["tc"]; seeds = data[cid]["seeds"]
            n = len(seeds)
            fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12, 2.6 * n + 0.5))
            axes = np.atleast_1d(axes)
            tx = sorted(twin[cid])
            for si, (ax, s) in enumerate(zip(axes, seeds)):
                ax.axvspan(0.5, W - 0.5, color=OKABE["yellow"], alpha=.12, lw=0)
                ax.axvspan(lo - 0.5, hi + 0.5, color=OKABE["green"], alpha=.16, lw=0)
                ax.axvline(W - 0.5, color="0.5", ls="--", lw=1)
                if tx:
                    ax.plot(tx, [twin[cid][r] for r in tx], color=C_TWIN, lw=1.6, ls=(0, (1, 1)),
                            zorder=3, label=(f"{LBL_HONEST_TWIN} (cls {tc})" if si == 0 else None))
                sx = sorted(s["srv"]); sy = [s["srv"][r] for r in sx]
                px = sorted(s["probe"]); py = [s["probe"][r] for r in px]
                ax.plot(sx, sy, color=C_FR, lw=2.2, marker="o", ms=3, zorder=4,
                        label=(LBL_FR_SERVER if si == 0 else None))
                ax.plot(px, py, color=OKABE["orange"], lw=1.2, ls=(0, (4, 2)), zorder=3,
                        label=(LBL_FR_PROBE if si == 0 else None))
                srv_at = dict(zip(sx, sy))
                tapx = [rd for rd, ac in s["act"].items() if ac == "tap" and rd in srv_at]
                coax = [rd for rd, ac in s["act"].items() if ac == "coast" and rd in srv_at]
                ax.scatter(tapx, [srv_at[rd] for rd in tapx], marker="v", s=64, color=OKABE["blue"],
                           edgecolor="white", zorder=6, label=(LBL_TAP if si == 0 else None))
                ax.scatter(coax, [srv_at[rd] for rd in coax], marker="s", s=40, facecolor="white",
                           edgecolor=C_FR, zorder=6, label=(LBL_COAST if si == 0 else None))
                ax.axhline(eta_l, color=C_HONEST, ls="--", lw=1.4)
                ax.axhline(eta_t, color="black", ls=":", lw=1.1)
                fr = [ac for ac in s["act"].values()]
                frac = (fr.count("tap") / len(fr)) if fr else float("nan")
                ax.set_ylabel(f"seed {runs[si].get('seed', si)}\ntap {frac:.0%}", fontsize=9)
                ax.set_ylim(-0.03, max(0.62, eta_l + 0.06)); ax.grid(alpha=.3)
            axes[0].set_title(f"{f.split('_rep')[0]}  ·  cid{cid} · class {tc}  ·  per-seed",  # TEXT
                              fontsize=11)
            axes[0].legend(fontsize=7, loc="upper right", ncol=2, framealpha=.95)
            axes[-1].set_xlabel(LBL_ROUND)
            finish(fig, f"{base}_{f}_cid{cid}_perseed")


def tap_effort(a):
    """VIEW 3 (BER + effort side by side): ONE figure per free-rider, two panels:
    left = server BER over rounds (mean±band) with the eta lines + honest twin;
    right = cumulative SAMPLES, free-rider vs honest mean (effort in one look).
    -> *_cid<cid>_effort."""
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family or --families")
    runs_all = load(a.inp)
    hon_ext = load(a.honest_in) if getattr(a, "honest_in", None) else None
    for f in fams:
        runs = pick(runs_all, f)
        if not runs:
            print(f"  (skip {f} -- no runs)"); continue
        nseed = len(runs)
        eta_t, eta_l = eta_pair(a, runs)
        fr_cids, tclass, data, twin = _submarine(runs, hon_ext, a.honest_family)
        if not fr_cids:
            print(f"  (skip {f} -- no free-riders)"); continue
        # honest-mean cumulative samples (all honest cids in the attack run)
        all_cids = sorted({int(k) for r in runs
                           for k in ((r.get("compute", {}) or {}).get("per_client", {}) or {})})
        honest_cids = [c for c in all_cids if c not in fr_cids]

        def avg_cum(runs_, cids, fld):
            acc = defaultdict(list)
            for r in runs_:
                for cid in cids:
                    for rd, v in _cumulative_by_round(r, cid, fld).items():
                        acc[rd].append(v)
            xs = sorted(acc)
            return xs, [float(np.mean(acc[rd])) for rd in xs]

        hx, hcum = avg_cum(runs, honest_cids, "samples")
        base = str(a.out or "tap_perfr").rstrip(".png")

        for cid in fr_cids:
            tc = data[cid]["tc"]; seeds = data[cid]["seeds"]
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2))
            # LEFT: BER
            tx = sorted(twin[cid])
            if tx:
                axL.plot(tx, [twin[cid][r] for r in tx], color=C_TWIN, lw=2.0, ls=(0, (1, 1)),
                         label=f"{LBL_HONEST_TWIN} (cls {tc})")
            sx, smean, sstd = _mean_over_seeds(seeds, "srv")
            axL.plot(sx, smean, color=C_FR, lw=2.4, marker="o", ms=3, label=LBL_FR_SERVER)
            if nseed > 1:
                axL.fill_between(sx, np.array(smean) - np.array(sstd), np.array(smean) + np.array(sstd),
                                 color=C_FR, alpha=.15)
            axL.axhline(eta_l, color=C_HONEST, ls="--", lw=1.6, label=f"{LBL_ETA_LOOSE} = {eta_l:.3f}")
            axL.axhline(eta_t, color="black", ls=":", lw=1.2, label=f"{LBL_ETA_TIGHT} = {eta_t:.3f}")
            axL.set_xlabel(LBL_ROUND); axL.set_ylabel(LBL_BER_SHORT)
            axL.set_ylim(-0.03, max(0.62, eta_l + 0.06))
            axL.set_title(f"stealth · cid{cid} · class {tc}", fontsize=11); axL.legend(fontsize=7.5)  # TEXT
            # RIGHT: cumulative samples (effort)
            fx, fcum = avg_cum(runs, [cid], "samples")
            axR.plot(hx, hcum, color=C_HONEST, lw=2.4, marker="o", ms=3, label="honest mean")   # TEXT
            axR.plot(fx, fcum, color=C_FR, lw=2.4, marker="s", ms=3, label=f"free-rider cid{cid}")  # TEXT
            if fx and hx:
                axR.fill_between(fx, fcum, [dict(zip(hx, hcum)).get(rd, np.nan) for rd in fx],
                                 color=C_GOOD, alpha=.12, label="compute saved")   # TEXT
                saved = 1 - (fcum[-1] / dict(zip(hx, hcum)).get(fx[-1], np.nan)) if hx else float("nan")
                axR.set_title(f"effort · cid{cid} · saves {saved:.0%}", fontsize=11)   # TEXT
            axR.set_xlabel(LBL_ROUND); axR.set_ylabel(LBL_SAMPLES); axR.legend(fontsize=8)
            fig.suptitle(f"{TITLE_TAP}: stealth vs effort  ·  {f.split('_rep')[0]}  cid{cid}",  # TEXT
                         fontsize=12, fontweight="bold")
            finish(fig, f"{base}_{f}_cid{cid}_effort")


# ###########################################################################
# ##  CLI                                                                   ##
# ###########################################################################
CMDS = {
    "honest_lines": honest_lines,
    "honest_per_round": honest_per_round,
    "class_acc": class_acc,
    "sweep": sweep,
    "timeline": timeline,
    "accuracy": accuracy,
    "dirichlet_dist": dirichlet_dist,
    "gpu_savings": gpu_savings,
    "iso_pair": iso_pair,
    "iso_acc": iso_acc,
    "tap_perfr": tap_perfr,
    "tap_perseed": tap_perseed,
    "tap_effort": tap_effort,
}


def main():
    apply_style()
    ap = argparse.ArgumentParser(description="all FareMark plotting in one place")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in CMDS:
        s = sub.add_parser(name)
        s.add_argument("--in", dest="inp", nargs="+", default=None,
                       help="glob(s) of result.json (ignored by dirichlet_dist / iso_*)")
        s.add_argument("--family", default=None)
        s.add_argument("--families", nargs="+", default=None)
        s.add_argument("--out", default=None)
        s.add_argument("--title", default="")
        s.add_argument("--seed", default=None)
        s.add_argument("--level", default=None)
        s.add_argument("--tail", type=int, default=TAIL)
        s.add_argument("--honest_in", nargs="+", default=None,
                       help="honest result.json glob(s): floor overlay / same-class twin.")
        s.add_argument("--honest_family", default=None)
        s.add_argument("--fr_in", nargs="+", default=None, help="attack result.json glob(s) for iso_*.")
        s.add_argument("--eta", type=float, default=None, help="single calibrated eta (honest_lines).")
        s.add_argument("--eta_tight", type=float, default=None)
        s.add_argument("--eta_loose", type=float, default=None)
        s.add_argument("--classes", default=None, help="comma list: restrict trigger classes / n for dirichlet.")
        s.add_argument("--class", dest="cls", type=int, default=None, help="single trigger class (iso_*).")
        s.add_argument("--per_seed", action="store_true", help="faint per-seed lines (honest_lines).")
        s.add_argument("--warmup", type=int, default=None, help="free-riding start round (iso_pair marker).")
    a = ap.parse_args()
    if a.inp is None and a.cmd not in ("dirichlet_dist", "iso_pair", "iso_acc"):
        ap.error(f"{a.cmd} needs --in")
    if a.out is None and a.inp:
        a.out = default_out(a.inp)
    CMDS[a.cmd](a)


if __name__ == "__main__":
    main()