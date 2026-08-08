"""
plots:
TODO: update the docstring to document all available plotting


  threshold   Is the mu+3sigma calculation SOLID? Shows WHY honest points sit
              above the "tight" eta line even though 3-sigma "should" cover
              ~99.7%: the tight eta is mu+3sigma over ROUND-MEAN BER (variance
              shrunk by ~sqrt(#clients)), but you test it against PER-CLIENT
              BER (full variance). Also shows BER quantisation/skew (why the
              Gaussian 99.7% never holds exactly) and the swingy cumulative eta.

  positions   Are some trigger classes harder? Per-trigger-class BER (bar +
              over-time), so the bimodal honest floor (a few hard classes) that
              inflates the loose eta is visible. 

  fidelity    Global test accuracy over rounds + per-client converged BER
              (honest vs free-rider, if the run has any) + per-client effort.
              NOTE: FedAvg yields one global model, so there is no per-client
              *test accuracy* in result.json -- only global test_acc and
              per-client BER/effort. Per-class accuracy and per-round loss are
              NOT logged either; see the note printed by `fidelity`.

  all         run all three.

Usage
  python plot_diag.py all --in '/path/to/results/*/result.json'
  python plot_diag.py threshold --in '/path/*/result.json' --family t1_iid
  # --out defaults to <common input dir>/figs
"""
import os, sys, glob, json, argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, "faremark"); sys.path.insert(0, "scripts")
    import plotstyle as ps
    ps.apply()
    C_HONEST = ps.C_HONEST; C_FR = ps.C_FR
    C_BAD = getattr(ps, "C_BAD", "#B23A2E"); C_GOOD = getattr(ps, "C_GOOD", "#009E73")
    OK = ps.OKABE
    def finish(fig, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True); ps.finish(fig, path)
    def stacked_panels(n, **kw): return ps.stacked_panels(n, **kw)
except Exception:
    OK = {"black": "#000000", "grey": "#888888", "orange": "#E69F00",
          "skyblue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
          "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
    C_HONEST = OK["blue"]; C_FR = OK["vermillion"]
    C_BAD = "#B23A2E"; C_GOOD = OK["green"]
    plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    def finish(fig, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        print("  wrote", path)
    def stacked_panels(n, figsize=(11, 8), height_ratios=None):
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n, 1, height_ratios=height_ratios or [1]*n, hspace=0.35)
        return fig, tuple(fig.add_subplot(gs[i]) for i in range(n))
    import types as _t
    ps = _t.SimpleNamespace(C_HONEST=C_HONEST, C_FR=C_FR, C_BAD=C_BAD, C_GOOD=C_GOOD,
                            OKABE=OK, apply=lambda: None, finish=finish,
                            stacked_panels=stacked_panels)

# MERGED: threshold.py + separability.py are now one module, scripts/detection.py.
# Both aliases point at it, so every `th.*` and `sep.*` call below is unchanged.
import detection as th
sep = th

def lvl(r):
    m = r.get('manifest', {}) or {}
    v = m.get('sweep_level')
    if v is None: v = (r.get('config', {}) or {}).get('autop_common_per_class')
    try: return float(v)
    except (TypeError, ValueError): return None

def lvl_label(v):
    if v is None: return '?'
    if v < 0: return 'full\nshard'
    if v == 0: return 'triggers\nonly'
    return f'+{int(v)}/cls'

def data_lvl(r):
    """Data budget the run's attacker ACTUALLY used, for titles/labels.
    reduced/submarine -> autop_common_per_class (the +N spectrum knob, via lvl());
    adaptive_tap      -> tap_data_cpc (the per-tap budget). Without this an
    adaptive_tap run mislabels as cpc=autop_common_per_class (=-1 default), a field
    the adaptive_tap attacker never reads."""
    cfg = r.get('config', {}) or {}
    atk = cfg.get('attack') or r.get('attack')
    if atk == 'adaptive_tap':
        v = cfg.get('tap_data_cpc')
        try: return float(v)
        except (TypeError, ValueError): return None
    return lvl(r)

GREY = OK.get("grey", "#888888")
BLACK = OK.get("black", "#000000")
TAIL = 20   # "converged" window = last N rounds

# Two canonical detection thresholds kept for reference and drawn on EVERY timeline:
#   ETA_TIGHT = the frozen aggressive line the server actually used (WM_ETA_FIXED, ~0.064,
#               below 1/m so degenerate: "flag if >= 1 bit wrong").
#   ETA_LOOSE = the loosest sane deployable rule = POOLED mu+3sigma over honest round-means
#               (~0.264, non-degenerate). If honest runs are passed via --honest_in the loose
#               line is recomputed from them (detection.py 'pooled'); otherwise this default
#               is used. Override either with --eta_tight / --eta_loose.
ETA_TIGHT_DEFAULT = 0.064
ETA_LOOSE_DEFAULT = 0.264   # loose reference = per-client mu+3s (true 3-sigma, ~2% honest FPR).
                            # (was 0.264 pooled round-means; switched to the lenient per-client
                            # bound so the timelines bracket 0.064 aggressive .. 0.264 lenient.)


# ---------------------------------------------------------------- io / helpers
def load(globs):
    out = []
    for g in globs:
        for f in sorted(glob.glob(g)):
            try:
                out.append(json.load(open(f)))
            except Exception as e:
                print("  (skip", f, "->", e, ")")
    return out


def fam(r):
    return (r.get("manifest", {}) or {}).get("family")


def pick(runs, family):
    if not family:
        return runs
    return [r for r in runs if fam(r) == family]


def mu3s(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(np.mean(xs)) + 3.0 * (float(np.std(xs)) if len(xs) > 1 else 0.0)


def m_bits(runs):
    for r in runs:
        v = r.get("wm_bits_m")
        if v:
            return int(v)
    return 10


def default_out(inp):
    """<common dir of the input glob(s)>/figs"""
    paths = []
    for g in inp:
        paths += glob.glob(g)
    if not paths:
        base = os.path.dirname(inp[0].split("*")[0].rstrip("/")) or "."
    else:
        base = os.path.commonpath([os.path.dirname(p) for p in paths])
        # step up out of the per-run subdir into the results root
        parent = os.path.dirname(base)
        base = parent or base
    return os.path.join(base, "figs")


def converged_perclient(runs, tail=TAIL, free_rider=False):
    """All individual (client,round) BERs over the converged tail."""
    out = []
    for r in runs:
        for h in r.get("history", [])[-tail:]:
            for p in (h.get("wm_per_client") or []):
                if bool(p.get("is_free_rider")) == free_rider:
                    out.append(p["ber"])
    return out


def converged_roundmeans(runs, tail=TAIL):
    """Per-(run,round) MEAN honest BER over the converged tail."""
    out = []
    for r in runs:
        for h in r.get("history", [])[-tail:]:
            vals = [p["ber"] for p in (h.get("wm_per_client") or [])
                    if not p.get("is_free_rider")]
            if vals:
                out.append(float(np.mean(vals)))
    return out


# ============================================================================
# 1. THRESHOLD SOUNDNESS
# ============================================================================

def threshold(a):
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return
    m = m_bits(runs)
    step = 1.0 / m

    indiv = converged_perclient(runs, free_rider=False)      # per-client (wide)
    rmeans = converged_roundmeans(runs)                      # round-mean (narrow)
    indiv = np.array(indiv); rmeans = np.array(rmeans)

    eta_tight = mu3s(rmeans)          # mu+3s over round-means  (what live detector approximates)
    eta_loose = mu3s(indiv)           # mu+3s over per-client   (the fair-to-honest one)

    # cumulative "live" paper eta trajectory (mean over runs, per round)
    cum = defaultdict(list)
    for r in runs:
        for h in r["history"]:
            if h.get("wm_eta_round") is not None:
                cum[h["round"]].append(h["wm_eta_round"])
    cum_rounds = sorted(cum)
    cum_eta = [np.mean(cum[rd]) for rd in cum_rounds]

    # honest mean BER band per round
    hb = defaultdict(list)
    for r in runs:
        for h in r["history"]:
            vals = [p["ber"] for p in (h.get("wm_per_client") or [])
                    if not p.get("is_free_rider")]
            if vals:
                hb[h["round"]].append(np.mean(vals))
    hb_rounds = sorted(hb)
    hb_mean = [np.mean(hb[rd]) for rd in hb_rounds]

    # coverage curve: fraction of PER-CLIENT honest BERs below a sweep of eta
    grid = np.linspace(0, max(0.5, float(indiv.max()) + step), 300)
    cov_indiv = [np.mean(indiv < e) for e in grid]
    cov_rmean = [np.mean(rmeans < e) for e in grid]

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (a) the two distributions that get confused
    axA = ax[0, 0]
    bins = np.arange(-step / 2, indiv.max() + 1.5 * step, step)
    axA.hist(indiv, bins=bins, color=C_HONEST, alpha=0.55, density=True,
             label=f"per-client BER  (n={len(indiv)}, std={indiv.std():.3f})")
    axA.hist(rmeans, bins=bins, color=OK["orange"], alpha=0.65, density=True,
             label=f"round-MEAN BER  (n={len(rmeans)}, std={rmeans.std():.3f})")
    axA.axvline(eta_tight, color=C_BAD, ls="--", lw=2.2,
                label=f"eta_tight = mu+3s(round-mean) = {eta_tight:.3f}")
    axA.axvline(eta_loose, color=C_GOOD, ls="-", lw=2.2,
                label=f"eta_loose = mu+3s(per-client) = {eta_loose:.3f}")
    axA.set_xlabel("bit-error-rate (converged tail)")
    axA.set_ylabel("density")
    axA.set_title("(a) Two different distributions.\nAveraging clients shrinks the spread -> a much tighter eta")
    axA.legend(fontsize=8, loc="upper right")

    # (b) empirical coverage vs the Gaussian 99.7% target
    axB = ax[0, 1]
    axB.plot(grid, np.array(cov_indiv) * 100, color=C_HONEST, lw=2.4,
             label="coverage of PER-CLIENT BER")
    axB.plot(grid, np.array(cov_rmean) * 100, color=OK["orange"], lw=2.0, ls="--",
             label="coverage of round-MEAN BER")
    axB.axhline(99.87, color=GREY, ls=":", lw=1.6, label="one-sided 3-sigma target = 99.87%")
    for e, c, lab in [(eta_tight, C_BAD, "eta_tight"), (eta_loose, C_GOOD, "eta_loose")]:
        cov = 100 * np.mean(indiv < e)
        axB.axvline(e, color=c, ls="--", lw=1.6)
        axB.annotate(f"{lab}\n{cov:.0f}% of per-client\nbelow", (e, cov),
                     textcoords="offset points", xytext=(8, -30), fontsize=8, color=c)
    axB.set_xlabel("candidate eta")
    axB.set_ylabel("% of honest BERs below eta")
    axB.set_ylim(0, 103)
    axB.set_title("(b) Test sigma on the SAME distribution you calibrate on.\n"
                  "eta_tight covers only ~59% of per-client points -> false positives")
    axB.legend(fontsize=8, loc="lower right")

    # (c) discreteness / non-normality of per-client BER
    axC = ax[1, 0]
    vals, counts = np.unique(indiv, return_counts=True)
    axC.bar(vals, counts / counts.sum(), width=step * 0.8, color=C_HONEST,
            alpha=0.7, label="empirical BER pmf")
    mu, sd = indiv.mean(), indiv.std()
    xs = np.linspace(indiv.min() - step, indiv.max() + step, 200)
    if sd > 0:
        gauss = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        axC.plot(xs, gauss * step, color=C_BAD, lw=2,
                 label="Gaussian(mu,sigma) x bin")
    axC.axvline(0, color=BLACK, lw=0.8)
    axC.set_xlabel(f"BER (quantised in steps of 1/m = {step:.2f}, m={m})")
    axC.set_ylabel("probability")
    axC.set_title("(c) BER is discrete, bounded at 0, right-skewed.\n"
                  "3-sigma is only ever an approximation here")
    axC.legend(fontsize=8)

    # (d) eta over rounds: cumulative (paper) vs frozen tight/loose
    axD = ax[1, 1]
    axD.plot(hb_rounds, hb_mean, color=C_HONEST, lw=2.2, label="honest mean BER")
    if cum_rounds:
        axD.plot(cum_rounds, cum_eta, color=OK["purple"], lw=2.4,
                 label="eta_cumul (paper, live) - swings high")
    axD.axhline(eta_tight, color=C_BAD, ls="--", lw=2, label=f"eta_tight={eta_tight:.3f}")
    axD.axhline(eta_loose, color=C_GOOD, ls="-", lw=2, label=f"eta_loose={eta_loose:.3f}")
    axD.set_xlabel("communication round")
    axD.set_ylabel("BER / eta")
    axD.set_title("(d) The paper's cumulative eta is inflated by pre-convergence\n"
                  "rounds -> very loose -> trivial for a free-rider to sit under")
    axD.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Threshold soundness - {a.family or 'all runs'}  "
                 f"(converged tail = last {TAIL} rounds)", fontsize=13, y=1.005)
    finish(fig, os.path.join(a.out, f"threshold_{a.family or 'all'}.png"))

    print(f"  eta_tight (round-mean mu+3s) = {eta_tight:.4f}  "
          f"-> covers {100*np.mean(indiv<eta_tight):.1f}% of per-client honest BERs")
    print(f"  eta_loose (per-client mu+3s) = {eta_loose:.4f}  "
          f"-> covers {100*np.mean(indiv<eta_loose):.1f}% of per-client honest BERs")
    if cum_eta:
        print(f"  cumulative live eta (final) = {cum_eta[-1]:.4f}")


# ============================================================================
# 2. HARDER TRIGGER CLASS IDS
# ============================================================================
def positions(a):
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return

    # per-trigger-class BER over the converged tail
    byc_tail = defaultdict(list)
    # per-trigger-class BER over time (round -> class -> [bers])
    over_time = defaultdict(lambda: defaultdict(list))
    for r in runs:
        n = len(r.get("history", []))
        for i, h in enumerate(r["history"]):
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider"):
                    continue
                c = int(p["trigger_class"])
                over_time[h["round"]][c].append(p["ber"])
                if i >= n - TAIL:
                    byc_tail[c].append(p["ber"])

    classes = sorted(byc_tail)
    means = [np.mean(byc_tail[c]) for c in classes]
    stds = [np.std(byc_tail[c]) for c in classes]
    order = np.argsort(means)
    classes_s = [classes[i] for i in order]
    means_s = [means[i] for i in order]
    stds_s = [stds[i] for i in order]

    rounds = sorted(over_time)

    # --- OPTIONAL: per-class difficulty diagnostics (present only if the run was
    # produced by the updated wm_verify hook: pmax / entropy / dominance / trig_acc)
    diag_by_class = defaultdict(lambda: defaultdict(list))  # class -> field -> [vals]
    have_diag = False
    for r in runs:
        n = len(r.get("history", []))
        for i, h in enumerate(r["history"]):
            if i < n - TAIL:
                continue
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider"):
                    continue
                for k in ("pmax", "entropy", "dominance", "trig_acc"):
                    if p.get(k) is not None:
                        diag_by_class[int(p["trigger_class"])][k].append(p[k])
                        have_diag = True

    ncol = 3 if have_diag else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.8 * ncol, 5.6))
    axL, axR = axes[0], axes[1]

    # left: sorted per-class converged BER bar
    cols = [C_GOOD if m_ < 0.05 else (OK["orange"] if m_ < 0.15 else C_BAD)
            for m_ in means_s]
    axL.bar(range(len(classes_s)), means_s, yerr=stds_s, color=cols, alpha=0.85,
            capsize=3, error_kw={"lw": 1})
    axL.set_xticks(range(len(classes_s)))
    axL.set_xticklabels([f"cls {c}" for c in classes_s])
    axL.set_xlabel("trigger class (sorted easy -> hard)")
    axL.set_ylabel(f"converged honest BER (last {TAIL} rounds)")
    axL.set_title("(a) Some trigger classes never reach BER~0.\n"
                  "This bimodal floor is what inflates the loose eta")
    overall = np.mean([b for v in byc_tail.values() for b in v])
    axL.axhline(overall, color=GREY, ls=":", lw=1.5, label=f"overall mean = {overall:.3f}")
    axL.legend(fontsize=8)

    # right: per-class BER over rounds (highlight hardest)
    hard = set(classes_s[-3:])
    for c in classes:
        y = [np.mean(over_time[rd][c]) if over_time[rd].get(c) else np.nan for rd in rounds]
        if c in hard:
            axR.plot(rounds, y, lw=2.4, label=f"cls {c} (hard)")
        else:
            axR.plot(rounds, y, lw=0.9, alpha=0.35, color=GREY)
    axR.set_xlabel("communication round")
    axR.set_ylabel("mean honest BER for that class")
    axR.set_title("(b) Hard classes converge slower / plateau above 0\n"
                  "(grey = the easy classes that reach 0)")
    axR.legend(fontsize=8, loc="upper right")

    # (c) WHY: per-class BER vs softmax peakiness (only if diagnostics logged)
    if have_diag:
        axD = axes[2]
        cx = sorted(diag_by_class)
        ber_c = [np.mean(byc_tail[c]) if byc_tail.get(c) else np.nan for c in cx]
        pmax_c = [np.mean(diag_by_class[c]["pmax"]) if diag_by_class[c].get("pmax") else np.nan for c in cx]
        ent_c = [np.mean(diag_by_class[c]["entropy"]) if diag_by_class[c].get("entropy") else np.nan for c in cx]
        # scatter BER vs pmax, colour by entropy
        sc = axD.scatter(pmax_c, ber_c, c=ent_c, s=90, cmap="viridis",
                         edgecolor=BLACK, lw=0.5, zorder=3)
        for c, x_, y_ in zip(cx, pmax_c, ber_c):
            axD.annotate(f"cls {c}", (x_, y_), fontsize=8,
                         textcoords="offset points", xytext=(5, 4))
        cb = fig.colorbar(sc, ax=axD); cb.set_label("softmax entropy (flatter ->)")
        axD.set_xlabel("mean top-1 softmax confidence on trigger samples (p_max)")
        axD.set_ylabel("converged honest BER")
        axD.set_title("(c) WHY: confident (peaky) classes have no tail to\n"
                      "shape -> higher BER. Right+low-entropy = hard")
    else:
        # keep the note only while diagnostics are absent
        fig.text(0.5, -0.02,
                 "add per-class diagnostics (pmax/entropy/dominance/trig_acc) via the "
                 "updated wm_verify hook to get panel (c): BER vs softmax peakiness",
                 ha="center", fontsize=9, color=GREY)

    fig.suptitle(f"Trigger-class difficulty - {a.family or 'all runs'}", fontsize=13, y=1.02)
    finish(fig, os.path.join(a.out, f"positions_{a.family or 'all'}.png"))

    print("  per-class converged BER (easy->hard):")
    for c, m_ in zip(classes_s, means_s):
        print(f"    cls {c}: {m_:.3f}")
    if have_diag:
        print("  per-class diagnostics ARE present -> panel (c) shows BER vs p_max/entropy.")
    else:
        print("  NOTE: per-class diagnostics not in this run. Re-run with the updated")
        print("        wm_verify hook to log pmax/entropy/dominance/trig_acc per client.")


# ============================================================================
# 3. FIDELITY: accuracy + per-client BER + effort (honest vs free-rider)
# ============================================================================
def fidelity(a):
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return

    has_fr = any(p.get("is_free_rider")
                 for r in runs for h in r.get("history", [])
                 for p in (h.get("wm_per_client") or []))

    # global test accuracy over rounds (mean across runs)
    acc = defaultdict(list)
    for r in runs:
        for h in r["history"]:
            if h.get("test_acc") is not None:
                acc[h["round"]].append(h["test_acc"])
    ar = sorted(acc); am = [np.mean(acc[rd]) for rd in ar]
    astd = [np.std(acc[rd]) for rd in ar]

    # per-client converged BER, honest vs FR
    ho = converged_perclient(runs, free_rider=False)
    fr = converged_perclient(runs, free_rider=True)

    # effort from compute.summary (mean across runs)
    def csum(key):
        vals = [(r.get("compute", {}) or {}).get("summary", {}).get(key)
                for r in runs]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else np.nan
    eff = {k: csum(k) for k in ("honest_mean_samples", "fr_mean_samples",
                                "honest_mean_gpu_ms", "fr_mean_gpu_ms")}

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5))

    # (a) global accuracy
    ax[0].plot(ar, am, color=C_HONEST, lw=2.6)
    ax[0].fill_between(ar, np.array(am) - np.array(astd), np.array(am) + np.array(astd),
                       color=C_HONEST, alpha=0.15)
    ax[0].set_xlabel("communication round")
    ax[0].set_ylabel("global test accuracy (%)")
    final = am[-1] if am else float("nan")
    ax[0].set_title(f"(a) Fidelity: global model accuracy\nfinal = {final:.2f}%")

    # (b) per-client converged BER distributions
    axB = ax[1]
    parts = [ho] + ([fr] if fr else [])
    labels = ["honest"] + (["free-rider"] if fr else [])
    colors = [C_HONEST] + ([C_FR] if fr else [])
    for i, (vals, lab, c) in enumerate(zip(parts, labels, colors)):
        xj = i + (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.35
        axB.scatter(xj, vals, s=16, alpha=0.4, color=c)
        axB.hlines(np.mean(vals), i - 0.28, i + 0.28, color=BLACK, lw=2)
    axB.set_xticks(range(len(labels)))
    axB.set_xticklabels(labels)
    axB.set_ylabel(f"converged per-client BER (last {TAIL} rounds)")
    ttl = "(b) Honest vs free-rider BER"
    if not fr:
        ttl += "\n(this run is ALL-HONEST - no FR present)"
    axB.set_title(ttl)

    # (c) effort
    axC = ax[2]
    if has_fr and not np.isnan(eff["honest_mean_samples"]):
        xs = [0, 1]
        axC.bar(xs, [eff["honest_mean_samples"], eff["fr_mean_samples"]],
                color=[C_HONEST, C_FR], alpha=0.85, width=0.6)
        axC.set_xticks(xs); axC.set_xticklabels(["honest", "free-rider"])
        axC.set_ylabel("mean image-passes (whole run)")
        ratio = (eff["fr_mean_samples"] / eff["honest_mean_samples"]
                 if eff["honest_mean_samples"] else float("nan"))
        axC.set_title(f"(c) Training effort\nFR / honest = {ratio:.2%}")
    else:
        axC.axis("off")
        axC.text(0.5, 0.5, "No free-rider in this run,\nso no effort comparison.\n\n"
                           "Point --in at a run with\nFREE_RIDER_IDS set to populate this.",
                 ha="center", va="center", fontsize=11, color=GREY)

    fig.suptitle(f"Fidelity & per-client comparison - {a.family or 'all runs'}",
                 fontsize=13, y=1.02)
    finish(fig, os.path.join(a.out, f"fidelity_{a.family or 'all'}.png"))

    print(f"  final global accuracy = {final:.2f}%")
    print(f"  honest converged BER: mean={np.mean(ho):.3f}  n={len(ho)}")
    if fr:
        print(f"  free-rider converged BER: mean={np.mean(fr):.3f}  n={len(fr)}")
    else:
        print("  (all-honest run: no free-rider BER, no effort ratio)")
    print("  NOTE: FedAvg produces ONE global model - result.json has no per-client")
    print("        test accuracy. 'accuracy of each client' isn't logged; only global")
    print("        test_acc + per-client BER + per-client compute effort exist.")


# ============================================================================
# 4. CANONICAL THRESHOLD -- intuitive derivation (mean-over-clients, then mu+3s over rounds)
# ============================================================================
def thresholds(a):
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return
    tail = TAIL

    # step 1: per-round mean over clients (honest), over the converged tail
    m_r, m_r_rounds = [], []
    for r in runs:
        hist = r.get("history", [])
        for h in hist[-tail:]:
            vals = [p["ber"] for p in (h.get("wm_per_client") or [])
                    if not p.get("is_free_rider")]
            if vals:
                m_r.append(float(np.mean(vals))); m_r_rounds.append(h.get("round"))
    m_r = np.array(m_r)
    mu = m_r.mean(); sigma = m_r.std()
    eta = mu + 3 * sigma                                   # steps 2-4

    # frozen constant, if the pre-calibrated file sits next to the runs
    frozen = None
    for cand in (os.path.join(a.out, "..", "eta_calibrated.json"),
                 os.path.join(os.path.dirname(a.out), "eta_calibrated.json")):
        if os.path.exists(cand):
            try:
                frozen = float(json.load(open(cand))["eta"]); break
            except Exception:
                pass

    # honest per-client & FR per-client BER (converged) to show where eta lands
    ho = np.array(converged_perclient(runs, free_rider=False))
    fr = np.array(converged_perclient(runs, free_rider=True))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # (a) derivation
    axA.scatter(range(len(m_r)), m_r, s=30, color=C_HONEST, alpha=0.7,
                label="m_r = mean BER over clients, each round", zorder=3)
    axA.axhline(mu, color=BLACK, lw=2, label=f"grand mean mu = {mu:.3f}")
    for k, al in [(1, 0.18), (2, 0.12), (3, 0.07)]:
        axA.axhspan(mu - k * sigma, mu + k * sigma, color=OK["orange"], alpha=al, lw=0)
    axA.axhline(eta, color=C_BAD, lw=1.8, ls=":",
                label=f"POOLED mu+3sigma (reference) = {eta:.3f}")
    if frozen is not None:
        axA.axhline(frozen, color=C_GOOD, lw=2.6, ls="-",
                    label=f"eta USED = avg of per-seed etas = {frozen:.3f}")
    axA.set_xlabel(f"round index within converged tail (last {tail}, pooled over seeds)")
    axA.set_ylabel("bit-error-rate")
    axA.set_title("(a) dots = per-round mean BER (pooled over seeds). The USED eta (green)\n"
                  "is the AVERAGE of each seed's own (mu_s+3sigma_s); pooled (dotted) is looser")
    axA.legend(fontsize=8, loc="upper right")

    # (b) where eta lands vs honest & free-rider
    axB.hist(ho, bins=np.arange(-0.05, max(ho.max() if len(ho) else 0.5,
                                           fr.max() if len(fr) else 0.5) + 0.15, 0.1),
             color=C_HONEST, alpha=0.6, density=True, label=f"honest per-client (n={len(ho)})")
    if len(fr):
        axB.hist(fr, bins=np.arange(-0.05, max(fr.max(), 0.5) + 0.15, 0.1),
                 color=C_FR, alpha=0.55, density=True, label=f"free-rider per-client (n={len(fr)})")
    axB.axvline(eta, color=C_BAD, lw=1.8, ls=":", label=f"pooled (ref) = {eta:.3f}")
    if frozen is not None:
        axB.axvline(frozen, color=C_GOOD, lw=2.4, label=f"eta USED = {frozen:.3f}")
    axB.set_xlabel("converged per-client BER")
    axB.set_ylabel("density")
    fp = 100 * np.mean(ho >= (frozen if frozen is not None else eta)) if len(ho) else 0
    rc = 100 * np.mean(fr >= (frozen if frozen is not None else eta)) if len(fr) else float("nan")
    axB.set_title(f"(b) Where the line lands.  honest flagged (FPR) = {fp:.0f}%"
                  + (f",  FR caught (recall) = {rc:.0f}%" if len(fr) else ""))
    axB.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Canonical threshold - {a.family or 'all runs'}", fontsize=13, y=1.02)
    finish(fig, os.path.join(a.out, f"thresholds_{a.family or 'all'}.png"))
    print(f"  eta (recomputed) = {eta:.4f}  (mu={mu:.4f}, sigma={sigma:.4f}, n_round_means={len(m_r)})")
    if frozen is not None:
        print(f"  frozen constant in use = {frozen:.4f}")


# ============================================================================
# 5. CLASS DYNAMICS -- loss & accuracy per trigger class (proves "hard" classes)
# ============================================================================
def class_dynamics(a):
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return

    # --- client-side wm_stats (per-round cls_loss / wm_loss / trig_train_acc) ---
    # keyed by trigger class (each client owns one), converged tail only.
    by_cls = defaultdict(lambda: defaultdict(list))   # cls -> field -> [vals]
    loss_curves = defaultdict(lambda: defaultdict(dict))  # cls -> field -> {round:val}
    have_client = False
    for r in runs:
        pcs = (r.get("compute", {}) or {}).get("per_client", {}) or {}
        nrounds = len(r.get("history", []))
        for cid, c in pcs.items():
            ws = c.get("wm_stats")
            if not ws:
                continue
            have_client = True
            items = sorted(((int(rd), s) for rd, s in ws.items()), key=lambda t: t[0])
            for rd, s in items:
                cls = s.get("trigger_class")
                if cls is None:
                    continue
                for k in ("cls_loss", "wm_loss", "trig_train_acc"):
                    if s.get(k) is not None:
                        loss_curves[cls][k][rd] = s[k]
                        if rd >= nrounds - TAIL:
                            by_cls[cls][k].append(s[k])

    # --- server-side diagnostics (pmax/entropy) + BER per class from history ---
    ber_by, pmax_by, ent_by = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in runs:
        n = len(r.get("history", []))
        for i, h in enumerate(r["history"]):
            if i < n - TAIL:
                continue
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider"):
                    continue
                c = int(p["trigger_class"])
                ber_by[c].append(p["ber"])
                if p.get("pmax") is not None:
                    pmax_by[c].append(p["pmax"]); ent_by[c].append(p.get("entropy"))
    have_server_diag = any(pmax_by.values())

    classes = sorted(ber_by)
    ber_c = [np.mean(ber_by[c]) for c in classes]
    order = np.argsort(ber_c)
    cls_sorted = [classes[i] for i in order]

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # (a) watermark loss per class (client-side) -- high = hard to embed
    axA = ax[0, 0]
    if have_client:
        wm_c = [np.mean(by_cls[c]["wm_loss"]) if by_cls[c].get("wm_loss") else np.nan for c in cls_sorted]
        axA.bar(range(len(cls_sorted)), wm_c, color=C_BAD, alpha=0.8)
        axA.set_xticks(range(len(cls_sorted))); axA.set_xticklabels([f"cls {c}" for c in cls_sorted])
        axA.set_ylabel("converged watermark loss  L_wm")
        axA.set_title("(a) Watermark-embedding loss per class\n(client-side; higher = harder to embed)")
    else:
        axA.axis("off")
        axA.text(0.5, 0.5, "no client-side wm_stats yet\n(re-run with updated clients.py)",
                 ha="center", va="center", color=GREY)

    # (b) trigger-class TRAIN accuracy (client-side) -- low = fuzzy boundary
    axB = ax[0, 1]
    if have_client:
        acc_c = [np.mean(by_cls[c]["trig_train_acc"]) if by_cls[c].get("trig_train_acc") else np.nan
                 for c in cls_sorted]
        axB.bar(range(len(cls_sorted)), acc_c, color=C_HONEST, alpha=0.8)
        axB.set_xticks(range(len(cls_sorted))); axB.set_xticklabels([f"cls {c}" for c in cls_sorted])
        axB.set_ylabel("trigger-class train accuracy")
        axB.set_title("(b) Classification accuracy on the trigger class\n(low = fuzzier boundary)")
    else:
        axB.axis("off")
        axB.text(0.5, 0.5, "no client-side wm_stats yet", ha="center", va="center", color=GREY)

    # (c) BER vs softmax peakiness (server-side) -- the mechanism
    axC = ax[1, 0]
    if have_server_diag:
        cx = [c for c in classes if pmax_by[c]]
        px = [np.mean(pmax_by[c]) for c in cx]
        by = [np.mean(ber_by[c]) for c in cx]
        ec = [np.mean([e for e in ent_by[c] if e is not None]) if ent_by[c] else np.nan for c in cx]
        sc = axC.scatter(px, by, c=ec, s=90, cmap="viridis", edgecolor=BLACK, lw=0.5, zorder=3)
        for c, x_, y_ in zip(cx, px, by):
            axC.annotate(f"cls {c}", (x_, y_), fontsize=8, textcoords="offset points", xytext=(5, 4))
        fig.colorbar(sc, ax=axC).set_label("softmax entropy")
        axC.set_xlabel("mean top-1 softmax confidence (p_max)")
        axC.set_ylabel("converged honest BER")
        axC.set_title("(c) WHY: confident (peaky) classes = higher BER")
    else:
        axC.axis("off")
        axC.text(0.5, 0.5, "no server-side diagnostics yet\n(re-run with updated wm_verify)",
                 ha="center", va="center", color=GREY)

    # (d) loss curves over rounds for hardest vs easiest class
    axD = ax[1, 1]
    if have_client and len(cls_sorted) >= 2:
        for c, lab, col in [(cls_sorted[-1], "hardest", C_BAD), (cls_sorted[0], "easiest", C_GOOD)]:
            cur = loss_curves[c].get("wm_loss", {})
            if cur:
                xs = sorted(cur); axD.plot(xs, [cur[x] for x in xs], color=col, lw=2.2,
                                           label=f"cls {c} ({lab})")
        axD.set_xlabel("communication round"); axD.set_ylabel("watermark loss L_wm")
        axD.set_title("(d) L_wm over training: hard class stays high")
        axD.legend(fontsize=8)
    else:
        axD.axis("off")
        axD.text(0.5, 0.5, "no client-side loss curves yet", ha="center", va="center", color=GREY)

    fig.suptitle(f"Trigger-class embedding difficulty - {a.family or 'all runs'}", fontsize=13, y=1.01)
    finish(fig, os.path.join(a.out, f"class_dynamics_{a.family or 'all'}.png"))
    if not (have_client or have_server_diag):
        print("  NOTE: neither client wm_stats nor server diagnostics present. Re-run with")
        print("        the updated clients.py + wm_verify to populate these panels.")
    else:
        print(f"  client-side stats: {have_client} | server-side diagnostics: {have_server_diag}")


# ============================================================================

BLACK = OK.get('black', '#000000')
CONVERGED_TAIL = TAIL  # plot_tests alias

def eta_defs(runs, tail=TAIL, fixed_path=None):
    """CANONICAL threshold only. Recompute eta = mean-over-clients-then-mu+3sigma
    -over-rounds from the honest runs (for display); prefer the frozen constant in
    eta_calibrated.json when a path is given. Keeps the old dict keys so existing
    plot bodies still run -- but only 'eta_tight' (= the canonical eta) is set;
    'eta_loose' is None (its line is skipped everywhere)."""
    live = []
    for r in runs:
        for h in r.get("history", [])[-tail:]:
            if h.get("wm_eta_round") is not None:
                live.append(h["wm_eta_round"])
    eta = th.frozen_eta(runs, tail=tail)
    fixed = th.load_fixed(fixed_path) if fixed_path else None
    return {
        "eta_tight": fixed if fixed is not None else eta,   # THE canonical/frozen eta
        "eta_loose": None,                                  # per-client version: DROPPED
        "eta_cumul": float(np.mean(live)) if live else None,  # what the server actually used
        "eta_fixed": fixed if fixed is not None else eta,
    }


def honest_class_floor(honest_runs, classes=None, tail=TAIL):
    """Per-trigger-class honest BER floor, pooled over honest runs & the converged
    tail. Returns {class_id: [ber, ...]} restricted to `classes` if given.
    Used to overlay 'what an honest client at THIS trigger class would floor at'
    on the free-rider timeline, so the FR line is read against its own class
    rather than the (harder-class-inflated) honest mixture."""
    from collections import defaultdict as _dd
    by_cls = _dd(list)
    want = set(classes) if classes is not None else None
    for r in honest_runs:
        for h in r.get("history", [])[-tail:]:
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider"):
                    continue
                c = int(p["trigger_class"])
                if want is None or c in want:
                    by_cls[c].append(p["ber"])
    return by_cls


def timeline(a):
    # Load runs based on family, level, and optional seed.
    # If seed is None (default), it loads ALL seeds matching the criteria.
    runs = [r for r in load(a.inp) if (a.family is None or fam(r) == a.family)
            and (a.level is None or lvl(r) == float(a.level))
            and (a.seed is None or r.get("seed") == int(a.seed))]
    if not runs: print("no matching run"); return

    num_seeds = len(runs)
    is_aggregated = num_seeds > 1
    
    # Use the first run for static info (rounds, warmup window)
    r_ref = runs[0]
    hist = r_ref.get("history", [])
    rounds = [h["round"] for h in hist]
    
    # --- Aggregate actions (taps/coasts) across all seeds ---
    taps = defaultdict(int)
    coasts = defaultdict(int)
    for r in runs:
        pc = (r.get("compute", {}) or {}).get("per_client", {}) or {}
        for cid, c in pc.items():
            for t in c.get("trace", []):
                if t.get("action") == "tap": taps[t["round"]] += 1
                elif t.get("action") == "coast": coasts[t["round"]] += 1

    # Collect honest and free-rider mean BERs for every seed, every round
    honest_means_per_seed = []
    freer_means_per_seed = []
    # ALSO collect each individual free-rider's BER trace, keyed by (seed_idx, cid),
    # so we can draw thin reference lines showing per-FR spread behind the mean.
    fr_indiv = {}   # (seed_idx, cid) -> list of ber per round
    # AND the honest client(s) sharing a trigger class with any free-rider -- the
    # true same-class comparison (blue mean is over ALL honest clients, too coarse).
    fr_classes = set()
    for r in runs:
        for h in r.get("history", []):
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider") and p.get("trigger_class") is not None:
                    fr_classes.add(p["trigger_class"])
    honest_sameclass = {}   # (seed_idx, cid) -> list of ber per round
    for si, r in enumerate(runs):
        h_means, f_means = [], []
        for h in r.get("history", []):
            pcs = (h.get("wm_per_client") or [])
            h_vals = [p["ber"] for p in pcs if not p.get("is_free_rider")]
            f_vals = [p["ber"] for p in pcs if p.get("is_free_rider")]
            h_means.append(np.mean(h_vals) if h_vals else np.nan)
            f_means.append(np.mean(f_vals) if f_vals else np.nan)
            for p in pcs:
                if p.get("is_free_rider"):
                    fr_indiv.setdefault((si, p.get("cid")), []).append(p["ber"])
                elif p.get("trigger_class") in fr_classes:
                    honest_sameclass.setdefault((si, p.get("cid")), []).append(p["ber"])
        honest_means_per_seed.append(h_means)
        freer_means_per_seed.append(f_means)

    # Convert to numpy arrays for statistics over the "seed" axis (axis 0)
    honest_arr = np.array(honest_means_per_seed)
    freer_arr = np.array(freer_means_per_seed)
    
    h_mean = np.nanmean(honest_arr, axis=0)
    h_std = np.nanstd(honest_arr, axis=0)
    f_mean = np.nanmean(freer_arr, axis=0)
    f_std = np.nanstd(freer_arr, axis=0)

    # Calibration window schedule
    lo, hi = th.calib_window(r_ref)
    W = hi + 1

    E = eta_defs(runs)
    fig, ax = plt.subplots(figsize=(12, 6.2))

    # --- PLOTTING LOGIC ---
    if is_aggregated:
        # Aggregated mode: Plot Mean + Standard Deviation shaded bands
        ax.fill_between(rounds, h_mean - h_std, h_mean + h_std, 
                         color=ps.C_HONEST, alpha=0.2, lw=0, label="honest mean ± std")
        ax.plot(rounds, h_mean, color=ps.C_HONEST, lw=3, label="honest mean BER")
        
        ax.fill_between(rounds, f_mean - f_std, f_mean + f_std, 
                         color=ps.C_FR, alpha=0.2, lw=0, label="free-rider mean ± std")
        ax.plot(rounds, f_mean, color=ps.C_FR, lw=3, label="free-rider mean BER")

        # thin per-free-rider reference lines (one per cid per seed). Shows how much
        # of the mean is spread: individual FRs can straddle eta even when the mean does not.
        _lbl_done = False
        for (si, cid), trace in sorted(fr_indiv.items()):
            n = min(len(trace), len(rounds))
            ax.plot(rounds[:n], trace[:n], color=ps.C_FR, lw=0.7, alpha=0.35,
                    zorder=2, label=("individual free-riders" if not _lbl_done else None))
            _lbl_done = True
        # honest client(s) at the SAME trigger class as the free-rider -- the real
        # apples-to-apples comparison, drawn as thin blue lines (mean is over ALL honest).
        _lbl_h = False
        for (si, cid), trace in sorted(honest_sameclass.items()):
            n = min(len(trace), len(rounds))
            ax.plot(rounds[:n], trace[:n], color=ps.C_HONEST, lw=0.9, alpha=0.5,
                    ls=(0, (4, 2)), zorder=3,
                    label=("honest client(s) at FR's class" if not _lbl_h else None))
            _lbl_h = True

        # Plot tap/coast markers if at least 50% of seeds performed that action in a round
        if len(taps) > 0:
            tap_x = [rd for rd, cnt in taps.items() if cnt > num_seeds / 2]
            tap_y = [f_mean[rounds.index(rd)] for rd in tap_x]
            ax.scatter(tap_x, tap_y, marker="v", s=34, color=ps.C_FR, edgecolor="white", zorder=5, label="tap (re-embed) [majority]")
        if len(coasts) > 0:
            coast_x = [rd for rd, cnt in coasts.items() if cnt > num_seeds / 2]
            coast_y = [f_mean[rounds.index(rd)] for rd in coast_x]
            ax.scatter(coast_x, coast_y, marker="s", s=30, color="#FFFFFF", edgecolor=ps.C_FR, zorder=5, label="coast (no train) [majority]")

    else:
        # Single Seed mode (Your original code) - Keep individual client lines
        honest, freer = {}, {}
        for h in hist:
            for p in (h.get("wm_per_client") or []):
                (freer if p.get("is_free_rider") else honest).setdefault(p["cid"], {})[h["round"]] = p["ber"]
        def series(d, cid): return [d[cid].get(rd, np.nan) for rd in rounds]
        
        for cid in honest:
            ax.plot(rounds, series(honest, cid), color=ps.C_HONEST, lw=0.8, alpha=0.25)
        for cid in freer:
            ax.plot(rounds, series(freer, cid), color=ps.C_FR, lw=0.9, alpha=0.5,
                    label=f"free-rider cid {cid} (cls {cid%100})")
        
        # Plot single seed mean
        ax.plot(rounds, h_mean, color=ps.C_HONEST, lw=2.8, label="honest mean BER")
        ax.plot(rounds, f_mean, color=ps.C_FR, lw=2.8, label="free-rider mean BER")
        _lbl_done = False
        for (si, cid), trace in sorted(fr_indiv.items()):
            n = min(len(trace), len(rounds))
            ax.plot(rounds[:n], trace[:n], color=ps.C_FR, lw=0.7, alpha=0.4,
                    zorder=2, label=("individual free-riders" if not _lbl_done else None))
            _lbl_done = True
        
        # Single Seed markers
        tap_x = [rd for rd in rounds if rd in taps]
        if tap_x:
            tap_y = [f_mean[rounds.index(rd)] for rd in tap_x]
            ax.scatter(tap_x, tap_y, marker="v", s=34, color=ps.C_FR, edgecolor="white", zorder=5, label="tap (re-embed)")
        coast_x = [rd for rd in rounds if rd in coasts]
        if coast_x:
            coast_y = [f_mean[rounds.index(rd)] for rd in coast_x]
            ax.scatter(coast_x, coast_y, marker="s", s=30, color="#FFFFFF", edgecolor=ps.C_FR, zorder=5, label="coast (no train)")

    # --- VISUAL GUIDES (Background, Thresholds, Labels) ---
    ytop = ax.get_ylim()[1]
    ax.axvspan(min(rounds), lo - 0.5, color="#FADFA6", alpha=0.30, lw=0, label="forced-honest warmup")
    ax.axvspan(lo - 0.5, hi + 0.5, color="#BFE3C6", alpha=0.55, lw=0, label=f"calibration window [{lo},{hi}] (η frozen here)")
    ax.axvline(lo - 0.5, color="#2C7A3F", ls="-", lw=1.4)
    ax.axvline(W - 0.5, color=GREY, ls="--", lw=1.6)
    ax.text(lo - 0.4, ytop*0.97, " converged → calibrate η", color="#2C7A3F", fontsize=8.5, va="top")
    ax.text(W - 0.4, ytop*0.90, " free-riding starts", color=GREY, fontsize=8.5, va="top")

    # ONLY the calibrated (frozen) eta the server ACTUALLY used for detection.
    # Prefer the value the server logged each round (wm_eta_round == WM_ETA_FIXED,
    # flat), then config.wm_eta_fixed, then the recomputed frozen eta. Do NOT use
    # eta_defs()'s recomputation first: on an ATTACK run it recomputes mu+3s from
    # the 8 honest clients (which include hard classes) and lands ~0.099, not the
    # frozen 0.064 the detector applied.
    live = [h.get("wm_eta_round") for h in hist if h.get("wm_eta_round") is not None]
    cfg_fixed = (r_ref.get("config") or {}).get("wm_eta_fixed")
    if live:
        eta_cal = float(np.median(live))
    elif cfg_fixed:
        eta_cal = float(cfg_fixed)
    else:
        eta_cal = E.get("eta_fixed")
    # tight (frozen) eta -- what the server actually used
    eta_tight = getattr(a, "eta_tight", None)
    if eta_tight is None:
        eta_tight = eta_cal if eta_cal is not None else ETA_TIGHT_DEFAULT
    if eta_tight is not None:
        ax.axhline(eta_tight, color=BLACK, ls="--", lw=2.2,
                   label=f"η tight (frozen, used) = {eta_tight:.3f}")

    # loose eta -- the loosest sane deployable rule; drawn on EVERY timeline.
    # CONSISTENCY FIX: recompute mu+3s over honest PER-CLIENT BERs (the same rule
    # tap_perfr and plot_sameclass_pair use), NOT over round-means. Before this,
    # timeline drew mu+3s(round-means) (~0.075 IID / ~0.178 non-IID) while the
    # tap/iso plots drew mu+3s(per-client) (0.264 IID) -- the SAME label "η loose"
    # showed two different values across figures. Per-client is the fair-to-honest
    # rule (matches ETA_LOOSE_DEFAULT) and scales correctly with the partition.
    eta_loose = getattr(a, "eta_loose", None)
    if eta_loose is None and getattr(a, "honest_in", None):
        try:
            href2 = [r for r in load(a.honest_in) if th.is_honest_run(r)
                     and (a.honest_family is None or fam(r) == a.honest_family)]
            indiv = converged_perclient(href2, tail=getattr(a, "tail", TAIL),
                                        free_rider=False)
            if len(indiv):
                eta_loose = mu3s(np.asarray(indiv))
        except Exception:
            eta_loose = None
    if eta_loose is None:
        eta_loose = ETA_LOOSE_DEFAULT
    ax.axhline(eta_loose, color="#3B6FB5", ls=(0, (5, 2)), lw=2.0,
               label=f"η loose (per-client μ+3σ) = {eta_loose:.3f}")

    # --- OVERLAY: honest floor for the free-rider's OWN trigger classes ---------
    # Read the FR line against an honest client at the SAME class, not the honest
    # mixture (which is dragged up by hard classes like cls 6). Honest runs come
    # from a separate directory via --honest_in.
    fr_classes = sorted({int(p["trigger_class"])
                         for r in runs for h in r.get("history", [])
                         for p in (h.get("wm_per_client") or [])
                         if p.get("is_free_rider")})
    if getattr(a, "honest_in", None) and fr_classes:
        href = [r for r in load(a.honest_in) if th.is_honest_run(r)
                and (a.honest_family is None or fam(r) == a.honest_family)]
        floor = honest_class_floor(href, classes=fr_classes)
        per_cls_mean = {c: float(np.mean(v)) for c, v in floor.items() if v}
        if per_cls_mean:
            vals = list(per_cls_mean.values())
            lo_f, hi_f, mid_f = min(vals), max(vals), float(np.mean(vals))
            if len(per_cls_mean) <= 4:
                cls_str = ", ".join(f"cls {c} {per_cls_mean[c]:.2f}" for c in sorted(per_cls_mean))
            else:
                cls_str = f"{len(per_cls_mean)} classes, floor {lo_f:.2f}-{hi_f:.2f}"
            ax.axhspan(lo_f, hi_f, color=ps.C_HONEST, alpha=0.12, lw=0,
                       label=f"honest floor @ FR classes ({cls_str})")
            ax.axhline(mid_f, color=ps.C_HONEST, ls=(0, (2, 2)), lw=1.6, alpha=0.8)

    # Data usage label (calculate average over seeds)
    eff_ratios = []
    for r in runs:
        cs = (r.get("compute", {}) or {}).get("summary", {}) or {}
        if cs.get("effort_ratio_samples") is not None:
            eff_ratios.append(cs["effort_ratio_samples"])
    avg_eff = np.mean(eff_ratios) if eff_ratios else None
    
    if avg_eff is not None:
        note = f"Data used: {avg_eff*100:.0f}% of honest total"
        if is_aggregated: note += f"\n(Avg over {num_seeds} seeds)"
        note += f"\n(Config cpc={data_lvl(r_ref)})"
        ax.text(0.02, 0.05, note, transform=ax.transAxes, fontsize=9, 
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))

    ax.set_xlabel("communication round"); ax.set_ylabel("bit-error-rate (lower = mark present)")
    agg_seed_str = f"aggregated over {num_seeds} seeds" if is_aggregated else f"seed={r_ref.get('seed')}"
    ax.set_title(a.title or f"BER vs round  ·  {fam(r_ref)}  ·  cpc={data_lvl(r_ref)}  ·  {agg_seed_str}")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    _deg = "  (< 1/m, degenerate)" if (eta_tight is not None and eta_tight < 0.1) else ""
    _cfg = r_ref.get("config") or {}
    _cm = _cfg.get("tap_coast_mode") if (_cfg.get("attack") == "adaptive_tap") else None
    note = (f"Black dashed = η tight (frozen, used) = {eta_tight:.3f}{_deg}. "
            f"Blue dashed = η loose (pooled μ+3σ) = {eta_loose:.3f}."
            + (f"  Coast mode = {_cm}." if _cm else "")
            + "\nA free-rider whose BER stays below a line is not flagged by it. "
            "Warmup (yellow) = forced-honest; green = calibration window; grey dashed = free-riding starts.")
    ax.text(0.005, -0.18, note, transform=ax.transAxes, fontsize=8.5, color=GREY)
    ps.finish(fig, a.out + ".png")
    print(f"calib window [{lo},{hi}] | free-ride from {W} | n_taps={len(taps)} n_coasts={len(coasts)} | seeds={num_seeds}")


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
    ax.set_xlabel("setting (scope × trigger-class-id set)"); ax.set_ylabel("training data / round")
    ps.finish(fig, a.out + ".png")
    print("eta_tight:", et)


# ================================================================= THRESHOLDS

_level_key = lvl
_label_for_level = lvl_label

def honest_fpr(a):
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return

    # gather converged-round per-client honest BERs, keyed by trigger_class
    by_class = {}                 # trigger_class -> [ber, ...]
    round_means = []              # per-(run,round) mean honest BER (the coded calibration input)
    all_indiv = []                # every individual honest (client,round) BER
    for r in runs:
        hist = r.get("history", [])[-CONVERGED_TAIL:]
        for h in hist:
            pcs = [p for p in (h.get("wm_per_client") or []) if not p.get("is_free_rider")]
            if not pcs:
                continue
            round_means.append(np.mean([p["ber"] for p in pcs]))
            for p in pcs:
                by_class.setdefault(int(p["trigger_class"]), []).append(p["ber"])
                all_indiv.append(p["ber"])

    if not by_class:
        print("no per-client BER in history for", a.family); return

    # two eta definitions
    # eta FROZEN on ALL clients during the converged warmup window (before any
    # free-riding) -> independent of who the free-riders are.
    eta_roundmean = th.frozen_eta(runs); eta_perclient = None  # per-client (loose) DROPPED

    fpr_rm = np.mean([b >= eta_roundmean for b in all_indiv]) if eta_roundmean else 0.0
    fpr_pc = np.mean([b >= eta_perclient for b in all_indiv]) if eta_perclient else 0.0

    classes = sorted(by_class)
    fig, ax = plt.subplots(figsize=(11, 6))
    # individual points (jittered) + per-class mean bar
    for i, c in enumerate(classes):
        vals = by_class[c]
        xj = i + (np.random.rand(len(vals)) - 0.5) * 0.5
        ax.scatter(xj, vals, s=14, alpha=0.35, color=ps.C_HONEST,
                   label="honest client-round" if i == 0 else None)
        ax.hlines(np.mean(vals), i - 0.3, i + 0.3, color=ps.OKABE["black"], lw=2)
    overall = np.mean(all_indiv)
    ax.axhline(overall, color=ps.OKABE["grey"], ls=":", lw=1.5,
               label=f"overall honest mean = {overall:.3f}")
    ax.axhline(eta_roundmean, color=ps.C_BAD, ls="--", lw=2.2,
               label=f"η = μ+3σ over round-MEANS = {eta_roundmean:.3f}  (as coded → FPR {fpr_rm:.0%})")
    if eta_perclient is not None:
        ax.axhline(eta_perclient, color=ps.C_GOOD, ls="-", lw=2.2,
               label=f"η = μ+3σ over PER-CLIENT BERs = {eta_perclient:.3f}")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"cls {c}" for c in classes])
    ax.set_xlabel("trigger class id")
    ax.set_ylabel("honest bit-error-rate (converged rounds)")
    ax.set_title("TEST 1 — honest clients vs η.  Points above a line = that η flags honest clients.\n"
                 "round-mean η catches hard class ids AND false-positives; per-client η spares them but is looser")
    ax.legend(loc="upper right", fontsize=8)
    ps.finish(fig, a.out + ".png")
    print(f"eta_roundmean={eta_roundmean:.4f} FPR={fpr_rm:.3f}")


# =====================================================================
# TEST 2 / 3 — data sweep: per-FR & per-honest BER + effort
# =====================================================================
def test_data(a):
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return

    # group by sweep level
    levels = {}
    for r in runs:
        lv = _level_key(r)
        levels.setdefault(lv, []).append(r)
    # order: triggers-only(0), +N ascending, full shard(-1) last
    order = sorted(levels, key=lambda v: (v == -1, v if v is not None else 1e9))

    fr_mean, fr_std, ho_mean = [], [], []
    fr_indiv = {}          # level_index -> list of per-FR mean BERs (each FR distinguishable)
    g_ms_fr, g_ms_ho, s_fr, s_ho = [], [], [], []
    pool_rm, pool_pc = [], []   # honest ROUND-MEAN and PER-CLIENT BERs -> fair eta (see below)

    for li, lv in enumerate(order):
        rs = levels[lv]
        fr_vals, ho_vals = [], []
        per_fr = {}        # cid -> [ber across seeds/rounds]
        gmf, gmh, smf, smh = [], [], [], []
        for r in rs:
            hist = r.get("history", [])[-CONVERGED_TAIL:]
            for h in hist:
                hround = []
                for p in (h.get("wm_per_client") or []):
                    if p.get("is_free_rider"):
                        fr_vals.append(p["ber"]); per_fr.setdefault(p["cid"], []).append(p["ber"])
                    else:
                        ho_vals.append(p["ber"]); hround.append(p["ber"]); pool_pc.append(p["ber"])
                if hround:
                    pool_rm.append(float(np.mean(hround)))   # this round's honest MEAN
            cs = (r.get("compute", {}) or {}).get("summary", {}) or {}
            if cs.get("fr_mean_gpu_ms") is not None:      gmf.append(cs["fr_mean_gpu_ms"])
            if cs.get("honest_mean_gpu_ms") is not None:  gmh.append(cs["honest_mean_gpu_ms"])
            if cs.get("fr_mean_samples") is not None:     smf.append(cs["fr_mean_samples"])
            if cs.get("honest_mean_samples") is not None: smh.append(cs["honest_mean_samples"])

        fr_mean.append(np.mean(fr_vals) if fr_vals else np.nan)
        fr_std.append(np.std(fr_vals) if fr_vals else 0.0)
        ho_mean.append(np.mean(ho_vals) if ho_vals else np.nan)
        fr_indiv[li] = [np.mean(v) for v in per_fr.values()]
        g_ms_fr.append(np.mean(gmf) if gmf else np.nan); g_ms_ho.append(np.mean(gmh) if gmh else np.nan)
        s_fr.append(np.mean(smf) if smf else np.nan);   s_ho.append(np.mean(smh) if smh else np.nan)

    # FAIR thresholds, calibrated on the HONEST clients' CONVERGED BERs (last N rounds):
    #   eta_tight = mu+3sigma over per-ROUND-MEAN honest BER  -> what the live detector approximates
    #   eta_loose = mu+3sigma over PER-CLIENT honest BER      -> the looser alternative
    # (NOT the swingy cumulative wm_eta_round, which over-states evasion.)
    eta_tight = th.frozen_eta(runs); eta_loose = None  # canonical only; loose DROPPED

    x = np.arange(len(order))
    xlabels = [_label_for_level(lv) for lv in order]

    fig, (axB, axG, axS) = ps.stacked_panels(3, figsize=(11, 10),
                                             height_ratios=[2, 1, 1])

    # --- panel 1: BER ---
    axB.plot(x, fr_mean, color=ps.C_FR, lw=2.6, marker="o", label="free-rider mean BER")
    axB.fill_between(x, np.array(fr_mean) - np.array(fr_std),
                     np.array(fr_mean) + np.array(fr_std), color=ps.C_FR, alpha=0.15)
    for li in fr_indiv:                       # each free-rider distinguishable
        for v in fr_indiv[li]:
            axB.scatter(li, v, s=26, color=ps.C_FR, alpha=0.5, marker="x")
    axB.plot(x, ho_mean, color=ps.C_HONEST, lw=2.6, marker="s", label="honest mean BER")
    if eta_tight is not None:
        axB.axhline(eta_tight, color=ps.OKABE["black"], ls="--", lw=2,
                    label=f"fair η (frozen, all clients) = {eta_tight:.3f}")
    if eta_loose is not None:
        axB.axhline(eta_loose, color=ps.OKABE.get("grey", "888888"), ls=":", lw=1.8,
                    label=f"loose η (per-client) = {eta_loose:.3f}")
    axB.set_ylabel("bit-error-rate\n(converged)")
    axB.set_title(a.title or "per-free-rider & per-honest BER vs training-data amount")
    axB.legend(loc="upper right", fontsize=8)

    # --- panel 2: GPU-ms effort (scope-sensitive) ---
    axG.plot(x, g_ms_fr, color=ps.C_FR, lw=2.4, marker="o", label="free-rider GPU-ms (total)")
    axG.plot(x, g_ms_ho, color=ps.C_HONEST, lw=2.4, marker="s", label="honest GPU-ms (total)")
    axG.set_ylabel("total GPU-ms\n(whole run)")
    axG.legend(loc="upper right", fontsize=8)

    # --- panel 3: samples effort (scope-blind) ---
    axS.plot(x, s_fr, color=ps.C_FR, lw=2.4, marker="o", label="free-rider image-passes (total)")
    axS.plot(x, s_ho, color=ps.C_HONEST, lw=2.4, marker="s", label="honest image-passes (total)")
    axS.set_ylabel("total image-passes\n(whole run)")
    axS.set_xlabel("training data per round (triggers-only → +N/common-class → full shard)")
    axS.set_xticks(x); axS.set_xticklabels(xlabels)
    axS.legend(loc="upper left", fontsize=8)

    ps.finish(fig, a.out + ".png")
    print("levels:", xlabels)


# =====================================================================
def class_difficulty(a):
    """CONFIRM the assumption "some trigger-class IDs are harder" using the
    watermark-INDEPENDENT per-class test accuracy + loss (result['per_class']),
    correlated against per-trigger-class watermark BER. If hard-to-embed classes
    (high BER) are also the low-accuracy / high-loss classes, the boundary-fuzziness
    explanation holds."""
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return

    # per-class test acc/loss (final model), averaged over seeds
    acc_by, loss_by = defaultdict(list), defaultdict(list)
    have_pc = False
    for r in runs:
        pc = r.get("per_class")
        if not pc or not pc.get("by_class"):
            continue
        have_pc = True
        for c, d in pc["by_class"].items():
            acc_by[int(c)].append(d["acc"]); loss_by[int(c)].append(d["loss"])

    # per-trigger-class converged watermark BER (only classes some client holds)
    ber_by = defaultdict(list)
    for r in runs:
        n = len(r.get("history", []))
        for i, h in enumerate(r["history"]):
            if i < n - TAIL:
                continue
            for p in (h.get("wm_per_client") or []):
                if not p.get("is_free_rider"):
                    ber_by[int(p["trigger_class"])].append(p["ber"])

    trig = sorted(ber_by)                         # the trigger classes clients hold
    ber = np.array([np.mean(ber_by[c]) for c in trig])
    if not have_pc:
        print("  NOTE: result['per_class'] absent -> re-run with the updated "
              "run_experiment.py to log per-class acc/loss. Plotting BER only.")

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    order = np.argsort(ber)
    ts = [trig[i] for i in order]

    # (a) per-trigger-class BER
    axA = ax[0, 0]
    axA.bar(range(len(ts)), [ber_by and np.mean(ber_by[c]) for c in ts], color=C_FR, alpha=0.85)
    axA.set_xticks(range(len(ts))); axA.set_xticklabels([f"cls {c}" for c in ts])
    axA.set_ylabel(f"watermark BER (last {TAIL} rounds)")
    axA.set_xlabel("trigger class id (sorted easy -> hard)")
    axA.set_title("(a) Watermark difficulty per trigger class id")

    # (b) per-class test accuracy
    axB = ax[0, 1]
    if have_pc:
        accs = [np.mean(acc_by[c]) if acc_by.get(c) else np.nan for c in ts]
        axB.bar(range(len(ts)), accs, color=C_HONEST, alpha=0.85)
        axB.set_xticks(range(len(ts))); axB.set_xticklabels([f"cls {c}" for c in ts])
        axB.set_ylabel("per-class TEST accuracy (%)")
        axB.set_xlabel("trigger class id (same order as (a))")
        axB.set_title("(b) Classification accuracy per class id\n(low here + high BER in (a) = fuzzy boundary)")
    else:
        axB.axis("off"); axB.text(0.5, 0.5, "no per_class in result.json", ha="center", color=GREY)

    # (c) BER vs per-class ERROR (100-acc), with correlation
    axC = ax[1, 0]
    if have_pc:
        err = np.array([100 - np.mean(acc_by[c]) if acc_by.get(c) else np.nan for c in trig])
        good = ~np.isnan(err)
        axC.scatter(err[good], ber[good], s=80, color=OK["purple"], edgecolor=BLACK, zorder=3)
        for c, x_, y_ in zip(trig, err, ber):
            axC.annotate(f"cls {c}", (x_, y_), fontsize=8, textcoords="offset points", xytext=(5, 3))
        if good.sum() >= 2:
            rho = float(np.corrcoef(err[good], ber[good])[0, 1])
            axC.set_title(f"(c) BER vs classification error  (Pearson r = {rho:.2f})")
        axC.set_xlabel("per-class test error = 100 - acc (%)")
        axC.set_ylabel("watermark BER")
    else:
        axC.axis("off")

    # (d) BER vs per-class LOSS
    axD = ax[1, 1]
    if have_pc:
        lo = np.array([np.mean(loss_by[c]) if loss_by.get(c) else np.nan for c in trig])
        good = ~np.isnan(lo)
        axD.scatter(lo[good], ber[good], s=80, color=OK["orange"], edgecolor=BLACK, zorder=3)
        for c, x_, y_ in zip(trig, lo, ber):
            axD.annotate(f"cls {c}", (x_, y_), fontsize=8, textcoords="offset points", xytext=(5, 3))
        if good.sum() >= 2:
            rho = float(np.corrcoef(lo[good], ber[good])[0, 1])
            axD.set_title(f"(d) BER vs classification loss  (Pearson r = {rho:.2f})")
        axD.set_xlabel("per-class test cross-entropy loss")
        axD.set_ylabel("watermark BER")
    else:
        axD.axis("off")

    fig.suptitle(f"Assumption check: harder class ids - {a.family or 'all runs'}",
                 fontsize=13, y=1.01)
    finish(fig, os.path.join(a.out, f"class_difficulty_{a.family or 'all'}.png"))
    print("  per-trigger-class BER (easy->hard):", [(c, round(np.mean(ber_by[c]), 3)) for c in ts])
    if have_pc:
        print("  per-class acc:", [(c, round(np.mean(acc_by[c]), 1)) for c in ts])


def class_acc(a):
    """PER-CLIENT trigger-class accuracy check (all-honest run).

    One panel PER CLIENT. Each panel shows three test-accuracy bars for the SINGLE
    shared global model, read off `result['per_class']['by_class']` (averaged over
    seeds):
      * trigger class   -- this client's own assigned trigger class
      * non-trigger     -- mean over all the OTHER classes
      * global          -- overall test accuracy (per_class.overall_acc)
    The cid->trigger-class map comes from the last round's wm_per_client rows.

    Why: it isolates trigger-class DIFFICULTY from the watermark. If a client's
    watermark BER is high, this plot says whether that is because its trigger class
    is intrinsically hard to classify (low trigger-class bar, well below global) or
    not (trigger-class bar ~ global). A hard draw shows up as a short orange bar.
    Runs on an all-honest family so no free-rider effects are in play.
    """
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return

    # per-class test acc (final model), averaged over seeds
    acc_by = defaultdict(list)
    overall = []
    for r in runs:
        pc = r.get("per_class")
        if pc and pc.get("by_class"):
            for c, d in pc["by_class"].items():
                acc_by[int(c)].append(d["acc"])
        if pc and pc.get("overall_acc") is not None:
            overall.append(float(pc["overall_acc"]))
    if not acc_by:
        print("  NOTE: result['per_class'] absent -> re-run run_experiment.py "
              "(it logs per_class.by_class). Cannot draw class_acc.")
        return
    acc_mean = {c: float(np.mean(v)) for c, v in acc_by.items()}
    global_acc = float(np.mean(overall)) if overall else float(np.mean(list(acc_mean.values())))

    # cid -> trigger class from the last round's rows (any run; they agree)
    cid_tc = {}
    for r in runs:
        h = r.get("history") or []
        if not h:
            continue
        for p in (h[-1].get("wm_per_client") or []):
            if p.get("trigger_class") is not None:
                cid_tc[int(p["cid"])] = int(p["trigger_class"])
    if not cid_tc:
        print("  NOTE: no wm_per_client rows -> cannot map clients to trigger classes.")
        return
    cids = sorted(cid_tc)

    ncol = min(5, len(cids))
    nrow = int(np.ceil(len(cids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 3.0 * nrow),
                             squeeze=False, sharey=True)
    rows_md = []
    for i, cid in enumerate(cids):
        ax = axes[i // ncol][i % ncol]
        tc = cid_tc[cid]
        trig = acc_mean.get(tc, float("nan"))
        others = [acc_mean[c] for c in acc_mean if c != tc]
        nontrig = float(np.mean(others)) if others else float("nan")
        vals = [trig, nontrig, global_acc]
        colors = [C_FR, C_HONEST, GREY]
        bars = ax.bar([0, 1, 2], vals, color=colors, width=0.7, edgecolor="white")
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8)
        ax.axhline(global_acc, color=GREY, ls="--", lw=1.0, zorder=0)
        gap = trig - global_acc
        flag = "  (HARD draw)" if (not np.isnan(gap) and gap <= -10) else ""
        ax.set_title(f"cid {cid} · trig cls {tc}{flag}", fontsize=9.5,
                     color=(C_BAD if flag else "black"))
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["trig", "non-trig", "global"], fontsize=8)
        ax.set_ylim(0, 100)
        if i % ncol == 0:
            ax.set_ylabel("test acc (%)")
        rows_md.append((cid, tc, trig, nontrig, global_acc, gap))
    # blank any unused panels
    for j in range(len(cids), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fam_ = a.family or "all"
    fig.suptitle(
        f"Per-client trigger-class accuracy (all-honest) · {fam_}\n"
        f"orange = client's trigger class · blue = mean of other classes · grey = global "
        f"({global_acc:.1f}%). A short orange bar = a hard trigger-class draw, not a watermark effect.",
        fontsize=11, y=1.005)
    out = a.out if str(a.out).endswith(".png") else str(a.out) + ".png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    finish(fig, out)

    md = out[:-4] + ".md"
    L = [f"# Per-client trigger-class accuracy — {fam_}", "",
         f"Global test acc: **{global_acc:.2f}%**. A trigger-class bar far below global "
         "means that client sits on an intrinsically hard class (its watermark BER floor is "
         "a class-difficulty artifact, not evidence about the mark).", "",
         "| cid | trigger class | trig-class acc | non-trigger acc | global | trig − global |",
         "|---|---|---|---|---|---|"]
    for cid, tc, trig, nontrig, g, gap in rows_md:
        hard = " ⚠️ hard" if (not np.isnan(gap) and gap <= -10) else ""
        L.append(f"| {cid} | {tc} | {trig:.1f}% | {nontrig:.1f}% | {g:.1f}% | {gap:+.1f}{hard} |")
    open(md, "w").write("\n".join(L))
    print("wrote", md)


def sanity(a):
    """TEXT report (no figure) that flags suspicious/degenerate runs BEFORE you
    trust any plot -- the failure modes you hit before: flat or zero BER, an FR that
    never taps, a non-frozen eta, or missing loss logging. Prints WARN/OK per run and
    points you at the loss plot when a BER looks fishy."""
    runs = pick(load(a.inp), a.family)
    if not runs:
        print("no runs for", a.family); return
    for r in runs:
        tag = (r.get("manifest", {}) or {}).get("family", "?")
        seed = r.get("seed")
        hist = r.get("history", [])
        tail = hist[-TAIL:]
        ho = [p["ber"] for h in tail for p in (h.get("wm_per_client") or []) if not p.get("is_free_rider")]
        fr = [p["ber"] for h in tail for p in (h.get("wm_per_client") or []) if p.get("is_free_rider")]
        etas = [h.get("wm_eta_round") for h in hist if h.get("wm_eta_round") is not None]
        msgs = []
        # honest degeneracy
        if ho and max(ho) == 0:
            msgs.append("WARN honest BER == 0 everywhere (watermark trivial or extraction bug?)")
        if ho and float(np.std(ho)) == 0 and max(ho) > 0:
            msgs.append(f"WARN honest BER perfectly flat at {ho[0]} (no per-position spread?)")
        # free-rider suspicion
        if fr:
            if float(np.std(fr)) < 1e-9:
                msgs.append(f"WARN free-rider BER perfectly FLAT at {fr[0]:.3f} "
                            "(is it actually tapping/coasting? -> inspect loss)")
            if max(fr) == 0:
                msgs.append("WARN free-rider BER == 0 everywhere (no embedding cost? check taps)")
            if min(fr) >= 0.45:
                msgs.append("WARN free-rider BER ~0.5 everywhere (never embeds -> not a real submarine)")
        # frozen-eta check
        if etas and (max(etas) - min(etas)) > 1e-6:
            msgs.append(f"WARN wm_eta_round NOT flat ({min(etas):.3f}..{max(etas):.3f}) "
                        "-> this run did NOT use the frozen constant (WM_ETA_FIXED)")
        # taps / coasts from the FR trace
        traces = [c.get("trace", []) for c in ((r.get("compute", {}) or {}).get("per_client", {}) or {}).values()
                  if c.get("is_free_rider")]
        for t in traces:
            acts = [x.get("action") for x in t]
            n_tap = acts.count("tap"); n_coast = acts.count("coast")
            if t and n_tap == 0 and n_coast > 0:
                msgs.append(f"NOTE FR coasted {n_coast}x, tapped 0x (stay_min coasting only)")
            if t and n_tap == 0 and n_coast == 0 and "honest_clone" not in acts:
                msgs.append("WARN FR trace has no tap/coast actions (did free-riding start?)")
        # loss logging present?
        has_ws = any(c.get("wm_stats") for c in ((r.get("compute", {}) or {}).get("per_client", {}) or {}).values())
        if not has_ws:
            msgs.append("NOTE no client wm_stats (loss/acc) logged -> class_dynamics loss panels blank")
        if r.get("per_class") is None:
            msgs.append("NOTE no per_class (acc/loss) -> class_difficulty needs the updated run_experiment")
        head = f"[{tag} seed={seed}] honest_meanBER={np.mean(ho):.3f} " + \
               (f"fr_meanBER={np.mean(fr):.3f} " if fr else "(all-honest) ") + \
               (f"eta={etas[-1]:.3f}" if etas else "eta=?")
        print(head)
        for m in msgs:
            print("   ", m)
        if not msgs:
            print("    OK")
        if any("free-rider BER" in m for m in msgs):
            print(f"    -> inspect: python plots.py class_dynamics --in '<glob>' --family {tag}")


def eta_stability(a):
    runs = [r for r in pick(load(a.inp), a.family) if th.is_honest_run(r)]
    if not runs:
        print("no honest runs for", a.family); return
    tail = TAIL

    series, etas, seeds = [], [], []
    for r in runs:
        hist = r.get("history", [])
        mr = []
        for h in hist:
            vals = [p["ber"] for p in (h.get("wm_per_client") or []) if not p.get("is_free_rider")]
            mr.append(np.mean(vals) if vals else np.nan)
        e = th.frozen_eta([r], tail=tail)
        series.append(np.array(mr)); seeds.append(r.get("seed"))
        if e is not None:
            etas.append(e)

    if not etas:
        print("no eta values computed; skipping")
        return

    eta_final = float(np.mean(etas)); eta_std = float(np.std(etas))
    maxlen = max(len(m) for m in series)
    M = np.full((len(series), maxlen), np.nan)
    for i, m in enumerate(series):
        M[i, :len(m)] = m
    mean_mr = np.nanmean(M, axis=0); std_mr = np.nanstd(M, axis=0)
    x = np.arange(1, maxlen + 1)

    # ---- Figure 1: BER over rounds ----
    fig1, axA = plt.subplots(figsize=(12, 5.5))
    for m in series:
        axA.plot(range(1, len(m) + 1), m, lw=0.8, alpha=0.4, color=C_HONEST)
    axA.plot(x, mean_mr, color=BLACK, lw=2.6, label="mean over seeds")
    axA.fill_between(x, mean_mr - std_mr, mean_mr + std_mr, color=GREY, alpha=0.25,
                     label="+/- std across seeds")
    axA.axvspan(maxlen - tail + 1, maxlen, color=OK["orange"], alpha=0.08,
                label=f"converged tail ({tail} rounds)")
    for e in etas:
        axA.axhline(e, color=C_HONEST, lw=0.7, alpha=0.35)
    axA.axhline(eta_final, color=C_GOOD, lw=2.6, label=f"eta final = {eta_final:.3f}")
    axA.axhspan(eta_final - eta_std, eta_final + eta_std, color=C_GOOD, alpha=0.13,
                label=f"eta +/- std = {eta_std:.3f}")
    axA.set_xlabel("communication round")
    axA.set_ylabel("honest mean-over-clients BER  (m_r)")
    axA.set_title(f"Honest BER over rounds – {a.family or 'honest'}  "
                  f"(faint lines = per-seed, blue h-lines = per-seed etas)")
    axA.legend(fontsize=8, loc="upper right")
    finish(fig1, os.path.join(a.out, f"eta_stability_ber_{a.family or 'honest'}.png"))

    # ---- Figure 2: Per‑seed eta spread ----
    fig2, axB = plt.subplots(figsize=(6, 5))
    rng = np.random.RandomState(0)
    axB.scatter(rng.rand(len(etas)) * 0.3 - 0.15, etas, s=55, color=C_HONEST,
                edgecolor=BLACK, lw=0.5, zorder=3)
    for e, sd in zip(etas, seeds):
        axB.annotate(str(sd), (0.16, e), fontsize=7, va="center", color=GREY)
    axB.axhline(eta_final, color=C_GOOD, lw=2.4, label=f"final = {eta_final:.3f}")
    axB.axhspan(eta_final - eta_std, eta_final + eta_std, color=C_GOOD, alpha=0.15,
                label=f"+/- std = {eta_std:.3f}")
    axB.set_xlim(-0.4, 0.5); axB.set_xticks([])
    axB.set_ylabel("per-seed eta")
    rng_lo, rng_hi = min(etas), max(etas)
    axB.set_title(f"Per‑seed eta spread\n{rng_lo:.3f}..{rng_hi:.3f}  "
                  f"({rng_hi/max(rng_lo,1e-9):.1f}x)")
    axB.legend(fontsize=8, loc="upper right")
    finish(fig2, os.path.join(a.out, f"eta_stability_eta_{a.family or 'honest'}.png"))

    print(f"  eta_final={eta_final:.4f} +/- {eta_std:.4f}  "
          f"per-seed range {rng_lo:.4f}..{rng_hi:.4f} ({rng_hi/max(rng_lo,1e-9):.1f}x), "
          f"n_seeds={len(etas)}")
    print(f"  saved BER plot and eta spread plot to {a.out}")

def _honest_runs(a):
    return [r for r in load(a.inp) if th.is_honest_run(r)
            and (a.family is None or fam(r) == a.family)]


def honest_lines(a):
    """Honest client BER over rounds, ONE line per trigger class (merged from
    honest_class_lines.py). The right-hand end of each line == that class's
    converged floor (the value the timeline/overlay collapses to)."""
    tail = getattr(a, "tail", None) or TAIL
    runs = _honest_runs(a)
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
    if tail and tail > 0 and max_round > tail:
        ax.axvspan(max_round - tail + 0.5, max_round + 0.5, color="#DDDDDD",
                   alpha=0.35, lw=0, label=f"converged tail (last {tail})")
    floors = {}
    for i, c in enumerate(classes):
        col = cmap(i % cmap.N)
        mean = np.array([np.mean(by_cr[c][rd]) if by_cr[c].get(rd) else np.nan for rd in rounds])
        std = np.array([np.std(by_cr[c][rd]) if by_cr[c].get(rd) else np.nan for rd in rounds])
        if getattr(a, "per_seed", False):
            for si in per_seed[c]:
                ys = [per_seed[c][si].get(rd, np.nan) for rd in rounds]
                ax.plot(rounds, ys, color=col, lw=0.6, alpha=0.20)
        else:
            ax.fill_between(rounds, mean - std, mean + std, color=col, alpha=0.12, lw=0)
        tailvals = [np.mean(by_cr[c][rd]) for rd in rounds[-tail:] if by_cr[c].get(rd)]
        floor = float(np.mean(tailvals)) if tailvals else float("nan")
        floors[c] = floor
        ax.plot(rounds, mean, color=col, lw=2.2, label=f"cls {c}  (floor {floor:.3f})")
        if floor == floor:
            ax.annotate(f"{floor:.2f}", xy=(rounds[-1], mean[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=col)
    eta = getattr(a, "eta", None)
    if eta is not None:
        ax.axhline(eta, color="black", ls="--", lw=2, label=f"calibrated η = {eta:.3f}")
    ax.set_xlabel("communication round")
    ax.set_ylabel("honest bit-error-rate (lower = mark embeds)")
    ttl = f"Honest BER per trigger class  ·  {a.family or 'honest'}  ·  {len(runs)} seeds"
    if only:
        ttl += f"  ·  classes {sorted(only)}"
    ax.set_title(ttl)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    out = a.out if str(a.out).endswith(".png") else a.out + ".png"
    finish(fig, out)
    print("converged floors:", {c: round(floors[c], 4) for c in classes})


def class_probe(a):
    """TEXT: per-trigger-class BER vs candidate predictors + correlations (merged
    from class_difficulty_probe.py). Confirms difficulty is a softmax-SHAPE effect
    (entropy/dominance/pmax), not a class-accuracy effect."""
    tail = getattr(a, "tail", None) or TAIL
    runs = _honest_runs(a)
    if not runs:
        print("no honest runs matched (check --in / --family)."); return
    print(f"class_probe: {len(runs)} honest run(s)"
          + (f" [{a.family}]" if a.family else "") + f"; tail={tail}")

    ber = defaultdict(list); pmax = defaultdict(list); ent = defaultdict(list)
    dom = defaultdict(list); tacc = defaultdict(list)
    test_acc = defaultdict(list); test_loss = defaultdict(list)
    for r in runs:
        for h in r.get("history", [])[-tail:]:
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider"):
                    continue
                c = int(p["trigger_class"]); ber[c].append(p["ber"])
                for src, key in ((pmax, "pmax"), (ent, "entropy"),
                                 (dom, "dominance"), (tacc, "trig_acc")):
                    v = p.get(key)
                    if v is not None:
                        src[c].append(v)
        pc = (r.get("per_class") or {}).get("by_class") or {}
        for c, d in pc.items():
            c = int(c)
            if d.get("acc") is not None:
                test_acc[c].append(d["acc"])
            if d.get("loss") is not None:
                test_loss[c].append(d["loss"])

    def m(dct, c):
        return float(np.mean(dct[c])) if dct.get(c) else float("nan")

    classes = sorted(ber)
    rows = []
    for c in classes:
        acc = m(test_acc, c)
        rows.append(dict(cls=c, n=len(ber[c]), ber=m(ber, c), test_acc=acc,
                         test_error=(100 - acc if acc == acc else float("nan")),
                         test_loss=m(test_loss, c), trig_acc=m(tacc, c),
                         pmax=m(pmax, c), entropy=m(ent, c), dominance=m(dom, c)))
    rows.sort(key=lambda d: (float("inf") if d["ber"] != d["ber"] else d["ber"]))
    cols = ["cls", "n", "ber", "test_acc", "test_error", "test_loss",
            "trig_acc", "pmax", "entropy", "dominance"]
    w = {c: max(len(c), 8) for c in cols}
    print("PER-CLASS (easy -> hard by BER):")
    print("  " + "  ".join(f"{c:>{w[c]}}" for c in cols))
    for d in rows:
        cells = []
        for c in cols:
            v = d[c]
            cells.append(f"{v:>{w[c]}d}" if isinstance(v, int)
                         else (f"{'--':>{w[c]}}" if v != v else f"{v:>{w[c]}.4f}"))
        print("  " + "  ".join(cells))

    def pearson(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    y = [d["ber"] for d in rows]
    print(f"\nCORRELATION of per-class BER vs predictor (over {len(classes)} classes):")
    for p in ["test_acc", "test_error", "test_loss", "trig_acc", "pmax", "entropy", "dominance"]:
        x = [d[p] for d in rows]
        mask = [xi == xi and yi == yi for xi, yi in zip(x, y)]
        xs = [xi for xi, ok in zip(x, mask) if ok]
        ys = [yi for yi, ok in zip(y, mask) if ok]
        r = pearson(xs, ys)
        print(f"  {p:>10}  pearson_r={'  --  ' if r != r else f'{r:+.3f}'}  (n={len(xs)})")

    if getattr(a, "csv", None):
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for d in rows:
                fh.write(",".join(("" if d[c] != d[c] else str(d[c])) for c in cols) + "\n")
        print(f"wrote {a.csv}")


def separability_plot(a):
    """Non-separability figure: honest vs free-rider converged-BER histograms with
    the coded and best-possible (Youden) η lines, plus the FPR/recall of the whole
    threshold regime. Reads honest from --family, free-riders from --attack_family.
    Uses separability.py for the numbers (single source of truth)."""
    if sep is None:
        print("separability module unavailable"); return
    tail = getattr(a, "tail", None) or TAIL
    honest_runs = sep.select(load(a.inp), a.family, honest=True)
    attack_runs = sep.select(load(a.inp), a.attack_family, honest=False)
    if not honest_runs or not attack_runs:
        print(f"need both honest [{a.family}] and attack [{a.attack_family}] runs "
              f"(got {len(honest_runs)}/{len(attack_runs)})"); return
    H = [b for _c, b in sep.per_client_bers(honest_runs, tail, free_rider=False)]
    F = [b for _c, b in sep.per_client_bers(attack_runs, tail, free_rider=True)]
    res = sep.summarise(H, F, honest_runs, tail, label=a.attack_family or "attack")

    fig, (axh, axb) = stacked_panels(2, figsize=(11, 8), height_ratios=[1.1, 1])
    lo, hi = 0.0, max(max(H, default=0.5), max(F, default=0.5), 0.5)
    bins = np.linspace(lo, hi, 30)
    axh.hist(H, bins=bins, color=C_HONEST, alpha=0.55, label=f"honest (n={len(H)})", density=True)
    axh.hist(F, bins=bins, color=C_FR, alpha=0.55, label=f"free-rider (n={len(F)})", density=True)
    coded = res["rules"].get("coded (mu+3s round-mean)")
    best = res["rules"].get("Youden-optimal (best)")
    if coded:
        axh.axvline(coded["eta"], color=C_BAD, ls="--", lw=2,
                    label=f"coded η={coded['eta']:.3f} (FPR {coded['fpr']}, recall {coded['recall']})")
    if best:
        axh.axvline(best["eta"], color=OK["black"], ls=":", lw=2,
                    label=f"best η={best['eta']:.3f} (bal_acc {best['bal_acc']})")
    axh.set_xlabel("converged bit-error-rate")
    axh.set_ylabel("density")
    axh.set_title(f"Non-separability — honest vs free-rider BER  ·  {a.attack_family}\n"
                  f"overlap OVL={res['overlap_coefficient']}   "
                  f"best-possible balanced error={res['best_threshold_balanced_error']} "
                  f"(0.5 = no η helps)")
    axh.legend(fontsize=8, loc="upper right")

    names, fprs, recs = [], [], []
    for name, d in res["rules"].items():
        if d is None:
            continue
        names.append(name.replace(" (", "\n(")); fprs.append(d["fpr"] or 0); recs.append(d["recall"] or 0)
    x = np.arange(len(names)); ww = 0.4
    axb.bar(x - ww / 2, fprs, ww, color=C_BAD, label="honest FPR (lower=better)")
    axb.bar(x + ww / 2, recs, ww, color=C_GOOD, label="free-rider recall (higher=better)")
    axb.set_xticks(x); axb.set_xticklabels(names, fontsize=7, rotation=0)
    axb.set_ylabel("rate"); axb.set_ylim(0, 1.0)
    axb.set_title("Threshold regime — every rule trades FPR against recall")
    axb.legend(fontsize=8, loc="upper right")
    out = a.out if str(a.out).endswith(".png") else a.out + ".png"
    finish(fig, out)
    print(f"OVL={res['overlap_coefficient']}  best_bal_err={res['best_threshold_balanced_error']}")


def sweep_plot(a):
    """+N sweep: how much data must a free-rider spend before its mark passes?

    panel 1: free-rider BER over rounds, one line per N (post-warmup)
    panel 2: converged free-rider BER vs N, with the honest floor and eta drawn
    panel 3: effort actually spent (samples/round) vs N -- the x-axis that matters,
             since N is a knob but samples is the cost.
    N = -1 means FULL shard (honest-equivalent); it is drawn as the right-hand anchor.
    """
    tail = getattr(a, "tail", None) or TAIL
    runs = load(a.inp)
    # collect sweep runs. The +N level normally lives in manifest.sweep_var/
    # sweep_level, but older/renamed runs encode it only in the family-name suffix
    # `_n<val>` (e.g. D1_reduced_c100_c36_n5, _n-1 = full shard). Recover it either
    # way so a completed batch never needs re-running just to plot.
    import re as _re

    def _sweep_level(man):
        # 1) explicit manifest field (paper-faithful path)
        if man.get("sweep_var") == "common_per_class":
            try:
                return int(man.get("sweep_level"))
            except (TypeError, ValueError):
                pass
        # 2) fall back to the `_n<val>` suffix on the family name
        mobj = _re.search(r"_n(-?\d+)$", man.get("family") or "")
        return int(mobj.group(1)) if mobj else None

    # which families to include: explicit --families list, else --family as a prefix,
    # else any family that carries an `_n<val>` sweep suffix.
    want = set(a.families) if getattr(a, "families", None) else None
    prefix = a.family if (want is None and a.family) else None

    byN = defaultdict(list)
    for r in runs:
        man = r.get("manifest", {}) or {}
        family = man.get("family") or ""
        if want is not None and family not in want:
            continue
        if prefix is not None and not family.startswith(prefix):
            continue
        n = _sweep_level(man)
        if n is None:
            continue
        byN[n].append(r)
    if not byN:
        print("no +N sweep runs found: need either manifest.sweep_var=common_per_class "
              "or a family name ending in `_n<val>` (e.g. D1_reduced_c100_c36_n5). "
              "Pass the members via --families or a common prefix via --family."); return

    Ns = sorted(byN)                      # -1 sorts first; move it to the end as the anchor
    order = [n for n in Ns if n >= 0] + [n for n in Ns if n < 0]

    def fr_ber_series(rs):
        """round -> mean FR BER across seeds."""
        acc = defaultdict(list)
        for r in rs:
            for h in r.get("history", []):
                rd = h.get("round")
                vals = [p["ber"] for p in (h.get("wm_per_client") or []) if p.get("is_free_rider")]
                if rd and vals:
                    acc[rd].append(float(np.mean(vals)))
        return {rd: float(np.mean(v)) for rd, v in acc.items()}

    def conv_fr_ber(rs):
        vals = []
        for r in rs:
            for h in r.get("history", [])[-tail:]:
                vals += [p["ber"] for p in (h.get("wm_per_client") or []) if p.get("is_free_rider")]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)

    def conv_hon_ber(rs):
        vals = []
        for r in rs:
            for h in r.get("history", [])[-tail:]:
                vals += [p["ber"] for p in (h.get("wm_per_client") or []) if not p.get("is_free_rider")]
        return float(np.mean(vals)) if vals else float("nan")

    def fr_samples(rs):
        """mean samples/round spent by the free-riders (device-independent effort)."""
        vals = []
        for r in rs:
            cs = (r.get("compute", {}) or {}).get("summary", {}) or {}
            v = cs.get("fr_mean_samples")
            if v: vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    import matplotlib.gridspec as _gridspec
    fig = plt.figure(figsize=(11, 10))
    # _gs = _gridspec.GridSpec(3, 1, height_ratios=[1.2, 1, 1], hspace=0.55)
    _gs = _gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.55)
    ax1 = fig.add_subplot(_gs[0])
    ax2 = fig.add_subplot(_gs[1])   # NOT sharex -- categorical axis
    # ax3 = fig.add_subplot(_gs[2])   # NOT sharex -- categorical axis
    cmap = plt.get_cmap("viridis")
    lab = lambda n: ("full shard" if n < 0 else ("triggers only" if n == 0 else f"+{n}/class"))

    for i, n in enumerate(order):
        ser = fr_ber_series(byN[n])
        if not ser: continue
        rds = sorted(ser)
        col = OK["black"] if n < 0 else cmap(i / max(len(order) - 1, 1))
        ax1.plot(rds, [ser[r] for r in rds], lw=2,
                 ls="--" if n < 0 else "-", color=col,
                 label=f"{lab(n)}  (n={len(byN[n])} seeds)")
    # Two DEPLOYABLE thresholds instead of the (misleading) honest floor. The honest floor
    # here averaged the EASY non-free-rider clients, which understates honest BER at the hard
    # class the FR sits on. The tight/loose η are the same two lines every timeline uses, so
    # the reader can see the FR sits BETWEEN them: above η_tight (which also flags many honest
    # clients) but below η_loose (the low-false-alarm line a server would actually deploy).
    eta_t = a.eta_tight if getattr(a, "eta_tight", None) is not None else \
            (a.eta if getattr(a, "eta", None) is not None else ETA_TIGHT_DEFAULT)
    eta_l = a.eta_loose if getattr(a, "eta_loose", None) is not None else ETA_LOOSE_DEFAULT
    ax1.axhline(eta_t, color=OK["black"], ls="--", lw=2, label=f"η tight = {eta_t:.3f}")
    ax1.axhline(eta_l, color="#3B6FB5", ls=(0, (5, 2)), lw=2, label=f"η loose = {eta_l:.3f}")
    ax1.set_ylabel("free-rider BER")
    _cfg0 = (next(iter(byN.values()))[0].get("config") or {}) if byN else {}
    _bs = _cfg0.get("batch_size"); _nc = _cfg0.get("num_clients")
    _sub = "".join(x for x in [f"  ·  batch={_bs}" if _bs else "",
                                f", {_nc} clients" if _nc else ""] )
    ax1.set_title("Free-rider BER over rounds, per data budget" + _sub)
    ax1.legend(fontsize=8, ncol=2, loc="upper right")

    xs = list(range(len(order)))
    mus = [conv_fr_ber(byN[n])[0] for n in order]
    sds = [conv_fr_ber(byN[n])[1] for n in order]
    ax2.errorbar(xs, mus, yerr=sds, marker="o", lw=2, color=C_FR, capsize=3,
                 label="free-rider (converged)")
    ax2.axhline(eta_t, color=OK["black"], ls="--", lw=2, label=f"η tight = {eta_t:.3f}")
    ax2.axhline(eta_l, color="#3B6FB5", ls=(0, (5, 2)), lw=2, label=f"η loose = {eta_l:.3f}")
    ax2.set_xlim(-0.5, len(order) - 0.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([lab(n) for n in order], fontsize=9, rotation=30, ha="right")
    ax2.set_ylabel("converged BER")
    ax2.set_title("Converged free-rider BER vs data budget")
    ax2.legend(fontsize=8)

    # NOTE: skip the third panel graph
    # sm = [fr_samples(byN[n]) for n in order]
    # ax3.bar(xs, sm, color=OK["blue"], width=0.7)
    # ax3.set_xlim(-0.5, len(order) - 0.5)
    # ax3.set_xticks(xs)
    # ax3.set_xticklabels([lab(n) for n in order], fontsize=9, rotation=30, ha="right")
    # ax3.set_ylabel("free-rider samples / round")
    # ax3.set_title("Actual effort spent (device-independent)")
    out = a.out if str(a.out).endswith(".png") else a.out + ".png"
    finish(fig, out)
    for n, m in zip(order, mus):
        print(f"  {lab(n):>16}: converged FR BER = {m:.4f}")


# ===========================================================================
#  operating_point -- the "no threshold works" plot.
#  For a single GLOBAL honest-calibrated eta at each FPR budget, what recall does
#  each attack family get? 
# ===========================================================================
def _pc_bers(runs, free_rider, tail, trigger_class=None):
    """flat per-client BER list over the tail rounds (honest or FR)."""
    out = []
    for r in runs:
        h = r.get("history", []) or []
        for rec in (h[-tail:] if tail else h):
            for p in (rec.get("wm_per_client") or []):
                if bool(p.get("is_free_rider")) != bool(free_rider):
                    continue
                if trigger_class is not None and int(p.get("trigger_class", -1)) != int(trigger_class):
                    continue
                if p.get("ber") is not None:
                    out.append(float(p["ber"]))
    return out


def _fr_class(runs):
    """the trigger class the free-rider(s) sit on in an attack family."""
    for r in runs:
        for rec in reversed(r.get("history", []) or []):
            for p in (rec.get("wm_per_client") or []):
                if p.get("is_free_rider") and p.get("trigger_class") is not None:
                    return int(p["trigger_class"])
    return None


def _eta_for_fpr(H, budget):
    """smallest eta with honest FPR = P(H>=eta) <= budget; returns (eta, actual_fpr)."""
    if not H:
        return None, None
    cand = sorted(set(H) | {0.0, 1.0})
    best = (1.0, 0.0)
    for e in cand:
        fpr = float(np.mean([h >= e for h in H]))
        if fpr <= budget:
            return e, fpr
        best = (e, fpr)
    return best


def operating_point(a):
    runs = load(a.inp)
    hon = [r for r in pick(runs, a.honest_family)]
    if not hon:
        raise SystemExit(f"no honest runs for --honest_family {a.honest_family}")
    fams = a.families or []
    if not fams:
        raise SystemExit("pass --families A2_... A3_... [H3_... for contrast]")
    budgets = [0.01, 0.05, 0.10]
    H = _pc_bers(hon, free_rider=False, tail=a.tail)              # global honest pool
    etas = {b: _eta_for_fpr(H, b) for b in budgets}

    rows = []            # (family, fr_class, {budget: (recall_global, recall_perclass)})
    for f in fams:
        ar = pick(runs, f)
        if not ar:
            print(f"  (skip {f} -- no runs)"); continue
        c = _fr_class(ar)
        F = _pc_bers(ar, free_rider=True, tail=a.tail)
        Hc = _pc_bers(hon, free_rider=False, tail=a.tail, trigger_class=c) if c is not None else []
        rec = {}
        for b in budgets:
            eg, _ = etas[b]
            rg = float(np.mean([x >= eg for x in F])) if F else float("nan")
            # per-class oracle eta (calibrated only on honest clients of the FR's class)
            ec, _ = _eta_for_fpr(Hc, b) if Hc else (None, None)
            rc = (float(np.mean([x >= ec for x in F])) if (F and ec is not None) else float("nan"))
            rec[b] = (rg, rc)
        rows.append((f, c, rec))

    # ---- figure: grouped horizontal bars, recall per family per FPR budget ----
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(rows) + 2.2))
    ys = np.arange(len(rows))
    hgt = 0.8 / len(budgets)
    cols = [ps.OKABE["red"], ps.OKABE["orange"], ps.OKABE["yellow"]]
    for i, b in enumerate(budgets):
        vals = [r[2][b][0] for r in rows]
        eg, af = etas[b]
        ax.barh(ys + (i - (len(budgets) - 1) / 2) * hgt, vals, height=hgt,
                color=cols[i % len(cols)], edgecolor="black", linewidth=0.4,
                label=f"honest FPR\u2264{int(b*100)}%  (\u03b7={eg:.2f}, actual {af:.0%})")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0].split('_')[0]}  (cls {r[1]})" for r in rows])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("free-rider recall  (fraction of free-riders caught)")
    ax.set_title("Recall at a fixed honest false-positive budget\n"
                 "one deployable \u03b7 per budget \u2014 no line catches the insiders")
    ax.axvline(0.9, color="0.5", ls=":", lw=1, zorder=0)
    ax.text(0.9, len(rows) - 0.4, " target 0.9", color="0.4", fontsize=8, va="top")
    ax.grid(axis="x", alpha=.3); ax.legend(fontsize=8, loc="lower right", framealpha=.95)
    out = a.out if str(a.out).endswith(".png") else str(a.out) + ".png"
    ps.finish(fig, out)

    # ---- paste-ready table (global + per-class oracle) ----
    lines = ["# Operating points \u2014 recall at fixed honest FPR", "",
             f"- honest family: `{a.honest_family}`  ({len(hon)} run(s), tail {a.tail})",
             "- eta per budget (global honest pool): "
             + ", ".join(f"{int(b*100)}%\u2192\u03b7={etas[b][0]:.3f}" for b in budgets),
             "- **global** = one deployable eta for all classes; **per-class** = oracle eta "
             "calibrated on honest clients of the FR's own class (upper bound).", "",
             "| family | FR class | " + " | ".join(f"recall@{int(b*100)}%" for b in budgets)
             + " | " + " | ".join(f"perclass@{int(b*100)}%" for b in budgets) + " |",
             "|---|---|" + "---|" * (2 * len(budgets))]
    for f, c, rec in rows:
        g = " | ".join(f"{rec[b][0]:.2f}" for b in budgets)
        pc = " | ".join(("-" if np.isnan(rec[b][1]) else f"{rec[b][1]:.2f}") for b in budgets)
        lines.append(f"| {f} | {c} | {g} | {pc} |")
    lines += ["", "**Read:** at any usable FPR the insider recall stays low; the only families "
              "near 1.0 are the crude paper baselines (previous-models / gaussian). Even the "
              "per-class oracle (which the server cannot actually run) barely moves it."]
    md = out[:-4] + ".md"
    open(md, "w").write("\n".join(lines))
    print("wrote", md)


# ===========================================================================
#  tap_dynamics -- fade & recovery from the adaptive-tap trace.
#  Reads compute.per_client[fr].trace ({round, action, ber_before, ber_after,
#  target}). ONE family -> a trace plot (see a tap dip and the coast climb).
#  MANY families -> the stealth frontier: tap_fraction (compute spent) vs
#  rounds-between-taps (how long a tap lasts).
# ===========================================================================
def _fr_traces(run):
    """-> list of (cid, trace_list) for every free-rider in a run."""
    out = []
    comp = ((run.get("compute", {}) or {}).get("per_client", {}) or {})
    for cid, c in comp.items():
        if c.get("is_free_rider") and c.get("trace"):
            out.append((cid, c["trace"]))
    return out


def _tap_stats(trace):
    """fade/recovery numbers from one free-rider's trace."""
    fr = [t for t in trace if t.get("action") in ("tap", "coast")]
    if not fr:
        return None
    taps = [t["round"] for t in fr if t.get("action") == "tap"]
    n_tap = len(taps); n_coast = sum(1 for t in fr if t.get("action") == "coast")
    n = n_tap + n_coast
    gaps = [taps[i + 1] - taps[i] for i in range(len(taps) - 1)]   # rounds between taps
    drops = [t["ber_before"] - t["ber_after"] for t in fr
             if t.get("action") == "tap" and t.get("ber_before") is not None
             and t.get("ber_after") is not None]
    # fade slope: BER rise between CONSECUTIVE coasts only (a tap resets the run,
    # so the post-tap drop must not count as "fade").
    rises = []
    prev = None
    for t in fr:
        if t.get("action") == "coast" and t.get("ber_before") is not None:
            if prev is not None:
                rises.append(t["ber_before"] - prev)
            prev = t["ber_before"]
        else:
            prev = None                      # tap breaks the coast run
    targ = next((t.get("target") for t in fr if t.get("target") is not None), None)
    below = [t.get("ber_before") for t in fr if t.get("ber_before") is not None]
    stay = (float(np.mean([b <= (targ if targ is not None else 1) for b in below]))
            if below else float("nan"))
    return {
        "n_freeride": n, "n_taps": n_tap, "n_coasts": n_coast,
        "tap_fraction": (n_tap / n) if n else float("nan"),        # compute actually spent
        "rounds_between_taps": (float(np.mean(gaps)) if gaps else float("nan")),  # a tap's lifetime
        "ber_drop_per_tap": (float(np.mean(drops)) if drops else float("nan")),   # recovery magnitude
        "fade_per_coast": (float(np.mean(rises)) if rises else 0.0),              # climb while coasting
        "stayed_below_target": stay,
    }


def _cumulative_by_round(run, cid, field):
    """Cumulative sum of a per-round compute field (samples|gpu_ms) for one client."""
    comp = (run.get("compute", {}) or {}).get("per_client", {}) or {}
    c = comp.get(str(cid)) or comp.get(cid) or {}
    pr = c.get("per_round") or {}
    rounds = sorted(int(k) for k in pr)
    cum, run_tot = {}, 0.0
    for rd in rounds:
        run_tot += float(pr[str(rd)].get(field, 0.0) if str(rd) in pr else pr[rd].get(field, 0.0))
        cum[rd] = run_tot
    return cum


def gpu_savings(a):
    """Cumulative GPU-cycles (and samples) per communication round: each free-rider
    vs the honest-client mean, plus the running "saved vs honest" gap. Answers the
    meeting's question -- 'how much compute is the free-rider actually saving?' -- in
    the cluster's real cost unit (gpu_ms), not just image count.

    Two stacked panels: (top) cumulative gpu_ms per round, honest-mean line + each FR;
    (bottom) cumulative FRACTION of honest compute spent (FR_cum / honest_cum) so the
    'X% of honest effort' headline is a curve, not a single endpoint number.

    Reads compute.per_client[cid].per_round.{gpu_ms,samples} (already logged by
    ComputeMeter for BOTH taps and coasts -- a coast records gpu_ms~0, samples=0).
        python plots.py gpu_savings --in 'results/*/result.json' --family J2_saw_graft_head_c36
    """
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family <fam> or --families f1 f2 ...")
    runs_all = load(a.inp)
    field = "gpu_ms"      # cluster cost unit; samples plotted alongside

    for f in fams:
        runs = pick(runs_all, f)
        if not runs:
            print(f"  (skip {f} -- no runs)"); continue
        nseed = len(runs)

        fr_cids = sorted({int(p["cid"]) for r in runs for h in r.get("history", [])
                          for p in (h.get("wm_per_client") or []) if p.get("is_free_rider")})
        all_cids = sorted({int(cidk) for r in runs
                           for cidk in ((r.get("compute", {}) or {}).get("per_client", {}) or {})})
        honest_cids = [c for c in all_cids if c not in fr_cids]
        if not fr_cids:
            print(f"  (skip {f} -- no free-riders)"); continue

        # gather cumulative curves per cid per seed, then average over seeds
        def _avg_cum(cids, fld):
            per_round_acc = defaultdict(list)   # round -> [cum over (cid,seed)]
            for r in runs:
                for cid in cids:
                    cum = _cumulative_by_round(r, cid, fld)
                    for rd, v in cum.items():
                        per_round_acc[rd].append(v)
            xs = sorted(per_round_acc)
            return xs, [float(np.mean(per_round_acc[rd])) for rd in xs]

        hx, honest_cum = _avg_cum(honest_cids, field)
        honest_at = dict(zip(hx, honest_cum))
        # also compute the contention-free samples axis (always valid across pods)
        hx_s, honest_cum_s = _avg_cum(honest_cids, "samples")
        honest_at_s = dict(zip(hx_s, honest_cum_s))
        # is absolute gpu_ms trustworthy? (concurrency==1). ratio always is.
        concs = [(r.get("compute", {}) or {}).get("summary", {}).get("gpu_concurrency", 1)
                 for r in runs]
        gpu_reliable = all((c or 1) <= 1 for c in concs)

        fig, axes = ps.stacked_panels(3, figsize=(11, 9), height_ratios=[2, 2, 1])
        ax0, axS, ax1 = axes
        # --- panel 0: cumulative gpu_ms (cluster cost) ---
        ax0.plot(hx, honest_cum, color=ps.C_HONEST, lw=2.6, marker="o", ms=3,
                 label="honest mean (cumulative)")
        gap_rows = []
        for i, cid in enumerate(fr_cids):
            fx, fr_cum = _avg_cum([cid], field)
            ax0.plot(fx, fr_cum, lw=2.2, color=ps.CYCLE[(i + 1) % len(ps.CYCLE)],
                     marker="s", ms=3, label=f"free-rider cid{cid} (cumulative)")
            frac = [fr_cum[j] / honest_at[rd] if honest_at.get(rd) else np.nan
                    for j, rd in enumerate(fx)]
            ax1.plot(fx, frac, lw=2.0, color=ps.CYCLE[(i + 1) % len(ps.CYCLE)],
                     marker="s", ms=3, label=f"cid{cid}")
            if fx:
                fend, hend = fr_cum[-1], honest_at.get(fx[-1], float("nan"))
                # samples endpoint too
                fx_s, fr_cum_s = _avg_cum([cid], "samples")
                fend_s = fr_cum_s[-1] if fr_cum_s else float("nan")
                hend_s = honest_at_s.get(fx_s[-1], float("nan")) if fx_s else float("nan")
                gap_rows.append((cid, fend, hend, (1 - fend / hend) if hend else float("nan"),
                                 fend_s, hend_s, (1 - fend_s / hend_s) if hend_s else float("nan")))
            # samples panel
            axS.plot(fx_s, fr_cum_s, lw=2.0, color=ps.CYCLE[(i + 1) % len(ps.CYCLE)],
                     marker="s", ms=3, label=f"cid{cid}")
        axS.plot(hx_s, honest_cum_s, color=ps.C_HONEST, lw=2.4, marker="o", ms=3,
                 label="honest mean")

        rel = "" if gpu_reliable else "  \u26a0 gpu_ms inflated by GPU sharing (WORKERS>1) -- use the samples panel / ratio"
        ax0.set_ylabel(f"cumulative {field}")
        ax0.set_title(f"Cumulative compute per round  \u00b7  {f.split('_rep')[0]}  "
                      f"({nseed} seed{'s' if nseed != 1 else ''})  \u00b7  "
                      f"gap below honest = compute saved{rel}", fontsize=10)
        ax0.grid(alpha=.3); ax0.legend(fontsize=8, loc="upper left")
        if fr_cids:
            fx, fr_cum = _avg_cum([fr_cids[0]], field)
            ax0.fill_between(fx, fr_cum, [honest_at.get(rd, np.nan) for rd in fx],
                             color=ps.C_GOOD, alpha=.12, label="_nolegend_")
        axS.set_ylabel("cumulative samples\n(contention-free)")
        axS.grid(alpha=.3); axS.legend(fontsize=8, loc="upper left")

        ax1.axhline(1.0, color=ps.C_HONEST, ls="--", lw=1.2)
        ax1.set_ylabel("cum FR / cum honest")
        ax1.set_xlabel("communication round")
        ax1.set_ylim(0, 1.15)
        ax1.grid(alpha=.3); ax1.legend(fontsize=8, loc="upper right")

        out = (a.out if str(a.out).endswith(".png") else str(a.out) + ".png") if a.out else f"gpu_savings_{f}.png"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        ps.finish(fig, out)

        md = out[:-4] + ".md"
        L = [f"# GPU-cycles saved by free-riding \u2014 {f}  ({nseed} seed(s))", "",
             f"Cumulative **{field}** and **samples** over the whole run. 'saved' = 1 \u2212 "
             "(FR cumulative / honest cumulative) at the final round. Whole-run figure "
             "(warmup included); for the marginal attack-phase cost see the tap-fraction in "
             "`tap_perfr`.", ""]
        if not gpu_reliable:
            L += ["> \u26a0 **gpu_ms absolute values are inflated** because these runs shared a GPU "
                  "(WORKERS>1). The **ratio** (saved %) is still valid (same-process inflation "
                  "cancels), and **samples** is contention-free. Prefer the samples column for "
                  "cross-run cost.", ""]
        L += ["| cid | FR gpu_ms | honest gpu_ms | saved (gpu) | FR samples | honest samples | saved (samples) |",
              "|---|---|---|---|---|---|---|"]
        for cid, fend, hend, saved, fend_s, hend_s, saved_s in gap_rows:
            L.append(f"| {cid} | {fend:,.0f} | {hend:,.0f} | "
                     f"{'n/a' if np.isnan(saved) else f'{saved:.0%}'} | "
                     f"{fend_s:,.0f} | {hend_s:,.0f} | "
                     f"{'n/a' if np.isnan(saved_s) else f'{saved_s:.0%}'} |")
        open(md, "w").write("\n".join(L))
        print("wrote", md)


def trigger_fairness(a):
    """BER vs the number of trigger-class images a client HOLDS -- the non-IID
    fairness check. Under round-robin assignment a client is often starved on its
    own trigger class (few images -> high BER); under distribution assignment the
    server gives it a class it holds a lot of. This scatter shows whether BER is
    still dictated by how many trigger samples the client got.

    x = trigger-class images held (from summary.wm_trigger_holdings);
    y = converged honest BER of that client (tail mean). One point per client per
    seed; free-riders marked separately. Overlays the two assignment policies if
    both families are passed via --families.
        python plots.py trigger_fairness --in 'results/*/result.json' \
            --families E1_honest_niid_c100 E4_honest_niid_distrib_c100
    """
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family <fam> or --families f1 f2 ...")
    runs_all = load(a.inp)
    tail = getattr(a, "tail", None) or TAIL

    fig, ax = plt.subplots(figsize=(9, 6))
    md_rows = []
    skipped = []
    for i, f in enumerate(fams):
        runs = pick(runs_all, f)
        if not runs:
            print(f"  (skip {f} -- no runs)"); skipped.append(f + " (no runs)"); continue
        col = ps.CYCLE[i % len(ps.CYCLE)]
        xs, ys, is_fr = [], [], []
        assign_mode = None
        for r in runs:
            summ = r.get("summary") or {}
            holdings = summ.get("wm_trigger_holdings") or r.get("wm_trigger_holdings") or {}
            assign_mode = summ.get("wm_trigger_assign") or r.get("wm_trigger_assign") or assign_mode
            # tail BER per cid
            perc = defaultdict(list)
            frflag = {}
            for h in r.get("history", [])[-tail:]:
                for p in (h.get("wm_per_client") or []):
                    if p.get("ber") is not None:
                        perc[int(p["cid"])].append(float(p["ber"]))
                        frflag[int(p["cid"])] = bool(p.get("is_free_rider"))
            for cid, bers in perc.items():
                hold = holdings.get(str(cid), holdings.get(cid))
                if hold is None:
                    continue
                xs.append(int(hold)); ys.append(float(np.mean(bers))); is_fr.append(frflag.get(cid, False))
        if not xs:
            print(f"  (skip {f} -- no trigger_holdings in result.json; run backfill_holdings.py "
                  f"or re-run with the patched runner)")
            skipped.append(f + " (no holdings -- backfill or re-run)")
            continue
        hx = [x for x, fr in zip(xs, is_fr) if not fr]
        hy = [y for y, fr in zip(ys, is_fr) if not fr]
        fx = [x for x, fr in zip(xs, is_fr) if fr]
        fy = [y for y, fr in zip(ys, is_fr) if fr]
        lab = f"{f.split('_rep')[0]}" + (f" [{assign_mode}]" if assign_mode else "")
        ax.scatter(hx, hy, s=55, color=col, alpha=.8, edgecolor="white", label=lab)
        if fx:
            ax.scatter(fx, fy, s=90, color=col, marker="X", edgecolor="black",
                       zorder=5, label=f"{f.split('_rep')[0]} free-riders")
        # correlation
        if len(xs) > 2:
            r_pear = float(np.corrcoef(xs, ys)[0, 1])
            md_rows.append((lab, len(xs), min(xs), max(xs), r_pear))

    ax.axhline(ETA_LOOSE_DEFAULT, color=ps.C_HONEST, ls="--", lw=1.4,
               label=f"\u03b7 loose {ETA_LOOSE_DEFAULT:.3f}")
    ax.set_xlabel("trigger-class images the client HOLDS  (fewer = more starved)")
    ax.set_ylabel("converged BER (tail mean)  \u00b7  higher = worse mark")
    ax.set_title("BER vs trigger-sample holdings (non-IID fairness check)\n"
                 "flat/low = fair (holdings don't dictate BER); steep = starvation-driven")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper right")
    if skipped:
        ax.text(0.5, 0.4, "No holdings for:\n" + "\n".join(skipped) +
                "\n\nFix: python backfill_holdings.py --in '<those runs>/result.json'",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                color=ps.C_BAD, bbox=dict(boxstyle="round", fc="white", ec=ps.C_BAD, alpha=.95))
    out = (a.out if str(a.out).endswith(".png") else str(a.out) + ".png") if a.out else "trigger_fairness.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ps.finish(fig, out)

    md = out[:-4] + ".md"
    L = ["# BER vs trigger-sample holdings (non-IID fairness)", "",
         "Correlation r near 0 = fair (a client's BER doesn't depend on how many trigger "
         "images it happened to hold). Strong positive/negative r = starvation-driven BER, "
         "which distribution-aware assignment is meant to remove.", "",
         "| family [assign] | n points | min held | max held | corr(held, BER) |",
         "|---|---|---|---|---|"]
    for lab, n, mn, mx, r in md_rows:
        L.append(f"| {lab} | {n} | {mn} | {mx} | {r:+.3f} |")
    open(md, "w").write("\n".join(L))
    print("wrote", md)


def accuracy(a):
    """Global test accuracy over rounds: attack run vs an honest reference, plus the
    free-rider's own trigger-class TEST accuracy (from per_class of the final model)
    when available. This is the 'Fig B' panel the meeting asked to auto-generate --
    it shows the free-rider barely dents global accuracy (the whole point: it steals a
    good model) while its own trigger class may be sacrificed.

    Overlays every family passed. Honest reference via --honest_in/--honest_family
    (falls back to any honest run in --in).
        python plots.py accuracy --in 'results/*/result.json' --family J2_saw_graft_head_c36 \
            --honest_in 'results/*/result.json' --honest_family A1_honest_c100
    """
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family <fam> or --families f1 f2 ...")
    runs_all = load(a.inp)

    def _acc_curve(runs):
        acc = defaultdict(list)
        for r in runs:
            for h in r.get("history", []):
                if h.get("test_acc") is not None:
                    acc[h["round"]].append(float(h["test_acc"]))
        xs = sorted(acc)
        return xs, [float(np.mean(acc[rd])) for rd in xs], [float(np.std(acc[rd])) for rd in xs]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # honest reference
    hon = []
    if getattr(a, "honest_in", None):
        hon = [r for r in load(a.honest_in) if th.is_honest_run(r)
               and (a.honest_family is None or fam(r) == a.honest_family)]
    if not hon:
        hon = [r for r in runs_all if th.is_honest_run(r)]
    if hon:
        hx, hm, hs = _acc_curve(hon)
        ax.plot(hx, hm, color=ps.C_HONEST, lw=2.6, label="honest run (global)")
        ax.fill_between(hx, np.array(hm) - np.array(hs), np.array(hm) + np.array(hs),
                        color=ps.C_HONEST, alpha=.12)

    md_rows = []
    for i, f in enumerate(fams):
        runs = pick(runs_all, f)
        if not runs:
            print(f"  (skip {f} -- no runs)"); continue
        fx, fm, fsd = _acc_curve(runs)
        col = ps.CYCLE[(i + 1) % len(ps.CYCLE)]
        ax.plot(fx, fm, color=col, lw=2.2, ls="--",
                label=f"{f.split('_rep')[0]} (global)")
        final = fm[-1] if fm else float("nan")
        # free-rider trigger-class TEST accuracy from per_class of the final model
        fr_tc_acc = []
        for r in runs:
            frcids = {int(p["cid"]): int(p["trigger_class"])
                      for h in r.get("history", []) for p in (h.get("wm_per_client") or [])
                      if p.get("is_free_rider") and p.get("trigger_class") is not None}
            by = ((r.get("per_class") or {}).get("by_class") or {})
            for cid, tc in frcids.items():
                cell = by.get(str(tc)) or by.get(tc)
                if cell and cell.get("acc") is not None:
                    fr_tc_acc.append((tc, float(cell["acc"])))
        tc_txt = ", ".join(f"cls{tc}:{acc:.0f}%" for tc, acc in sorted(set(fr_tc_acc))) or "n/a"
        md_rows.append((f.split("_rep")[0], final, tc_txt))

    honest_final = hm[-1] if hon and hm else float("nan")
    ax.set_xlabel("communication round")
    ax.set_ylabel("global test accuracy (%)")
    ax.set_title("Global test accuracy: attack vs honest\n"
                 "(free-riders barely dent global accuracy -- they steal a good model)")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
    out = (a.out if str(a.out).endswith(".png") else str(a.out) + ".png") if a.out else "accuracy.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ps.finish(fig, out)

    md = out[:-4] + ".md"
    L = ["# Accuracy: attack vs honest", "",
         f"Honest final global accuracy: **{honest_final:.2f}%**.", "",
         "| family | final global acc | free-rider trigger-class TEST acc (final model) |",
         "|---|---|---|"]
    for name, final, tc in md_rows:
        L.append(f"| {name} | {final:.2f}% | {tc} |")
    L += ["", "The free-rider's *global* accuracy tracks honest (it rides the shared model); "
          "its own trigger-class accuracy is the sacrificed/hidden cost (see wrap-up 3.5)."]
    open(md, "w").write("\n".join(L))
    print("wrote", md)


def dirichlet_dist(a):
    """Reference figure: what a Dirichlet(alpha) label-skew partition actually looks
    like, for alpha in {0.1, 0.5, 1.0}. NO result files needed -- it re-draws the
    exact partition datasets.py::dirichlet_partition would produce (same rule), so
    the reader can SEE why small alpha starves clients on most classes. One heatmap
    per alpha: rows = clients, cols = classes, colour = fraction of that class the
    client holds.  --classes sets n (default 10), --seed the RNG.
        python plots.py dirichlet_dist --in x --out figs/dirichlet_dist
    (--in is ignored; the subparser requires it, so pass any placeholder.)
    """
    alphas = [0.1, 0.5, 1.0]
    n_classes = int(a.classes) if getattr(a, "classes", None) else 10
    n_clients = 10
    seed = int(a.seed) if getattr(a, "seed", None) else 0
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 4.2))
    if len(alphas) == 1:
        axes = [axes]
    for ax, alpha in zip(axes, alphas):
        # props[class] ~ Dirichlet(alpha over clients); matrix[client, class] = share
        mat = np.zeros((n_clients, n_classes))
        for c in range(n_classes):
            props = rng.dirichlet(alpha * np.ones(n_clients))
            mat[:, c] = props
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"\u03b1 = {alpha}", fontsize=12)
        ax.set_xlabel("class")
        if ax is axes[0]:
            ax.set_ylabel("client")
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_clients))
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("fraction of the class held by the client")
    fig.suptitle("Dirichlet(\u03b1) label-skew partition (rows=clients, cols=classes)\n"
                 "small \u03b1 = one client hogs each class (severe skew); larger \u03b1 = more even",
                 fontsize=12, fontweight="bold")
    out = (a.out if str(a.out).endswith(".png") else str(a.out) + ".png") if a.out else "dirichlet_dist.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ps.finish(fig, out)


def tap_dynamics(a):
    runs = load(a.inp)
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family <adaptive_tap fam> or --families f1 f2 ...")

    # aggregate stats per family
    table = {}
    for f in fams:
        rr = pick(runs, f)
        st = [s for r in rr for _, tr in _fr_traces(r) if (s := _tap_stats(tr))]
        if not st:
            print(f"  (skip {f} -- no adaptive-tap trace)"); continue
        agg = {k: float(np.nanmean([s[k] for s in st])) for k in st[0]}
        table[f] = agg

    if not table:
        raise SystemExit("no traces found (are these adaptive_tap runs?)")

    out = a.out if str(a.out).endswith(".png") else str(a.out) + ".png"
    if len(table) == 1:
        # --- single family: draw the actual BER trace with taps & coasts ---
        f = next(iter(table)); rr = pick(runs, f)
        tr = _fr_traces(rr[0])[0][1]
        fr = [t for t in tr if t.get("action") in ("tap", "coast")]
        xs = [t["round"] for t in fr]
        yb = [t.get("ber_before") for t in fr]
        targ = next((t.get("target") for t in fr if t.get("target") is not None), None)
        eta = next((t.get("eta_frozen") for t in fr if t.get("eta_frozen") is not None), None)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(xs, yb, color=ps.OKABE["red"], lw=1.8, marker="o", ms=3,
                label="free-rider BER (before acting)")
        tapx = [t["round"] for t in fr if t.get("action") == "tap"]
        tapy = [t.get("ber_after") for t in fr if t.get("action") == "tap"]
        ax.scatter(tapx, tapy, marker="v", s=60, color=ps.OKABE["blue"], zorder=5,
                   label="tap (BER after re-embed)")
        if targ is not None:
            ax.axhline(targ, color="0.4", ls="--", lw=1.2, label=f"target (\u03b7\u2212margin) = {targ:.3f}")
        if eta is not None:
            ax.axhline(eta, color="black", ls="--", lw=1.6, label=f"\u03b7 = {eta:.3f}")
        s = table[f]
        cap = (f"tap fraction {s['tap_fraction']:.0%} of rounds  \u00b7  a tap lasts "
               f"~{s['rounds_between_taps']:.1f} rounds  \u00b7  drop/tap {s['ber_drop_per_tap']:.3f}  "
               f"\u00b7  fade +{s['fade_per_coast']:.3f}/coast")
        ax.set_title(f"Adaptive-tap dynamics  \u00b7  {f.split('_rep')[0]}\n{cap}")
        ax.set_xlabel("communication round"); ax.set_ylabel("bit-error-rate")
        ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper right")
        ps.finish(fig, out)
    else:
        # --- many families: the stealth frontier (compute vs persistence) ---
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, (f, s) in enumerate(sorted(table.items())):
            ax.scatter(s["tap_fraction"], s["rounds_between_taps"],
                       s=90, color=ps.CYCLE[i % len(ps.CYCLE)], edgecolor="black", zorder=5)
            ax.annotate(f.split("_c")[0].replace("I_", "").replace("J_", ""),
                        (s["tap_fraction"], s["rounds_between_taps"]),
                        fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("tap fraction  (compute actually spent; 1.0 = trains every round)")
        ax.set_ylabel("rounds a tap lasts  (mark persistence)")
        ax.set_title("Adaptive free-rider stealth frontier\nlower-left & higher = cheaper and longer-lived")
        ax.grid(alpha=.3)
        ps.finish(fig, out)

    # --- table ---
    ks = ["n_freeride", "tap_fraction", "rounds_between_taps", "ber_drop_per_tap",
          "fade_per_coast", "stayed_below_target"]
    lines = ["# Adaptive-tap dynamics", "",
             "- **tap_fraction** = compute the FR actually spent (fraction of freeride rounds it trained)",
             "- **rounds_between_taps** = how many rounds one tap keeps the mark under target (fade time)",
             "- **ber_drop_per_tap** = how far one tap pushes BER down (recovery magnitude)",
             "- **fade_per_coast** = BER rise per coasting round  \u00b7  **stayed_below_target** = safety", "",
             "| family | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
    for f, s in sorted(table.items()):
        lines.append("| " + f.split("_rep")[0] + " | "
                     + " | ".join(f"{s[k]:.3f}" for k in ks) + " |")
    md = out[:-4] + ".md"
    open(md, "w").write("\n".join(lines))
    print("wrote", md)


# ===========================================================================
#  tap_perfr -- ONE panel per free-rider cid, so each free-rider is visible
#  SEPARATELY on its own trigger class: when it taps vs coasts, its
#  server-measured BER (the ground truth that gets flagged) AND its own
#  self-probe (ber_before, what drives the tap/coast decision) on the same axis
#  -> you SEE the self-probe-vs-server gap and the per-class asymmetry that the
#  aggregate tap_dynamics plot hides.  Aggregates over seeds.
#
#    python plots.py tap_perfr --in 'results/*/result.json' \
#        --family J2_saw_graft_head_c36 --out figs/tap_perfr_J2
# ===========================================================================
def _honest_ber_by_round(runs, tclass_filter=None):
    """{round: [ber over honest clients across seeds]}.

    tclass_filter=None -> ALL honest clients (the global honest cloud).
    tclass_filter=c    -> only honest clients whose trigger_class == c (the
                          same-class honest twin: the fair comparison for a FR on
                          class c). A client is honest iff is_free_rider is false.
    """
    out = defaultdict(list)
    for r in runs:
        for h in r.get("history", []):
            rd = h["round"]
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider"):
                    continue
                if p.get("ber") is None:
                    continue
                if tclass_filter is not None and int(p.get("trigger_class", -1)) != int(tclass_filter):
                    continue
                out[rd].append(float(p["ber"]))
    return out


def tap_perfr(a):
    """ONE FIGURE PER FREE-RIDER (not stacked). Each figure shows, for a single
    free-rider on its own trigger class:
      * the FR's server-measured BER (what actually gets flagged),
      * the FR's self-probe (what drives its tap/coast decision),
      * tap / coast markers per round,
      * the GLOBAL honest BER cloud (mean over ALL honest clients) -- kept on BOTH
        free-rider figures as the common reference, and
      * the SAME-CLASS honest twin (honest clients on the FR's own trigger class):
        the fair, apples-to-apples comparison the meeting asked for -- it isolates
        "did the FR do less work?" from "is this class just hard for everyone?".
    Aggregates over seeds. Emits <out>_<family>_cid<cid>.png + a combined .md table.
    """
    fams = a.families or ([a.family] if a.family else None)
    if not fams:
        raise SystemExit("pass --family <adaptive_tap fam> or --families f1 f2 ...")
    runs_all = load(a.inp)
    eta_t = a.eta_tight if getattr(a, "eta_tight", None) is not None else ETA_TIGHT_DEFAULT
    eta_l = a.eta_loose if getattr(a, "eta_loose", None) is not None else ETA_LOOSE_DEFAULT

    for f in fams:
        runs = pick(runs_all, f)
        if not runs:
            print(f"  (skip {f} -- no runs)"); continue

        # free-rider cids and their trigger classes (from the server-side per-client rows)
        fr_cids, tclass = set(), {}
        for r in runs:
            for h in r.get("history", []):
                for p in (h.get("wm_per_client") or []):
                    if p.get("is_free_rider"):
                        cid = int(p["cid"]); fr_cids.add(cid)
                        if p.get("trigger_class") is not None:
                            tclass[cid] = int(p["trigger_class"])
        fr_cids = sorted(fr_cids)
        if not fr_cids:
            print(f"  (skip {f} -- no free-riders in trace)"); continue

        nseed = len(runs)
        # per (cid, round): server BER (history), self-probe + actions (trace), across seeds
        srv    = {c: defaultdict(list) for c in fr_cids}
        probe  = {c: defaultdict(list) for c in fr_cids}
        tap_ct = {c: defaultdict(int)  for c in fr_cids}
        coa_ct = {c: defaultdict(int)  for c in fr_cids}
        target_val = {}
        for r in runs:
            for h in r.get("history", []):
                rd = h["round"]
                for p in (h.get("wm_per_client") or []):
                    if p.get("is_free_rider") and p.get("ber") is not None:
                        srv[int(p["cid"])][rd].append(float(p["ber"]))
            comp = (r.get("compute", {}) or {}).get("per_client", {}) or {}
            for cidk, c in comp.items():
                if not c.get("is_free_rider"):
                    continue
                cid = int(cidk)
                if cid not in srv:
                    continue
                for t in (c.get("trace") or []):
                    rd, act = t.get("round"), t.get("action")
                    if t.get("ber_before") is not None:
                        probe[cid][rd].append(float(t["ber_before"]))
                    if act == "tap":
                        tap_ct[cid][rd] += 1
                    elif act == "coast":
                        coa_ct[cid][rd] += 1
                    if t.get("target") is not None:
                        target_val[cid] = float(t["target"])

        # honest references: the global cloud (all honest) once, and the same-class
        # twin per free-rider trigger class.
        #   In the standard 10-client setup the FR cids ARE cid3/cid6, so NO honest
        #   client sits on classes 3/6 within the attack run -> the same-class twin
        #   would be empty. Fall back to an external honest family (e.g. A1) passed
        #   via --honest_in/--honest_family, exactly like `timeline` does, so the
        #   fair same-class comparison is always available.
        honest_all = _honest_ber_by_round(runs, tclass_filter=None)
        honest_same = {tc: _honest_ber_by_round(runs, tclass_filter=tc)
                       for tc in set(tclass.values())}
        # external honest family (same pattern as timeline's --honest_in/--honest_family)
        hon_ext = []
        if getattr(a, "honest_in", None):
            hon_ext = [r for r in load(a.honest_in) if th.is_honest_run(r)
                       and (a.honest_family is None or fam(r) == a.honest_family)]
        if hon_ext:
            if not any(honest_all.values()):
                honest_all = _honest_ber_by_round(hon_ext, tclass_filter=None)
            for tc in set(tclass.values()):
                if not honest_same.get(tc):
                    honest_same[tc] = _honest_ber_by_round(hon_ext, tclass_filter=tc)

        lo, hi = th.calib_window(runs[0]); W = hi + 1
        base = a.out or "tap_perfr"
        base = base[:-4] if str(base).endswith(".png") else str(base)
        md_rows = []

        # ---- ONE FIGURE PER FREE-RIDER ----
        for cid in fr_cids:
            tc = tclass.get(cid, "?")
            fig, ax = plt.subplots(figsize=(12, 5.2))
            xr = sorted(set(list(srv[cid]) + list(probe[cid])))
            srv_mean = [np.mean(srv[cid][rd])   if srv[cid].get(rd)   else np.nan for rd in xr]
            prb_mean = [np.mean(probe[cid][rd]) if probe[cid].get(rd) else np.nan for rd in xr]

            # schedule bands
            ax.axvspan(0.5, W - 0.5, color=ps.OKABE["yellow"], alpha=.12, lw=0,
                       label="warmup")
            ax.axvspan(lo - 0.5, hi + 0.5, color=ps.OKABE["green"], alpha=.16, lw=0,
                       label="calib window")
            ax.axvline(W - 0.5, color="0.5", ls="--", lw=1)

            # honest references (on EVERY free-rider figure)
            hx = sorted(honest_all)
            ax.plot(hx, [np.mean(honest_all[rd]) for rd in hx],
                    color=ps.C_HONEST, lw=1.6, alpha=.85, zorder=2,
                    label="honest GLOBAL mean BER (all honest clients)")
            hs = honest_same.get(tc, {})
            if hs:
                hsx = sorted(hs)
                ax.plot(hsx, [np.mean(hs[rd]) for rd in hsx],
                        color=ps.OKABE["purple"], lw=2.0, ls=(0, (1, 1)), zorder=3,
                        label=f"honest SAME-CLASS twin (class {tc}) -- fair comparison")

            # the free-rider
            ax.plot(xr, srv_mean, color=ps.C_FR, lw=2.4, marker="o", ms=3, zorder=4,
                    label="FR server-measured BER (what gets flagged)")
            ax.plot(xr, prb_mean, color=ps.OKABE["orange"], lw=1.3, ls=(0, (4, 2)),
                    alpha=.95, zorder=3, label="FR self-probe (drives tap/coast)")

            def _yat(rd):
                if srv[cid].get(rd):   return float(np.mean(srv[cid][rd]))
                if probe[cid].get(rd): return float(np.mean(probe[cid][rd]))
                return np.nan
            # a round is a TAP/COAST if the majority of seeds tapped/coasted there
            tapx = [rd for rd in xr if tap_ct[cid].get(rd, 0) > nseed / 2]
            coax = [rd for rd in xr if coa_ct[cid].get(rd, 0) > nseed / 2]
            ax.scatter(tapx, [_yat(rd) for rd in tapx], marker="v", s=72,
                       color=ps.OKABE["blue"], edgecolor="white", zorder=6,
                       label="TAP (trains)")
            ax.scatter(coax, [_yat(rd) for rd in coax], marker="s", s=44,
                       facecolor="white", edgecolor=ps.C_FR, zorder=6,
                       label="COAST (no train)")

            ax.axhline(eta_l, color=ps.C_HONEST, ls="--", lw=1.7,
                       label=f"\u03b7 loose {eta_l:.3f} (operating line)")
            ax.axhline(eta_t, color="black", ls=":", lw=1.3,
                       label=f"\u03b7 tight {eta_t:.3f} (degenerate)")
            if cid in target_val:
                ax.axhline(target_val[cid], color="0.45", ls="-.", lw=1.0,
                           label=f"target {target_val[cid]:.3f}")

            # per-seed attack-phase tap fraction (denominator = freeride rounds)
            fracs = []
            for r in runs:
                comp = (r.get("compute", {}) or {}).get("per_client", {}) or {}
                c = comp.get(str(cid)) or comp.get(cid)
                if not c:
                    continue
                fr = [t for t in (c.get("trace") or []) if t.get("action") in ("tap", "coast")]
                nt = sum(1 for t in fr if t["action"] == "tap")
                if fr:
                    fracs.append(nt / len(fr))
            frac = float(np.mean(fracs)) if fracs else float("nan")
            tail = [np.mean(srv[cid][rd]) for rd in xr if rd >= W and srv[cid].get(rd)]
            srv_tail = float(np.mean(tail)) if tail else float("nan")
            # same-class honest twin tail (fair reference value)
            htail = [np.mean(hs[rd]) for rd in sorted(hs) if rd >= W and hs.get(rd)]
            hon_tail = float(np.mean(htail)) if htail else float("nan")
            verdict = ("UNDER \u03b7_loose (evades)" if srv_tail < eta_l
                       else "OVER \u03b7_loose (per-client CAUGHT)")
            cmp = ("cleaner than" if srv_tail < hon_tail else "dirtier than") \
                if not np.isnan(hon_tail) else "n/a"
            ax.set_title(f"{f.split('_rep')[0]}  \u00b7  cid{cid} \u00b7 class {tc}  "
                         f"({nseed} seed{'s' if nseed != 1 else ''})\n"
                         f"tap-fraction {frac:.0%} (attack phase)  \u00b7  tail FR-BER {srv_tail:.2f} "
                         f"vs same-class honest {hon_tail:.2f} ({cmp})  \u2192  {verdict}",
                         fontsize=10)
            ax.set_xlabel("communication round")
            ax.set_ylabel("bit-error-rate")
            ax.set_ylim(-0.03, max(0.62, eta_l + 0.06))
            ax.grid(alpha=.3)
            ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=.95)

            out = f"{base}_{f}_cid{cid}.png"
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            ps.finish(fig, out)
            md_rows.append((cid, tc, frac, srv_tail, hon_tail, verdict))

        # combined markdown table for the family
        md = f"{base}_{f}.md"
        L = [f"# Per-free-rider tap/coast \u2014 {f}  ({nseed} seed(s))", "",
             "One FIGURE per free-rider (`*_cid<cid>.png`); each carries the global honest "
             "mean AND the same-class honest twin. **tap-fraction** uses the attack-phase "
             "denominator (freeride rounds only). **tail FR-BER** is the server's read over the "
             "converged tail; compare to the **same-class honest** column (the fair baseline) and "
             f"to \u03b7 loose = {eta_l:.3f}. A free-rider is only truly hidden if its *server* tail "
             "BER is under \u03b7_loose; if it also sits at/under the same-class honest twin it is "
             "*inseparable* from an honest client on that class.", "",
             "| cid | class | tap-fraction | tail FR-BER | same-class honest BER | verdict |",
             "|---|---|---|---|---|---|"]
        for cid, tc, frac, srv_tail, hon_tail, verdict in md_rows:
            L.append(f"| {cid} | {tc} | {frac:.0%} | {srv_tail:.3f} | "
                     f"{'n/a' if np.isnan(hon_tail) else f'{hon_tail:.3f}'} | {verdict} |")
        open(md, "w").write("\n".join(L))
        print("wrote", md)


def gpu_inflation(a):
    """GPU-SHARING INFLATION CHECK. Validates that the free-rider 'compute saved %'
    ratio is trustworthy even though absolute gpu_ms is inflated when many runs share a
    GPU (MPS, WORKERS>1). It splits a family's runs into a SHARED bucket
    (gpu_concurrency>1, gpu_ms inflated) and a SINGLE-TENANT bucket (gpu_concurrency==1,
    gpu_ms reliable) -- the two K4 runs differ only in this -- and shows:
      (A) honest absolute cumulative gpu_ms (shared >> single) while samples are
          identical (samples are contention-free);
      (B) per-free-rider compute-saved %, computed from gpu_ms AND from samples, for
          BOTH buckets. If gpu_ms-saved == samples-saved WITHIN each bucket, the ratio
          is inflation-invariant, so the shared-GPU saved-% headline is trustworthy.
    Run it on the family that has both a WORKERS>1 and a WORKERS=1 run in the pool:
        python plots.py gpu_inflation --in 'results/*/result.json' --family K4_alldyn_block2_c36
    """
    fam_ = a.family or (a.families[0] if a.families else None)
    if not fam_:
        raise SystemExit("pass --family <fam that has a WORKERS=1 and a WORKERS>1 run>")
    runs = pick(load(a.inp), fam_)
    if not runs:
        raise SystemExit(f"no runs for {fam_}")

    def _conc(r):
        return int((r.get("compute", {}) or {}).get("summary", {}).get("gpu_concurrency", 1) or 1)
    shared = [r for r in runs if _conc(r) > 1]
    single = [r for r in runs if _conc(r) <= 1]

    def _cum(run, cid, field):
        pr = (run.get("compute", {}) or {}).get("per_client", {}).get(str(cid), {}).get("per_round", {}) or {}
        return float(sum(float(pr[k].get(field, 0.0)) for k in pr))
    def _bucket(rs):
        if not rs:
            return None
        fr = sorted({int(p["cid"]) for r in rs for h in r.get("history", [])
                     for p in (h.get("wm_per_client") or []) if p.get("is_free_rider")})
        allc = sorted({int(c) for r in rs for c in ((r.get("compute", {}) or {}).get("per_client", {}) or {})})
        hon = [c for c in allc if c not in fr]
        hg = float(np.mean([np.mean([_cum(r, c, "gpu_ms") for c in hon]) for r in rs]))
        hs = float(np.mean([np.mean([_cum(r, c, "samples") for c in hon]) for r in rs]))
        d = {"hon_gpu": hg, "hon_smp": hs, "conc": _conc(rs[0]), "fr": {}}
        for c in fr:
            fg = float(np.mean([_cum(r, c, "gpu_ms") for r in rs]))
            fs = float(np.mean([_cum(r, c, "samples") for r in rs]))
            d["fr"][c] = {"sg": 1 - fg / hg if hg else float("nan"),
                          "ss": 1 - fs / hs if hs else float("nan")}
        return d
    M = _bucket(shared); S = _bucket(single)
    if M is None and S is None:
        raise SystemExit("no usable runs")
    if M is None or S is None:
        have = "single-tenant (WORKERS=1)" if S else "shared (WORKERS>1)"
        print(f"  (only the {have} bucket is present for {fam_}; plotting it alone. "
              f"Add the other WORKERS setting to the pool for the full inflation check.)")

    out = (a.out if str(a.out).endswith(".png") else str(a.out) + ".png") if a.out \
        else f"gpu_inflation_{fam_}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    CYCLE = getattr(ps, "CYCLE", [OK["blue"], OK["vermillion"], OK["green"], OK["orange"]])
    C_M, C_S = OK["vermillion"], OK["blue"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 6.2),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    # Panel A: honest absolute cumulative compute
    x = np.arange(2); w = 0.36
    if M:
        axA.bar(x - w / 2, [M["hon_gpu"] / 1e6, M["hon_smp"] / 1e6], w, color=C_M,
                label=f"shared (concurrency {M['conc']})")
    if S:
        axA.bar(x + w / 2, [S["hon_gpu"] / 1e6, S["hon_smp"] / 1e6], w, color=C_S,
                label="single-tenant (WORKERS=1)")
    if M and S and S["hon_gpu"]:
        axA.annotate(f"gpu_ms inflated \u00d7{M['hon_gpu']/S['hon_gpu']:.2f}\nby GPU sharing",
                     xy=(0 - w / 2, M["hon_gpu"] / 1e6), xytext=(0.02, M['hon_gpu'] / 1e6 * 0.78),
                     fontsize=9.5, color=C_M,
                     arrowprops=dict(arrowstyle="->", color=C_M, lw=1.3))
    axA.set_xticks(x); axA.set_xticklabels(["gpu_ms\n(cluster cost)", "samples\n(contention-free)"])
    axA.set_ylabel("honest-client cumulative compute  (\u00d710\u2076)")
    axA.set_title("A. Inflation is real and only in gpu_ms\n(samples identical)", fontsize=10.5)
    axA.legend(fontsize=8.5, loc="upper right"); axA.grid(alpha=.3)
    # Panel B: per-cid saved %, gpu vs samples, both buckets
    cids = sorted(set(list((M or {"fr": {}})["fr"]) + list((S or {"fr": {}})["fr"])))
    xc = np.arange(len(cids)); bw = 0.19
    series = [("shared \u00b7 gpu_ms", C_M, M, "sg", None),
              ("shared \u00b7 samples", C_M, M, "ss", "///"),
              ("single \u00b7 gpu_ms", C_S, S, "sg", None),
              ("single \u00b7 samples", C_S, S, "ss", "///")]
    for i, (lab, col, D, key, hatch) in enumerate(series):
        if D is None:
            continue
        vals = [D["fr"].get(c, {}).get(key, np.nan) * 100 for c in cids]
        bars = axB.bar(xc + (i - 1.5) * bw, vals, bw, color=col, edgecolor="white",
                       hatch=hatch, label=lab, alpha=0.6 if hatch else 1.0)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                axB.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}",
                         ha="center", va="bottom", fontsize=8)
    axB.set_xticks(xc); axB.set_xticklabels([f"cid{c} (cls {c})" for c in cids])
    axB.set_ylabel("compute SAVED vs honest  (%)"); axB.set_ylim(0, 100)
    axB.set_title("B. Within each bucket, gpu_ms-saved \u2248 samples-saved\n"
                  "\u21d2 saved-% ratio is inflation-invariant \u2192 trustworthy", fontsize=10.5)
    axB.legend(fontsize=8.5, ncol=2, loc="lower center"); axB.grid(alpha=.3)
    for c, xi in zip(cids, xc):
        gaps = []
        if M and c in M["fr"]:
            gaps.append(f"shared gap {abs(M['fr'][c]['sg']-M['fr'][c]['ss'])*100:.1f}pp")
        if S and c in S["fr"]:
            gaps.append(f"single gap {abs(S['fr'][c]['sg']-S['fr'][c]['ss'])*100:.1f}pp")
        if gaps:
            axB.text(xi, 10, "\n".join(gaps), ha="center", va="bottom", fontsize=8,
                     color="0.3", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.9))
    fig.suptitle(f"GPU-sharing inflation check \u00b7 {fam_}  \u00b7  single-tenant vs shared\n"
                 "absolute gpu_ms is inflated by MPS sharing, but FR/honest saved-% "
                 "tracks the contention-free samples ratio in both regimes", fontsize=11.5, y=1.02)
    fig.tight_layout(); ps.finish(fig, out)

    md = out[:-4] + ".md"
    L = [f"# GPU-sharing inflation check \u2014 {fam_}", "",
         "Splits the family's runs by `gpu_concurrency` (>1 = shared/inflated, ==1 = "
         "single-tenant/reliable). The saved-% ratio is trustworthy iff gpu_ms-saved "
         "\u2248 samples-saved within each bucket.", "",
         "| bucket | concurrency | honest gpu_ms | honest samples | cid | saved (gpu) | saved (samples) | gap |",
         "|---|---|---|---|---|---|---|---|"]
    for D, name in [(M, "shared"), (S, "single-tenant")]:
        if D is None:
            continue
        for c in cids:
            if c in D["fr"]:
                L.append(f"| {name} | {D['conc']} | {D['hon_gpu']:,.0f} | {D['hon_smp']:,.0f} | "
                         f"cid{c} | {D['fr'][c]['sg']*100:.1f}% | {D['fr'][c]['ss']*100:.1f}% | "
                         f"{abs(D['fr'][c]['sg']-D['fr'][c]['ss'])*100:.1f}pp |")
    open(md, "w").write("\n".join(L))
    print("wrote", md)


def _ovl_and_balerr(honest_bers, fr_bers, m_bits_=10):
    """Threshold-independent separability of two BER samples.
    OVL = shared histogram area (1.0 = identical distributions, inseparable);
    best_balanced_error = the LOWEST balanced error any threshold achieves, even
    an oracle one that peeks at the free-riders (0.5 = a coin flip = inseparable).
    Both self-contained (no dependency on detection.py) so this renders anywhere."""
    H = np.asarray([b for b in honest_bers if b is not None], float)
    F = np.asarray([b for b in fr_bers if b is not None], float)
    if len(H) == 0 or len(F) == 0:
        return float("nan"), float("nan")
    # BER is quantised to k/m; bin on those exact levels
    edges = (np.arange(0, m_bits_ + 2) - 0.5) / m_bits_
    ph, _ = np.histogram(H, bins=edges, density=False)
    pf, _ = np.histogram(F, bins=edges, density=False)
    ph = ph / ph.sum(); pf = pf / pf.sum()
    ovl = float(np.minimum(ph, pf).sum())
    # best balanced error over every candidate threshold (flag if BER >= eta):
    # honest wrongly flagged (FPR) vs FR correctly caught (recall).
    cand = sorted(set(np.round(np.concatenate([H, F]) * m_bits_).astype(int)))
    best = 1.0
    for c in list(cand) + [(max(cand) + 1 if cand else 1)]:
        eta = c / m_bits_
        fpr = float(np.mean(H >= eta))
        recall = float(np.mean(F >= eta))
        best = min(best, 0.5 * (fpr + (1.0 - recall)))
    return ovl, float(best)


def ea_fair(a):
    """GROUP EA -- distribution-aware non-IID fair comparison.

    The pro-server variant of the non-IID story (PROJECT_WRAPUP 6.6 / RESULTS_INDEX
    Group E). Under `wm_trigger_assign=distribution` the server hands every client a
    trigger class it *holds a lot of*, so honest clients are NO LONGER data-starved on
    their own class and embed cleanly. The question this plot answers, per class:

        does removing honest starvation open a separating threshold, or does the
        reduced free-rider simply embed cleanly too and still sit on top of the honest
        client assigned the SAME trigger class?

    It matches every free-rider in the ATTACK family (EA2) to the honest client in the
    HONEST family (EA1) that was assigned the *same* trigger class -- the fair,
    apples-to-apples twin -- and shows (top) both BER curves over rounds and (bottom)
    the converged per-client BER clouds with the threshold-independent separability
    numbers (OVL, best balanced-error). FR at/under the honest twin => inseparable even
    without starvation, which is the EA hypothesis.

        python plots.py ea_fair --in 'results/*/result.json' \
            --family EA2_reduced_niid_distrib_c36 \
            --honest_family EA1_honest_niid_distrib_c100
    """
    CYCLE = getattr(ps, "CYCLE", [OK["blue"], OK["vermillion"], OK["green"],
                                  OK["orange"], OK["purple"], OK["skyblue"]])
    fr_fam = a.family or (a.families[0] if a.families else None)
    if not fr_fam:
        raise SystemExit("pass --family <EA2 reduced-FR family> "
                         "(and --honest_family <EA1 honest family>)")
    runs_all = load(a.inp)
    fr_runs = pick(runs_all, fr_fam)
    if not fr_runs:
        raise SystemExit(f"no runs for FR family {fr_fam}")
    # honest family: explicit --honest_family in --in, else --honest_in glob, else
    # any honest run sharing the pool.
    hon_runs = pick(runs_all, a.honest_family) if a.honest_family else []
    if not hon_runs and getattr(a, "honest_in", None):
        hon_runs = load(a.honest_in)
        if a.honest_family:
            hon_runs = [r for r in hon_runs if fam(r) == a.honest_family]
    if not hon_runs:
        hon_runs = [r for r in runs_all
                    if not any(p.get("is_free_rider")
                               for h in r.get("history", [])
                               for p in (h.get("wm_per_client") or []))]
    m = m_bits(fr_runs)
    tail = getattr(a, "tail", None) or TAIL
    eta_l = a.eta_loose if getattr(a, "eta_loose", None) is not None else ETA_LOOSE_DEFAULT
    eta_t = a.eta_tight if getattr(a, "eta_tight", None) is not None else ETA_TIGHT_DEFAULT

    assign_fr = (fr_runs[0].get("summary") or {}).get("wm_trigger_assign", "?")
    assign_hon = (hon_runs[0].get("summary") or {}).get("wm_trigger_assign", "?") if hon_runs else "?"

    # FR cid -> its assigned trigger class (from the server-side rows)
    fr_class = {}
    for r in fr_runs:
        for h in r.get("history", []):
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider") and p.get("trigger_class") is not None:
                    fr_class[int(p["cid"])] = int(p["trigger_class"])
    if not fr_class:
        raise SystemExit(f"{fr_fam} has no free-riders in its history rows")

    def _fr_ber_by_round(cid):
        out = defaultdict(list)
        for r in fr_runs:
            for h in r.get("history", []):
                for p in (h.get("wm_per_client") or []):
                    if p.get("is_free_rider") and int(p.get("cid", -1)) == cid \
                            and p.get("ber") is not None:
                        out[h["round"]].append(float(p["ber"]))
        return out

    base = a.out or "ea_fair"
    base = base[:-4] if str(base).endswith(".png") else str(base)
    md_rows = []

    for cid in sorted(fr_class):
        tc = fr_class[cid]
        fr_by = _fr_ber_by_round(cid)
        twin_by = _honest_ber_by_round(hon_runs, tclass_filter=tc)   # SAME assigned class
        if not twin_by:
            print(f"  (warn: no honest twin on class {tc} in the honest family -- "
                  f"is EA1 assigning class {tc} to some honest client?)")

        fig, (axT, axB) = plt.subplots(
            2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2.2, 1]})

        # ---- top: BER over rounds, FR vs same-class honest twin ----
        fx = sorted(fr_by)
        axT.plot(fx, [np.mean(fr_by[rd]) for rd in fx], color=ps.C_FR, lw=2.4,
                 marker="o", ms=3, zorder=4,
                 label=f"reduced free-rider (EA2) \u00b7 cid{cid} \u00b7 class {tc}")
        if twin_by:
            tx = sorted(twin_by)
            axT.plot(tx, [np.mean(twin_by[rd]) for rd in tx], color=CYCLE[0], lw=2.2,
                     ls=(0, (4, 2)), zorder=3,
                     label=f"honest twin (EA1) assigned class {tc} \u2014 fair baseline")
            tlo = [np.percentile(twin_by[rd], 10) if len(twin_by[rd]) > 1 else twin_by[rd][0] for rd in tx]
            thi = [np.percentile(twin_by[rd], 90) if len(twin_by[rd]) > 1 else twin_by[rd][0] for rd in tx]
            axT.fill_between(tx, tlo, thi, color=CYCLE[0], alpha=.12, lw=0)
        axT.axhline(eta_l, color=ps.C_HONEST, ls="--", lw=1.6,
                    label=f"\u03b7 loose {eta_l:.3f} (operating line)")
        axT.axhline(eta_t, color="black", ls=":", lw=1.2,
                    label=f"\u03b7 tight {eta_t:.3f} (degenerate)")
        axT.set_ylabel("bit-error-rate")
        axT.set_ylim(-0.03, max(0.62, eta_l + 0.06))
        axT.grid(alpha=.3); axT.legend(fontsize=8.5, loc="upper right", framealpha=.95)

        # ---- bottom: converged per-client BER clouds + separability ----
        fr_tail, twin_tail = [], []
        for r in fr_runs:
            for h in r.get("history", [])[-tail:]:
                for p in (h.get("wm_per_client") or []):
                    if p.get("is_free_rider") and int(p.get("cid", -1)) == cid \
                            and p.get("ber") is not None:
                        fr_tail.append(float(p["ber"]))
        for r in hon_runs:
            for h in r.get("history", [])[-tail:]:
                for p in (h.get("wm_per_client") or []):
                    if (not p.get("is_free_rider")) and p.get("ber") is not None \
                            and int(p.get("trigger_class", -1)) == tc:
                        twin_tail.append(float(p["ber"]))
        ovl, balerr = _ovl_and_balerr(twin_tail, fr_tail, m)
        fr_mu = float(np.mean(fr_tail)) if fr_tail else float("nan")
        tw_mu = float(np.mean(twin_tail)) if twin_tail else float("nan")

        rng = np.random.default_rng(0)
        jit = lambda n: (rng.random(n) - 0.5) * 0.5
        if twin_tail:
            axB.scatter(np.zeros(len(twin_tail)) + jit(len(twin_tail)), twin_tail,
                        s=30, color=CYCLE[0], alpha=.5, edgecolor="none",
                        label=f"honest twin (class {tc})  mean {tw_mu:.3f}")
            axB.plot([-0.35, 0.35], [tw_mu, tw_mu], color=CYCLE[0], lw=2.5)
        if fr_tail:
            axB.scatter(np.ones(len(fr_tail)) + jit(len(fr_tail)), fr_tail,
                        s=30, color=ps.C_FR, alpha=.5, edgecolor="none",
                        label=f"reduced FR (class {tc})  mean {fr_mu:.3f}")
            axB.plot([0.65, 1.35], [fr_mu, fr_mu], color=ps.C_FR, lw=2.5)
        axB.axhline(eta_l, color=ps.C_HONEST, ls="--", lw=1.4)
        axB.set_xticks([0, 1]); axB.set_xticklabels(["honest twin\n(EA1)", "reduced FR\n(EA2)"])
        axB.set_ylabel("converged BER\n(per client-round, tail)")
        cleaner = ("FR cleaner \u2192 inseparable" if fr_mu <= tw_mu else "FR dirtier")
        verdict = ("inseparable (coin flip)" if (not np.isnan(balerr) and balerr >= 0.45)
                   else f"best balanced-error {balerr:.2f}")
        axB.set_title(f"converged separability  \u00b7  OVL {ovl:.2f}  \u00b7  "
                      f"best balanced-error {balerr:.2f}  ({verdict})  \u00b7  {cleaner}",
                      fontsize=9.5)
        axB.grid(alpha=.3); axB.legend(fontsize=8, loc="upper right")

        evade = ("UNDER \u03b7_loose (evades)" if (not np.isnan(fr_mu) and fr_mu < eta_l)
                 else "OVER \u03b7_loose")
        fig.suptitle(
            f"EA fair comparison \u00b7 {fr_fam.split('_rep')[0]} vs "
            f"{(a.honest_family or 'honest').split('_rep')[0]}  \u00b7  class {tc}\n"
            f"assignment: FR [{assign_fr}] / honest [{assign_hon}]  \u00b7  "
            f"FR tail-BER {fr_mu:.3f} vs honest-twin {tw_mu:.3f}  \u2192  {evade}",
            fontsize=11, y=1.0)
        fig.tight_layout()
        out = f"{base}_{fr_fam}_cls{tc}.png"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        ps.finish(fig, out)
        md_rows.append((cid, tc, tw_mu, fr_mu, ovl, balerr, evade))

    md = f"{base}_{fr_fam}.md"
    L = [f"# Group EA \u2014 distribution-aware fair comparison ({fr_fam})", "",
         f"FR family **{fr_fam}** [{assign_fr}] vs honest family "
         f"**{a.honest_family or '(auto)'}** [{assign_hon}]. Each free-rider is matched to "
         "the honest client assigned the **same** trigger class (the fair twin). Under "
         "distribution assignment the honest twin is NOT starved, so if the reduced FR still "
         "sits at/under it, removing starvation did **not** open a separating threshold \u2014 "
         "the overlap moved, it didn't open (the EA hypothesis).", "",
         "| FR cid | class | honest-twin BER | FR BER | OVL | best balanced-error | verdict |",
         "|---|---|---|---|---|---|---|"]
    for cid, tc, tw, fr_, ovl, be, ev in md_rows:
        L.append(f"| {cid} | {tc} | {tw:.3f} | {fr_:.3f} | {ovl:.2f} | {be:.2f} | {ev} |")
    open(md, "w").write("\n".join(L))
    print("wrote", md)


CMDS = {
    "class_acc": class_acc,                # NEW: per-client trigger-class vs non-trigger vs global test-acc (all-honest)
    "ea_fair": ea_fair,                    # NEW: Group EA distribution-aware fair FR-vs-same-class-honest comparison
    "gpu_inflation": gpu_inflation,        # NEW: single-tenant vs shared-GPU saved-% validation (ratio inflation-invariant)
    "operating_point": operating_point,    # NEW: recall @ fixed honest FPR across attacks (the money plot)
    "tap_perfr": tap_perfr,                # NEW: one FIGURE per free-rider (per class) -- taps/coasts, server BER vs self-probe, + same-class honest twin
    "gpu_savings": gpu_savings,            # NEW: cumulative gpu_ms per round, FR vs honest, compute saved
    "trigger_fairness": trigger_fairness,  # NEW: BER vs trigger-sample holdings (non-IID fairness)
    "dirichlet_dist": dirichlet_dist,      # NEW: reference heatmap of the Dirichlet partition per alpha
    "accuracy": accuracy,                  # NEW: global test acc attack-vs-honest + FR trigger-class acc
    "tap_dynamics": tap_dynamics,          # fade/recovery + stealth frontier (aggregate; use tap_perfr for per-cid)
    "eta_stability": eta_stability,        # per-seed BER curves + eta spread (threshold noise)
    "sanity": sanity,                      # TEXT: flag suspicious/degenerate runs first
    "class_difficulty": class_difficulty,  # CONFIRM harder class ids (acc/loss vs BER)
    "thresholds": thresholds,          # intuitive derivation of the ONE eta
    "class_dynamics": class_dynamics,  # loss/acc per class -> hard classes
    "positions": positions,            # per-class BER (easy vs hard)
    "fidelity": fidelity,              # accuracy + per-client BER + effort
    "timeline": timeline,              # BER over rounds, taps/coasts, eta lines
    "honest_fpr": honest_fpr,          # honest false-positive rate vs eta
    "honest_lines": honest_lines,      # MERGED: honest BER per class over rounds (was honest_class_lines.py)
    "class_probe": class_probe,        # MERGED: per-class BER vs predictors + correlations (was class_difficulty_probe.py)
    "separability": separability_plot, # NEW: honest vs FR BER overlap + threshold-regime FPR/recall
    "sweep": sweep_plot,               # NEW: +N free-riding spectrum (BER vs data budget)
    "threshold": threshold,            # (legacy) two-distribution soundness view
    # legacy sweep plots (kept for reuse)
    "frontier": frontier,
    "scorecard": scorecard,
    "test_data": test_data,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="all FareMark plotting in one place")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in list(CMDS) + ["all"]:
        s = sub.add_parser(name)
        s.add_argument("--in", dest="inp", nargs="+", required=True)
        s.add_argument("--family", default=None)
        s.add_argument("--out", default=None)
        s.add_argument("--title", default="")
        s.add_argument("--level", default=None)
        s.add_argument("--seed", default=None)
        s.add_argument("--families", nargs="+", default=None)
        s.add_argument("--honest_family", default=None)
        s.add_argument("--honest_in", nargs="+", default=None,
                       help="glob(s) of honest result.json for the per-class floor overlay "
                            "on the timeline (e.g. 'results/sub_16_3/*/result.json').")
        s.add_argument("--scope", default=None)
        s.add_argument("--tail", type=int, default=TAIL)
        s.add_argument("--eta", type=float, default=None)
        s.add_argument("--eta_tight", type=float, default=None,
                       help="tight (frozen) eta line on timelines; default = the run's "
                            "WM_ETA_FIXED, else ETA_TIGHT_DEFAULT.")
        s.add_argument("--eta_loose", type=float, default=None,
                       help="loose (pooled) eta line on timelines; default = pooled mu+3s "
                            "recomputed from --honest_in, else ETA_LOOSE_DEFAULT.")
        s.add_argument("--classes", default=None,
                       help="comma list to restrict trigger classes (honest_lines).")
        s.add_argument("--per-seed", dest="per_seed", action="store_true",
                       help="faint per-seed lines in honest_lines.")
        s.add_argument("--attack_family", default=None,
                       help="free-rider family for the separability plot.")
        s.add_argument("--csv", default=None, help="optional CSV out (class_probe).")
    a = ap.parse_args()
    if a.out is None:
        a.out = default_out(a.inp)
    # finish() creates the needed directory for both dir-style and prefix-style out
    # "all" = the current headline set
    if a.cmd == "all":
        for name in ("thresholds", "class_difficulty", "class_dynamics", "positions", "fidelity"):
            print(f"== {name} =="); CMDS[name](a)
    else:
        CMDS[a.cmd](a)