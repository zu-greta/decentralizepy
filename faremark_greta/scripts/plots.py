"""
plots:
  threshold   Is the mu+3sigma calculation SOLID? Shows WHY honest points sit
              above the "tight" eta line even though 3-sigma "should" cover
              ~99.7%: the tight eta is mu+3sigma over ROUND-MEAN BER (variance
              shrunk by ~sqrt(#clients)), but you test it against PER-CLIENT
              BER (full variance). Also shows BER quantisation/skew (why the
              Gaussian 99.7% never holds exactly) and the swingy cumulative eta.

  positions   Are some trigger classes harder? Per-trigger-class BER (bar +
              over-time), so the bimodal honest floor (a few hard classes) that
              inflates the loose eta is visible. TODO: dont name it position, its class index.

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

import threshold as th

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

GREY = OK.get("grey", "#888888")
BLACK = OK.get("black", "#000000")
TAIL = 20   # "converged" window = last N rounds


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
    axA.axhline(eta, color=C_BAD, lw=2.4, ls="--",
                label=f"eta = mu + 3 sigma = {eta:.3f}")
    if frozen is not None:
        axA.axhline(frozen, color=C_GOOD, lw=2, ls="-",
                    label=f"FROZEN constant (used) = {frozen:.3f}")
    axA.set_xlabel(f"round index within converged tail (last {tail}, pooled over seeds)")
    axA.set_ylabel("bit-error-rate")
    axA.set_title("(a) How eta is built: average clients within each round -> dots,\n"
                  "then mu + 3 sigma OF THOSE DOTS across rounds")
    axA.legend(fontsize=8, loc="upper right")

    # (b) where eta lands vs honest & free-rider
    axB.hist(ho, bins=np.arange(-0.05, max(ho.max() if len(ho) else 0.5,
                                           fr.max() if len(fr) else 0.5) + 0.15, 0.1),
             color=C_HONEST, alpha=0.6, density=True, label=f"honest per-client (n={len(ho)})")
    if len(fr):
        axB.hist(fr, bins=np.arange(-0.05, max(fr.max(), 0.5) + 0.15, 0.1),
                 color=C_FR, alpha=0.55, density=True, label=f"free-rider per-client (n={len(fr)})")
    axB.axvline(eta, color=C_BAD, lw=2.4, ls="--", label=f"eta = {eta:.3f}")
    if frozen is not None:
        axB.axvline(frozen, color=C_GOOD, lw=2, label=f"frozen = {frozen:.3f}")
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
        axA.text(0.5, 0.5, "no client-side wm_stats yet\n(re-run with updated wm_client)",
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
        print("        the updated wm_client + wm_verify to populate these panels.")
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
    for r in runs:
        h_means, f_means = [], []
        for h in r.get("history", []):
            pcs = (h.get("wm_per_client") or [])
            h_vals = [p["ber"] for p in pcs if not p.get("is_free_rider")]
            f_vals = [p["ber"] for p in pcs if p.get("is_free_rider")]
            h_means.append(np.mean(h_vals) if h_vals else np.nan)
            f_means.append(np.mean(f_vals) if f_vals else np.nan)
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

    if E["eta_tight"]: ax.axhline(E["eta_tight"], color=BLACK, ls="--", lw=2,
        label=f"fair η tight (round-mean) = {E['eta_tight']:.3f}")
    if E["eta_loose"]: ax.axhline(E["eta_loose"], color=GREY, ls=":", lw=1.8,
        label=f"loose η (per-client) = {E['eta_loose']:.3f}")

    live_eta = [h.get("wm_eta_round") for h in hist]
    ax.plot(rounds, live_eta, color="#E69F00", linestyle="-.", lw=1.5, alpha=0.9, label="Server Live η (cumulative μ+3σ)")

    # Calculate FR Estimated eta (uses first run's trace usually, but if aggregated, average them)
    fr_est_vals = []
    for r in runs:
        pc = (r.get("compute", {}) or {}).get("per_client", {}) or {}
        for c in pc.values():
            if c.get("is_free_rider"):
                est = [t["eta_frozen"] for t in c.get("trace", []) if t.get("action") == "calib" and t.get("eta_frozen") is not None]
                fr_est_vals.extend(est)
                break
    if fr_est_vals:
        fr_est_eta = np.mean(fr_est_vals)
        ax.axhline(fr_est_eta, color="#CC79A7", linestyle="-", lw=1.8,
                   label=f"FR Estimated η (self-calib) = {fr_est_eta:.3f}")

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
        note += f"\n(Config cpc={lvl(r_ref)})"
        ax.text(0.02, 0.05, note, transform=ax.transAxes, fontsize=9, 
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))

    ax.set_xlabel("communication round"); ax.set_ylabel("bit-error-rate (lower = mark present)")
    agg_seed_str = f"aggregated over {num_seeds} seeds" if is_aggregated else f"seed={r_ref.get('seed')}"
    ax.set_title(a.title or f"BER vs round  ·  {fam(r_ref)}  ·  cpc={lvl(r_ref)}  ·  {agg_seed_str}")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    note = ("η frozen on the calibration window (green, all clients honest):  tight = μ+3σ of the "
            "per-round mean BER;  loose = μ+3σ of all per-client BERs.\n"
            "FR Estimated η is the free-rider's own calculated μ+3σ from its observation during forced warmup.")
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


CMDS = {
    # canonical / current
    "class_difficulty": class_difficulty,  # CONFIRM harder class ids (acc/loss vs BER)
    "thresholds": thresholds,          # intuitive derivation of the ONE eta
    "class_dynamics": class_dynamics,  # loss/acc per class -> hard classes
    "positions": positions,            # per-class BER (easy vs hard)
    "fidelity": fidelity,              # accuracy + per-client BER + effort
    "timeline": timeline,              # BER over rounds, taps/coasts, eta lines
    "honest_fpr": honest_fpr,          # honest false-positive rate vs eta
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
        s.add_argument("--scope", default=None)
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