"""plot_tests.py — plots for the 3-test suite (see run_tests.sh).

Reads result.json files (schema written by run_experiment.py):
  run["manifest"]  = {family, sweep_var, sweep_level, note}
  run["history"]   = [{round, wm_benign_ber, wm_fr_ber, wm_eta_round,
                       wm_benign_ber_list, wm_fr_ber_list,
                       wm_per_client:[{cid, trigger_class, ber, is_free_rider, flagged}]}]
  run["compute"]["summary"] = {honest_mean_gpu_ms, fr_mean_gpu_ms,
                               honest_mean_samples, fr_mean_samples, ...}

Two subcommands:
  test1_fpr   per-client honest BER vs TWO eta definitions + false-positive rate
  test_data   per-FR & per-honest BER (each distinguishable) + GPU/samples effort,
              swept over training-data amount (autop_common_per_class)

NOTE: written against the schema but not executed here — smoke-test on one
result.json before the full run.  3 seeds expected; std shown as error bars/bands.
"""
import json, glob, sys, argparse, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "faremark"); sys.path.insert(0, "scripts")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotstyle as ps
ps.apply()

CONVERGED_TAIL = 20   # use the last N rounds as the "converged" window


def load(globs):
    out = []
    for g in globs:
        for f in sorted(glob.glob(g)):
            try:
                out.append(json.load(open(f)))
            except Exception:
                pass
    return out


def pick(runs, family):
    return [r for r in runs if (r.get("manifest", {}) or {}).get("family") == family]


def mu3s(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    mu = float(np.mean(xs))
    sd = float(np.std(xs)) if len(xs) > 1 else 0.0
    return mu + 3.0 * sd


# =====================================================================
# TEST 1 — honest false-positive check
# =====================================================================
def test1_fpr(a):
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
    eta_roundmean = mu3s(round_means)    # mu+3sigma over per-round MEANS  (as coded)
    eta_perclient = mu3s(all_indiv)      # mu+3sigma over INDIVIDUAL client BERs (alternative)

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
    ax.axhline(eta_perclient, color=ps.C_GOOD, ls="-", lw=2.2,
               label=f"η = μ+3σ over PER-CLIENT BERs = {eta_perclient:.3f}  (alt → FPR {fpr_pc:.0%})")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"cls {c}" for c in classes])
    ax.set_xlabel("trigger class (position)")
    ax.set_ylabel("honest bit-error-rate (converged rounds)")
    ax.set_title("TEST 1 — honest clients vs η.  Points above a line = that η flags honest clients.\n"
                 "round-mean η catches hard positions AND false-positives; per-client η spares them but is looser")
    ax.legend(loc="upper right", fontsize=8)
    ps.finish(fig, a.out + ".png")
    print(f"eta_roundmean={eta_roundmean:.4f} FPR={fpr_rm:.3f} | "
          f"eta_perclient={eta_perclient:.4f} FPR={fpr_pc:.3f}")


# =====================================================================
# TEST 2 / 3 — data sweep: per-FR & per-honest BER + effort
# =====================================================================
def _level_key(r):
    m = r.get("manifest", {}) or {}
    lv = m.get("sweep_level")
    if lv is None:                       # fall back to the config field
        lv = (r.get("config", {}) or {}).get("autop_common_per_class")
    try:
        return float(lv)
    except (TypeError, ValueError):
        return None


def _label_for_level(lv):
    if lv is None:
        return "?"
    if lv < 0:
        return "full\nshard"
    if lv == 0:
        return "triggers\nonly"
    return f"+{int(lv)}/cls"


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

    fr_mean, fr_std, ho_mean, eta_line = [], [], [], []
    fr_indiv = {}          # level_index -> list of per-FR mean BERs (each FR distinguishable)
    g_ms_fr, g_ms_ho, s_fr, s_ho = [], [], [], []

    for li, lv in enumerate(order):
        rs = levels[lv]
        fr_vals, ho_vals, etas = [], [], []
        per_fr = {}        # cid -> [ber across seeds/rounds]
        gmf, gmh, smf, smh = [], [], [], []
        for r in rs:
            hist = r.get("history", [])[-CONVERGED_TAIL:]
            for h in hist:
                for p in (h.get("wm_per_client") or []):
                    if p.get("is_free_rider"):
                        fr_vals.append(p["ber"]); per_fr.setdefault(p["cid"], []).append(p["ber"])
                    else:
                        ho_vals.append(p["ber"])
                if h.get("wm_eta_round") is not None:
                    etas.append(h["wm_eta_round"])
            cs = (r.get("compute", {}) or {}).get("summary", {}) or {}
            if cs.get("fr_mean_gpu_ms") is not None:      gmf.append(cs["fr_mean_gpu_ms"])
            if cs.get("honest_mean_gpu_ms") is not None:  gmh.append(cs["honest_mean_gpu_ms"])
            if cs.get("fr_mean_samples") is not None:     smf.append(cs["fr_mean_samples"])
            if cs.get("honest_mean_samples") is not None: smh.append(cs["honest_mean_samples"])

        fr_mean.append(np.mean(fr_vals) if fr_vals else np.nan)
        fr_std.append(np.std(fr_vals) if fr_vals else 0.0)
        ho_mean.append(np.mean(ho_vals) if ho_vals else np.nan)
        eta_line.append(np.mean(etas) if etas else np.nan)
        fr_indiv[li] = [np.mean(v) for v in per_fr.values()]
        g_ms_fr.append(np.mean(gmf) if gmf else np.nan); g_ms_ho.append(np.mean(gmh) if gmh else np.nan)
        s_fr.append(np.mean(smf) if smf else np.nan);   s_ho.append(np.mean(smh) if smh else np.nan)

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
    axB.plot(x, eta_line, color=ps.OKABE["black"], ls="--", lw=2, label="server η (mean)")
    axB.set_ylabel("bit-error-rate\n(converged)")
    axB.set_title(a.title or "per-free-rider & per-honest BER vs training-data amount")
    axB.legend(loc="upper right", fontsize=8)

    # --- panel 2: GPU-ms effort (scope-sensitive) ---
    axG.plot(x, g_ms_fr, color=ps.C_FR, lw=2.4, marker="o", label="free-rider GPU-ms/round")
    axG.plot(x, g_ms_ho, color=ps.C_HONEST, lw=2.4, marker="s", label="honest GPU-ms/round")
    axG.set_ylabel("GPU-ms\nper round")
    axG.legend(loc="upper right", fontsize=8)

    # --- panel 3: samples effort (scope-blind) ---
    axS.plot(x, s_fr, color=ps.C_FR, lw=2.4, marker="o", label="free-rider samples/round")
    axS.plot(x, s_ho, color=ps.C_HONEST, lw=2.4, marker="s", label="honest samples/round")
    axS.set_ylabel("image-passes\nper round")
    axS.set_xlabel("training data per round (triggers-only → +N/common-class → full shard)")
    axS.set_xticks(x); axS.set_xticklabels(xlabels)
    axS.legend(loc="upper left", fontsize=8)

    ps.finish(fig, a.out + ".png")
    print("levels:", xlabels)
    print("fr_mean_BER:", [round(v, 3) for v in fr_mean])
    print("ho_mean_BER:", [round(v, 3) for v in ho_mean])


# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("test1_fpr", "test_data"):
        s = sub.add_parser(name)
        s.add_argument("--in", dest="inp", nargs="+", required=True)
        s.add_argument("--family", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--title", default="")
        s.add_argument("--scope", default=None)   # descriptive only
    a = ap.parse_args()
    {"test1_fpr": test1_fpr, "test_data": test_data}[a.cmd](a)
