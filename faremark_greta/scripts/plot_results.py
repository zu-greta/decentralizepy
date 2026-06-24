#!/usr/bin/env python3
"""Turn result.json file(s) into figures.

Every experiment should emit a graph, not just final numbers. Point this at a
single result.json or a directory of them (scanned recursively) and it writes
PNGs to --out.

Figures produced (whichever the data supports):
  1. ber_trajectory_<tag>.png   per-run benign vs free-rider BER over rounds,
                                with the eta threshold and detection accuracy.
  2. sweep.png                  auto-detects the swept variable (num_free_riders,
                                n_trigger_samples, attack_round, dataset, attack)
                                and plots converged det-acc / recall / FPR / BER.
  3. separability.png           THE thesis figure: the converged benign-BER and
                                free-rider-BER distributions side by side, plus a
                                detection-accuracy-vs-eta curve. If the two
                                distributions overlap, no eta separates them and
                                detection is impossible -- the curve shows it.

Usage:
  python scripts/plot_results.py --in results/ --out figs/
  python scripts/plot_results.py --in a.json b.json --out figs/ --window 10
"""
import argparse, json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- house style (kept neutral; no project-specific deps) ------------------
BENIGN = "#2a7de1"      # blue  = honest
FR = "#d1495b"          # red   = free-rider
ETA = "#3a3a3a"         # threshold line
GRID = "#e6e6e6"
plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})


def load_results(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "result.json"), recursive=True)
            files += glob.glob(os.path.join(p, "**", "*.json"), recursive=True)
        else:
            files.append(p)
    runs, seen = [], set()
    for f in sorted(set(files)):
        if f in seen:
            continue
        seen.add(f)
        try:
            d = json.load(open(f))
            if "history" in d and "config" in d:
                d["_path"] = f
                runs.append(d)
        except Exception as e:
            print(f"  skip {f}: {e}")
    return runs


def converged_pairs(run, window):
    """Return (benign_bers, fr_bers) pooled over the last `window` rounds."""
    hist = run["history"][-window:]
    b = [h["wm_benign_ber"] for h in hist if h.get("wm_benign_ber") is not None]
    f = [h["wm_fr_ber"] for h in hist if h.get("wm_fr_ber") is not None]
    return b, f


def tag_of(run):
    c = run["config"]
    return (f"{c['dataset']}_{run.get('attack','none')}"
            f"_fr{run.get('num_free_riders',0)}"
            f"_ns{c.get('n_trigger_samples','-')}"
            f"_ar{c.get('attack_round','-')}")


# ---- 1. per-run BER trajectory --------------------------------------------
def plot_trajectory(run, outdir):
    h = run["history"]
    rounds = [x["round"] for x in h]
    benign = [x.get("wm_benign_ber") for x in h]
    fr = [x.get("wm_fr_ber") for x in h]
    eta = [x.get("wm_eta_round") for x in h]
    det = [x.get("wm_detect_acc") for x in h]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.plot(rounds, benign, color=BENIGN, lw=2, label="honest BER")
    if any(v is not None for v in fr):
        ax1.plot(rounds, fr, color=FR, lw=2, label="free-rider BER")
    ax1.plot(rounds, eta, color=ETA, ls="--", lw=1.4, label="\u03b7 threshold")
    ax1.axhspan(0, 0, color="none")
    ax1.set_xlabel("communication round")
    ax1.set_ylabel("bit error rate")
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(rounds, det, color="#6aa84f", lw=1.2, alpha=0.7, label="detection acc")
    # global model test accuracy (normalised to [0,1]) so it shares the right axis
    tacc = [(x.get("test_acc") or 0) / 100.0 for x in h]
    ax2.plot(rounds, tacc, color="#e69138", lw=1.2, ls=":", alpha=0.9,
             label="global test acc /100")
    ax2.set_ylabel("detection acc  /  test acc(\u00f7100)", color="#6aa84f")
    ax2.set_ylim(-0.02, 1.05); ax2.grid(False)
    c = run["config"]
    ax1.set_title(f"{c['model']} / {c['dataset']} \u00b7 {run.get('attack')} "
                  f"\u00b7 {run.get('num_free_riders')} free-riders", fontsize=11)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(outdir, f"ber_trajectory_{tag_of(run)}.png")
    fig.savefig(out); plt.close(fig)
    return out


# ---- 2. swept-variable summary --------------------------------------------
SWEEP_KEYS = ["num_free_riders", "n_trigger_samples", "attack_round"]


def detect_sweep_key(runs):
    for k in SWEEP_KEYS:
        vals = {r["config"].get(k, r.get(k)) for r in runs}
        if len(vals) > 1:
            return k
    for k in ["dataset", "attack"]:
        vals = {r["config"].get(k) for r in runs}
        if len(vals) > 1:
            return k
    return None


def plot_sweep(runs, outdir, window):
    key = detect_sweep_key(runs)
    if key is None or len(runs) < 2:
        return None
    rows = []
    for r in runs:
        x = r["config"].get(key, r.get(key))
        rows.append((x, r.get("wm_detect_acc"), r.get("wm_fr_recall"),
                     r.get("wm_fpr"), r.get("wm_benign_ber"), r.get("wm_fr_ber")))
    numeric = all(isinstance(x[0], (int, float)) for x in rows)
    rows.sort(key=lambda t: t[0] if numeric else str(t[0]))
    xs = [str(r[0]) for r in rows]
    xi = range(len(xs))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    axL.plot(xi, [r[1] for r in rows], "o-", color="#6aa84f", label="detection acc")
    axL.plot(xi, [r[2] for r in rows], "s-", color=BENIGN, label="recall (TPR)")
    axL.plot(xi, [r[3] for r in rows], "^-", color=FR, label="FPR")
    axL.set_xticks(list(xi)); axL.set_xticklabels(xs)
    axL.set_ylim(-0.03, 1.05); axL.set_xlabel(key); axL.set_ylabel("rate")
    axL.set_title("detection vs " + key); axL.legend(fontsize=9)

    axR.plot(xi, [r[4] for r in rows], "o-", color=BENIGN, label="honest BER")
    axR.plot(xi, [r[5] for r in rows], "o-", color=FR, label="free-rider BER")
    axR.axhline(0.25, color=ETA, ls="--", lw=1.2, label="\u03b7 cap = 0.25")
    axR.set_xticks(list(xi)); axR.set_xticklabels(xs)
    axR.set_ylim(-0.02, 0.75); axR.set_xlabel(key); axR.set_ylabel("converged BER")
    axR.set_title("BER separation vs " + key); axR.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(outdir, "sweep.png")
    fig.savefig(out); plt.close(fig)
    return out


# ---- 3. separability / eta analysis (the thesis figure) -------------------
def plot_separability(runs, outdir, window):
    benign_all, fr_all = [], []
    for r in runs:
        b, f = converged_pairs(r, window)
        benign_all += b; fr_all += f
    benign_all = np.array([x for x in benign_all if x is not None], float)
    fr_all = np.array([x for x in fr_all if x is not None], float)
    if len(benign_all) == 0 or len(fr_all) == 0:
        return None

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # left: the two distributions
    bins = np.linspace(0, 1, 26)
    axL.hist(benign_all, bins=bins, color=BENIGN, alpha=0.7, label=f"honest (n={len(benign_all)})", density=True)
    axL.hist(fr_all, bins=bins, color=FR, alpha=0.6, label=f"free-rider (n={len(fr_all)})", density=True)
    mu, sd = benign_all.mean(), benign_all.std()
    eta_3s = mu + 3 * sd
    axL.axvline(eta_3s, color=ETA, ls="--", lw=1.5, label=f"\u03bc+3\u03c3 = {eta_3s:.3f}")
    axL.axvline(0.25, color="#999", ls=":", lw=1.5, label="cap = 0.25")
    axL.set_xlabel("bit error rate (converged)"); axL.set_ylabel("density")
    axL.set_title("honest vs free-rider BER distributions"); axL.legend(fontsize=9)

    # right: detection accuracy as a function of eta (separability curve)
    etas = np.linspace(0, 1, 201)
    labels = np.concatenate([np.zeros(len(benign_all)), np.ones(len(fr_all))])
    bers = np.concatenate([benign_all, fr_all])
    accs = []
    for e in etas:
        pred_fr = bers >= e
        tp = np.sum(pred_fr & (labels == 1)); tn = np.sum(~pred_fr & (labels == 0))
        accs.append((tp + tn) / len(labels))
    accs = np.array(accs)
    best_i = int(np.argmax(accs))
    best_eta, best_acc = etas[best_i], accs[best_i]
    overlap = benign_all.max() >= fr_all.min()   # distributions touch/overlap?

    axR.plot(etas, accs, color="#6a4ca8", lw=2)
    axR.axvline(best_eta, color="#6a4ca8", ls="--", lw=1.3,
                label=f"best \u03b7 = {best_eta:.3f} \u2192 acc {best_acc:.2f}")
    axR.axvline(eta_3s, color=ETA, ls="--", lw=1.2, label=f"\u03bc+3\u03c3 = {eta_3s:.3f}")
    axR.scatter([benign_all.max()], [0.5], color=BENIGN, zorder=5, s=20)
    axR.scatter([fr_all.min()], [0.5], color=FR, zorder=5, s=20)
    axR.set_xlabel("\u03b7 threshold"); axR.set_ylabel("detection accuracy")
    axR.set_ylim(0, 1.03)
    verdict = ("OVERLAP \u2192 no clean \u03b7 (separation impossible here)"
               if overlap else "SEPARABLE \u2192 a margin of \u03b7 gives perfect detection")
    axR.set_title(verdict, fontsize=10,
                  color=(FR if overlap else "#3a7d34"))
    axR.legend(fontsize=9, loc="lower center")
    fig.tight_layout()
    out = os.path.join(outdir, "separability.png")
    fig.savefig(out); plt.close(fig)
    print(f"  honest BER: mu={mu:.3f} sd={sd:.3f} max={benign_all.max():.3f} | "
          f"fr BER min={fr_all.min():.3f} | margin={fr_all.min()-benign_all.max():+.3f} "
          f"({'overlap' if overlap else 'separable'})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--window", type=int, default=10, help="converged-window size")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_results(args.inp)
    print(f"loaded {len(runs)} run(s)")
    if not runs:
        return
    made = []
    for r in runs:                       # one trajectory per run
        made.append(plot_trajectory(r, args.out))
    s = plot_sweep(runs, args.out, args.window)
    if s: made.append(s)
    sep = plot_separability(runs, args.out, args.window)
    if sep: made.append(sep)
    print("wrote:")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()