#!/usr/bin/env python
"""
honest_class_lines.py -- honest client BER over rounds, ONE line per trigger class.

This is the per-class version of the timeline's honest-floor overlay: the overlay
band collapses each class to a single converged number (mean of the last --tail
rounds); this plot shows the whole trajectory so you can see how each trigger
class converges and where it floors. The right-hand end of each line == that
class's value in the overlay band.

Pulls, per honest client, per round: trigger_class + ber (from
history[*].wm_per_client, is_free_rider=False). With round-robin assignment
(cid == class) there is one honest client per class per seed, so each line is the
mean over seeds (optionally with a ±std band or faint per-seed lines).

USAGE
  python honest_class_lines.py --in 'results/threshold_calibrate/*/result.json' \
       --family honest_iid --tail 20 \
       --out results/threshold_calibrate/figs/honest_class_lines.png

  # only the free-rider positions you want to compare against an overlay:
  python honest_class_lines.py --in '.../*/result.json' --classes 1,7 --out fig.png

  # draw the calibrated eta line too:
  python honest_class_lines.py --in '.../*/result.json' --eta 0.06397 --out fig.png




  python scripts/honest_class_lines.py \
  --in 'results/threshold_calibrate/*/result.json' \
  --family honest_iid --tail 20 --eta 0.06397 \
  --out results/sub_17/figs/honest_class_lines.png
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(globs):
    out = []
    for g in globs:
        for f in sorted(glob.glob(g)):
            try:
                out.append(json.load(open(f)))
            except Exception as e:
                print(f"  (skip {f} -> {e})")
    return out


def family(r):
    return (r.get("manifest", {}) or {}).get("family")


def is_honest(r):
    if r.get("free_rider_indices"):
        return False
    for h in r.get("history", []):
        for p in (h.get("wm_per_client") or []):
            if p.get("is_free_rider"):
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--family", default=None,
                    help="restrict to this manifest family (e.g. honest_iid). "
                         "Omit to use every honest run found.")
    ap.add_argument("--classes", default=None,
                    help="comma list to restrict (e.g. '1,7'); default = all present.")
    ap.add_argument("--tail", type=int, default=20,
                    help="converged window shaded on the right + used for the floor label.")
    ap.add_argument("--eta", type=float, default=None,
                    help="draw a horizontal calibrated-eta line for reference.")
    ap.add_argument("--per-seed", action="store_true",
                    help="also draw a faint line per seed (not just the seed-mean).")
    ap.add_argument("--out", default="honest_class_lines.png")
    a = ap.parse_args()

    runs = [r for r in load(a.inp) if is_honest(r)
            and (a.family is None or family(r) == a.family)]
    if not runs:
        raise SystemExit("no honest runs matched (check --in / --family).")
    only = set(int(c) for c in a.classes.split(",")) if a.classes else None

    # class -> round -> [ber over seeds];  and per-seed: class -> seed_idx -> {round: ber}
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
        raise SystemExit("no matching trigger classes.")
    rounds = list(range(1, max_round + 1))
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(11, 6.2))

    # shade the converged tail used for the floor / overlay
    if a.tail and a.tail > 0 and max_round > a.tail:
        ax.axvspan(max_round - a.tail + 0.5, max_round + 0.5,
                   color="#DDDDDD", alpha=0.35, lw=0,
                   label=f"converged tail (last {a.tail})")

    # per-class mean line (+ optional per-seed faint lines) + floor label
    floors = {}
    for i, c in enumerate(classes):
        col = cmap(i % cmap.N)
        mean = np.array([np.mean(by_cr[c][rd]) if by_cr[c].get(rd) else np.nan
                         for rd in rounds])
        std = np.array([np.std(by_cr[c][rd]) if by_cr[c].get(rd) else np.nan
                        for rd in rounds])
        if a.per_seed:
            for si in per_seed[c]:
                ys = [per_seed[c][si].get(rd, np.nan) for rd in rounds]
                ax.plot(rounds, ys, color=col, lw=0.6, alpha=0.20)
        else:
            ax.fill_between(rounds, mean - std, mean + std, color=col, alpha=0.12, lw=0)

        # converged floor = mean over last `tail` rounds
        tailvals = [np.mean(by_cr[c][rd]) for rd in rounds[-a.tail:] if by_cr[c].get(rd)]
        floor = float(np.mean(tailvals)) if tailvals else float("nan")
        floors[c] = floor
        ax.plot(rounds, mean, color=col, lw=2.2,
                label=f"cls {c}  (floor {floor:.3f})")
        # label the floor at the right edge
        if floor == floor:
            ax.annotate(f"{floor:.2f}", xy=(rounds[-1], mean[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=col)

    if a.eta is not None:
        ax.axhline(a.eta, color="black", ls="--", lw=2,
                   label=f"calibrated η = {a.eta:.3f}")

    ax.set_xlabel("communication round")
    ax.set_ylabel("honest bit-error-rate (lower = mark embeds)")
    ttl = f"Honest BER per trigger class  ·  {a.family or 'honest'}  ·  {len(runs)} seeds"
    if only:
        ttl += f"  ·  classes {sorted(only)}"
    ax.set_title(ttl)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("converged floors (== overlay band values):")
    for c in classes:
        print(f"  cls {c}: {floors[c]:.4f}")


if __name__ == "__main__":
    main()
