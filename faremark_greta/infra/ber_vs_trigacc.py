#!/usr/bin/env python
"""ber_vs_trigacc.py -- the BER-vs-accuracy relationship in 2D (labmate Q3b).

Instead of a 3D (BER, trig_acc, round) plot, we collapse the round axis into COLOUR:
each point is one (client, round); x = watermark BER, y = trigger-class accuracy;
marker = honest (o) vs free-rider (x); colour = communication round.

This shows whether BER and trig_acc are related, and how honest vs FR separate, without
an uninterpretable 3D axis.

    python ber_vs_trigacc.py --in 'results/*/result.json' --class 7 --out figs/bva_c7
    python ber_vs_trigacc.py --in 'results/*/result.json' --out figs/bva_all   # all classes pooled
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(patterns):
    runs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                runs.append(json.load(open(f)))
            except Exception:
                pass
    return runs


def collect(runs, cls, tail):
    """-> dict with x=BER, y=trig_acc, r=round, fr=is_free_rider, for every client-round."""
    X, Y, R, FR = [], [], [], []
    for run in runs:
        hist = run.get("history", [])
        sel = hist[-tail:] if tail else hist
        for h in sel:
            rd = h.get("round")
            for p in (h.get("wm_per_client") or []):
                if cls is not None and p.get("trigger_class") != cls:
                    continue
                b, a = p.get("ber"), p.get("trig_acc")
                if b is None or a is None:
                    continue
                X.append(float(b)); Y.append(float(a)); R.append(int(rd or 0))
                FR.append(bool(p.get("is_free_rider")))
    return (np.array(X), np.array(Y), np.array(R), np.array(FR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--class", dest="cls", type=int, default=None,
                    help="restrict to one trigger class; omit to pool all")
    ap.add_argument("--tail", type=int, default=0, help="use only the last N rounds (0 = all)")
    ap.add_argument("--out", default="ber_vs_trigacc")
    a = ap.parse_args()

    runs = load(a.inp)
    X, Y, R, FR = collect(runs, a.cls, a.tail)
    if len(X) == 0:
        raise SystemExit("no (ber, trig_acc) points found -- check --in / --class")

    fig, ax = plt.subplots(figsize=(8, 6))
    for mask, mk, lab in [(~FR, "o", "honest"), (FR, "x", "free-rider")]:
        if mask.any():
            sc = ax.scatter(X[mask], Y[mask], c=R[mask], cmap="viridis",
                            marker=mk, s=36, alpha=.8, label=lab,
                            edgecolors="none" if mk == "x" else "k", linewidths=.3)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("communication round")
    ttl = "trigger-class accuracy vs watermark BER"
    ttl += f"  ·  class {a.cls}" if a.cls is not None else "  ·  all classes"
    if a.tail: ttl += f"  ·  last {a.tail} rounds"
    ax.set_xlabel("watermark BER  (0 = mark present, 0.5 = none)")
    ax.set_ylabel("trigger-class accuracy  (0 = argmax destroyed)")
    ax.set_title(ttl)
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    png = a.out if a.out.endswith(".png") else a.out + ".png"
    import os; os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    fig.savefig(png, dpi=150)
    print(f"wrote {png}  ({len(X)} points; "
          f"{int((~FR).sum())} honest, {int(FR.sum())} free-rider)")


if __name__ == "__main__":
    main()