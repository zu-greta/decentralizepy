#!/usr/bin/env python
"""iso_accuracy.py -- the accuracy companions to the iso BER plots (labmate Q3).

Fig A: accuracy on the TRIGGER samples vs round, honest vs free-rider.
       This is already logged per client per round as wm_per_client[i]["trig_acc"]
       (the verifier's accuracy on the held-out trigger bank). No re-run needed.

Fig B: accuracy on the REST OF THE TEST SET vs round.
       Only the GLOBAL model's test_acc is logged per round (history[r]["test_acc"]),
       one number per run -- NOT per client and NOT split trigger-vs-rest. So we plot
       the honest RUN's global test_acc vs the free-rider RUN's global test_acc. For a
       true per-client, trigger-excluded test accuracy you must add an eval hook in the
       verifier and re-run (see the note printed at the end).

    python iso_accuracy.py \
        --honest_in 'results/A1_honest_c100_rep*/result.json' \
        --fr_in     'results/A3_reduced_c100_c36_rep*/result.json' \
        --class 6 --out figs/iso_acc_c6
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
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


def trig_acc_series(runs, cls, free_rider):
    """round -> mean trig_acc over seeds, for the client whose trigger_class == cls
    and whose is_free_rider flag matches."""
    acc = defaultdict(list)
    for r in runs:
        for h in r.get("history", []):
            rd = h.get("round")
            for p in (h.get("wm_per_client") or []):
                if p.get("trigger_class") == cls and bool(p.get("is_free_rider")) == free_rider:
                    if p.get("trig_acc") is not None and rd:
                        acc[rd].append(float(p["trig_acc"]))
    xs = sorted(acc)
    return xs, [float(np.mean(acc[x])) for x in xs]


def test_acc_series(runs):
    """global model test accuracy per round (fallback)."""
    acc = defaultdict(list)
    for r in runs:
        for h in r.get("history", []):
            rd = h.get("round")
            if rd and h.get("test_acc") is not None:
                acc[rd].append(float(h["test_acc"]))
    xs = sorted(acc)
    return xs, [float(np.mean(acc[x])) for x in xs]


def nontrig_acc_series(runs, cls, free_rider):
    """per-client accuracy on the NON-trigger test set, if the verifier logs
    wm_per_client[i]['test_acc_nontrigger'] (see the hook in the chat message).
    Returns ([],[]) if the field isn't present."""
    acc = defaultdict(list)
    for r in runs:
        for h in r.get("history", []):
            rd = h.get("round")
            for p in (h.get("wm_per_client") or []):
                if p.get("trigger_class") == cls and bool(p.get("is_free_rider")) == free_rider:
                    v = p.get("test_acc_nontrigger")
                    if v is not None and rd:
                        acc[rd].append(float(v))
    xs = sorted(acc)
    return xs, [float(np.mean(acc[x])) for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--honest_in", nargs="+", required=True)
    ap.add_argument("--fr_in", nargs="+", required=True)
    ap.add_argument("--class", dest="cls", type=int, required=True)
    ap.add_argument("--out", default="iso_acc")
    a = ap.parse_args()

    honest = load(a.honest_in)
    fr = load(a.fr_in)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Fig A: trigger-sample accuracy ----
    hx, hy = trig_acc_series(honest, a.cls, free_rider=False)
    fx, fy = trig_acc_series(fr, a.cls, free_rider=True)
    if hx: axA.plot(hx, hy, lw=2, color="#1f77b4", label=f"honest (cls {a.cls})")
    if fx: axA.plot(fx, fy, lw=2, color="#D55E00", label=f"free-rider (cls {a.cls})")
    axA.set_title(f"Fig A — accuracy on TRIGGER samples · class {a.cls}")
    axA.set_xlabel("communication round"); axA.set_ylabel("trigger-class accuracy")
    axA.grid(alpha=.3); axA.legend()

    # ---- Fig B: prefer per-client non-trigger test acc; fall back to global ----
    hnx, hny = nontrig_acc_series(honest, a.cls, free_rider=False)
    fnx, fny = nontrig_acc_series(fr, a.cls, free_rider=True)
    if hnx or fnx:   # verifier hook present -> the RIGHT Fig B (per-client, trigger-excluded)
        if hnx: axB.plot(hnx, hny, lw=2, color="#1f77b4", label=f"honest (cls {a.cls}, non-trigger test)")
        if fnx: axB.plot(fnx, fny, lw=2, color="#D55E00", label=f"free-rider (cls {a.cls}, non-trigger test)")
        axB.set_title(f"Fig B — per-client accuracy on the NON-trigger test set · class {a.cls}")
        axB.set_ylabel("non-trigger test accuracy")
    else:            # fallback: global model accuracy per run
        hbx, hby = test_acc_series(honest); fbx, fby = test_acc_series(fr)
        if hbx: axB.plot(hbx, hby, lw=2, color="#1f77b4", label="honest run (global)")
        if fbx: axB.plot(fbx, fby, lw=2, color="#D55E00", label="free-rider run (global)")
        axB.set_title("Fig B — accuracy on the test set (global model per run) [add verifier hook for per-client]")
        axB.set_ylabel("test accuracy (%)")
    axB.set_xlabel("communication round")
    axB.grid(alpha=.3); axB.legend()

    fig.tight_layout()
    png = a.out if a.out.endswith(".png") else a.out + ".png"
    import os; os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")
    print("NOTE Fig B is the GLOBAL model's accuracy per run (all classes), not per-client and "
          "not trigger-excluded. For per-client, trigger-vs-rest test accuracy, add an eval hook "
          "in the verifier (it already holds each client's submitted local model) and re-run.")


if __name__ == "__main__":
    main()