#!/usr/bin/env python
"""
class_difficulty_probe.py  --  "why are some trigger classes harder to watermark?"

WHAT THIS DOES (plain terms)
----------------------------
Every honest client in your runs is asked to hide a secret bit-string (its
"watermark") inside the model's softmax output on images of ONE class (its
"trigger class"). How well that succeeds is measured by the BER
(bit-error-rate): 0.0 = watermark recovered perfectly, 0.5 = pure coin-flip /
no watermark. Your honest runs show the *converged* BER is not the same for
every trigger class -- some sit near 0, one (cls 6) sits near 0.2. This script
takes your result.json files and tries to explain that spread: it lines up, per
trigger class, the BER against several candidate predictors and reports which
predictor actually tracks the BER (correlation).

Candidate predictors (all pulled straight from result.json, nothing recomputed):
  test_acc    per-class TEST accuracy of the final global model
              (result['per_class']). This is your LABMATE'S hypothesis:
              "classes the model classifies badly are harder to watermark."
  test_error  = 100 - test_acc  (same signal, flipped sign)
  test_loss   per-class cross-entropy of the final model (higher = harder class)
  trig_acc    fraction of the trigger IMAGES the model labels as the trigger
              class (how confidently/correctly THOSE specific images land)
  pmax        mean top-1 softmax probability on trigger images = "confidence".
              High pmax = the model dumps almost all probability on one class.
  entropy     mean softmax entropy on trigger images = "spread". High entropy =
              probability spread over many classes (a flat distribution).
  dominance   Eq.6/10 ratio f(p_max)/sum_j f(p_j): how much the single biggest
              (smoothed) probability dominates the projection the watermark is
              read from. The paper wants this < 0.5. This is the most
              *mechanistic* predictor -- it is literally the read-out margin.

WHY THESE (the mechanism, one paragraph)
----------------------------------------
The watermark bits are read from the SHAPE of the non-top softmax values on
trigger images. If the model is very confident on a class (pmax high / entropy
low / dominance high), almost all probability sits on one class and the rest
collapse to near-uniform tiny values -- there is no "shape" left to write bits
into, so bits become noise and BER rises. If the model is less peaky, the tail
carries structure the embedding loss can move, so BER falls. Crucially this is
about the softmax *shape*, NOT about whether the class is classified correctly,
so test accuracy need not predict BER at all. This script quantifies which
story your data actually supports.

INPUT  : one or more result.json produced by run_experiment.py (honest runs).
OUTPUT : a ranked correlation table on stdout, and an optional CSV.

USAGE
-----
  python class_difficulty_probe.py --in '/path/to/results/*/result.json'
  python class_difficulty_probe.py --in '/path/results/*/result.json' \
        --family honest_iid --tail 20 --csv /path/results/class_difficulty.csv

Only numpy is required.
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
import numpy as np


# --------------------------------------------------------------------- loading
def load(globs):
    runs = []
    for g in globs:
        for f in sorted(glob.glob(g)):
            try:
                runs.append((f, json.load(open(f))))
            except Exception as e:
                print(f"  (skip {f} -> {e})")
    return runs


def family(run):
    return (run.get("manifest", {}) or {}).get("family")


def is_honest(run):
    """No free-riders anywhere in the run."""
    if run.get("free_rider_indices"):
        return False
    for h in run.get("history", []):
        for p in (h.get("wm_per_client") or []):
            if p.get("is_free_rider"):
                return False
    return True


# ----------------------------------------------------------------- aggregation
def collect(runs, tail):
    """Return per-trigger-class dict of lists, pooled over runs & the converged
    tail. Keys: ber, pmax, entropy, dominance, trig_acc (watermark-side,
    per-round) and test_acc, test_loss (class-side, one value per run)."""
    ber = defaultdict(list)
    pmax = defaultdict(list)
    ent = defaultdict(list)
    dom = defaultdict(list)
    tacc = defaultdict(list)            # trig_acc (server diagnostic)
    test_acc = defaultdict(list)        # per_class final-model accuracy
    test_loss = defaultdict(list)

    for _f, r in runs:
        # ---- watermark-side, per honest client per round (converged tail) ----
        for h in r.get("history", [])[-tail:]:
            for p in (h.get("wm_per_client") or []):
                if p.get("is_free_rider"):
                    continue
                c = int(p["trigger_class"])
                ber[c].append(p["ber"])
                for src, key in ((pmax, "pmax"), (ent, "entropy"),
                                 (dom, "dominance"), (tacc, "trig_acc")):
                    v = p.get(key)
                    if v is not None:
                        src[c].append(v)
        # ---- class-side, one value per run (final global model) ----
        pc = (r.get("per_class") or {}).get("by_class") or {}
        for c, d in pc.items():
            c = int(c)
            if d.get("acc") is not None:
                test_acc[c].append(d["acc"])
            if d.get("loss") is not None:
                test_loss[c].append(d["loss"])
    return ber, pmax, ent, dom, tacc, test_acc, test_loss


# ------------------------------------------------------------------ statistics
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """Pearson on ranks (monotonic association, robust to outliers)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True,
                    help="glob(s) of result.json, e.g. '/path/results/*/result.json'")
    ap.add_argument("--family", default="honest_iid",
                    help="manifest family to restrict to (default honest_iid). "
                         "Pass '' to use every honest run regardless of family.")
    ap.add_argument("--tail", type=int, default=20,
                    help="converged window: last N rounds per run (default 20).")
    ap.add_argument("--csv", default=None, help="optional path to write a per-class CSV.")
    a = ap.parse_args()

    runs = load(a.inp)
    honest = [(f, r) for f, r in runs if is_honest(r)
              and (not a.family or family(r) == a.family)]
    if not honest:
        raise SystemExit("no honest runs matched (check --in / --family).")
    print(f"loaded {len(honest)} honest run(s)"
          + (f" in family '{a.family}'" if a.family else "")
          + f"; converged tail = last {a.tail} rounds\n")

    ber, pmax, ent, dom, tacc, test_acc, test_loss = collect(honest, a.tail)
    classes = sorted(ber)

    def m(dct, c):
        return float(np.mean(dct[c])) if dct.get(c) else float("nan")

    # ---- per-class table (means) --------------------------------------------
    rows = []
    for c in classes:
        acc = m(test_acc, c)
        rows.append(dict(cls=c, n=len(ber[c]), ber=m(ber, c),
                         test_acc=acc, test_error=(100 - acc if acc == acc else float("nan")),
                         test_loss=m(test_loss, c), trig_acc=m(tacc, c),
                         pmax=m(pmax, c), entropy=m(ent, c), dominance=m(dom, c)))
    rows.sort(key=lambda d: (float("inf") if d["ber"] != d["ber"] else d["ber"]))

    cols = ["cls", "n", "ber", "test_acc", "test_error", "test_loss",
            "trig_acc", "pmax", "entropy", "dominance"]
    w = {c: max(len(c), 8) for c in cols}
    print("PER-CLASS (sorted easy -> hard by BER):")
    print("  " + "  ".join(f"{c:>{w[c]}}" for c in cols))
    for d in rows:
        cells = []
        for c in cols:
            v = d[c]
            if isinstance(v, int):
                cells.append(f"{v:>{w[c]}d}")
            elif v != v:                       # NaN
                cells.append(f"{'--':>{w[c]}}")
            else:
                cells.append(f"{v:>{w[c]}.4f}")
        print("  " + "  ".join(cells))

    # ---- correlations: BER vs each predictor --------------------------------
    y = [d["ber"] for d in rows]
    predictors = ["test_acc", "test_error", "test_loss", "trig_acc",
                  "pmax", "entropy", "dominance"]
    corr = []
    for p in predictors:
        x = [d[p] for d in rows]
        mask = [xi == xi and yi == yi for xi, yi in zip(x, y)]   # drop NaN pairs
        xs = [xi for xi, ok in zip(x, mask) if ok]
        ys = [yi for yi, ok in zip(y, mask) if ok]
        corr.append((p, pearson(xs, ys), spearman(xs, ys), len(xs)))
    corr.sort(key=lambda t: (float("-inf") if t[1] != t[1] else -abs(t[1])))

    print(f"\nCORRELATION of per-class BER vs each predictor "
          f"(over {len(classes)} classes):")
    print(f"  {'predictor':>10}  {'pearson_r':>10}  {'spearman':>9}  {'n':>3}   note")
    hyp = {"test_acc": "labmate: low acc -> hard?",
           "test_error": "labmate (flipped)",
           "test_loss": "labmate: high loss -> hard?",
           "trig_acc": "confidence proxy",
           "pmax": "confidence (peakiness)",
           "entropy": "spread (flatness)",
           "dominance": "read-out margin (mechanistic)"}
    for p, r, rho, n in corr:
        rs = "  --  " if r != r else f"{r:+.3f}"
        rhos = "  --  " if rho != rho else f"{rho:+.3f}"
        print(f"  {p:>10}  {rs:>10}  {rhos:>9}  {n:>3}   {hyp[p]}")

    # ---- verdict ------------------------------------------------------------
    ranked = [t for t in corr if t[1] == t[1]]
    print("\nREADING IT:")
    if len(classes) < 8:
        print(f"  * Only {len(classes)} classes -> correlations are noisy; treat as directional.")
    acc_r = dict((p, r) for p, r, _, _ in corr).get("test_acc", float("nan"))
    if acc_r == acc_r and abs(acc_r) < 0.3:
        print(f"  * test_acc (labmate's idea) correlates weakly with BER (r={acc_r:+.2f}): "
              "class accuracy does NOT explain difficulty.")
    if ranked:
        best = ranked[0]
        print(f"  * Strongest predictor: {best[0]} (r={best[1]:+.2f}). "
              + ("Positive r with pmax/dominance/trig_acc, or negative r with entropy, "
                 "means peaky/confident classes are the hard ones."
                 if best[0] in ("pmax", "dominance", "trig_acc", "entropy") else ""))
    print("  * Watermark difficulty here is a softmax-SHAPE effect (peakiness), "
          "not a class-accuracy effect.")

    # ---- optional CSV -------------------------------------------------------
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for d in rows:
                fh.write(",".join(("" if d[c] != d[c] else str(d[c])) for c in cols) + "\n")
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
