#!/usr/bin/env python
"""paper_check -- compare against FareMark paper's published rows.

PAPER ROWS
----------
  c10   Table I+II   ResNet-18 / CIFAR-10  /  10 clients   wm 99.72 / acc 90.78
  c100  Table I+II   ResNet-18 / CIFAR-100 / 100 clients   wm 99.71 / acc 75.31
  t9    Table IX     ResNet-18 / CIFAR-10  /  50 clients   wm 95.78 / acc 88.42

USAGE
  python scripts/paper_check.py --row t9 --in 'results/*/result.json' \\
      --family paper_t9_nc50_client_train [--heldout-family paper_t9_nc50_class]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resultio as rio  # noqa: E402

# (paper watermark acc %, paper main-task acc %, clients, config idx, trigger mode, label)
ROWS = {
    "t9":   (95.78, 88.42,  50, 11, "client_train",
             "Table IX   ResNet-18 / CIFAR-10 / 50 clients (capacity)"),
    "c10":  (99.72, 90.78,  10, 11, "class",
             "Table I+II ResNet-18 / CIFAR-10 / 10 clients"),
    "c100": (99.71, 75.31, 100, 14, "class",
             "Table I+II ResNet-18 / CIFAR-100 / 100 clients"),
}


def _mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


def _sd(v):
    return float(np.std(v)) if len(v) > 1 else 0.0


def report(runs, label, row, want_mode, expect_clients, expect_nt, expect_rounds):
    """One family vs one paper row. Returns {wm, acc} or None."""
    p_wm, p_acc = row[0], row[1]
    if not runs:
        print(f"\n(no runs yet for this family)")
        return None
    print(f"\n=== {label} ===")
    print(f"seeds found: {len(runs)}")

    r0 = rio.runs_only(runs)[0]
    cpc = rio.trigger_classes(r0)
    n_cl = sum(cpc.values())
    lo, hi = (min(cpc.values()), max(cpc.values())) if cpc else (0, 0)

    print("\nSETUP (vs paper)")

    def chk(name, got, want):
        ok = "ok " if (want is None or str(got) == str(want)) else "!! "
        print(f"  {ok}{name:<26} {str(got):<14} paper: {want}")

    chk("clients", n_cl, expect_clients)
    chk("clients per trigger class", lo if lo == hi else f"{lo}-{hi}",
        expect_clients // 10 if expect_clients >= 10 else 1)
    chk("trigger samples N_T", rio.cfg(r0, "wm_num_triggers"), expect_nt)
    chk("rounds", rio.cfg(r0, "rounds"), expect_rounds)
    chk("local epochs", rio.cfg(r0, "local_epochs"), 5)
    chk("lr / batch", f"{rio.cfg(r0,'lr')} / {rio.cfg(r0,'batch_size')}", "0.01 / 16")
    chk("free-riders", len(r0.get("free_rider_indices") or []), 0)
    chk("trigger mode", rio.cfg(r0, "wm_trigger_mode") or "class", want_mode)

    # structural watermark ceiling: with random keys, a fraction of key rows are
    # all-same-sign and can never be embedded, capping achievable accuracy.
    s = rio.summary_of(r0)
    m, l = s.get("wm_bits_m"), s.get("wm_group_size_l")
    un = s.get("wm_unembeddable_frac")
    if m:
        print(f"     watermark bits m={m}, group l={l}, unembeddable rows={un}")
        if un is not None:
            print(f"     -> structural wm-accuracy ceiling ~ {100*(1-0.5*un):.2f}%  "
                  f"(paper reports {p_wm})")

    fin_wm = [v for v in (rio.wm_accuracy(r, 1) for r in rio.runs_only(runs)) if v is not None]
    fin_ac = [v for v in (rio.test_acc(r, 1) for r in rio.runs_only(runs)) if v is not None]
    t_wm = [v for v in (rio.wm_accuracy(r, 10) for r in rio.runs_only(runs)) if v is not None]
    t_ac = [v for v in (rio.test_acc(r, 10) for r in rio.runs_only(runs)) if v is not None]

    print(f"\nRESULTS (mean over {len(runs)} seed(s))")
    print(f"  {'metric':<24}{'paper':>8}{'yours':>9}{'+/-':>7}{'diff':>8}   {'tail-10':>8}")
    for name, fv, tv, paper in (("watermark accuracy %", fin_wm, t_wm, p_wm),
                                ("classification acc %", fin_ac, t_ac, p_acc)):
        mv = _mean(fv)
        print(f"  {name:<24}{paper:>8.2f}{mv:>9.2f}{_sd(fv):>7.2f}"
              f"{mv - paper:>+8.2f}   {_mean(tv):>8.2f}")

    ok_wm = abs(_mean(fin_wm) - p_wm) <= 2.0
    ok_ac = abs(_mean(fin_ac) - p_acc) <= 2.0
    print(f"\n  VERDICT: watermark {'MATCH' if ok_wm else 'OFF'} | "
          f"classification {'MATCH' if ok_ac else 'OFF'}  (tolerance +/-2pp)")
    if not ok_wm and want_mode == "client_train":
        print("   - wm far below? check m/l above: m=5(l=2)->~75% ceiling, m=10(l=1)->~50%.")
        print("   - wm ~50%? the mark is not embedding at all (check wm_lambda, trigger mode).")
    if not ok_wm and want_mode == "class":
        print("   - (expected: this is the held-out control, NOT the paper's protocol.)")
    if not ok_ac:
        print("   - acc low? more clients = smaller shards; FedAvg needs the full 50 rounds.")
    return {"wm": _mean(fin_wm), "acc": _mean(fin_ac)}


def main():
    ap = argparse.ArgumentParser(description="grade runs against the FareMark paper")
    ap.add_argument("--row", choices=sorted(ROWS), default="t9")
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--heldout-family", default=None,
                    help="optional held-out-bank twin (trigger mode 'class') for the "
                         "memorisation-vs-generalisation gap")
    ap.add_argument("--clients", type=int, default=None)
    ap.add_argument("--nt", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=50)
    a = ap.parse_args()

    row = ROWS[a.row]
    nclients = a.clients or row[2]
    all_runs = rio.load(a.inp, with_path=False)
    if not all_runs:
        sys.exit(f"no result.json under {a.inp}")
    print(f"== {row[5]} ==")

    main_runs = rio.select(all_runs, family=a.family)
    res_a = report(main_runs, "PAPER-FAITHFUL (trigger-sample consistency, V-F3)",
                   row, row[4], nclients, a.nt, a.rounds)

    res_b = None
    if a.heldout_family:
        ho = rio.select(all_runs, family=a.heldout_family)
        res_b = report(ho, "HELD-OUT BANK (generalisation control)",
                       row, "class", nclients, a.nt, a.rounds)

    if res_a and res_b:
        print("\n=== memorisation gap ===")
        print(f"  watermark acc: paper-mode {res_a['wm']:.2f}%  vs held-out "
              f"{res_b['wm']:.2f}%   -> gap {res_a['wm'] - res_b['wm']:+.2f} pp")
        print("  a large positive gap = the mark is memorised on the client's own trigger")
        print("  images, not a generalising property of the class (paper Table V caveat).")


if __name__ == "__main__":
    main()