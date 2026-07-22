#!/usr/bin/env python
"""
TODO: remove file if not used?

Aggregate result.json files into mean +/- std summaries

One RunAI job writes one result.json into its own timestamped directory. This
script walks a results root, finds every result.json, and groups them so you
get the paper's table format without merging anything by hand

Grouping key = (config name, attack, num_free_riders). That means:
  * submit_sweep.sh  (same config x many repeats, free-riders fixed)
        -> repeats collapse into ONE row: mean +/- std over seeds (Tables I/II).
  * submit_fig7.sh   (one config, many free-rider counts)
        -> ONE row per free-rider count, so you can read the Fig. 7 trend
           (accuracy falls as free-riders rise). If you also did repeats at
           each count, they are averaged within the count.

Watermark metrics (benign BER, free-rider BER, detection accuracy) are
shown automatically when present

Usage:
    python scripts/aggregate_results.py /mnt/nfs/home/zu/results
    python scripts/aggregate_results.py /mnt/nfs/home/zu/results --fig7   # trend view
"""
import argparse
import json
import os
import statistics
from collections import defaultdict


def find_results(root):
    for dirpath, _, files in os.walk(root):
        if "result.json" in files:
            with open(os.path.join(dirpath, "result.json")) as f:
                try:
                    yield json.load(f)
                except json.JSONDecodeError:
                    print(f"  (skipping unreadable {dirpath}/result.json)")


def _ms(values):
    """mean, std (0 if a single value)."""
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def _mean_opt(runs, key):
    """Mean of an optional numeric field over the runs that have it, else None."""
    vals = [r[key] for r in runs if r.get(key) is not None]
    return round(statistics.mean(vals), 4) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory containing per-run result.json files")
    ap.add_argument("--fig7", action="store_true",
                    help="group by config and print the accuracy-vs-free-rider trend")
    args = ap.parse_args()

    # Group by (name, attack, num_free_riders) so Fig.7 counts stay separate
    # while plain repeats of one setting are averaged together.
    groups = defaultdict(list)
    for r in find_results(args.root):
        name = r["config"]["name"]
        attack = r.get("attack", r["config"].get("attack", "none"))
        nfr = r.get("num_free_riders", r["config"].get("num_free_riders", 0))
        groups[(name, attack, nfr)].append(r)

    if not groups:
        print(f"No result.json found under {args.root}")
        return

    has_wm = any(any(rr.get("watermark") for rr in rs) for rs in groups.values())

    if args.fig7:
        # Trend view: for each (config, attack), list free-rider count -> acc.
        by_cfg = defaultdict(list)
        for (name, attack, nfr), runs in groups.items():
            m, s = _ms([x["final_acc"] for x in runs])
            by_cfg[(name, attack)].append((nfr, m, s, len(runs)))
        for (name, attack), points in sorted(by_cfg.items()):
            print(f"\nFig. 7 trend  —  {name}  (attack={attack})")
            print(f"  {'#free-riders':>12}  {'final_acc (mean+/-std)':<22}{'n':>3}")
            print("  " + "-" * 42)
            for nfr, m, s, n in sorted(points):
                print(f"  {nfr:>12}  {m:6.2f} +/- {s:4.2f}{'':<8}{n:>3}")
        return

    # Default table view.
    header = f"\n{'config':<26}{'atk':<16}{'FR':>3}{'n':>3}  {'final_acc (mean+/-std)':<22}{'pass':>6}"
    if has_wm:
        header += f"  {'benignBER':>9}{'frBER':>7}{'detAcc':>7}"
    print(header)
    print("-" * (len(header) + 4))
    for (name, attack, nfr) in sorted(groups):
        runs = groups[(name, attack, nfr)]
        m, s = _ms([x["final_acc"] for x in runs])
        n_pass = sum(1 for x in runs if x.get("correctness_pass"))
        row = (f"{name:<26}{attack:<16}{nfr:>3}{len(runs):>3}  "
               f"{m:6.2f} +/- {s:4.2f}{'':<6}{n_pass}/{len(runs)}")
        if has_wm:
            bb = _mean_opt(runs, "wm_benign_ber")
            fb = _mean_opt(runs, "wm_fr_ber")
            da = _mean_opt(runs, "wm_detect_acc")
            row += (f"  {('' if bb is None else f'{bb:.3f}'):>9}"
                    f"{('' if fb is None else f'{fb:.3f}'):>7}"
                    f"{('' if da is None else f'{da:.2f}'):>7}")
        print(row)
    print()


if __name__ == "__main__":
    main()