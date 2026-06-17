#!/usr/bin/env python
"""Aggregate result.json files (one per repeat) into mean +/- std per config.

The paper reports accuracy as mean +/- std over 10 repeats (the +/- values in
Tables I and II). This script reproduces that summary.

Usage:
    python scripts/aggregate_results.py /mnt/nfs/home/zu/results
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory containing per-run result.json files")
    args = ap.parse_args()

    by_config = defaultdict(list)
    for r in find_results(args.root):
        key = r["config"]["name"]
        by_config[key].append(r)

    if not by_config:
        print(f"No result.json found under {args.root}")
        return

    print(f"\n{'config':<26}{'n':>3}  {'final_acc (mean+/-std)':<24}{'pass':>6}")
    print("-" * 62)
    for name in sorted(by_config):
        runs = by_config[name]
        accs = [x["final_acc"] for x in runs]
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        n_pass = sum(1 for x in runs if x.get("correctness_pass"))
        print(f"{name:<26}{len(runs):>3}  "
              f"{mean:6.2f} +/- {std:4.2f}{'':<10}{n_pass}/{len(runs):>1}")


if __name__ == "__main__":
    main()
