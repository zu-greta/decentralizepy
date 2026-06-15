#!/usr/bin/env python3
"""
FareMark — Results Collection & Paper Table Comparison
=======================================================
Reads all summary.json files from the results directory,
aggregates over repetitions (mean ± std), and prints
formatted tables matching every table in the paper.

Usage:
    python analysis/collect_results.py --results_dir /results
    python analysis/collect_results.py --results_dir ./results --table all
    python analysis/collect_results.py --results_dir ./results --table 1
    python analysis/collect_results.py --results_dir ./results --table 3 --save_csv
"""

import os, sys, json, argparse, glob
import numpy as np

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("[WARN] tabulate not installed. Install with: pip install tabulate")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_summaries(results_dir, prefix):
    """Load all summary.json files whose parent folder starts with prefix."""
    pattern = os.path.join(results_dir, f"{prefix}*/summary.json")
    files = glob.glob(pattern)
    data = []
    for f in files:
        try:
            with open(f) as fh:
                data.append(json.load(fh))
        except Exception as e:
            print(f"  [WARN] Could not load {f}: {e}")
    return data

def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return "—", "—"
    return f"{np.mean(values)*100:.2f}", f"{np.std(values)*100:.2f}"

def fmt(mean, std):
    if mean == "—":
        return "—"
    return f"{mean} ± {std}"

def print_table(title, headers, rows, tablefmt="grid"):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt=tablefmt))
    else:
        print("  " + " | ".join(headers))
        print("  " + "-" * 60)
        for row in rows:
            print("  " + " | ".join(str(x) for x in row))


# ─── Table I ──────────────────────────────────────────────────────────────────

def collect_table1(results_dir):
    """
    Table I paper values (Ours column):
      ResNet-18 CIFAR-10  n=10:  90.78
      ResNet-18 MNIST     n=10:  98.03
      ResNet-18 CIFAR-100 n=100: 75.31
      AlexNet   CIFAR-10  n=10:  85.89
      AlexNet   MNIST     n=10:  90.89
      AlexNet   CIFAR-100 n=10:  67.89
    """
    PAPER = {
        ("resnet18","cifar10",10):  90.78,
        ("resnet18","mnist",10):    98.03,
        ("resnet18","cifar100",100):75.31,
        ("alexnet","cifar10",10):   85.89,
        ("alexnet","mnist",10):     90.89,
        ("alexnet","cifar100",10):  67.89,
    }
    summaries = load_summaries(results_dir, "table1_")
    if not summaries:
        print("[Table I] No results found yet.")
        return

    # Group by (model, dataset, num_clients)
    groups = {}
    for s in summaries:
        key = (s["model"], s["dataset"], s["num_clients"])
        groups.setdefault(key, []).append(s)

    rows = []
    for key, items in sorted(groups.items()):
        model, dataset, nc = key
        accs = [s["final_main_acc"]  for s in items]
        wms  = [s["final_wm_acc_benign"] for s in items]
        paper_val = PAPER.get(key, "—")
        m, s_ = mean_std(accs)
        wm_m, wm_s = mean_std(wms)
        rows.append([
            model, dataset, nc,
            fmt(m, s_),
            f"{paper_val:.2f}" if isinstance(paper_val, float) else "—",
            f"{float(m)-paper_val:+.2f}" if isinstance(paper_val,float) and m!="—" else "—",
            fmt(wm_m, wm_s),
            len(items),
        ])

    print_table(
        "TABLE I — Model Accuracy With and Without Watermark Embedding",
        ["Model","Dataset","N","Ours Acc (%)","Paper (%)","Δ","WM Acc (%)","Reps"],
        rows
    )


# ─── Table II ─────────────────────────────────────────────────────────────────

def collect_table2(results_dir):
    """
    Table II paper values (Ours column):
      ResNet-18 CIFAR-10  n=10:  99.72 ± 0.08
      ResNet-18 MNIST     n=10:  99.84 ± 0.10
      ResNet-18 CIFAR-100 n=100: 99.71 ± 0.18
      AlexNet   CIFAR-10  n=10:  99.91 ± 0.16
      AlexNet   MNIST     n=10:  99.89 ± 0.12
      AlexNet   CIFAR-100 n=10:  99.89 ± 0.04
    """
    PAPER = {
        ("resnet18","cifar10",10):   (99.72, 0.08),
        ("resnet18","mnist",10):     (99.84, 0.10),
        ("resnet18","cifar100",100): (99.71, 0.18),
        ("alexnet","cifar10",10):    (99.91, 0.16),
        ("alexnet","mnist",10):      (99.89, 0.12),
        ("alexnet","cifar100",10):   (99.89, 0.04),
    }
    summaries = load_summaries(results_dir, "table2_")
    if not summaries:
        print("[Table II] No results found yet.")
        return

    groups = {}
    for s in summaries:
        key = (s["model"], s["dataset"], s["num_clients"])
        groups.setdefault(key, []).append(s)

    rows = []
    for key, items in sorted(groups.items()):
        model, dataset, nc = key
        wms = [s["wm_acc_mean"] for s in items]
        m, s_ = mean_std(wms)
        paper = PAPER.get(key)
        p_str = f"{paper[0]:.2f} ± {paper[1]:.2f}" if paper else "—"
        rows.append([model, dataset, nc, fmt(m, s_), p_str, len(items)])

    print_table(
        "TABLE II — Comparison of Watermark Detection Accuracy",
        ["Model","Dataset","N","Ours WM Acc (%)","Paper (%)","Reps"],
        rows
    )


# ─── Table III ────────────────────────────────────────────────────────────────

def collect_table3(results_dir):
    """Table III: FR detection accuracy and FPR at varying FR ratios."""
    summaries = load_summaries(results_dir, "table3_")
    if not summaries:
        print("[Table III] No results found yet.")
        return

    groups = {}
    for s in summaries:
        key = (s.get("fr_type","?"), round(s.get("fr_ratio",0),1))
        groups.setdefault(key, []).append(s)

    rows = []
    for (fr_type, ratio), items in sorted(groups.items()):
        dets = [s["fr_detection_acc"] for s in items]
        fprs = [s["fpr"]              for s in items]
        dm, ds = mean_std(dets)
        fm, fs = mean_std(fprs)
        rows.append([
            fr_type, f"{int(ratio*100)}%",
            fmt(dm, ds), fmt(fm, fs), len(items)
        ])

    print_table(
        "TABLE III — Free-Rider Detection Analysis",
        ["FR Type","FR Ratio","Det Acc (%)","FPR (%)","Reps"],
        rows
    )


# ─── Table VI ─────────────────────────────────────────────────────────────────

def collect_table6(results_dir):
    """Table VI: DP robustness."""
    summaries = load_summaries(results_dir, "table6_")
    if not summaries:
        print("[Table VI] No results found yet.")
        return

    groups = {}
    for s in summaries:
        key = (s.get("noise_mult",0), s.get("extra_epochs",0))
        groups.setdefault(key, []).append(s)

    rows = []
    for (nm, ep), items in sorted(groups.items()):
        accs = [s["main_acc"]    for s in items]
        wms  = [s["wm_acc_mean"] for s in items]
        am, as_ = mean_std(accs)
        wm, ws  = mean_std(wms)
        rows.append([nm, ep, fmt(am, as_), fmt(wm, ws), len(items)])

    print_table(
        "TABLE VI — Robustness Against Differential Privacy (ResNet-18, CIFAR-10)",
        ["Noise Mult","Extra Epochs","Main Acc (%)","WM Acc (%)","Reps"],
        rows
    )


# ─── Table VII ────────────────────────────────────────────────────────────────

def collect_table7(results_dir):
    """Table VII: WM accuracy vs N_T."""
    summaries = load_summaries(results_dir, "table7_")
    if not summaries:
        print("[Table VII] No results found yet.")
        return

    NT_VALUES = [1, 10, 50, 100, 150, 200, 300, 400]
    # Aggregate across repeats per (model, dataset, N_T)
    data = {}  # (model, dataset) → {nt → [acc]}
    for s in summaries:
        for nt in NT_VALUES:
            key_nt = f"nt_{nt}"
            for model_dataset, row in s.items():
                if not isinstance(row, dict):
                    continue
                if key_nt in row:
                    md = model_dataset
                    data.setdefault(md, {}).setdefault(nt, [])
                    if row[key_nt]["mean"] is not None:
                        data[md][nt].append(row[key_nt]["mean"])

    rows = []
    for md, nt_dict in sorted(data.items()):
        row = [md]
        for nt in NT_VALUES:
            vals = nt_dict.get(nt, [])
            if vals:
                row.append(f"{np.mean(vals)*100:.2f}")
            else:
                row.append("—")
        rows.append(row)

    print_table(
        "TABLE VII — Watermark Detection Accuracy (%) With Different Number of Triggers",
        ["Config"] + [f"N_T={n}" for n in NT_VALUES],
        rows
    )


# ─── Table VIII ───────────────────────────────────────────────────────────────

def collect_table8(results_dir):
    """Table VIII: memory-enhanced ablation."""
    summaries = load_summaries(results_dir, "table8_")
    if not summaries:
        print("[Table VIII] No results found yet.")
        return

    groups = {"with_memory": [], "without_memory": []}
    for s in summaries:
        for label in groups:
            if label in s:
                groups[label].append(s[label])

    rows = []
    for label, items in groups.items():
        accs = [i.get("final_main_acc") for i in items]
        wms  = [i.get("final_wm_acc")   for i in items]
        am, as_ = mean_std(accs)
        wm, ws  = mean_std(wms)
        rows.append([label, fmt(am, as_), fmt(wm, ws), len(items)])

    print_table(
        "TABLE VIII — Ablation Study of Memory-Enhanced Strategy (ResNet-18, CIFAR-10)",
        ["Strategy","Main Acc (%)","WM Acc (%)","Reps"],
        rows
    )


# ─── Table IX ─────────────────────────────────────────────────────────────────

def collect_table9(results_dir):
    """Table IX: capacity analysis."""
    summaries = load_summaries(results_dir, "table9_")
    if not summaries:
        print("[Table IX] No results found yet.")
        return

    groups = {}
    for s in summaries:
        for nc_key, v in s.items():
            if not isinstance(v, dict): continue
            nc = v.get("num_clients")
            if nc is None: continue
            groups.setdefault(nc, []).append(v)

    rows = []
    for nc, items in sorted(groups.items()):
        accs = [i.get("final_main_acc") for i in items]
        wms  = [i.get("final_wm_acc")   for i in items]
        am, as_ = mean_std(accs)
        wm, ws  = mean_std(wms)
        rows.append([nc, fmt(am, as_), fmt(wm, ws), len(items)])

    print_table(
        "TABLE IX — Capacity Analysis (ResNet-18, CIFAR-10)",
        ["Num Clients","Main Acc (%)","WM Acc (%)","Reps"],
        rows
    )


# ─── Figure 7 summary ─────────────────────────────────────────────────────────

def collect_fig7(results_dir):
    """Figure 7: accuracy at each FR count."""
    for sf in ["a","b","c","d"]:
        summaries = load_summaries(results_dir, f"fig7_{sf}_")
        if not summaries:
            continue
        groups = {}
        for s in summaries:
            groups.setdefault(s["num_fr"], []).append(s["final_main_acc"])

        rows = [[nfr, fmt(*mean_std(vals)), len(vals)]
                for nfr, vals in sorted(groups.items())]
        sfig_labels = {
            "a": "ResNet-18/CIFAR-10/prev_models",
            "b": "AlexNet/MNIST/prev_models",
            "c": "ResNet-18/CIFAR-10/gaussian",
            "d": "AlexNet/MNIST/gaussian",
        }
        print_table(
            f"FIGURE 7({sf.upper()}) — {sfig_labels[sf]}",
            ["Num Free-Riders","Main Acc (%)","Reps"],
            rows
        )


# ─── Figure 8 summary ─────────────────────────────────────────────────────────

def collect_fig8(results_dir):
    """Figure 8: detection rate at key rounds."""
    for sf in ["a","b"]:
        summaries = load_summaries(results_dir, f"fig8_{sf}_")
        if not summaries:
            continue

        # Aggregate at rounds 30, 60, 100
        key_rounds = [30, 60, 80, 100]
        rows = []
        for rnd in key_rounds:
            benign_vals, fr_vals = [], []
            for s in summaries:
                rounds = s.get("rounds", [])
                if rnd in rounds:
                    idx = rounds.index(rnd)
                    benign_vals.append(s["wm_acc_benign"][idx])
                    if s.get("wm_acc_freerider") and len(s["wm_acc_freerider"]) > idx:
                        fr_vals.append(s["wm_acc_freerider"][idx])
            bm, bs = mean_std(benign_vals)
            fm, fs = mean_std(fr_vals)
            rows.append([rnd, fmt(bm, bs), fmt(fm, fs)])

        sfig_labels = {"a": "ResNet-18/CIFAR-10", "b": "AlexNet/MNIST"}
        print_table(
            f"FIGURE 8({sf.upper()}) — {sfig_labels[sf]} — Detection Rate",
            ["Round","Benign WM Acc (%)","FR WM Acc (%)"],
            rows
        )


# ─── Figures 9 & 10 summary ───────────────────────────────────────────────────

def collect_fig9(results_dir):
    summaries = load_summaries(results_dir, "fig9_")
    if not summaries:
        print("[Fig 9] No results found yet.")
        return
    rows = []
    epoch_keys = [10, 20, 30, 40, 50, 60, 70, 80]
    for ep in epoch_keys:
        wms, accs = [], []
        for s in summaries:
            if ep in s.get("epoch", []):
                idx = s["epoch"].index(ep)
                wms.append(s["wm_acc"][idx])
                accs.append(s["main_acc"][idx])
        wm_m, wm_s = mean_std(wms)
        ac_m, ac_s = mean_std(accs)
        rows.append([ep, fmt(ac_m, ac_s), fmt(wm_m, wm_s)])
    print_table("FIGURE 9 — Fine-Tuning Robustness (ResNet-18, CIFAR-10)",
                ["Finetune Epoch","Main Acc (%)","WM Acc (%)"], rows)


def collect_fig10(results_dir):
    summaries = load_summaries(results_dir, "fig10_")
    if not summaries:
        print("[Fig 10] No results found yet.")
        return
    rows = []
    for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        wms, accs = [], []
        for s in summaries:
            if ratio in s.get("prune_ratio", []):
                idx = s["prune_ratio"].index(ratio)
                wms.append(s["wm_acc"][idx])
                accs.append(s["main_acc"][idx])
        wm_m, wm_s = mean_std(wms)
        ac_m, ac_s = mean_std(accs)
        rows.append([f"{int(ratio*100)}%", fmt(ac_m, ac_s), fmt(wm_m, wm_s)])
    print_table("FIGURE 10 — Pruning Robustness (ResNet-18, CIFAR-10)",
                ["Prune Ratio","Main Acc (%)","WM Acc (%)"], rows)


# ─── Progress checker ─────────────────────────────────────────────────────────

def check_progress(results_dir):
    """Print how many jobs have completed vs expected."""
    expected = {
        "table1": 60,   # 6 configs × 10 reps
        "table2": 60,
        "fig7":   360,  # 4 subfigs × 9 FR counts × 10 reps
        "fig8":   20,   # 2 subfigs × 10 reps
        "table3": 160,  # 8 ratios × 2 fr types × 10 reps
        "table4": 10,
        "table5": 10,
        "table6": 100,  # 5 noise × 2 ep settings × 10 reps
        "table7": 10,
        "table8": 10,
        "table9": 10,
        "fig9":   10,
        "fig10":  10,
    }
    print(f"\n{'='*50}")
    print("  Job Progress")
    print('='*50)
    total_done = 0
    total_exp  = 0
    for prefix, exp in expected.items():
        done = len(load_summaries(results_dir, f"{prefix}_"))
        total_done += done
        total_exp  += exp
        bar = "█" * int(done/exp*20) + "░" * (20-int(done/exp*20)) if exp > 0 else ""
        print(f"  {prefix:10s} [{bar}] {done:4d}/{exp}")
    print(f"\n  Total: {total_done}/{total_exp} "
          f"({total_done/total_exp*100:.1f}% complete)")


# ─── Main ─────────────────────────────────────────────────────────────────────

COLLECTORS = {
    "1":    collect_table1,
    "2":    collect_table2,
    "3":    collect_table3,
    "6":    collect_table6,
    "7":    collect_table7,
    "8":    collect_table8,
    "9":    collect_table9,
    "f7":   collect_fig7,
    "f8":   collect_fig8,
    "f9":   collect_fig9,
    "f10":  collect_fig10,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--table", type=str, default="all",
                        help="Which table/figure: 1,2,3,6,7,8,9,f7,f8,f9,f10,progress,all")
    parser.add_argument("--save_csv", action="store_true",
                        help="Also save tables as CSV files")
    args = parser.parse_args()

    if args.table == "progress":
        check_progress(args.results_dir)
        return

    if args.table == "all":
        check_progress(args.results_dir)
        for fn in COLLECTORS.values():
            fn(args.results_dir)
    elif args.table in COLLECTORS:
        COLLECTORS[args.table](args.results_dir)
    else:
        print(f"Unknown table: {args.table}")
        print(f"Available: {list(COLLECTORS.keys())} + all + progress")
        sys.exit(1)

if __name__ == "__main__":
    main()