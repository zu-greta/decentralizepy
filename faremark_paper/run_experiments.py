#!/usr/bin/env python3
"""
run_experiments.py — FareMark paper experiment runner.

Reproduces all main tables and figures from:
  "FareMark: Model-Watermark-Driven Free-Rider Detection in
   Federated Learning Model", IEEE IoT Journal, 2025.

Usage:
    # Run a single experiment:
    python run_experiments.py --exp table1

    # Run all experiments:
    python run_experiments.py --exp all

    # Quick smoke test (very short, CPU-friendly):
    python run_experiments.py --exp smoke

Available --exp values:
    smoke       quick sanity check (2 rounds, tiny config)
    table1      Table I  — fidelity (accuracy with/without watermark)
    table2      Table II — watermark detection accuracy comparison
    fig7        Figure 7 — accuracy vs number of free-riders
    fig8        Figure 8 — watermark detection rate over training rounds
    table3      Table III — free-rider detection at varying FR ratios
    table6      Table VI  — robustness against differential privacy
    table7      Table VII — effect of number of trigger samples
    table8      Table VIII — memory-enhanced strategy ablation
    fig9        Figure 9  — robustness against fine-tuning
    fig10       Figure 10 — robustness against pruning
    all         run all of the above sequentially
"""

import argparse
import copy
import json
import os
import sys
import torch
import numpy as np

# Make sure the package is importable when running from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faremark_mod import (
    FareMarkConfig, FareMarkTrainer,
    build_model_for_dataset, load_dataset, split_iid, make_trigger_loader,
    WatermarkKey,
)
from faremark_mod.config import (
    config_table1, config_table2, config_fig7, config_fig8,
    config_table3, config_robustness_dp, config_table7,
)
from faremark_mod.evaluate import evaluate_finetune_curve, evaluate_prune_curve
from faremark_mod.train import accuracy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved → {path}")


def print_header(title: str):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke(args):
    """2-round test on CIFAR-10 + ResNet-18, 3 clients, 1 free-rider."""
    print_header("SMOKE TEST")
    cfg = FareMarkConfig(
        model_name="resnet18",
        dataset_name="cifar10",
        num_clients=3,
        num_free_riders=1,
        global_rounds=2,
        local_epochs=1,
        batch_size=32,
        wm_bits=4,
        eval_every=1,
        save_every=999,
        device=args.device,
        output_dir=args.output_dir,
        exp_name="smoke_test",
    )
    trainer = FareMarkTrainer(cfg)
    results = trainer.run()
    print("\nSmoke test passed ✓")
    return results


# ---------------------------------------------------------------------------
# Table I — Fidelity
# ---------------------------------------------------------------------------

def run_table1(args):
    """
    Table I: model accuracy with and without watermark embedding.
    Configurations: ResNet-18 & AlexNet × CIFAR-10, MNIST, CIFAR-100
    with 10 clients (and 100 for CIFAR-100).
    """
    print_header("TABLE I — Fidelity")

    configs = [
        config_table1("resnet18", "cifar10",  num_clients=10),
        config_table1("resnet18", "mnist",    num_clients=10),
        config_table1("resnet18", "cifar100", num_clients=100),
        config_table1("alexnet",  "cifar10",  num_clients=10),
        config_table1("alexnet",  "mnist",    num_clients=10),
        config_table1("alexnet",  "cifar100", num_clients=10),
    ]

    table = {}
    for cfg in configs:
        cfg.device = args.device
        cfg.output_dir = args.output_dir
        trainer = FareMarkTrainer(cfg)
        results = trainer.run()
        key = f"{cfg.model_name}_{cfg.dataset_name}_n{cfg.num_clients}"
        table[key] = {
            "final_acc": results["main_acc"][-1] if results["main_acc"] else None,
            "wm_acc":    results["wm_acc_benign"][-1] if results["wm_acc_benign"] else None,
        }
        print(f"  {key}: acc={table[key]['final_acc']:.4f}, wm={table[key]['wm_acc']:.4f}")

    save_json(table, os.path.join(args.output_dir, "table1_fidelity.json"))
    return table


# ---------------------------------------------------------------------------
# Table II — Watermark detection accuracy
# ---------------------------------------------------------------------------

def run_table2(args):
    """
    Table II: watermark extraction accuracy comparison.
    Our method vs FedIPR (backdoor-based and feature-based).
    We only implement our method here; FedIPR is the paper's baseline.
    """
    print_header("TABLE II — Watermark Detection Accuracy")

    configs = [
        config_table2("resnet18", "cifar10",  num_clients=10),
        config_table2("resnet18", "mnist",    num_clients=10),
        config_table2("resnet18", "cifar100", num_clients=100),
        config_table2("alexnet",  "cifar10",  num_clients=10),
        config_table2("alexnet",  "mnist",    num_clients=10),
        config_table2("alexnet",  "cifar100", num_clients=10),
    ]

    table = {}
    for cfg in configs:
        cfg.device = args.device
        cfg.output_dir = args.output_dir
        trainer = FareMarkTrainer(cfg)
        results = trainer.run()
        key = f"{cfg.model_name}_{cfg.dataset_name}"
        table[key] = {
            "wm_acc_mean": results["wm_acc_benign"][-1] if results["wm_acc_benign"] else None,
        }
        print(f"  {key}: wm_acc={table[key]['wm_acc_mean']:.4f}")

    save_json(table, os.path.join(args.output_dir, "table2_wm_detection.json"))
    return table


# ---------------------------------------------------------------------------
# Figure 7 — Main task accuracy vs number of free-riders
# ---------------------------------------------------------------------------

def run_fig7(args):
    """
    Figure 7: classification accuracy as free-rider proportion increases
    (0 to 8 out of 10 clients), for both free-rider strategies.
    """
    print_header("FIGURE 7 — Accuracy vs Free-Rider Count")

    # Sub-figures: (a) ResNet-18 CIFAR-10 prev_models
    #              (b) AlexNet MNIST prev_models
    #              (c) ResNet-18 CIFAR-10 gaussian
    #              (d) AlexNet MNIST gaussian
    experiments = [
        ("resnet18", "cifar10",  "previous_models"),
        ("alexnet",  "mnist",    "previous_models"),
        ("resnet18", "cifar10",  "gaussian_noise"),
        ("alexnet",  "mnist",    "gaussian_noise"),
    ]

    results_all = {}
    for model, dataset, fr_type in experiments:
        key = f"{model}_{dataset}_{fr_type}"
        results_all[key] = {"num_free_riders": [], "main_acc": []}
        for nfr in range(0, 9):  # 0 to 8 free-riders out of 10
            cfg = config_fig7(model, dataset, nfr, fr_type)
            cfg.device = args.device
            cfg.output_dir = args.output_dir
            trainer = FareMarkTrainer(cfg)
            res = trainer.run()
            final_acc = res["main_acc"][-1] if res["main_acc"] else 0.0
            results_all[key]["num_free_riders"].append(nfr)
            results_all[key]["main_acc"].append(final_acc)
            print(f"  {key} FR={nfr}: acc={final_acc:.4f}")

    save_json(results_all, os.path.join(args.output_dir, "fig7_accuracy_vs_freeriders.json"))
    return results_all


# ---------------------------------------------------------------------------
# Figure 8 — Watermark detection rate over rounds
# ---------------------------------------------------------------------------

def run_fig8(args):
    """
    Figure 8: watermark detection rate of free-riders vs benign clients
    across training rounds.
    """
    print_header("FIGURE 8 — Detection Rate Over Rounds")

    experiments = [
        ("resnet18", "cifar10"),
        ("alexnet",  "mnist"),
    ]

    results_all = {}
    for model, dataset in experiments:
        key = f"{model}_{dataset}"
        cfg = config_fig8(model, dataset)
        cfg.device = args.device
        cfg.output_dir = args.output_dir
        trainer = FareMarkTrainer(cfg)
        res = trainer.run()
        results_all[key] = {
            "rounds":           res["rounds"],
            "wm_acc_benign":    res["wm_acc_benign"],
            "wm_acc_freerider": res["wm_acc_freerider"],
            "main_acc":         res["main_acc"],
        }
        print(f"  {key}: final benign wm={res['wm_acc_benign'][-1]:.3f}, "
              f"fr wm={res['wm_acc_freerider'][-1]:.3f}")

    save_json(results_all, os.path.join(args.output_dir, "fig8_detection_over_rounds.json"))
    return results_all


# ---------------------------------------------------------------------------
# Table III — Free-rider detection at varying FR ratios
# ---------------------------------------------------------------------------

def run_table3(args):
    """
    Table III: detection accuracy and FPR across FR ratios 20-80%.
    """
    print_header("TABLE III — Free-Rider Detection Analysis")

    fr_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    configs_per_setting = [
        ("resnet18", "cifar10", "previous_models"),
        ("alexnet",  "mnist",   "gaussian_noise"),
    ]

    table = {}
    for model, dataset, fr_type in configs_per_setting:
        for ratio in fr_ratios:
            cfg = config_table3(model, dataset, ratio)
            cfg.free_rider_type = fr_type
            cfg.device = args.device
            cfg.output_dir = args.output_dir
            trainer = FareMarkTrainer(cfg)
            res = trainer.run()
            key = f"{model}_{dataset}_{fr_type}_fr{int(ratio*100)}"
            table[key] = {
                "fr_detection_acc": res["fr_detection_acc"][-1] if res["fr_detection_acc"] else None,
                "fpr":              res["fpr"][-1] if res["fpr"] else None,
                "main_acc":         res["main_acc"][-1] if res["main_acc"] else None,
            }
            print(f"  {key}: det={table[key]['fr_detection_acc']:.3f}, "
                  f"fpr={table[key]['fpr']:.3f}")

    save_json(table, os.path.join(args.output_dir, "table3_fr_detection.json"))
    return table


# ---------------------------------------------------------------------------
# Table VI — Differential privacy robustness
# ---------------------------------------------------------------------------

def run_table6(args):
    """Table VI: watermark accuracy under differential privacy noise."""
    print_header("TABLE VI — Differential Privacy Robustness")
    from faremark_mod.evaluate import train_with_dp
    from faremark_mod.watermark import extract_watermark, bit_accuracy

    cfg = config_robustness_dp("resnet18", "cifar10")
    cfg.device = args.device
    cfg.output_dir = args.output_dir

    # First train normally to get a watermarked model
    trainer = FareMarkTrainer(cfg)
    trainer.run()
    watermarked_model = copy.deepcopy(trainer.server.get_global_model())
    device = trainer.device

    train_loader = torch.utils.data.DataLoader(
        trainer.train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    test_loader = trainer.test_loader

    results = {}
    for noise_mult in [0.5, 1.0, 1.5, 2.0]:
        dp_model = train_with_dp(
            watermarked_model,
            train_loader,
            device,
            epochs=5,
            noise_multiplier=noise_mult,
        )
        main_acc = accuracy(dp_model, test_loader, device)

        wm_accs = []
        for cid, tloader in trainer.trigger_loaders.items():
            with torch.no_grad():
                logits_list = []
                for imgs, _ in tloader:
                    logits_list.append(dp_model(imgs.to(device)))
                logits = torch.cat(logits_list)[:cfg.n_triggers]
                b_hat = extract_watermark(logits, trainer.keys[cid], cfg.smooth_fn, cfg.alpha_smooth)
                wm_accs.append(bit_accuracy(b_hat, trainer.keys[cid].B.to(device)))

        key = f"noise_{noise_mult}"
        results[key] = {
            "noise_multiplier": noise_mult,
            "main_acc": main_acc,
            "wm_acc": float(np.mean(wm_accs)),
        }
        print(f"  noise={noise_mult}: main={main_acc:.3f}, wm={results[key]['wm_acc']:.3f}")

    save_json(results, os.path.join(args.output_dir, "table6_dp_robustness.json"))
    return results


# ---------------------------------------------------------------------------
# Table VII — Effect of number of trigger samples
# ---------------------------------------------------------------------------

def run_table7(args):
    """Table VII: watermark detection accuracy vs N_T."""
    print_header("TABLE VII — Trigger Sample Count")

    n_trigger_values = [1, 10, 50, 100, 150, 200, 300, 400]

    # Train once, then evaluate with different N_T
    cfg = config_table7("resnet18", "cifar10", n_triggers=400)
    cfg.device = args.device
    cfg.output_dir = args.output_dir
    trainer = FareMarkTrainer(cfg)
    trainer.run()

    from faremark_mod.watermark import extract_watermark, bit_accuracy

    results = {}
    for nt in n_trigger_values:
        wm_accs = []
        for cid, tloader in trainer.trigger_loaders.items():
            with torch.no_grad():
                logits_list = []
                count = 0
                for imgs, _ in tloader:
                    logits_list.append(trainer.server.get_global_model()(imgs.to(trainer.device)))
                    count += imgs.size(0)
                    if count >= nt:
                        break
                if logits_list:
                    logits = torch.cat(logits_list)[:nt]
                    b_hat = extract_watermark(logits, trainer.keys[cid],
                                              cfg.smooth_fn, cfg.alpha_smooth)
                    wm_accs.append(bit_accuracy(b_hat, trainer.keys[cid].B.to(trainer.device)))
        results[f"nt_{nt}"] = float(np.mean(wm_accs)) if wm_accs else 0.0
        print(f"  N_T={nt:4d}: wm_acc={results[f'nt_{nt}']:.4f}")

    save_json(results, os.path.join(args.output_dir, "table7_trigger_count.json"))
    return results


# ---------------------------------------------------------------------------
# Table VIII — Memory-enhanced strategy ablation
# ---------------------------------------------------------------------------

def run_table8(args):
    """Table VIII: compare with vs without memory-enhanced updating."""
    print_header("TABLE VIII — Memory-Enhanced Strategy Ablation")

    results = {}
    for use_memory, label in [(True, "with_memory"), (False, "without_memory")]:
        cfg = FareMarkConfig(
            model_name="resnet18",
            dataset_name="cifar10",
            num_clients=10,
            num_free_riders=0,
            global_rounds=50,
            local_epochs=5,
            beta=0.9 if use_memory else 1.0,  # beta=1.0 = standard SGD
            device=args.device,
            output_dir=args.output_dir,
            exp_name=f"table8_{label}",
        )
        trainer = FareMarkTrainer(cfg)
        res = trainer.run()
        results[label] = {
            "main_acc": res["main_acc"][-1] if res["main_acc"] else None,
            "wm_acc":   res["wm_acc_benign"][-1] if res["wm_acc_benign"] else None,
        }
        print(f"  {label}: main={results[label]['main_acc']:.4f}, "
              f"wm={results[label]['wm_acc']:.4f}")

    save_json(results, os.path.join(args.output_dir, "table8_ablation.json"))
    return results


# ---------------------------------------------------------------------------
# Figure 9 — Fine-tuning robustness
# ---------------------------------------------------------------------------

def run_fig9(args):
    """Figure 9: watermark accuracy during fine-tuning attack."""
    print_header("FIGURE 9 — Fine-Tuning Robustness")

    cfg = FareMarkConfig(
        model_name="resnet18",
        dataset_name="cifar10",
        num_clients=10,
        num_free_riders=0,
        global_rounds=50,
        local_epochs=5,
        device=args.device,
        output_dir=args.output_dir,
        exp_name="fig9_base",
    )
    trainer = FareMarkTrainer(cfg)
    trainer.run()

    train_loader = torch.utils.data.DataLoader(
        trainer.train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
    )

    curve = evaluate_finetune_curve(
        model=trainer.server.get_global_model(),
        train_loader=train_loader,
        test_loader=trainer.test_loader,
        trigger_loaders=trainer.trigger_loaders,
        keys=trainer.keys,
        device=trainer.device,
        total_epochs=80,
        eval_every=10,
        smooth_fn=cfg.smooth_fn,
        alpha=cfg.alpha_smooth,
        n_triggers=cfg.n_triggers,
    )

    save_json(curve, os.path.join(args.output_dir, "fig9_finetune_robustness.json"))
    return curve


# ---------------------------------------------------------------------------
# Figure 10 — Pruning robustness
# ---------------------------------------------------------------------------

def run_fig10(args):
    """Figure 10: watermark accuracy after pruning at various sparsities."""
    print_header("FIGURE 10 — Pruning Robustness")

    cfg = FareMarkConfig(
        model_name="resnet18",
        dataset_name="cifar10",
        num_clients=10,
        num_free_riders=0,
        global_rounds=50,
        local_epochs=5,
        device=args.device,
        output_dir=args.output_dir,
        exp_name="fig10_base",
    )
    trainer = FareMarkTrainer(cfg)
    trainer.run()

    curve = evaluate_prune_curve(
        model=trainer.server.get_global_model(),
        test_loader=trainer.test_loader,
        trigger_loaders=trainer.trigger_loaders,
        keys=trainer.keys,
        device=trainer.device,
        prune_ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        smooth_fn=cfg.smooth_fn,
        alpha=cfg.alpha_smooth,
        n_triggers=cfg.n_triggers,
    )

    save_json(curve, os.path.join(args.output_dir, "fig10_pruning_robustness.json"))
    return curve


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "smoke":  run_smoke,
    "table1": run_table1,
    "table2": run_table2,
    "fig7":   run_fig7,
    "fig8":   run_fig8,
    "table3": run_table3,
    "table6": run_table6,
    "table7": run_table7,
    "table8": run_table8,
    "fig9":   run_fig9,
    "fig10":  run_fig10,
}


def main():
    parser = argparse.ArgumentParser(
        description="FareMark experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp", type=str, default="smoke",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Which experiment to run (default: smoke)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cuda' or 'cpu' (auto-detected by default)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Directory for saving results (default: ./results)",
    )
    args = parser.parse_args()

    print(f"\nDevice : {args.device}")
    print(f"Output : {args.output_dir}")
    print(f"Exp    : {args.exp}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.exp == "all":
        for name, fn in EXPERIMENTS.items():
            if name == "smoke":
                continue
            try:
                fn(args)
            except Exception as e:
                print(f"\n[ERROR] Experiment '{name}' failed: {e}")
                import traceback; traceback.print_exc()
    else:
        EXPERIMENTS[args.exp](args)


if __name__ == "__main__":
    main()