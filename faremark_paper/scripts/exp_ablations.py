#!/usr/bin/env python3
"""
Experiments: Table VII, VIII, IX and Figures 9, 10 — Ablation Studies

Table VII — Watermark Detection Accuracy vs N_T (trigger sample count)
  N_T in {1, 10, 50, 100, 150, 200, 300, 400}

Table VIII — Memory-Enhanced Strategy Ablation
  With beta=0.9 (memory-enhanced) vs beta=1.0 (standard SGD)

Table IX — Capacity Analysis
  Clients: 10, 20, 30, 40, 50 (over-subscribing trigger classes)

Figure 9 — Fine-Tuning Robustness
  lambda=0 fine-tuning for 80 epochs, WM and main acc every 10 epochs

Figure 10 — Pruning Robustness
  Prune 0%, 10%, ..., 70% of parameters
"""

import os, sys, json, argparse, copy, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faremark_mod import FareMarkConfig, FareMarkTrainer
from faremark_mod.watermark import extract_watermark, bit_accuracy
from faremark_mod.evaluate import evaluate_finetune_curve, evaluate_prune_curve
from faremark_mod.train import accuracy
import torch
from torch.utils.data import DataLoader


def run_table7(args):
    """Table VII: WM detection accuracy vs number of trigger samples N_T."""
    N_T_VALUES = [1, 10, 50, 100, 150, 200, 300, 400]

    # Train once with max N_T=400
    exp_name = f"table7_base_rep{args.repeat}"
    cfg = FareMarkConfig(
        model_name="resnet18", dataset_name="cifar10",
        num_clients=10, num_free_riders=0,
        global_rounds=50, local_epochs=5, batch_size=16,
        lr=0.01, wm_bits=8, n_triggers=400,
        seed=42 + args.repeat, device=args.device,
        data_root=args.data_root, output_dir=args.output_dir,
        exp_name=exp_name, eval_every=10, save_every=50,
    )
    # Also test for AlexNet
    results_all = {}
    for model in ["resnet18", "alexnet"]:
        for dataset in ["cifar10", "mnist", "cifar100"]:
            num_clients = 100 if dataset == "cifar100" else 10
            cfg.model_name = model
            cfg.dataset_name = dataset
            cfg.num_clients = num_clients
            cfg.exp_name = f"table7_{model}_{dataset}_base_rep{args.repeat}"
            trainer = FareMarkTrainer(cfg)
            trainer.run()
            global_model = trainer.server.get_global_model()
            device = trainer.device

            row = {}
            for nt in N_T_VALUES:
                wm_accs = []
                for cid, tloader in trainer.trigger_loaders.items():
                    with torch.no_grad():
                        logits_list, count = [], 0
                        for imgs, _ in tloader:
                            out = global_model(imgs.to(device))
                            if hasattr(out, 'logits'): out = out.logits
                            logits_list.append(out)
                            count += imgs.size(0)
                            if count >= nt: break
                        if logits_list:
                            logits = torch.cat(logits_list)[:nt]
                            b_hat = extract_watermark(logits, trainer.keys[cid],
                                                      cfg.smooth_fn, cfg.alpha_smooth)
                            wm_accs.append(bit_accuracy(b_hat, trainer.keys[cid].B.to(device)))
                row[f"nt_{nt}"] = {
                    "mean": float(np.mean(wm_accs)) if wm_accs else None,
                    "std":  float(np.std(wm_accs))  if wm_accs else None,
                }
                print(f"  {model}/{dataset} N_T={nt}: {row[f'nt_{nt}']['mean']*100:.2f}%")

            results_all[f"{model}_{dataset}_n{num_clients}"] = row

    out_dir = os.path.join(args.output_dir, f"table7_rep{args.repeat}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results_all, f, indent=2)


def run_table8(args):
    """Table VIII: memory-enhanced strategy ablation."""
    results_all = {}
    for label, beta in [("with_memory", 0.9), ("without_memory", 1.0)]:
        exp_name = f"table8_{label}_rep{args.repeat}"
        cfg = FareMarkConfig(
            model_name="resnet18", dataset_name="cifar10",
            num_clients=10, num_free_riders=0,
            global_rounds=50, local_epochs=5, batch_size=16,
            lr=0.01, wm_bits=8, beta=beta, n_triggers=100,
            seed=42 + args.repeat, device=args.device,
            data_root=args.data_root, output_dir=args.output_dir,
            exp_name=exp_name, eval_every=10, save_every=50,
        )
        trainer = FareMarkTrainer(cfg)
        res = trainer.run()
        results_all[label] = {
            "beta": beta,
            "final_main_acc":  res["main_acc"][-1]      if res["main_acc"]      else None,
            "final_wm_acc":    res["wm_acc_benign"][-1] if res["wm_acc_benign"] else None,
            "main_acc_curve":  res["main_acc"],
            "wm_acc_curve":    res["wm_acc_benign"],
        }
        print(f"\n[TABLE8] {label}: acc={results_all[label]['final_main_acc']:.3f}, "
              f"wm={results_all[label]['final_wm_acc']:.3f}")

    out_dir = os.path.join(args.output_dir, f"table8_rep{args.repeat}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results_all, f, indent=2)


def run_table9(args):
    """Table IX: capacity — more clients than classes (CIFAR-10 has 10 classes)."""
    client_counts = [10, 20, 30, 40, 50]
    results_all = {}
    for n in client_counts:
        exp_name = f"table9_{n}clients_rep{args.repeat}"
        cfg = FareMarkConfig(
            model_name="resnet18", dataset_name="cifar10",
            num_clients=n, num_free_riders=0,
            global_rounds=50, local_epochs=5, batch_size=16,
            lr=0.01, wm_bits=8, n_triggers=50,
            seed=42 + args.repeat, device=args.device,
            data_root=args.data_root, output_dir=args.output_dir,
            exp_name=exp_name, eval_every=10, save_every=50,
        )
        trainer = FareMarkTrainer(cfg)
        res = trainer.run()
        results_all[f"n{n}"] = {
            "num_clients": n,
            "final_main_acc": res["main_acc"][-1]      if res["main_acc"]      else None,
            "final_wm_acc":   res["wm_acc_benign"][-1] if res["wm_acc_benign"] else None,
        }
        print(f"\n[TABLE9] {n} clients: acc={results_all[f'n{n}']['final_main_acc']:.3f}, "
              f"wm={results_all[f'n{n}']['final_wm_acc']:.3f}")

    out_dir = os.path.join(args.output_dir, f"table9_rep{args.repeat}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results_all, f, indent=2)


def run_fig9(args):
    """Figure 9: fine-tuning robustness."""
    exp_name = f"fig9_base_rep{args.repeat}"
    cfg = FareMarkConfig(
        model_name="resnet18", dataset_name="cifar10",
        num_clients=10, num_free_riders=0,
        global_rounds=50, local_epochs=5, batch_size=16,
        lr=0.01, wm_bits=8, n_triggers=100,
        seed=42 + args.repeat, device=args.device,
        data_root=args.data_root, output_dir=args.output_dir,
        exp_name=exp_name, eval_every=10, save_every=50,
    )
    trainer = FareMarkTrainer(cfg)
    trainer.run()

    train_loader = DataLoader(trainer.train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=2)
    curve = evaluate_finetune_curve(
        model=trainer.server.get_global_model(),
        train_loader=train_loader,
        test_loader=trainer.test_loader,
        trigger_loaders=trainer.trigger_loaders,
        keys=trainer.keys,
        device=trainer.device,
        total_epochs=80, eval_every=10,
        smooth_fn=cfg.smooth_fn, alpha=cfg.alpha_smooth,
        n_triggers=cfg.n_triggers,
    )
    out_dir = os.path.join(args.output_dir, f"fig9_rep{args.repeat}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(curve, f, indent=2)
    print(f"\n[FIG9] Final WM after fine-tuning: {curve['wm_acc'][-1]:.3f}")


def run_fig10(args):
    """Figure 10: pruning robustness."""
    exp_name = f"fig10_base_rep{args.repeat}"
    cfg = FareMarkConfig(
        model_name="resnet18", dataset_name="cifar10",
        num_clients=10, num_free_riders=0,
        global_rounds=50, local_epochs=5, batch_size=16,
        lr=0.01, wm_bits=8, n_triggers=100,
        seed=42 + args.repeat, device=args.device,
        data_root=args.data_root, output_dir=args.output_dir,
        exp_name=exp_name, eval_every=10, save_every=50,
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
        smooth_fn=cfg.smooth_fn, alpha=cfg.alpha_smooth,
        n_triggers=cfg.n_triggers,
    )
    out_dir = os.path.join(args.output_dir, f"fig10_rep{args.repeat}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(curve, f, indent=2)
    print(f"\n[FIG10] WM at 50% pruning: {curve['wm_acc'][5]:.3f}")


RUNNERS = {
    "table7": run_table7,
    "table8": run_table8,
    "table9": run_table9,
    "fig9":   run_fig9,
    "fig10":  run_fig10,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",        type=str, required=True, choices=list(RUNNERS))
    parser.add_argument("--repeat",     type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=os.environ.get("RESULTS_DIR", "./results"))
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--data_root",  type=str, default="./data")
    args = parser.parse_args()
    RUNNERS[args.exp](args)

if __name__ == "__main__":
    main()