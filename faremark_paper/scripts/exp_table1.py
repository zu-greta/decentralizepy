#!/usr/bin/env python3
"""
Experiment: Table I — Fidelity Analysis
"Analysis of Model Accuracy With and Without Watermark Embedding"

Paper settings (Section V-A, V-B):
  - Models: ResNet-18, AlexNet
  - Datasets: CIFAR-10 (10 clients), MNIST (10 clients), CIFAR-100 (100 clients)
  - Global rounds: 50
  - Local epochs: 5 (per round)
  - Batch size: 16
  - No free-riders (all clients are benign)
  - Repeated 10 times; report mean ± std

Outputs: results/table1/<config>/results.json  for each of the 6 configurations.
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faremark_mod import FareMarkConfig, FareMarkTrainer

CONFIGS = [
    dict(model_name="resnet18", dataset_name="cifar10",  num_clients=10),
    dict(model_name="resnet18", dataset_name="mnist",    num_clients=10),
    dict(model_name="resnet18", dataset_name="cifar100", num_clients=100),
    dict(model_name="alexnet",  dataset_name="cifar10",  num_clients=10),
    dict(model_name="alexnet",  dataset_name="mnist",    num_clients=10),
    dict(model_name="alexnet",  dataset_name="cifar100", num_clients=10),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_idx", type=int, required=True,
                        help="Index into CONFIGS list (0-5). Submit one job per config.")
    parser.add_argument("--repeat",     type=int, default=0,
                        help="Repetition index (0-9). The paper averages 10 runs.")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("RESULTS_DIR", "./results"))
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--data_root",  type=str, default="./data")
    args = parser.parse_args()

    c = CONFIGS[args.config_idx]
    exp_name = f"table1_{c['model_name']}_{c['dataset_name']}_n{c['num_clients']}_rep{args.repeat}"

    cfg = FareMarkConfig(
        model_name   = c["model_name"],
        dataset_name = c["dataset_name"],
        num_clients  = c["num_clients"],
        num_free_riders = 0,
        global_rounds   = 50,
        local_epochs    = 5,
        batch_size      = 16,
        lr              = 0.01,
        wm_bits         = 8,
        lam             = 1.0,
        beta            = 0.9,
        smooth_fn       = "frac_power",
        alpha_smooth    = 0.5,
        n_triggers      = 100,
        seed            = 42 + args.repeat,   # different seed per repeat
        device          = args.device,
        data_root       = args.data_root,
        output_dir      = args.output_dir,
        exp_name        = exp_name,
        eval_every      = 10,
        save_every      = 50,
    )

    trainer = FareMarkTrainer(cfg)
    results = trainer.run()

    # Save a compact summary alongside the full results
    summary = {
        "config_idx":    args.config_idx,
        "repeat":        args.repeat,
        "model":         c["model_name"],
        "dataset":       c["dataset_name"],
        "num_clients":   c["num_clients"],
        "final_main_acc":       results["main_acc"][-1]       if results["main_acc"]       else None,
        "final_wm_acc_benign":  results["wm_acc_benign"][-1]  if results["wm_acc_benign"]  else None,
    }
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[TABLE1] {exp_name}")
    print(f"  Final acc : {summary['final_main_acc']:.4f}")
    print(f"  WM acc    : {summary['final_wm_acc_benign']:.4f}")

if __name__ == "__main__":
    main()