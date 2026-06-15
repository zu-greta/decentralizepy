#!/usr/bin/env python3
"""
Experiment: Figure 7 — Classification Accuracy vs Number of Free-Riders
"Classification accuracy under different number of free-riders"

Paper settings (Section V-D):
  - 10 total clients, varying free-riders: 0, 1, 2, ..., 8
  - (a) ResNet-18 / CIFAR-10 / previous_models
  - (b) AlexNet  / MNIST    / previous_models
  - (c) ResNet-18 / CIFAR-10 / gaussian_noise
  - (d) AlexNet  / MNIST    / gaussian_noise
  - Global rounds: 100, local epochs: 2
  - Reports final main task accuracy for each free-rider count
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faremark_mod import FareMarkConfig, FareMarkTrainer

# Sub-figure index → (model, dataset, fr_type)
SUBFIGS = {
    "a": ("resnet18", "cifar10",  "previous_models"),
    "b": ("alexnet",  "mnist",    "previous_models"),
    "c": ("resnet18", "cifar10",  "gaussian_noise"),
    "d": ("alexnet",  "mnist",    "gaussian_noise"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfig",     type=str, required=True, choices=["a","b","c","d"])
    parser.add_argument("--num_fr",     type=int, required=True, help="Number of free-riders (0-8)")
    parser.add_argument("--repeat",     type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=os.environ.get("RESULTS_DIR", "./results"))
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--data_root",  type=str, default="./data")
    args = parser.parse_args()

    model, dataset, fr_type = SUBFIGS[args.subfig]
    exp_name = f"fig7_{args.subfig}_{model}_{dataset}_fr{args.num_fr}_rep{args.repeat}"

    cfg = FareMarkConfig(
        model_name      = model,
        dataset_name    = dataset,
        num_clients     = 10,
        num_free_riders = args.num_fr,
        free_rider_type = fr_type,
        global_rounds   = 100,
        local_epochs    = 2,
        batch_size      = 16,
        lr              = 0.01,
        wm_bits         = 8,
        n_triggers      = 100,
        seed            = 42 + args.repeat,
        device          = args.device,
        data_root       = args.data_root,
        output_dir      = args.output_dir,
        exp_name        = exp_name,
        eval_every      = 10,
        save_every      = 100,
    )

    trainer = FareMarkTrainer(cfg)
    results = trainer.run()

    summary = {
        "subfig": args.subfig, "model": model, "dataset": dataset,
        "fr_type": fr_type, "num_fr": args.num_fr, "repeat": args.repeat,
        "final_main_acc":   results["main_acc"][-1]   if results["main_acc"]   else None,
        "final_wm_benign":  results["wm_acc_benign"][-1] if results["wm_acc_benign"] else None,
        "fr_detection_acc": results["fr_detection_acc"][-1] if results["fr_detection_acc"] else None,
        "rounds": results["rounds"],
        "main_acc_curve": results["main_acc"],
    }
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[FIG7{args.subfig.upper()}] FR={args.num_fr} → acc={summary['final_main_acc']:.4f}")

if __name__ == "__main__":
    main()