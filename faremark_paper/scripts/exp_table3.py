#!/usr/bin/env python3
"""
Experiment: Table III — Free-Rider Detection Analysis
"Free Rider Detection Analysis"

Paper settings (Section V-D-2):
  - 10 clients, FR ratios: 20%, 30%, 40%, 50%, 60%, 70%, 80%
  - FR type: both previous_models and gaussian_noise
  - N_T = 50 triggers for detection
  - 100 global rounds, 2 local epochs
  - Metrics: detection accuracy, FPR

Also covers single FR case (1 FR out of 10) for Table III top rows.
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faremark_mod import FareMarkConfig, FareMarkTrainer

FR_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fr_ratio",   type=float, required=True,
                        help="Free-rider ratio (0.1, 0.2, ..., 0.8)")
    parser.add_argument("--fr_type",    type=str, default="previous_models",
                        choices=["previous_models", "gaussian_noise"])
    parser.add_argument("--model",      type=str, default="resnet18")
    parser.add_argument("--dataset",    type=str, default="cifar10")
    parser.add_argument("--repeat",     type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=os.environ.get("RESULTS_DIR", "./results"))
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--data_root",  type=str, default="./data")
    args = parser.parse_args()

    num_clients = 10
    num_fr = max(1, round(num_clients * args.fr_ratio))
    fr_pct = int(args.fr_ratio * 100)
    exp_name = (f"table3_{args.model}_{args.dataset}_{args.fr_type}"
                f"_fr{fr_pct}pct_rep{args.repeat}")

    cfg = FareMarkConfig(
        model_name      = args.model,
        dataset_name    = args.dataset,
        num_clients     = num_clients,
        num_free_riders = num_fr,
        free_rider_type = args.fr_type,
        global_rounds   = 100,
        local_epochs    = 2,
        batch_size      = 16,
        lr              = 0.01,
        wm_bits         = 8,
        n_triggers      = 50,
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
        "model": args.model, "dataset": args.dataset,
        "fr_type": args.fr_type, "fr_ratio": args.fr_ratio,
        "num_fr": num_fr, "repeat": args.repeat,
        "final_main_acc":   results["main_acc"][-1]          if results["main_acc"]          else None,
        "fr_detection_acc": results["fr_detection_acc"][-1]  if results["fr_detection_acc"]  else None,
        "fpr":              results["fpr"][-1]                if results["fpr"]               else None,
        "wm_acc_benign":    results["wm_acc_benign"][-1]     if results["wm_acc_benign"]     else None,
    }
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[TABLE3] FR={fr_pct}% | det={summary['fr_detection_acc']:.3f} | "
          f"fpr={summary['fpr']:.3f} | acc={summary['final_main_acc']:.3f}")

if __name__ == "__main__":
    main()