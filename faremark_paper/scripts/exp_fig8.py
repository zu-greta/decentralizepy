#!/usr/bin/env python3
"""
Experiment: Figure 8 — Watermark Detection Rate Over Training Rounds
"Watermark detection rate of free-riders and benign clients"

Paper settings (Section V-D, Figure 8):
  - 10 clients, 1 free-rider (previous_models)
  - Local models uploaded every 10 training epochs = 1 communication round
  - 100 rounds total, evaluate every round
  - (a) ResNet-18 / CIFAR-10
  - (b) AlexNet   / MNIST
  - Reports: benign WM detection rate, FR WM detection rate, main task acc per round
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faremark_mod import FareMarkConfig, FareMarkTrainer

SUBFIGS = {
    "a": ("resnet18", "cifar10"),
    "b": ("alexnet",  "mnist"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfig",     type=str, required=True, choices=["a","b"])
    parser.add_argument("--repeat",     type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=os.environ.get("RESULTS_DIR", "./results"))
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--data_root",  type=str, default="./data")
    args = parser.parse_args()

    model, dataset = SUBFIGS[args.subfig]
    exp_name = f"fig8_{args.subfig}_{model}_{dataset}_rep{args.repeat}"

    cfg = FareMarkConfig(
        model_name      = model,
        dataset_name    = dataset,
        num_clients     = 10,
        num_free_riders = 1,
        free_rider_type = "previous_models",
        global_rounds   = 100,
        local_epochs    = 10,   # "every 10 training epochs = 1 communication round"
        batch_size      = 16,
        lr              = 0.01,
        wm_bits         = 8,
        n_triggers      = 50,   # paper uses 50 triggers for Fig 8 detection
        seed            = 42 + args.repeat,
        device          = args.device,
        data_root       = args.data_root,
        output_dir      = args.output_dir,
        exp_name        = exp_name,
        eval_every      = 1,    # evaluate every round to get the full curve
        save_every      = 100,
    )

    trainer = FareMarkTrainer(cfg)
    results = trainer.run()

    summary = {
        "subfig": args.subfig, "model": model, "dataset": dataset,
        "repeat": args.repeat,
        "rounds":               results["rounds"],
        "main_acc":             results["main_acc"],
        "wm_acc_benign":        results["wm_acc_benign"],
        "wm_acc_freerider":     results["wm_acc_freerider"],
        "fr_detection_acc":     results["fr_detection_acc"],
        "fpr":                  results["fpr"],
    }
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[FIG8{args.subfig.upper()}] Final benign WM={results['wm_acc_benign'][-1]:.3f}, "
          f"FR WM={results['wm_acc_freerider'][-1]:.3f}")

if __name__ == "__main__":
    main()