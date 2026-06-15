#!/usr/bin/env python3
"""
Experiment: Table II — Watermark Detection Accuracy
"Comparison of Watermark Detection Accuracy"

Paper settings (Section V-C):
  - Models: ResNet-18, AlexNet
  - Datasets: CIFAR-10 (10 clients), MNIST (10 clients), CIFAR-100 (100 clients)
  - Global rounds: 50, local epochs: 5
  - N_T = 100 trigger samples
  - All clients benign; measures Acc_wm mean ± std across clients
  - Repeated 10 times

Note: Paper also reports FedIPR (backdoor N_T=100, feature N_w=50/400).
      This script only runs FareMark (Ours). FedIPR numbers come from the paper.
"""

import os, sys, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faremark_mod import FareMarkConfig, FareMarkTrainer
from faremark_mod.watermark import extract_watermark, bit_accuracy
import torch

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
    parser.add_argument("--config_idx", type=int, required=True, help="0-5")
    parser.add_argument("--repeat",     type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=os.environ.get("RESULTS_DIR", "./results"))
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--data_root",  type=str, default="./data")
    args = parser.parse_args()

    c = CONFIGS[args.config_idx]
    exp_name = f"table2_{c['model_name']}_{c['dataset_name']}_n{c['num_clients']}_rep{args.repeat}"

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
        n_triggers      = 100,
        seed            = 42 + args.repeat,
        device          = args.device,
        data_root       = args.data_root,
        output_dir      = args.output_dir,
        exp_name        = exp_name,
        eval_every      = 10,
        save_every      = 50,
    )

    trainer = FareMarkTrainer(cfg)
    results = trainer.run()

    # Per-client watermark bit accuracies at final round
    global_model = trainer.server.get_global_model()
    device = trainer.device
    client_wm_accs = []
    for cid, tloader in trainer.trigger_loaders.items():
        with torch.no_grad():
            logits_list = []
            count = 0
            for imgs, _ in tloader:
                out = global_model(imgs.to(device))
                if hasattr(out, 'logits'): out = out.logits
                logits_list.append(out)
                count += imgs.size(0)
                if count >= cfg.n_triggers: break
            logits = torch.cat(logits_list)[:cfg.n_triggers]
            b_hat = extract_watermark(logits, trainer.keys[cid], cfg.smooth_fn, cfg.alpha_smooth)
            client_wm_accs.append(bit_accuracy(b_hat, trainer.keys[cid].B.to(device)))

    summary = {
        "model": c["model_name"], "dataset": c["dataset_name"],
        "num_clients": c["num_clients"], "repeat": args.repeat,
        "wm_acc_mean": float(np.mean(client_wm_accs)),
        "wm_acc_std":  float(np.std(client_wm_accs)),
        "wm_acc_per_client": [float(x) for x in client_wm_accs],
        "final_main_acc": results["main_acc"][-1] if results["main_acc"] else None,
    }
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[TABLE2] {exp_name}")
    print(f"  WM acc: {summary['wm_acc_mean']*100:.2f} ± {summary['wm_acc_std']*100:.2f}%")

if __name__ == "__main__":
    main()