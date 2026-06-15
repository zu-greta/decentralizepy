#!/usr/bin/env python3
"""
Experiment: Table VI — Robustness Against Differential Privacy
"Robustness Against Differential Privacy (ResNet-18 on CIFAR-10)"

Paper settings (Section V-E-1):
  - Model: ResNet-18, Dataset: CIFAR-10, 10 clients, no free-riders
  - DP applied via Opacus during training
  - Tests varying noise multipliers / epsilon values
  - Reports: main task accuracy, watermark extraction accuracy
"""

import os, sys, json, argparse, copy, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faremark_mod import FareMarkConfig, FareMarkTrainer
from faremark_mod.watermark import extract_watermark, bit_accuracy
from faremark_mod.train import accuracy
from faremark_mod.evaluate import train_with_dp
import torch
from torch.utils.data import DataLoader

# Noise multipliers to test (higher = stronger DP = more noise)
NOISE_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_mult",  type=float, required=True,
                        choices=NOISE_MULTIPLIERS,
                        help="DP noise multiplier (0.0 = no DP)")
    parser.add_argument("--extra_epochs", type=int, default=0,
                        help="Additional fine-tune epochs after DP training (paper varies this)")
    parser.add_argument("--repeat",     type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=os.environ.get("RESULTS_DIR", "./results"))
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--data_root",  type=str, default="./data")
    args = parser.parse_args()

    nm_str = str(args.noise_mult).replace(".", "p")
    exp_name = f"table6_dp_nm{nm_str}_ep{args.extra_epochs}_rep{args.repeat}"

    # Step 1: Train a normal watermarked model first
    base_cfg = FareMarkConfig(
        model_name="resnet18", dataset_name="cifar10",
        num_clients=10, num_free_riders=0,
        global_rounds=50, local_epochs=5, batch_size=16,
        lr=0.01, wm_bits=8, n_triggers=100,
        seed=42 + args.repeat,
        device=args.device, data_root=args.data_root,
        output_dir=args.output_dir,
        exp_name=f"table6_base_rep{args.repeat}",
        eval_every=10, save_every=50,
    )
    trainer = FareMarkTrainer(base_cfg)
    trainer.run()
    watermarked_model = copy.deepcopy(trainer.server.get_global_model())
    device = trainer.device

    # Step 2: Apply DP fine-tuning if noise > 0
    if args.noise_mult > 0:
        train_loader = DataLoader(
            trainer.train_dataset,
            batch_size=base_cfg.batch_size,
            shuffle=True,
            num_workers=2,
        )
        dp_model = train_with_dp(
            model=watermarked_model,
            train_loader=train_loader,
            device=device,
            epochs=5 + args.extra_epochs,
            lr=base_cfg.lr,
            noise_multiplier=args.noise_mult,
        )
    else:
        dp_model = watermarked_model

    dp_model = dp_model.to(device)

    # Step 3: Evaluate
    main_acc = accuracy(dp_model, trainer.test_loader, device)

    wm_accs = []
    for cid, tloader in trainer.trigger_loaders.items():
        with torch.no_grad():
            logits_list, count = [], 0
            for imgs, _ in tloader:
                out = dp_model(imgs.to(device))
                if hasattr(out, 'logits'): out = out.logits
                logits_list.append(out)
                count += imgs.size(0)
                if count >= base_cfg.n_triggers: break
            if logits_list:
                logits = torch.cat(logits_list)[:base_cfg.n_triggers]
                b_hat = extract_watermark(logits, trainer.keys[cid],
                                          base_cfg.smooth_fn, base_cfg.alpha_smooth)
                wm_accs.append(bit_accuracy(b_hat, trainer.keys[cid].B.to(device)))

    summary = {
        "noise_mult": args.noise_mult,
        "extra_epochs": args.extra_epochs,
        "repeat": args.repeat,
        "main_acc": main_acc,
        "wm_acc_mean": float(np.mean(wm_accs)) if wm_accs else None,
        "wm_acc_std":  float(np.std(wm_accs))  if wm_accs else None,
    }
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[TABLE6] noise={args.noise_mult} ep+{args.extra_epochs} | "
          f"acc={main_acc:.3f} | wm={summary['wm_acc_mean']:.3f}")

if __name__ == "__main__":
    main()