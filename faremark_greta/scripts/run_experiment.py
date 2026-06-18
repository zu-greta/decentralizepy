#!/usr/bin/env python
"""Stage 1 experiment runner: honest FedAvg, no free-riders, no watermark.

Usage (mirrors your existing submit script):
    python -u scripts/run_experiment.py \
        --config_idx 0 --repeat 0 --device cuda \
        --output_dir /path/out --data_root /path/data

It runs one (config, repeat), writes result.json to --output_dir, and prints a
PASS/FAIL correctness verdict against the expected accuracy band in config.py.
"""
import argparse
import json
import os
import sys
import time

import torch

# Make `import faremark` work whether run from repo root or scripts/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faremark.config import get_config, seed_for, CONFIGS
from faremark.utils import set_seed, get_logger
from faremark.models import build_model
from faremark.datasets import build_data
from faremark.client import Client
from faremark.attacks import build_clients
from faremark.server import Server
from faremark.wm_client import build_watermarked_clients
from faremark.wm_verify import WatermarkRegistry, build_trigger_bank, make_verifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_idx", type=int, default=None)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    # Optional overrides (handy for quick tests without editing the registry).
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    # Stage 2 overrides.
    p.add_argument("--attack", type=str, default=None,
                   choices=["none", "previous_models", "gaussian"])
    p.add_argument("--num_free_riders", type=int, default=None)
    p.add_argument("--noise_sigma", type=float, default=None)
    p.add_argument("--noise_decay", type=float, default=None)
    # Stage 3 overrides.
    p.add_argument("--watermark", dest="watermark", action="store_true", default=None)
    p.add_argument("--no_watermark", dest="watermark", action="store_false")
    p.add_argument("--wm_lambda", type=float, default=None)
    p.add_argument("--wm_beta", type=float, default=None)
    p.add_argument("--list_configs", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_configs:
        for i, c in enumerate(CONFIGS):
            print(f"{i}: {c.name}  ({c.model}/{c.dataset}, {c.num_clients} clients)")
        return

    missing = [n for n in ("config_idx", "output_dir", "data_root")
               if getattr(args, n) is None]
    if missing:
        sys.exit(f"error: missing required args: {', '.join('--' + m for m in missing)}")

    cfg = get_config(args.config_idx)
    # Apply overrides.
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.local_epochs is not None:
        cfg.local_epochs = args.local_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.attack is not None:
        cfg.attack = args.attack
    if args.num_free_riders is not None:
        cfg.num_free_riders = args.num_free_riders
    if args.noise_sigma is not None:
        cfg.noise_sigma = args.noise_sigma
    if args.noise_decay is not None:
        cfg.noise_decay = args.noise_decay
    if args.watermark is not None:
        cfg.watermark = args.watermark
    if args.wm_lambda is not None:
        cfg.wm_lambda = args.wm_lambda
    if args.wm_beta is not None:
        cfg.wm_beta = args.wm_beta

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(logfile=os.path.join(args.output_dir, "run.log"))

    seed = seed_for(cfg, args.repeat)
    set_seed(seed)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        logger.info(f"CUDA not available; falling back to {device}")

    logger.info(f"=== config[{args.config_idx}] {cfg.name} | repeat={args.repeat} "
                f"| seed={seed} | device={device} ===")
    logger.info(json.dumps(cfg.to_dict()))

    data = build_data(cfg.dataset, args.data_root, cfg.num_clients,
                      cfg.batch_size, seed, num_workers=args.num_workers)

    # Single shared model instance reused by every client (sequential sim).
    model = build_model(cfg.model, data.num_classes, data.in_channels).to(device)

    registry = None
    if getattr(cfg, "watermark", False):
        registry = WatermarkRegistry()
        clients, free_rider_indices = build_watermarked_clients(
            cfg, data.client_loaders, model, device, seed,
            data.num_classes, registry)
        logger.info(f"watermark ON: {len(registry)} clients registered, "
                    f"m={cfg.wm_bits or (data.num_classes - 1)//2} bits, "
                    f"lambda={cfg.wm_lambda}, beta={cfg.wm_beta}")
        if free_rider_indices:
            logger.info(f"free-riders ({cfg.attack}): clients {free_rider_indices}")
        # Dedicated model instance for extraction (don't disturb the trainer).
        verify_model = build_model(cfg.model, data.num_classes, data.in_channels)
        classes = sorted({e["trigger_class"] for e in registry.entries.values()})
        trigger_bank = build_trigger_bank(data.test_dataset, classes,
                                          cfg.wm_num_triggers, seed=seed)
        verify_hook = make_verifier(registry, trigger_bank, verify_model, device,
                                    free_rider_indices, eta=cfg.wm_eta,
                                    verify_every=cfg.wm_verify_every)
        server = Server(model, clients, data.test_loader, device, logger,
                        verify_hook=verify_hook)
    else:
        clients, free_rider_indices = build_clients(cfg, data.client_loaders,
                                                    model, device, seed)
        if free_rider_indices:
            logger.info(f"free-riders ({cfg.attack}): clients {free_rider_indices} "
                        f"of {cfg.num_clients}")
        server = Server(model, clients, data.test_loader, device, logger)

    t0 = time.time()
    history = server.run(cfg.rounds)
    elapsed = time.time() - t0

    final_acc = history[-1]["test_acc"]
    best_acc = max(h["test_acc"] for h in history)
    lo, hi = cfg.expected_acc
    passed = lo <= final_acc <= hi

    # Stage 3 watermark summary: take the most recent round that has metrics.
    wm_summary = {}
    if getattr(cfg, "watermark", False):
        wm_rounds = [h for h in history if "wm_benign_ber" in h]
        if wm_rounds:
            last = wm_rounds[-1]
            # Paper's detection threshold (Eq. 16): eta = mu + 3 sigma over the
            # benign bit-error-rate distribution across rounds.
            from faremark.watermark import calibrate_eta
            eta_cal = round(calibrate_eta([h.get("wm_benign_ber") for h in wm_rounds]), 4)
            wm_summary = {
                "wm_benign_ber": last.get("wm_benign_ber"),
                "wm_fr_ber": last.get("wm_fr_ber"),
                "wm_detect_acc": last.get("wm_detect_acc"),
                "wm_fpr": last.get("wm_fpr"),
                "wm_fr_recall": last.get("wm_fr_recall"),
                "wm_eta_used": cfg.wm_eta,
                "wm_eta_calibrated": eta_cal,   # Eq. 16 mu+3sigma
            }

    result = {
        "config_idx": args.config_idx,
        "config": cfg.to_dict(),
        "repeat": args.repeat,
        "seed": seed,
        "device": device,
        "attack": cfg.attack,
        "num_free_riders": cfg.num_free_riders,
        "free_rider_indices": free_rider_indices,
        "final_acc": final_acc,
        "best_acc": best_acc,
        "expected_acc": list(cfg.expected_acc),
        "correctness_pass": passed,
        "elapsed_sec": round(elapsed, 1),
        "watermark": getattr(cfg, "watermark", False),
        **wm_summary,
        "history": history,
    }
    out_path = os.path.join(args.output_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"--- final_acc={final_acc:.2f}%  best={best_acc:.2f}%  "
                f"expected={cfg.expected_acc}  elapsed={elapsed/60:.1f}min ---")
    verdict = "PASS" if passed else "FAIL"
    logger.info(f"CORRECTNESS CHECK: {verdict} "
                f"(final {final_acc:.2f}% vs expected {lo}-{hi}%)")
    logger.info(f"wrote {out_path}")

    # Non-zero exit on failure so a sweep / CI can detect it.
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
