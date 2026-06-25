#!/usr/bin/env python
"""Experiment runner

Usage:
    python -u scripts/run_experiment.py \
        --config_idx 0 --repeat 0 --device cuda \
        --output_dir /path/out --data_root /path/data

It runs one (config, repeat), writes result.json to --output_dir, and prints a
PASS/FAIL correctness verdict against the expected accuracy band in config.py
"""
import argparse
import json
import os
import sys
import time

import torch

# Make `import faremark` work whether run from repo root or scripts/
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
    # Optional overrides 
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--wm_num_triggers", type=int, default=None)
    p.add_argument("--wm_bits", type=int, default=None)
    p.add_argument("--attack_round", type=int, default=None)
    p.add_argument("--n_trigger_samples", type=int, default=None)
    p.add_argument("--honest_prob", type=float, default=None)
    p.add_argument("--blend", type=float, default=None)
    p.add_argument("--partition", type=str, default=None,
                   choices=["iid", "dirichlet", "noniid"])
    p.add_argument("--dirichlet_alpha", type=float, default=None)
    p.add_argument("--local_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    # Free-rider attacks overrides
    p.add_argument("--attack", type=str, default=None,
                   choices=["none", "previous_models", "gaussian",
                            "train_then_attack", "trigger_only",
                            "random_round", "mixed"])
    p.add_argument("--num_free_riders", type=int, default=None)
    p.add_argument("--noise_sigma", type=float, default=None)
    p.add_argument("--noise_decay", type=float, default=None)
    # Watermarking overrides
    p.add_argument("--watermark", dest="watermark", action="store_true", default=None)
    p.add_argument("--no_watermark", dest="watermark", action="store_false")
    p.add_argument("--wm_lambda", type=float, default=None)
    p.add_argument("--wm_beta", type=float, default=None)
    p.add_argument("--paper_faithful", dest="paper_faithful",
                   action="store_true", default=None,
                   help="strip our deviations: random keys, no trigger-class "
                        "exclusion, cumulative uncapped mu+3sigma threshold")
    p.add_argument("--calib_on_all", dest="calib_on_all",
                   action="store_true", default=None,
                   help="calibrate eta over ALL clients (free-riders poison it) "
                        "instead of the assumed trusted benign pool")
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
    # apply overrides from command line args (if not None)
    if args.model is not None:
        cfg.model = args.model
    if args.dataset is not None:
        cfg.dataset = args.dataset
    if args.wm_num_triggers is not None:
        cfg.wm_num_triggers = args.wm_num_triggers
    if args.wm_bits is not None:
        cfg.wm_bits = args.wm_bits
    if args.attack_round is not None:
        cfg.attack_round = args.attack_round
    if args.n_trigger_samples is not None:
        cfg.n_trigger_samples = args.n_trigger_samples
    if args.honest_prob is not None:
        cfg.honest_prob = args.honest_prob
    if args.blend is not None:
        cfg.blend = args.blend
    if args.partition is not None:
        cfg.partition = args.partition
    if args.dirichlet_alpha is not None:
        cfg.dirichlet_alpha = args.dirichlet_alpha
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
    if args.paper_faithful is not None:
        cfg.paper_faithful = args.paper_faithful
    if args.calib_on_all is not None:
        cfg.calib_on_all = args.calib_on_all

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
                      cfg.batch_size, seed, num_workers=args.num_workers,
                      partition=cfg.partition, dirichlet_alpha=cfg.dirichlet_alpha)

    # single shared model instance reused by every client (sequential sim)
    model = build_model(cfg.model, data.num_classes, data.in_channels).to(device)

    registry = None
    if getattr(cfg, "watermark", False):
        registry = WatermarkRegistry()
        clients, free_rider_indices = build_watermarked_clients(
            cfg, data.client_loaders, model, device, seed,
            data.num_classes, registry)
        logger.info(f"watermark ON: {len(registry)} clients registered, "
                    f"m={registry.m} bits, l={registry.l}, "
                    f"unembeddable={registry.unembeddable_frac:.2f}, "
                    f"lambda={cfg.wm_lambda}, beta={cfg.wm_beta}")
        if free_rider_indices:
            logger.info(f"free-riders ({cfg.attack}): clients {free_rider_indices}")
        # dedicated model instance for extraction (don't disturb the trainer)
        verify_model = build_model(cfg.model, data.num_classes, data.in_channels)
        classes = sorted({e["trigger_class"] for e in registry.entries.values()})
        trigger_bank = build_trigger_bank(data.test_dataset, classes,
                                          cfg.wm_num_triggers, seed=seed)
        verify_hook = make_verifier(registry, trigger_bank, verify_model, device,
                                    free_rider_indices, eta=cfg.wm_eta,
                                    verify_every=cfg.wm_verify_every,
                                    paper_faithful=getattr(cfg, "paper_faithful", False),
                                    calib_on_all=getattr(cfg, "calib_on_all", False))
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

    # watermark summary: report the converged decision (Table III), i.e.
    # averaged over the last K rounds, not a single noisy round
    wm_summary = {}
    if getattr(cfg, "watermark", False):
        wm_rounds = [h for h in history if "wm_benign_ber" in h]
        if wm_rounds:
            from faremark.watermark import calibrate_eta
            K = min(10, len(wm_rounds))
            tail = wm_rounds[-K:]                         # converged window

            def _avg(key):
                vals = [h.get(key) for h in tail if h.get(key) is not None]
                return round(sum(vals) / len(vals), 4) if vals else None

            wm_summary = {
                "wm_benign_ber": _avg("wm_benign_ber"),   # mean over last K rounds
                "wm_fr_ber": _avg("wm_fr_ber"),
                "wm_detect_acc": _avg("wm_detect_acc"),
                "wm_fpr": _avg("wm_fpr"),
                "wm_fr_recall": _avg("wm_fr_recall"),
                "wm_detect_window": K,                    # how many rounds averaged
                # Threshold actually used in the converged window (windowed+capped,
                # Eq. 16). NOT the cumulative mu+3sigma, which a transient model
                # collapse can poison
                "wm_eta_used": _avg("wm_eta_round"),
                # embeddability diagnostics (explain any honest-BER floor):
                "wm_bits_m": registry.m,
                "wm_group_size_l": registry.l,
                "wm_unembeddable_frac": registry.unembeddable_frac,
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

    # non-zero exit on failure so a sweep / CI can detect it
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()