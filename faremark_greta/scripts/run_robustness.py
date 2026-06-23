#!/usr/bin/env python
""" TODO:
Stage 4 (part 2): robustness experiment driver — Figs. 9-10 and Table VI.

Trains a fully-watermarked model (all honest clients, no free-riders), then
applies each watermark-removal operation at a range of strengths and records
(task accuracy, watermark accuracy) for each. Two curves come out:
  * fine-tune sweep   -> Fig. 9   (watermark decays as task acc returns to baseline)
  * pruning sweep     -> Fig. 10  (tolerant to ~50%, collapses past ~60%)
  * quantization / DP -> §V-E / Table VI

Usage (cluster):
    python scripts/run_robustness.py --config_idx 11 --repeat 0 \
        --output_dir /mnt/nfs/home/zu/results --data_root /mnt/nfs/home/zu/data

Writes robustness.json (per-operation task/watermark accuracy) to output_dir.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from faremark.config import get_config, seed_for
from faremark.datasets import build_data
from faremark.models import build_model
from faremark.server import Server
from faremark.wm_client import build_watermarked_clients
from faremark.wm_verify import WatermarkRegistry, build_trigger_bank
from faremark import watermark as wm, robustness as rob
from faremark.utils import set_seed, evaluate_accuracy, get_logger


def watermark_accuracy(model, registry, bank, device):
    """Mean watermark-recovery accuracy (1 - BER) over all registered clients."""
    model.to(device).eval()
    accs = []
    with torch.no_grad():
        for cid, e in registry.entries.items():
            tc = e["trigger_class"]
            if tc not in bank:
                continue
            p = F.softmax(model(bank[tc].to(device)), dim=1)
            bits = wm.extract_bits(p, e["key"].to(device), e["kind"], e["alpha"], exclude=tc)
            accs.append(1.0 - wm.bit_error_rate(bits, e["target_bits"]))
    return sum(accs) / max(len(accs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_idx", type=int, default=11)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--output_dir", default="./out")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = get_config(args.config_idx); cfg.watermark = True
    cfg.num_free_riders = 0                      # fidelity setup: everyone honest
    seed = seed_for(cfg, args.repeat); set_seed(seed)
    log = get_logger()
    data = build_data(cfg.dataset, args.data_root, cfg.num_clients, cfg.batch_size, seed)
    model = build_model(cfg.model, data.num_classes, data.in_channels)

    registry = WatermarkRegistry()
    clients, _ = build_watermarked_clients(cfg, data.client_loaders, model,
                                           args.device, seed, data.num_classes, registry)
    server = Server(model, clients, data.test_loader, args.device, log)
    server.run(cfg.rounds)                        # train the watermarked model

    classes = sorted({e["trigger_class"] for e in registry.entries.values()})
    bank = build_trigger_bank(data.test_dataset, classes, cfg.wm_num_triggers, seed)

    def measure(m):
        return (round(evaluate_accuracy(m, data.test_loader, args.device), 2),
                round(watermark_accuracy(m, registry, bank, args.device) * 100, 2))

    results = {"baseline": measure(model), "finetune": {}, "prune": {}, "quantize": {}}
    log.info(f"baseline task/wm = {results['baseline']}")

    # Fig. 9: fine-tune (lambda=0) for increasing epochs on client 0's data.
    ft_loader = data.client_loaders[0]
    for ep in (2, 5, 10, 20):
        results["finetune"][ep] = measure(
            rob.finetune(model, ft_loader, ep, cfg.lr, cfg.momentum, cfg.weight_decay, args.device))
        log.info(f"finetune {ep}ep -> {results['finetune'][ep]}")

    # Fig. 10: pruning sweep.
    for amt in (0.2, 0.4, 0.5, 0.6, 0.8):
        results["prune"][amt] = measure(rob.prune_model(model, amt))
        log.info(f"prune {int(amt*100)}% -> {results['prune'][amt]}")

    # §V-E: quantization.
    for bits in (8, 4, 2):
        results["quantize"][bits] = measure(rob.quantize(model, bits))
        log.info(f"quantize {bits}bit -> {results['quantize'][bits]}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "robustness.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
