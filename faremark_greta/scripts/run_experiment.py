#!/usr/bin/env python
"""Experiment runner.

    python -u scripts/run_experiment.py \
        --config_idx 14 --repeat 0 --device cuda \
        --output_dir /path/out --data_root /path/data

Runs one (config, repeat); writes result.json to --output_dir.
result.json carries "manifest" (self-describing metadata), "compute" (per-client
effort), and "history" (per-round metrics incl. wm_per_client BER lists).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faremark.config import get_config, seed_for, CONFIGS
from faremark.utils import set_seed, get_logger
from faremark.models import build_model
from faremark.datasets import build_data
from faremark.attacks import build_clients
from faremark.server import Server
from faremark.wm_client import build_watermarked_clients
from faremark.wm_verify import WatermarkRegistry, build_trigger_bank, make_verifier
from faremark.compute_meter import estimate_flops_per_sample_fwd
from faremark.manifest import build_manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_idx", type=int, default=None)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    # ---- general overrides ----
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--local_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--partition", type=str, default=None,
                   choices=["iid", "dirichlet", "noniid"])
    p.add_argument("--dirichlet_alpha", type=float, default=None)
    # ---- free-rider selection ----
    p.add_argument("--attack", type=str, default=None,
                   choices=["none", "previous_models", "gaussian", "autopilot"])
    p.add_argument("--num_free_riders", type=int, default=None)
    p.add_argument("--free_rider_ids", type=str, default=None,
                   help="pin which cids free-ride, e.g. '3,6' (overrides the seeded choice)")
    p.add_argument("--noise_sigma", type=float, default=None)
    p.add_argument("--noise_decay", type=float, default=None)
    # ---- autopilot overrides ----
    p.add_argument("--autop_oracle_eta", type=float, default=None)
    p.add_argument("--autop_honest_until", type=int, default=None)
    p.add_argument("--autop_conv_eps", type=float, default=None)
    p.add_argument("--autop_honest_extra", type=int, default=None)
    p.add_argument("--autop_eta_k", type=float, default=None)
    p.add_argument("--autop_protect_until", type=int, default=None)
    p.add_argument("--autop_warmup_cap", type=int, default=None)
    p.add_argument("--autop_max_batches", type=int, default=None)
    p.add_argument("--autop_margin0", type=float, default=None)
    p.add_argument("--autop_floor", type=float, default=None)
    p.add_argument("--autop_common_per_class", type=int, default=None)
    p.add_argument("--autop_scope", default=None, choices=["full", "block", "block2", "head"])
    p.add_argument("--autop_stay_min", action="store_true", default=None,
                   help="coast when safely under target, tap only when needed (default: tap every round)")
    p.add_argument("--autop_holdout_ratio", type=float, default=None)
    p.add_argument("--autop_honest_clone", action="store_true", default=None,
                   help="DIAGNOSTIC: embed via the exact honest path every round")
    # ---- watermarking overrides ----
    p.add_argument("--watermark", dest="watermark", action="store_true", default=None)
    p.add_argument("--no_watermark", dest="watermark", action="store_false")
    p.add_argument("--wm_bits", type=int, default=None)
    p.add_argument("--wm_num_triggers", type=int, default=None)
    p.add_argument("--wm_lambda", type=float, default=None)
    p.add_argument("--wm_beta", type=float, default=None)
    p.add_argument("--paper_faithful", dest="paper_faithful",
                   action="store_true", default=None)
    p.add_argument("--calib_on_all", dest="calib_on_all",
                   action="store_true", default=None,
                   help="calibrate eta over ALL clients (free-riders poison it)")
    # ---- manifest (descriptive only) ----
    p.add_argument("--manifest_family", type=str, default=None)
    p.add_argument("--manifest_note", type=str, default=None)
    p.add_argument("--sweep_var", type=str, default=None)
    p.add_argument("--sweep_level", type=str, default=None)
    p.add_argument("--list_configs", action="store_true")
    return p.parse_args()


_OVERRIDABLE = [
    "model", "dataset", "partition", "dirichlet_alpha", "rounds", "local_epochs",
    "batch_size", "lr", "attack", "num_free_riders", "free_rider_ids",
    "noise_sigma", "noise_decay",
    "autop_oracle_eta", "autop_honest_until", "autop_conv_eps", "autop_honest_extra",
    "autop_eta_k", "autop_protect_until", "autop_warmup_cap", "autop_max_batches",
    "autop_margin0", "autop_floor", "autop_common_per_class", "autop_scope",
    "autop_stay_min", "autop_holdout_ratio", "autop_honest_clone",
    "watermark", "wm_bits", "wm_num_triggers", "wm_lambda", "wm_beta",
    "paper_faithful", "calib_on_all",
]


def collect_compute(clients, free_rider_indices):
    """Per-client + summarized training effort. Crude free-riders that never
    train have no meter -> reported as zero compute."""
    fr_set = set(free_rider_indices)
    zero_total = {"samples": 0, "fwd_passes": 0, "bwd_passes": 0, "opt_steps": 0,
                  "gpu_ms": 0.0, "wall_ms": 0.0, "flops": 0.0,
                  "rounds_trained": 0, "rounds_total": 0, "duty_cycle": 0.0}
    per_client, honest_gpu, fr_gpu, honest_s, fr_s = {}, [], [], [], []
    for cid, c in enumerate(clients):
        meter = getattr(c, "meter", None)
        atk = getattr(c, "attack_name", "honest")
        isfr = cid in fr_set
        if meter is not None:
            s = meter.summary(attack_name=atk, is_free_rider=isfr)
        else:
            s = {"attack_name": atk, "is_free_rider": isfr,
                 "total": dict(zero_total), "per_round": {}}
        if getattr(c, "trace", None):
            s["trace"] = c.trace
        per_client[cid] = s
        tot = s["total"]
        (fr_gpu if isfr else honest_gpu).append(tot["gpu_ms"])
        (fr_s if isfr else honest_s).append(tot["samples"])

    def _mean(v):
        return round(sum(v) / len(v), 3) if v else 0.0

    hm_gpu, fm_gpu, hm_s, fm_s = _mean(honest_gpu), _mean(fr_gpu), _mean(honest_s), _mean(fr_s)
    summary = {
        "honest_mean_gpu_ms": hm_gpu, "fr_mean_gpu_ms": fm_gpu,
        "honest_mean_samples": hm_s, "fr_mean_samples": fm_s,
        "effort_ratio_gpu": round(fm_gpu / hm_gpu, 4) if hm_gpu else None,
        "effort_ratio_samples": round(fm_s / hm_s, 4) if hm_s else None,
    }
    return {"summary": summary, "per_client": per_client}


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
    for name in _OVERRIDABLE:
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg, name, v)

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

    model = build_model(cfg.model, data.num_classes, data.in_channels).to(device)

    try:
        sample_shape = tuple(data.test_dataset[0][0].shape)
        fps = estimate_flops_per_sample_fwd(model, sample_shape, device=device)
    except Exception as e:
        logger.info(f"FLOPs estimate skipped: {e}")
        fps = None
    if fps:
        logger.info(f"flops/sample (fwd) ~= {fps:.3e}")

    registry = None
    if getattr(cfg, "watermark", False):
        registry = WatermarkRegistry()
        clients, free_rider_indices = build_watermarked_clients(
            cfg, data.client_loaders, model, device, seed,
            data.num_classes, registry)
        logger.info(f"watermark ON: {len(registry)} clients, m={registry.m} bits, "
                    f"l={registry.l}, unembeddable={registry.unembeddable_frac:.2f}")
        if free_rider_indices:
            logger.info(f"free-riders ({cfg.attack}): clients {free_rider_indices}")
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
            logger.info(f"free-riders ({cfg.attack}): clients {free_rider_indices}")
        server = Server(model, clients, data.test_loader, device, logger)

    if fps:
        for c in clients:
            m = getattr(c, "meter", None)
            if m is not None:
                m.flops_per_sample_fwd = fps

    t0 = time.time()
    history = server.run(cfg.rounds)
    elapsed = time.time() - t0

    final_acc = history[-1]["test_acc"]
    best_acc = max(h["test_acc"] for h in history)
    lo, hi = cfg.expected_acc
    passed = lo <= final_acc <= hi

    wm_summary = {}
    if getattr(cfg, "watermark", False):
        wm_rounds = [h for h in history if "wm_benign_ber" in h]
        if wm_rounds:
            K = min(10, len(wm_rounds))
            tail = wm_rounds[-K:]

            def _avg(key):
                vals = [h.get(key) for h in tail if h.get(key) is not None]
                return round(sum(vals) / len(vals), 4) if vals else None

            wm_summary = {
                "wm_benign_ber": _avg("wm_benign_ber"),
                "wm_fr_ber": _avg("wm_fr_ber"),
                "wm_detect_acc": _avg("wm_detect_acc"),
                "wm_fpr": _avg("wm_fpr"),
                "wm_fr_recall": _avg("wm_fr_recall"),
                "wm_detect_window": K,
                "wm_eta_used": _avg("wm_eta_round"),
                "wm_bits_m": registry.m,
                "wm_group_size_l": registry.l,
                "wm_unembeddable_frac": registry.unembeddable_frac,
            }

    compute = collect_compute(clients, free_rider_indices)
    manifest = build_manifest(cfg, args)

    result = {
        "config_idx": args.config_idx,
        "config": cfg.to_dict(),
        "manifest": manifest,
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
        "flops_per_sample_fwd": fps,
        **wm_summary,
        "compute": compute,
        "history": history,
    }
    out_path = os.path.join(args.output_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"--- final_acc={final_acc:.2f}%  best={best_acc:.2f}%  "
                f"expected={cfg.expected_acc}  elapsed={elapsed/60:.1f}min ---")
    cs = compute["summary"]
    logger.info(f"compute: honest {cs['honest_mean_gpu_ms']:.0f} ms/client, "
                f"FR {cs['fr_mean_gpu_ms']:.0f} ms/client, "
                f"effort_ratio_gpu={cs['effort_ratio_gpu']}")
    logger.info(f"CORRECTNESS: {'PASS' if passed else 'FAIL'} "
                f"(final {final_acc:.2f}% vs {lo}-{hi}%)")
    logger.info(f"wrote {out_path}")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()