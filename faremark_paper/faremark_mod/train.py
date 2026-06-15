"""
FareMark: Main federated training loop.

Orchestrates the three stages of FareMark per communication round:
  Stage I  (once): Trigger assignment & key registration
  Stage II (each round): Local training → FedAvg → watermark verification
  Stage III (post-training): IPR / copyright verification
"""

import os
import copy
import json
import random
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from .config import FareMarkConfig
from .models import build_model_for_dataset
from .datasets import load_dataset, split_iid, make_trigger_loader
from .watermark import WatermarkKey, bit_accuracy
from .client import FLClient
from .server import FLServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute top-1 classification accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            try:
                out = model(images)
                # GoogLeNet returns namedtuple in train mode; handle both
                if hasattr(out, 'logits'):
                    out = out.logits
            except Exception:
                out = model(images)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class FareMarkTrainer:
    """
    Full FareMark training orchestrator.

    Usage:
        trainer = FareMarkTrainer(config)
        results = trainer.run()
    """

    def __init__(self, cfg: FareMarkConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        self.device = torch.device(
            cfg.device if torch.cuda.is_available() and cfg.device == "cuda"
            else "cpu"
        )
        print(f"Using device: {self.device}")

        os.makedirs(cfg.output_dir, exist_ok=True)

        # ----------------------------------------------------------------
        # Datasets
        # ----------------------------------------------------------------
        print(f"Loading dataset: {cfg.dataset_name}")
        self.train_dataset = load_dataset(cfg.dataset_name, cfg.data_root, train=True)
        self.test_dataset  = load_dataset(cfg.dataset_name, cfg.data_root, train=False)

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        # IID split across clients
        self.client_datasets = split_iid(
            self.train_dataset, cfg.num_clients, seed=cfg.seed
        )

        # ----------------------------------------------------------------
        # Global model & server
        # ----------------------------------------------------------------
        print(f"Building model: {cfg.model_name}")
        global_model = build_model_for_dataset(cfg.model_name, cfg.dataset_name)

        # Number of output classes
        self.num_classes = {
            "mnist": 10, "cifar10": 10, "cifar100": 100, "food100": 100
        }[cfg.dataset_name.lower()]

        self.server = FLServer(
            global_model=global_model,
            num_clients=cfg.num_clients,
            num_classes=self.num_classes,
            wm_bits=cfg.wm_bits,
            device=self.device,
            smooth_fn=cfg.smooth_fn,
            alpha=cfg.alpha_smooth,
            eta=cfg.eta,
        )

        # ----------------------------------------------------------------
        # Stage I: Trigger assignment & client key generation
        # ----------------------------------------------------------------
        client_ids = list(range(cfg.num_clients))
        trigger_map = self.server.assign_triggers(client_ids)

        self.keys: Dict[int, WatermarkKey] = {}
        for cid in client_ids:
            key = WatermarkKey(
                num_classes=self.num_classes,
                wm_bits=cfg.wm_bits,
                client_id=cid,
                device=self.device,
            )
            self.keys[cid] = key
            self.server.register_client(cid, key)

        # ----------------------------------------------------------------
        # Trigger loaders (per client, from test set)
        # ----------------------------------------------------------------
        self.trigger_loaders: Dict[int, DataLoader] = {}
        for cid in client_ids:
            self.trigger_loaders[cid] = make_trigger_loader(
                self.test_dataset,
                trigger_class=trigger_map[cid],
                batch_size=64,
                n_max=cfg.n_triggers + 50,
            )

        # ----------------------------------------------------------------
        # Determine free-rider indices
        # ----------------------------------------------------------------
        self.free_rider_ids = set(
            random.sample(client_ids, cfg.num_free_riders)
        ) if cfg.num_free_riders > 0 else set()
        if self.free_rider_ids:
            print(f"Free-rider clients: {sorted(self.free_rider_ids)}")

        # ----------------------------------------------------------------
        # Build FL clients
        # ----------------------------------------------------------------
        self.clients: Dict[int, FLClient] = {}
        for cid in client_ids:
            client_model = copy.deepcopy(
                self.server.get_global_model()
            )
            self.clients[cid] = FLClient(
                client_id=cid,
                model=client_model,
                trigger_class=trigger_map[cid],
                key=self.keys[cid],
                train_dataset=self.client_datasets[cid],
                device=self.device,
                lr=cfg.lr,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                lam=cfg.lam,
                beta=cfg.beta,
                smooth_fn=cfg.smooth_fn,
                alpha=cfg.alpha_smooth,
                is_free_rider=(cid in self.free_rider_ids),
                free_rider_type=cfg.free_rider_type,
            )

        # ----------------------------------------------------------------
        # Results tracking
        # ----------------------------------------------------------------
        self.results = {
            "rounds": [],
            "main_acc": [],
            "wm_acc_benign": [],    # per-round mean bit accuracy for benign clients
            "wm_acc_freerider": [], # per-round mean bit accuracy for free-riders
            "fr_detection_acc": [], # fraction of free-riders correctly detected
            "fpr": [],              # false positive rate (benign flagged as FR)
            "config": vars(cfg),
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.cfg
        client_ids = list(range(cfg.num_clients))
        t0 = time.time()

        print(f"\n{'='*60}")
        print(f"FareMark Training: {cfg.exp_name}")
        print(f"Clients={cfg.num_clients}, FreeRiders={cfg.num_free_riders}, "
              f"Rounds={cfg.global_rounds}, Dataset={cfg.dataset_name}, "
              f"Model={cfg.model_name}")
        print(f"{'='*60}\n")

        global_sd = self.server.get_global_state_dict()

        for rnd in range(1, cfg.global_rounds + 1):
            # --- Local training ---
            local_models = {}
            for cid in client_ids:
                local_sd = self.clients[cid].train(global_sd)
                local_models[cid] = local_sd

            # --- Stage II: watermark check before aggregation ---
            detected_fr = set()
            if rnd % cfg.eval_every == 0:
                for cid in client_ids:
                    # Temporarily load submitted model for verification
                    tmp_model = copy.deepcopy(self.server.get_global_model())
                    tmp_model.load_state_dict(local_models[cid])

                    error, b_hat, is_fr = self.server.verify_watermark(
                        model=tmp_model,
                        client_id=cid,
                        trigger_loader=self.trigger_loaders[cid],
                        n_triggers=cfg.n_triggers,
                    )

                    if cid not in self.free_rider_ids:
                        self.server.record_benign_error(error)

                    if is_fr:
                        detected_fr.add(cid)

            # --- FedAvg aggregation (include all — detection is advisory) ---
            global_sd = self.server.aggregate(
                [local_models[cid] for cid in client_ids]
            )

            # --- Periodic evaluation ---
            if rnd % cfg.eval_every == 0:
                self._evaluate_and_log(rnd, detected_fr)
                elapsed = time.time() - t0
                print(f"  Round {rnd:4d}/{cfg.global_rounds} | "
                      f"Acc={self.results['main_acc'][-1]:.3f} | "
                      f"WM_benign={self.results['wm_acc_benign'][-1]:.3f} | "
                      f"FR_det={self.results['fr_detection_acc'][-1]:.3f} | "
                      f"FPR={self.results['fpr'][-1]:.3f} | "
                      f"Elapsed={elapsed:.0f}s")

            # --- Checkpoint ---
            if rnd % cfg.save_every == 0:
                self._save_checkpoint(rnd)

        # Save final results
        self._save_results()
        print(f"\nTraining complete. Results saved to {cfg.output_dir}/{cfg.exp_name}/")
        return self.results

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_and_log(self, rnd: int, detected_fr: set):
        cfg = self.cfg
        global_model = self.server.get_global_model()
        global_sd = self.server.get_global_state_dict()

        # Main task accuracy
        main_acc = accuracy(global_model, self.test_loader, self.device)

        # Watermark extraction accuracy from the global model
        benign_accs = []
        fr_accs = []
        client_ids = list(range(cfg.num_clients))

        for cid in client_ids:
            tloader = self.trigger_loaders[cid]
            error, b_hat, _ = self.server.verify_watermark(
                model=copy.deepcopy(global_model),
                client_id=cid,
                trigger_loader=tloader,
                n_triggers=cfg.n_triggers,
            )
            acc = bit_accuracy(b_hat, self.keys[cid].B.to(self.device))
            if cid in self.free_rider_ids:
                fr_accs.append(acc)
            else:
                benign_accs.append(acc)

        # Free-rider detection accuracy and FPR
        true_fr = self.free_rider_ids
        benign_ids = set(client_ids) - true_fr

        if true_fr:
            fr_det_acc = len(detected_fr & true_fr) / len(true_fr)
        else:
            fr_det_acc = float('nan')

        if benign_ids:
            fpr = len(detected_fr & benign_ids) / len(benign_ids)
        else:
            fpr = 0.0

        self.results["rounds"].append(rnd)
        self.results["main_acc"].append(main_acc)
        self.results["wm_acc_benign"].append(
            float(np.mean(benign_accs)) if benign_accs else float('nan')
        )
        self.results["wm_acc_freerider"].append(
            float(np.mean(fr_accs)) if fr_accs else float('nan')
        )
        self.results["fr_detection_acc"].append(fr_det_acc)
        self.results["fpr"].append(fpr)

    def _save_checkpoint(self, rnd: int):
        out_dir = os.path.join(self.cfg.output_dir, self.cfg.exp_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"checkpoint_round{rnd}.pt")
        torch.save({
            "round": rnd,
            "global_model": self.server.get_global_state_dict(),
            "results": self.results,
        }, path)

    def _save_results(self):
        out_dir = os.path.join(self.cfg.output_dir, self.cfg.exp_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "results.json")
        # Convert any non-serializable values
        clean = {}
        for k, v in self.results.items():
            if isinstance(v, list):
                clean[k] = [float(x) if isinstance(x, (np.floating, float)) and not
                             (x != x) else ('nan' if (isinstance(x, float) and x != x)
                             else x) for x in v]
            else:
                clean[k] = v
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)