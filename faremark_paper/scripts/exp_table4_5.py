#!/usr/bin/env python3
"""
Experiments: Table IV & Table V — Advanced Free-Rider Scenarios

Table IV — "Detection Against Train-Then-Attack Free Rider"
  Free-rider participates for K rounds, then switches to free-riding at round 50.
  9 benign + 1 free-rider; detection measured at round 50.
  K values: 0, 10, 20, 30, 40, 50 (rounds of legitimate participation)

Table V — "Detection Against Training-Trigger-Sample-Only Free Rider"
  Free-rider trains only on N trigger samples (not full data).
  Tests: N = 1, 5, 10, 20, 50 trigger samples.
  This is handled inside the client as a data restriction.
"""

import os, sys, json, argparse, copy, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faremark_mod import FareMarkConfig, FareMarkTrainer, WatermarkKey
from faremark_mod.watermark import extract_watermark, bit_accuracy, watermark_similarity
from faremark_mod.client import FLClient
from faremark_mod.datasets import make_trigger_loader, Subset
from faremark_mod.train import FareMarkTrainer, accuracy


# -------------------------------------------------------------------------
# Table IV: Train-Then-Attack
# -------------------------------------------------------------------------

class TrainThenAttackTrainer(FareMarkTrainer):
    """
    Override: free-rider participates legitimately for `attack_round` rounds,
    then switches to free-riding from `attack_round` onwards.
    """
    def __init__(self, cfg, attack_round=50):
        super().__init__(cfg)
        self.attack_round = attack_round
        # The free-rider client starts as benign
        for cid in self.free_rider_ids:
            self.clients[cid].is_free_rider = False

    def run(self):
        import time
        cfg = self.cfg
        client_ids = list(range(cfg.num_clients))
        global_sd = self.server.get_global_state_dict()
        t0 = time.time()

        for rnd in range(1, cfg.global_rounds + 1):
            # Switch free-rider to actually free-riding at attack_round
            if rnd == self.attack_round:
                for cid in self.free_rider_ids:
                    self.clients[cid].is_free_rider = True
                    print(f"  [Round {rnd}] Free-rider {cid} switches to free-riding mode.")

            local_models = {}
            for cid in client_ids:
                local_models[cid] = self.clients[cid].train(global_sd)

            detected_fr = set()
            if rnd % cfg.eval_every == 0 or rnd == self.attack_round:
                for cid in client_ids:
                    tmp = copy.deepcopy(self.server.get_global_model())
                    tmp.load_state_dict(local_models[cid])
                    error, b_hat, is_fr = self.server.verify_watermark(
                        tmp, cid, self.trigger_loaders[cid], cfg.n_triggers)
                    if cid not in self.free_rider_ids:
                        self.server.record_benign_error(error)
                    if is_fr:
                        detected_fr.add(cid)

            global_sd = self.server.aggregate([local_models[c] for c in client_ids])

            if rnd % cfg.eval_every == 0:
                self._evaluate_and_log(rnd, detected_fr)
                elapsed = time.time() - t0
                print(f"  Round {rnd:4d}/{cfg.global_rounds} | "
                      f"Acc={self.results['main_acc'][-1]:.3f} | "
                      f"FR_det={self.results['fr_detection_acc'][-1]:.3f} | "
                      f"Elapsed={elapsed:.0f}s")

        self._save_results()
        return self.results


def run_table4(args):
    """Table IV: vary how many rounds the FR participates before switching."""
    attack_rounds = [0, 10, 20, 30, 40, 50]
    results_all = {}

    for k in attack_rounds:
        exp_name = f"table4_attack_at_round{k}_rep{args.repeat}"
        cfg = FareMarkConfig(
            model_name="resnet18", dataset_name="cifar10",
            num_clients=10, num_free_riders=1,
            free_rider_type="previous_models",
            global_rounds=100, local_epochs=2, batch_size=16,
            lr=0.01, wm_bits=8, n_triggers=50,
            seed=42 + args.repeat,
            device=args.device, data_root=args.data_root,
            output_dir=args.output_dir, exp_name=exp_name,
            eval_every=10, save_every=100,
        )
        trainer = TrainThenAttackTrainer(cfg, attack_round=k if k > 0 else 999)
        res = trainer.run()

        # Detection accuracy at/after the attack round
        relevant_idx = -1
        for i, r in enumerate(res["rounds"]):
            if r >= 50:
                relevant_idx = i
                break
        det = res["fr_detection_acc"][relevant_idx] if relevant_idx >= 0 else None

        results_all[f"attack_round_{k}"] = {
            "attack_round": k,
            "fr_detection_acc_at_round50": det,
            "final_main_acc": res["main_acc"][-1] if res["main_acc"] else None,
        }
        print(f"\n[TABLE4] Attack at round {k}: det@50={det}")

    out_dir = os.path.join(args.output_dir, f"table4_rep{args.repeat}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results_all, f, indent=2)


# -------------------------------------------------------------------------
# Table V: Trigger-Sample-Only Free Rider
# -------------------------------------------------------------------------

def run_table5(args):
    """
    Table V: FR trains on only N_trigger samples of the trigger class.
    We simulate this by limiting the FR's dataset to N_trigger samples.
    """
    n_trigger_samples_list = [1, 5, 10, 20, 50]
    results_all = {}

    for n_trig in n_trigger_samples_list:
        exp_name = f"table5_trigger_only_{n_trig}samples_rep{args.repeat}"
        cfg = FareMarkConfig(
            model_name="resnet18", dataset_name="cifar10",
            num_clients=10, num_free_riders=1,
            global_rounds=100, local_epochs=2, batch_size=16,
            lr=0.01, wm_bits=8, n_triggers=50,
            seed=42 + args.repeat,
            device=args.device, data_root=args.data_root,
            output_dir=args.output_dir, exp_name=exp_name,
            eval_every=10, save_every=100,
        )

        # Build trainer normally first, then override free-rider dataset
        trainer = FareMarkTrainer(cfg)

        # Override: free-rider client gets only n_trig trigger-class samples
        for cid in trainer.free_rider_ids:
            fr_client = trainer.clients[cid]
            trigger_class = trainer.server.trigger_classes[cid]
            # Find trigger samples in their dataset
            trigger_indices = []
            for i in range(len(fr_client.train_dataset)):
                _, label = fr_client.train_dataset[i]
                if label == trigger_class:
                    trigger_indices.append(i)
                if len(trigger_indices) >= n_trig:
                    break
            # Give FR only these samples (they train on them legitimately
            # to try to embed a watermark that passes inspection)
            restricted_dataset = Subset(fr_client.train_dataset, trigger_indices)
            fr_client.is_free_rider = False  # actually trains, but on tiny data
            fr_client.train_dataset = restricted_dataset
            fr_client._split_dataset()

        res = trainer.run()

        results_all[f"n_trigger_{n_trig}"] = {
            "n_trigger_samples": n_trig,
            "fr_detection_acc": res["fr_detection_acc"][-1] if res["fr_detection_acc"] else None,
            "fpr":              res["fpr"][-1] if res["fpr"] else None,
            "wm_acc_benign":    res["wm_acc_benign"][-1] if res["wm_acc_benign"] else None,
        }
        print(f"\n[TABLE5] N_trig={n_trig}: det={results_all[f'n_trigger_{n_trig}']['fr_detection_acc']:.3f}")

    out_dir = os.path.join(args.output_dir, f"table5_rep{args.repeat}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results_all, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table",      type=str, required=True, choices=["4","5"])
    parser.add_argument("--repeat",     type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=os.environ.get("RESULTS_DIR", "./results"))
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--data_root",  type=str, default="./data")
    args = parser.parse_args()

    if args.table == "4":
        run_table4(args)
    else:
        run_table5(args)

if __name__ == "__main__":
    main()