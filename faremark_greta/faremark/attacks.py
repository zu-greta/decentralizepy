"""Stage 2: free-rider attacks.

A free-rider submits a fabricated local model instead of doing real training, to
obtain the valuable global model "for free". We implement the paper's two
constructions (Section V-A2). Both subclass Client and override ONLY
`produce_update`, so they drop into the existing FedAvg loop with no other
changes.

PreviousModelsFreeRider (Eq. 17): build a plausible-looking update by
    extrapolating from the two most recent global models the client received,
        W_free = W_t + (W_t - W_{t-1}) = 2 W_t - W_{t-1}
    This is Lin et al.'s "delta-weights" free-rider: it mimics continued
    optimization progress without doing any. In round 1 (no W_{t-1}) it falls
    back to resubmitting W_t.

GaussianNoiseFreeRider (Eq. 18): perturb the current global model with noise,
        W_free = W_t + N(0, sigma^2)
    Optionally decay sigma across rounds (Fraboni et al. use shrinking noise to
    better disguise the free-rider): sigma_t = sigma0 * t^(-gamma).

Free-riders still "own" a data shard in the simulation (so they report a normal
sample count for FedAvg weighting) — they simply never train on it.
"""
import copy
import random

import torch

from .client import Client


# ---- weight-fabrication helpers (operate on CPU state dicts) ----------------
def _extrapolate(w_t: dict, w_prev: dict) -> dict:
    """Elementwise 2*W_t - W_{t-1} over float tensors; copy non-float buffers."""
    out = {}
    for k, v in w_t.items():
        if v.is_floating_point() and k in w_prev:
            out[k] = 2.0 * v - w_prev[k]
        else:
            out[k] = v.clone()
    return out


def _add_noise(state: dict, sigma: float, generator=None) -> dict:
    out = {}
    for k, v in state.items():
        if v.is_floating_point():
            out[k] = v + torch.randn(v.shape, generator=generator) * sigma
        else:
            out[k] = v.clone()
    return out


# ---- free-rider clients -----------------------------------------------------
class PreviousModelsFreeRider(Client):
    is_free_rider = True
    attack_name = "previous_models"

    def produce_update(self, global_state, prev_global_state, round_idx):
        if prev_global_state is None:
            fake = copy.deepcopy(global_state)
        else:
            fake = _extrapolate(global_state, prev_global_state)
        return fake, self.num_samples


class GaussianNoiseFreeRider(Client):
    is_free_rider = True
    attack_name = "gaussian"

    def __init__(self, *args, noise_sigma: float = 0.1,
                 noise_decay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_sigma = noise_sigma
        self.noise_decay = noise_decay

    def produce_update(self, global_state, prev_global_state, round_idx):
        sigma = self.noise_sigma
        if self.noise_decay > 0:
            sigma = self.noise_sigma * (round_idx ** (-self.noise_decay))
        # Deterministic per (client, round) so runs are reproducible.
        g = torch.Generator().manual_seed(1234 + self.cid * 1000 + round_idx)
        fake = _add_noise(global_state, sigma, generator=g)
        return fake, self.num_samples


# Honest Client carries the same flags so callers can treat all clients uniformly.
Client.is_free_rider = False
Client.attack_name = "honest"


ATTACKS = {
    "previous_models": PreviousModelsFreeRider,
    "gaussian": GaussianNoiseFreeRider,
}


def choose_free_riders(num_clients: int, num_free_riders: int, seed: int) -> list:
    """Pick which client ids are free-riders (deterministic given seed)."""
    if num_free_riders <= 0:
        return []
    if num_free_riders > num_clients:
        raise ValueError("num_free_riders cannot exceed num_clients")
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_clients), num_free_riders))


def build_clients(cfg, client_loaders, model, device, seed):
    """Construct a mix of honest and free-rider clients per the config.

    Returns (clients, free_rider_indices).
    """
    fr_idx = set(choose_free_riders(len(client_loaders),
                                    getattr(cfg, "num_free_riders", 0), seed))
    attack = getattr(cfg, "attack", "none")

    clients = []
    for cid, loader in enumerate(client_loaders):
        common = dict(cid=cid, model=model, train_loader=loader, device=device,
                      lr=cfg.lr, local_epochs=cfg.local_epochs,
                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        if cid in fr_idx:
            if attack not in ATTACKS:
                raise ValueError(
                    f"num_free_riders>0 but attack='{attack}' is not one of "
                    f"{list(ATTACKS)}")
            cls = ATTACKS[attack]
            if cls is GaussianNoiseFreeRider:
                clients.append(cls(noise_sigma=getattr(cfg, "noise_sigma", 0.1),
                                   noise_decay=getattr(cfg, "noise_decay", 0.0),
                                   **common))
            else:
                clients.append(cls(**common))
        else:
            clients.append(Client(**common))
    return clients, sorted(fr_idx)
