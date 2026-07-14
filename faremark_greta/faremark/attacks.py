"""Free-rider attacks (FareMark baseline)

A free-rider submits a fabricated local model instead of doing real training, to
obtain the global model without contributing to its training. 
Free-riders still own a data shard but never train on it. 

PreviousModelsFreeRider (Eq. 17): W_free = 2*W_t - W_{t-1}  (delta-weights).
GaussianNoiseFreeRider  (Eq. 18): W_free = W_t + N(0, sigma^2), optional decay.
"""
import copy
import random

import torch

from .client import Client


# ---- weight-fabrication helpers ----------------
def _is_norm_buffer(key: str) -> bool:
    """BatchNorm running stats. Extrapolating these can push running_var negative
    -> NaN -> model collapse. A real free-rider would keep valid stats, so copy
    them from the current global instead of extrapolating/perturbing."""
    return ("running_mean" in key) or ("running_var" in key)


def _extrapolate(w_t: dict, w_prev: dict) -> dict:
    """Elementwise 2*W_t - W_{t-1} over float weights; copy buffers/norm stats."""
    out = {}
    for k, v in w_t.items():
        if v.is_floating_point() and k in w_prev and not _is_norm_buffer(k):
            out[k] = 2.0 * v - w_prev[k]
        else:
            out[k] = v.clone()
    return out


def _add_noise(state: dict, sigma: float, generator=None) -> dict:
    """Add N(0, sigma^2) noise to float weights; copy buffers/norm stats."""
    out = {}
    for k, v in state.items():
        if v.is_floating_point() and not _is_norm_buffer(k):
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
            fake = copy.deepcopy(global_state)              # round 1: resubmit W_t (Eq. 17)
        else:
            fake = _extrapolate(global_state, prev_global_state)
        return fake, self.num_samples                       # no training


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
            sigma = self.noise_sigma * (round_idx ** (-self.noise_decay)) # decay noise over rounds
        g = torch.Generator().manual_seed(1234 + self.cid * 1000 + round_idx) 
        fake = _add_noise(global_state, sigma, generator=g) 
        return fake, self.num_samples                       # no training


# Honest Client carries the same flags so callers can treat all clients uniformly.
Client.is_free_rider = False
Client.attack_name = "honest"


ATTACKS = {
    "previous_models": PreviousModelsFreeRider,
    "gaussian": GaussianNoiseFreeRider,
}


def choose_free_riders(num_clients: int, num_free_riders: int, seed: int) -> list:
    """Pick which client ids are free-riders (deterministic given seed)"""
    if num_free_riders <= 0:
        return []
    if num_free_riders > num_clients:
        raise ValueError("num_free_riders cannot exceed num_clients")
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_clients), num_free_riders))


def resolve_free_riders(cfg, num_clients: int, seed: int) -> set:
    """Explicit cfg.free_rider_ids ("3,6") wins; else the seeded choice"""
    ids = getattr(cfg, "free_rider_ids", "") or ""
    if ids.strip():
        return set(int(x) for x in ids.split(",") if x.strip() != "")
    return set(choose_free_riders(num_clients,
                                  getattr(cfg, "num_free_riders", 0), seed))


def build_clients(cfg, client_loaders, model, device, seed):
    """Construct honest + (baseline) free-rider clients for the non-watermark path.
    Returns (clients, free_rider_indices)."""
    fr_idx = resolve_free_riders(cfg, len(client_loaders), seed)
    attack = getattr(cfg, "attack", "none")
    if attack in (None, "none", ""):
        fr_idx = set()

    clients = []
    for cid, loader in enumerate(client_loaders):
        common = dict(cid=cid, model=model, train_loader=loader, device=device,
                      lr=cfg.lr, local_epochs=cfg.local_epochs,
                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        if cid in fr_idx:
            if attack not in ATTACKS:
                raise ValueError(
                    f"num_free_riders>0 but attack='{attack}' is not one of "
                    f"{list(ATTACKS)} (the autopilot uses the watermark path)")
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