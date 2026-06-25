"""Free-rider attacks

A free-rider submits a fabricated local model instead of doing real training, to
obtain the valuable global model "for free". Free-riders still own a data shard
in the simulation (so they report a normal sample count for FedAvg weighting) but 
they simply never train on it. Implements the paper's two
attack methods - Previous model attack and Gaussian noise attack. 
Both subclass Client and override ONLY `produce_update`, so they drop into the 
existing FedAvg loop with no other changes. Can be extended to implement other attacks

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
"""
import copy
import random

import torch

from .client import Client


# ---- weight-fabrication helpers (operate on CPU state dicts) ----------------
def _is_norm_buffer(key: str) -> bool:
    """BatchNorm/other running statistics. Extrapolating these (2*v_t - v_{t-1})
    can push running_var negative -> sqrt(neg) -> NaN -> the model collapses to
    chance accuracy. A real free-rider submitting a usable model would keep valid
    normalization stats, so copy these from the current global instead of
    extrapolating/perturbing them. (Only matters for models with BatchNorm, e.g.
    ResNet; SmallCNN has none (MNIST smoke is unaffected))"""
    return ("running_mean" in key) or ("running_var" in key)


def _extrapolate(w_t: dict, w_prev: dict) -> dict:
    """Elementwise 2*W_t - W_{t-1} over float WEIGHTS; copy buffers/norm stats."""
    out = {}
    for k, v in w_t.items():
        if v.is_floating_point() and k in w_prev and not _is_norm_buffer(k):
            out[k] = 2.0 * v - w_prev[k]
        else:
            out[k] = v.clone()
    return out


def _add_noise(state: dict, sigma: float, generator=None) -> dict:
    """Add N(0, sigma^2) noise to float WEIGHTS; copy buffers/norm stats."""
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
            # first round, no W_{t-1} yet, so just resubmit W_t (Eq. 17)
            fake = copy.deepcopy(global_state)
        else:
            # W_free = 2*W_t - W_(t-1)
            fake = _extrapolate(global_state, prev_global_state)
        return fake, self.num_samples # no training


class GaussianNoiseFreeRider(Client):
    is_free_rider = True
    attack_name = "gaussian"

    def __init__(self, *args, noise_sigma: float = 0.1,
                 noise_decay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_sigma = noise_sigma # initial stddev of Gaussian noise
        self.noise_decay = noise_decay # exponent for decaying noise across rounds (Fraboni et al.'s shrinking noise)

    def produce_update(self, global_state, prev_global_state, round_idx):
        sigma = self.noise_sigma
        if self.noise_decay > 0: # decay sigma across rounds if requested
            sigma = self.noise_sigma * (round_idx ** (-self.noise_decay))
        # deterministic per (client, round) so runs are reproducible
        g = torch.Generator().manual_seed(1234 + self.cid * 1000 + round_idx)
        # W_free = W_t + N(0, sigma^2)
        fake = _add_noise(global_state, sigma, generator=g)
        return fake, self.num_samples # no training


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


# ============================================================================
# adaptive free-riders (paper section V-D3 / section V-D4, Tables IV-V).
# adapted free-riders subclass WatermarkClient 
# ============================================================================
def make_train_then_attack(base_cls):
    class TrainThenAttackFreeRider(base_cls):
        """Table IV: trains honestly (embedding its watermark) for the first
        `attack_round` rounds, then free-rides (fabricates) afterwards. The paper
        finds that if it trained only briefly, the embedded watermark fails to
        persist through later aggregation, so it is still detectable; the more
        rounds it trained, the harder it is to detect."""
        is_free_rider = True
        attack_name = "train_then_attack"

        def __init__(self, *a, attack_round: int = 50, **kw):
            super().__init__(*a, **kw)
            self.attack_round = attack_round

        def produce_update(self, global_state, prev_global_state, round_idx):
            if round_idx < self.attack_round:
                return super().produce_update(global_state, prev_global_state, round_idx)
            fake = (copy.deepcopy(global_state) if prev_global_state is None
                    else _extrapolate(global_state, prev_global_state))
            return fake, self.num_samples
    return TrainThenAttackFreeRider


def make_trigger_only(base_cls):
    class TriggerOnlyFreeRider(base_cls):
        """Table V: trains on only a few trigger-class samples to fake the
        watermark cheaply. The paper shows the watermark then OVERFITS those few
        samples and fails to generalize to other trigger-class images, so
        verification (which uses held-out trigger samples) still fails."""
        is_free_rider = True
        attack_name = "trigger_only"

        def __init__(self, *a, n_trigger_samples: int = 8, **kw):
            super().__init__(*a, **kw)
            self.n_trigger_samples = n_trigger_samples

        def _local_train_wm(self):
            # Restrict this client's loader to a handful of trigger-class samples.
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            xs, ys = [], []
            for x, y in self.loader:
                m = (y == self.trigger_class)
                if m.any():
                    xs.append(x[m]); ys.append(y[m])
                if sum(len(t) for t in ys) >= self.n_trigger_samples:
                    break
            if xs:
                x = torch.cat(xs)[: self.n_trigger_samples]
                y = torch.cat(ys)[: self.n_trigger_samples]
                self.loader = DataLoader(TensorDataset(x, y), batch_size=len(x))
            super()._local_train_wm()
    return TriggerOnlyFreeRider


def make_random_round_attack(base_cls):
    class RandomRoundFreeRider(base_cls):
        """Generalises train-then-attack: instead of defecting after a fixed
        round, the client free-rides on a RANDOM subset of rounds and trains
        honestly on the rest (each round honest with prob `honest_prob`, decided
        deterministically per (client, round)). Tests whether sporadic honest
        participation keeps the embedded watermark 'fresh' enough to evade a
        detector tuned for clean defectors."""
        is_free_rider = True
        attack_name = "random_round"

        def __init__(self, *a, honest_prob: float = 0.5, **kw):
            super().__init__(*a, **kw)
            self.honest_prob = honest_prob

        def produce_update(self, global_state, prev_global_state, round_idx):
            import random
            r = random.Random(1000 * getattr(self, "cid", 0) + round_idx)
            if r.random() < self.honest_prob:
                return super().produce_update(global_state, prev_global_state, round_idx)
            fake = (copy.deepcopy(global_state) if prev_global_state is None
                    else _extrapolate(global_state, prev_global_state))
            return fake, self.num_samples
    return RandomRoundFreeRider


def make_mixed_attack(base_cls):
    class MixedDisguiseFreeRider(base_cls):
        """Forgery disguise free-rider. Embeds a watermark cheaply, then hides it
        inside a mostly-replayed global update:
            submit  blend * (trained weights) + (1-blend) * extrapolated-global.

        Two training regimes for the embed:
          * default (full_trigger_class=False): train on only `n_trigger_samples`
            trigger images -> the mark OVERFITS and fails the held-out trigger
            bank (weak; same flaw as trigger_only).
          * full_trigger_class=True: train on ALL trigger-class images in the
            shard, plus `n_common_samples` random non-trigger images. The mark is
            embedded across the trigger-class distribution so it GENERALIZES to
            held-out triggers (stronger evasion), and the common-class samples
            keep the local update looking like balanced honest training and stop
            the model collapsing toward the trigger class. The cost is real
            training effort -> sweep (n_common_samples, local_epochs) vs evasion
            (fr_BER < eta, recall) to trace the forgery COST curve: cheap evasion
            => defense broken; near-honest cost to evade => defense holds.
        """
        is_free_rider = True
        attack_name = "mixed"

        def __init__(self, *a, n_trigger_samples: int = 8, blend: float = 0.5,
                     full_trigger_class: bool = False, n_common_samples: int = 0,
                     **kw):
            super().__init__(*a, **kw)
            self.n_trigger_samples = n_trigger_samples
            self.blend = blend
            self.full_trigger_class = full_trigger_class
            self.n_common_samples = n_common_samples

        def _local_train_wm(self):
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            trig_x, trig_y = [], []        # trigger-class samples (the embed target)
            comm_x, comm_y = [], []        # common-class samples (disguise + stability)
            for x, y in self.loader:
                tmask = (y == self.trigger_class)
                if tmask.any():
                    trig_x.append(x[tmask]); trig_y.append(y[tmask])
                if self.full_trigger_class and self.n_common_samples > 0:
                    omask = ~tmask
                    if omask.any():
                        comm_x.append(x[omask]); comm_y.append(y[omask])
                if not self.full_trigger_class:
                    # cheap mode: stop once we have the capped handful of triggers
                    if sum(len(t) for t in trig_y) >= self.n_trigger_samples:
                        break
            if not trig_x:
                return super()._local_train_wm()        # no triggers in shard; nothing to embed

            tx, ty = torch.cat(trig_x), torch.cat(trig_y)
            if not self.full_trigger_class:
                tx, ty = tx[: self.n_trigger_samples], ty[: self.n_trigger_samples]
                xs, ys = tx, ty
            else:
                # all trigger-class samples + a random slice of common-class samples
                if comm_x and self.n_common_samples > 0:
                    cx, cy = torch.cat(comm_x), torch.cat(comm_y)
                    k = min(self.n_common_samples, len(cx))
                    idx = torch.randperm(len(cx))[:k]
                    xs = torch.cat([tx, cx[idx]]); ys = torch.cat([ty, cy[idx]])
                else:
                    xs, ys = tx, ty
            bs = min(32, len(xs))                        # mini-batch (full-batch overfits)
            self.loader = DataLoader(TensorDataset(xs, ys), batch_size=bs, shuffle=True)
            super()._local_train_wm()

        def produce_update(self, global_state, prev_global_state, round_idx):
            import torch
            w_self, n = super().produce_update(global_state, prev_global_state, round_idx)
            fake = (copy.deepcopy(global_state) if prev_global_state is None
                    else _extrapolate(global_state, prev_global_state))
            b = self.blend
            blended = {k: b * w_self[k] + (1 - b) * fake[k]
                       if torch.is_floating_point(w_self[k]) else w_self[k]
                       for k in w_self}
            return blended, n
    return MixedDisguiseFreeRider