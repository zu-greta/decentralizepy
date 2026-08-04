"""clients -- honest and free-rider clients

SECTION 1  HONEST      Client, _to_cpu_state        
SECTION 2  WATERMARK   WatermarkClient (Eq.11-12 + Eq.14 memory)
                        build_watermarked_clients      
SECTION 3  ATTACKERS   _SimpleFRMixin, make_reduced_attack, make_tap_attack
                        [+ the DISABLED submarine]     

inheritance chain:
    Client                          honest FedAvg: load global -> local SGD -> return
      +-- WatermarkClient           ... + L_wm on trigger-class samples + Eq.14 memory update
        +-- ReducedFreeRider        ... but trains on a reduced shard after round W
        +-- OracleTapFreeRider      ... but taps only when its own BER nears eta
        +-- AdaptiveTapFreeRider   ... but taps only when its own BER nears eta, and estimates eta adaptively
        +-- [SubmarineFreeRider -- DISABLED]

imports:
  * watermark.py        Eq.1-16 Imported for embedding and wm_verify.py for extraction
  * compute_meter.py    effort used by clients 
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import watermark as wm
from .compute_meter import ComputeMeter


# ============================================================================
# SECTION 1 -- HONEST CLIENT  
# ============================================================================
# Honest behaviour: load the current global weights, run local SGD on the local shard, return the weights. 
# Plain FedAvg. `produce_update` is the overriden function

# helper to detach and move a state dict to CPU for aggregation
def _to_cpu_state(model) -> dict: 
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class Client:
    def __init__(self, cid: int, model, train_loader, device,
                 lr: float, local_epochs: int, momentum: float = 0.9,
                 weight_decay: float = 5e-4):
        self.cid = cid
        self.model = model            # shared model instance, reused each round
        self.loader = train_loader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        self.num_samples = len(train_loader.dataset)

    # ---- the seam ----------------------------------------------------------
    def produce_update(self, global_state: dict, prev_global_state: dict | None,
                       round_idx: int):
        """Return (cpu_state_dict, num_samples) for this round.

        Honest behaviour: load the global model, run local SGD, return weights.
        """
        self.model.load_state_dict(global_state)
        self._local_train() # SGD on local data
        return _to_cpu_state(self.model), self.num_samples

    # ---- honest local training --------------------------------------------
    def _local_train(self):
        self.model.train() 
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr,
            momentum=self.momentum, weight_decay=self.weight_decay,
        )
        for _ in range(self.local_epochs):
            for x, y in self.loader: # iter over local data only 
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad() 
                loss = self.criterion(self.model(x), y) # cross-entropy on local data
                loss.backward()
                optimizer.step()

# ============================================================================
# SECTION 2 -- WATERMARK CLIENT + FACTORY   
# ============================================================================

class WatermarkClient(Client):
    """Honest client that also embeds its private watermark."""

    def __init__(self, *args, trigger_class: int, key: torch.Tensor,
                 target_bits: torch.Tensor, wm_lambda: float = 5.0,
                 wm_kind: str = "power", wm_alpha: float = 0.4,
                 wm_beta: float = 0.6, label_smoothing: float = 0.1,
                 exclude: object = "trigger", **kw):
        super().__init__(*args, **kw)
        self.trigger_class = trigger_class
        self.key = key
        self.target_bits = target_bits
        self.wm_lambda = wm_lambda
        self.wm_kind = wm_kind
        self.wm_alpha = wm_alpha
        self.wm_beta = wm_beta
        self.label_smoothing = label_smoothing
        self.exclude = trigger_class if exclude == "trigger" else exclude
        self.memory: dict | None = None
        self.meter = ComputeMeter()

    # ---- the seam ----------------------------------------------------------
    def produce_update(self, global_state: dict, prev_global_state, round_idx):
        self.model.load_state_dict(global_state) # start from the global model
        self.meter.start_round(round_idx) # start timing the round
        self._local_train_wm(round_idx) # train L = L_cl + lambda * L_wm and log the two loss terms
        self.meter.end_round(trained=True) # end timing the round
        w_sgd = _to_cpu_state(self.model) # get the SGD-updated model
        w_new = self._memory_update(global_state, w_sgd) # memory-enhanced update
        return w_new, self.num_samples # return the new model and the number of samples used for weighting

    # ---- L = L_cl + lambda * L_wm  (Eq. 11-12) -----------------------------
    def _local_train_wm(self, round_idx=None):
        """Trains L = L_cl + lambda*L_wm and logs the two loss terms plus the
        trigger-class training accuracy per round into self.wm_stats
        -> class's BER floor: 
            (a) high watermark loss (hard to embed) and/or 
            (b) low trigger-class accuracy (fuzzy boundary)
        client-side counterparts to the server-side softmax diagnostics 
        logged in wm_verify (pmax/entropy/dominance)."""
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        key = self.key.to(self.device)
        bits = self.target_bits.to(self.device)
        # per-round accumulators (means over batches)
        cl_sum = wm_sum = tot_sum = 0.0
        n_batches = n_wm_batches = 0
        trig_correct = trig_total = 0
        # train over the local dataset for self.local_epochs
        for _ in range(self.local_epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                cl = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
                loss = cl
                tmask = (y == self.trigger_class)
                wm_val = 0.0
                # compute watermark loss only on the trigger-class samples 
                if tmask.any():
                    probs = F.softmax(logits[tmask], dim=1)
                    wml = wm.watermark_loss(probs, key, bits, self.wm_kind,
                                            self.wm_alpha, exclude=self.exclude)
                    loss = loss + self.wm_lambda * wml
                    wm_val = float(wml.detach())
                    wm_sum += wm_val; n_wm_batches += 1
                    with torch.no_grad():
                        pred = logits[tmask].argmax(1)
                        trig_correct += int((pred == self.trigger_class).sum())
                        trig_total += int(tmask.sum())
                loss.backward()
                # Guard the embedding term (the only non-standard loss). Two layers:
                # (1) skip a batch whose loss already went non-finite, so a single
                #     exploded watermark gradient cannot poison the weights and turn
                #     every subsequent softmax into nan (the R2 failure);
                # (2) clip the global grad norm as a smooth cap on spikes.
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                opt.step()
                cl_sum += float(cl.detach()); tot_sum += float(loss.detach())
                n_batches += 1
                # record the batch in the compute meter 
                if self.meter is not None and self.meter._cur is not None:
                    self.meter.record_batch(len(x))
        # record the per-round means in self.wm_stats
        if not hasattr(self, "wm_stats"):
            self.wm_stats = {}
        # record the round's mean losses and trigger-class accuracy
        self.wm_stats[int(round_idx) if round_idx is not None else len(self.wm_stats)] = {
            "cls_loss": round(cl_sum / max(n_batches, 1), 5),
            "wm_loss": round(wm_sum / max(n_wm_batches, 1), 5) if n_wm_batches else None,
            "total_loss": round(tot_sum / max(n_batches, 1), 5),
            "trig_train_acc": round(trig_correct / trig_total, 4) if trig_total else None,
            # number of trigger class samples client saw in the round (can be 0 or very few in non-iid)
            "n_trigger_samples": int(trig_total),
            "trigger_class": int(self.trigger_class),
        }

    # ---- memory-enhanced update (Eq. 14) -----------------------------------
    def _memory_update(self, global_state: dict, w_sgd: dict) -> dict:
        """W_new = beta*(memory + delta) + (1-beta)*global, delta = W_sgd - global.
        Keeps the client's watermarked trajectory alive through aggregation"""
        beta = self.wm_beta
        # initialize memory on the first round (clone to avoid in-place updates)
        if self.memory is None:
            self.memory = {k: v.clone() for k, v in global_state.items()}
        w_new = {}
        # update each parameter: if it's floating-point, do the memory-enhanced update; else just copy it
        for k, vg in global_state.items():
            if torch.is_floating_point(vg):
                delta = w_sgd[k] - vg
                w_new[k] = beta * (self.memory[k] + delta) + (1.0 - beta) * vg
            else:
                w_new[k] = w_sgd[k].clone()
        # update the memory to the new model for the next round
        self.memory = {k: v.clone() for k, v in w_new.items()}
        return w_new


def build_watermarked_clients(cfg, client_loaders, model, device, seed,
                              num_classes, registry):
    """Factory: Each client gets a unique trigger class + secret key + bits.
    Honest slots embed; free-rider slots are the submarine attack (watermark-capable) or
    a baseline. Returns (clients, free_rider_indices)."""

    # random (unbalanced) keys, full softmax (no trigger-class exclusion), m = n//10
    PF_GROUP = 10                                  # TODO hardcoded: bits-per-class divisor (m = num_classes // 10)
    m = cfg.wm_bits or max(2, num_classes // PF_GROUP)
    l = wm.grouping(num_classes, m)
    exclude_col = None                             # full softmax (no trigger-class exclusion)

    attack = getattr(cfg, "attack", "none")
    fr_idx = resolve_free_riders(cfg, len(client_loaders), seed)   # honours cfg.free_rider_ids
    if attack in (None, "none", ""):
        fr_idx = set()

    # optional trigger-class overrides: "0:6,1:6" -> {0: 6, 1: 6}. Lets a free-rider
    # share a trigger class with an honest client (same-class non-separability control).
    tmap = {}
    raw_map = (getattr(cfg, "trigger_class_map", "") or "").strip()
    if raw_map:
        for tok in raw_map.split(","):
            tok = tok.strip()
            if not tok:
                continue
            a, b = tok.split(":")
            tmap[int(a)] = int(b) % num_classes

    # KEY-TWIN map 
    key_twin = {}
    _kt = getattr(cfg, "wm_key_twins", None)
    if _kt:
        for pair in str(_kt).split(","):
            if ":" in pair:
                a, b = pair.split(":"); key_twin[int(a)] = int(b)

    clients, unembed = [], []
    # build each client with its trigger class, key, and target bits
    for cid, loader in enumerate(client_loaders):
        trigger_class = tmap.get(cid, cid % num_classes)  # round-robin, unless overridden
        # key balance config: balanced=True removes structurally-unembeddable same-sign rows 
        bal = bool(getattr(cfg, "wm_balanced_keys", False))
        key_cid = key_twin.get(cid, cid)   # derive key/bits from the twin's cid if set
        key = wm.make_key(m, l, seed=seed + 1000 * key_cid + 1, balanced=bal)
        unembed.append(wm.unembeddable_fraction(key)) # compute the fraction of same-sign rows (structurally unembeddable)
        bits = wm.make_bits(m, seed=seed + 1000 * key_cid + 1) # random target bits for the watermark
        reg_exclude = None                     # full softmax
        registry.register(cid, trigger_class, key, bits,
                          kind=cfg.wm_f, alpha=cfg.wm_alpha, exclude=reg_exclude) # register the client's watermark parameters in the registry

        # common arguments for all clients
        common = dict(cid=cid, model=model, train_loader=loader, device=device,
                      lr=cfg.lr, local_epochs=cfg.local_epochs,
                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)

        # build the client: honest or free-rider
        if cid in fr_idx:
            wm_args = dict(
                trigger_class=trigger_class, key=key, target_bits=bits,
                wm_lambda=cfg.wm_lambda, wm_kind=cfg.wm_f, wm_alpha=cfg.wm_alpha,
                wm_beta=cfg.wm_beta, label_smoothing=cfg.wm_label_smoothing,
                exclude=exclude_col)
            # ---- SUBMARINE DISABLED ----
            # Falls through to the final `else`, which raises a ValueError naming this block. 
#             if attack in ("submarine", "autopilot"):   # "autopilot" kept as a back-compat alias
#                 cls = make_submarine_attack(WatermarkClient)
#                 clients.append(cls(
#                     autop_oracle_eta=getattr(cfg, "autop_oracle_eta", 0.0),
#                     autop_honest_until=getattr(cfg, "autop_honest_until", 12),
#                     autop_calib_rounds=getattr(cfg, "autop_calib_rounds", 4),
#                     autop_eta_k=getattr(cfg, "autop_eta_k", 3.0),
#                     autop_eta_mode=getattr(cfg, "autop_eta_mode", "tight"),
#                     autop_num_clients_est=getattr(cfg, "autop_num_clients_est", 10),
#                     autop_margin0=getattr(cfg, "autop_margin0", 0.06),
#                     autop_safety=getattr(cfg, "autop_safety", 0.02),
#                     autop_max_coast=getattr(cfg, "autop_max_coast", 4),
#                     autop_floor=getattr(cfg, "autop_floor", 0.05),
#                     autop_common_per_class=getattr(cfg, "autop_common_per_class", -1),
#                     autop_scope=getattr(cfg, "autop_scope", "full"),
#                     autop_stay_min=getattr(cfg, "autop_stay_min", False),
#                     autop_holdout_ratio=getattr(cfg, "autop_holdout_ratio", 0.5),
#                     autop_honest_clone=getattr(cfg, "autop_honest_clone", False),
#                     autop_warmup_mode=getattr(cfg, "autop_warmup_mode", "dynamic"),
#                     autop_honest_min=getattr(cfg, "autop_honest_min", 6),
#                     autop_warmup_cap=getattr(cfg, "autop_warmup_cap", 15),
#                     autop_conv_eps=getattr(cfg, "autop_conv_eps", 0.03),
#                     autop_conv_patience=getattr(cfg, "autop_conv_patience", 2),
#                     **wm_args, **common))
            # (was `elif`: promoted to `if` because the submarine branch above is
            #  commented out. Revert to `elif` when reviving the submarine.)
            if attack == "reduced":
                cls = make_reduced_attack(WatermarkClient)
                clients.append(cls(
                    # -1: full data shard. 0 = trigger-class images only
                    common_per_class=int(getattr(cfg, "autop_common_per_class", 5)),
                    n_common_classes=int(getattr(cfg, "autop_n_common_classes", -1)),
                    honest_rounds=getattr(cfg, "autop_honest_until", 12),
                    calib_rounds=getattr(cfg, "autop_calib_rounds", 4),
                    trigger_train_n=int(getattr(cfg, "autop_trigger_train_n", -1)),
                    **wm_args, **common))
            elif attack == "tap_oracle":
                cls = make_tap_attack(WatermarkClient)
                clients.append(cls(
                    oracle_eta=getattr(cfg, "autop_oracle_eta", 0.0) or getattr(cfg, "wm_eta_fixed", 0.0),
                    honest_rounds=getattr(cfg, "autop_honest_until", 12),
                    calib_rounds=getattr(cfg, "autop_calib_rounds", 4),
                    common_per_class=max(0, getattr(cfg, "autop_common_per_class", 5)),
                    **wm_args, **common))
            elif attack == "adaptive_tap":
                cls = make_adaptive_tap_attack(WatermarkClient)
                clients.append(cls(
                    oracle_eta=getattr(cfg, "autop_oracle_eta", 0.0) or getattr(cfg, "wm_eta_fixed", 0.0),
                    honest_rounds=getattr(cfg, "autop_honest_until", 12),
                    calib_rounds=getattr(cfg, "autop_calib_rounds", 4),
                    eta_source=getattr(cfg, "tap_eta_source", "oracle"),
                    eta_k=getattr(cfg, "tap_eta_k", 3.0),
                    margin=getattr(cfg, "tap_margin", 0.02),
                    when=getattr(cfg, "tap_when", "threshold"),
                    period=getattr(cfg, "tap_period", 1),
                    max_coast=getattr(cfg, "tap_max_coast", 999),
                    data_cpc=getattr(cfg, "tap_data_cpc", 5),
                    scope=getattr(cfg, "tap_scope", "full"),
                    coast_mode=getattr(cfg, "tap_coast_mode", "resend"),
                    probe_holdout=getattr(cfg, "tap_probe_holdout", 16),
                    trigger_train_n=int(getattr(cfg, "autop_trigger_train_n", -1)),
                    **wm_args, **common))
            elif attack in ATTACKS:
                # paper baselines (previous_models / gaussian) - no embedding
                cls = ATTACKS[attack]
                if cls is GaussianNoiseFreeRider:
                    clients.append(cls(noise_sigma=getattr(cfg, "noise_sigma", 0.1),
                                       noise_decay=getattr(cfg, "noise_decay", 0.0),
                                       **common))
                else:
                    clients.append(cls(**common))
            else:
                hint = ("  NOTE: the SUBMARINE attacker is currently COMMENTED OUT "
                        "(see the banner in faremark/clients.py PART 3 for the 6 "
                        "sites to uncomment)."
                        if attack in ("submarine", "autopilot") else "")
                raise ValueError(
                    f"attack='{attack}' not supported in the watermark path "
                    f"(use 'reduced', 'tap_oracle', 'adaptive_tap', 'previous_models', 'gaussian', "
                    f"or 'none').{hint}")
        else:
            clients.append(WatermarkClient(
                trigger_class=trigger_class, key=key, target_bits=bits,
                wm_lambda=cfg.wm_lambda, wm_kind=cfg.wm_f, wm_alpha=cfg.wm_alpha,
                wm_beta=cfg.wm_beta, label_smoothing=cfg.wm_label_smoothing,
                exclude=exclude_col, **common))

    frac = sum(unembed) / len(unembed) if unembed else 0.0
    registry.m, registry.l = m, l
    registry.unembeddable_frac = round(frac, 4)
    if frac > 0.10:
        import warnings
        warnings.warn(
            f"[watermark] {frac:.0%} of key rows are same-sign and structurally "
            f"unembeddable (m={m}, l={l}); honest BER will floor near {0.5 * frac:.2f}.")
    return clients, sorted(fr_idx)

# ============================================================================
# SECTION 3 -- FREE-RIDER ATTACKERS   
# ============================================================================
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------------
# 3a. ATTACK BASELINES -- from FareMark paper
# ----------------------------------------------------------------------------
# Fabricate weights from the global history and never train:
#   PreviousModelsFreeRider (Eq. 17)  W_free = 2*W_t - W_{t-1}
#   GaussianNoiseFreeRider  (Eq. 18)  W_free = W_t + N(0, sigma^2)
# They embed no watermark, so their BER sits at ~0.5 and FareMark catches them

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
                    f"{list(ATTACKS)} (the submarine uses the watermark path)")
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

# ----------------------------------------------------------------------------
# 3b. REDUCED DATA ATTACKERS
# ----------------------------------------------------------------------------
# Honest clients that train for real but on a reduced shard - embeds a watermark

# --------------------------------------------------------------------------- #
#  shared helpers (data prep + self-probe)                                     #
# --------------------------------------------------------------------------- #
class _SimpleFRMixin:
    """Host is a WatermarkClient. Adds a reduced (trigger + N/common) loader and
    a self-BER probe on held-out trigger images. Nothing here touches training."""

    def _prepare(self, common_per_class: int, n_probe_holdout: int = 0,
                 n_common_classes: int = -1, trigger_train_n: int = -1):
        """Build the reduced loader once. Optionally hold out a few trigger
        images (never trained on) so the probe measures generalisation, matching
        how the server tests on a separate trigger bank."""
        if getattr(self, "_prepared", False):
            return
        self._prepared = True
        bs = getattr(self.loader, "batch_size", 16) or 16 # batch size for the reduced loader

        trig, comm_x, comm_y = [], [], []
        for x, y in self.loader:                      # original shard, once
            x = x.detach().cpu(); y = y.detach().cpu()
            tm = (y == self.trigger_class) # mask for trigger images
            if tm.any(): 
                trig.append(x[tm]) # trigger images
            if (~tm).any():
                comm_x.append(x[~tm]); comm_y.append(y[~tm]) # common-class images

        allt = torch.cat(trig) if trig else torch.empty(0) # trigger images
        # Hold out a slice of trigger images for the self-probe
        # Cap at half; and if the client has too few triggers skip the holdout
        # entirely -- _probe_x=None disables the probe, and threshold-tapping then falls back to always-tap 
        n_trig = len(allt)
        MIN_TRAIN_TRIG = 8                     # keep at least this many triggers to embed on
        if n_probe_holdout and n_trig >= 2 * MIN_TRAIN_TRIG:
            k = min(int(n_probe_holdout), n_trig - MIN_TRAIN_TRIG, n_trig // 2)
        else:
            k = 0
        self._probe_x = allt[:k].clone() if k > 0 else None # probe on held-out triggers
        trig_train = allt[k:] if k > 0 else allt # trigger images for training
        self._n_probe = int(k)                 # logged so starvation is visible in the trace
        # num of trigger class images trained on - -1 = all
        if trigger_train_n is not None and trigger_train_n >= 0:
            trig_train = trig_train[:trigger_train_n]
        self._trigger_train_n = len(trig_train)

        xs = [trig_train] # the reduced loader is trigger images + N common-class images
        ys = [torch.full((len(trig_train),), self.trigger_class, dtype=torch.long)] # labels for trigger images
        if common_per_class > 0 and comm_x: # if we have common-class images, sample N from each class
            cx = torch.cat(comm_x); cy = torch.cat(comm_y) # all common-class images and labels
            classes = cy.unique()
            # num images vs class diversity
            if n_common_classes is not None and 0 < n_common_classes < len(classes):
                sel = torch.randperm(len(classes))[:n_common_classes]
                classes = classes[sel]
            self._common_classes_used = [int(c) for c in classes]
            for cls in classes: # for each selected common class, sample N images
                idx = (cy == cls).nonzero(as_tuple=True)[0] # indices of this class
                take = idx[torch.randperm(len(idx))[:common_per_class]] # random sample of N indices
                xs.append(cx[take]); ys.append(cy[take]) # add to the reduced loader
        X, Y = torch.cat(xs), torch.cat(ys) # reduced dataset
        self._reduced_n = len(X) # number of samples in the reduced loader
        self._reduced_loader = DataLoader(TensorDataset(X, Y),
                                          batch_size=min(bs, max(1, len(X))),
                                          shuffle=True) 

    @torch.no_grad()
    def _probe_ber(self, state) -> float | None:
        """BER of this client's mark in `state`, on held-out trigger images.
        used by the OracleTapFreeRider to decide whether to coast or tap."""
        if getattr(self, "_probe_x", None) is None:
            return None
        self.model.load_state_dict(state)
        self.model.eval()
        probs = F.softmax(self.model(self._probe_x.to(self.device)), dim=1)
        bits = wm.extract_bits(probs, self.key.to(self.device),
                               self.wm_kind, self.wm_alpha, exclude=self.exclude)
        return wm.bit_error_rate(bits, self.target_bits)

    # window bookkeeping shared by both -------------------------------------
    def _phase_action(self, round_idx: int) -> str:
        """honest | calib (last K warmup rounds) | freeride."""
        W, K = self.honest_rounds, self.calib_rounds # W = honest warmup rounds, K = calibration rounds
        if round_idx >= W:
            return "freeride"
        return "calib" if round_idx >= (W - K) else "honest"


# --------------------------------------------------------------------------- #
#  Reduced Data Attack: honest, then honest-on-less-data                      #
# --------------------------------------------------------------------------- #
def make_reduced_attack(base_cls):

    class ReducedDataFreeRider(_SimpleFRMixin, base_cls):
        is_free_rider = True
        attack_name = "reduced"

        def __init__(self, *a, common_per_class: int = 5, honest_rounds: int = 12,
                     calib_rounds: int = 4, n_common_classes: int = -1,
                     trigger_train_n: int = -1, **kw):
            super().__init__(*a, **kw)
            self.common_per_class = int(common_per_class)
            self.n_common_classes = int(n_common_classes)
            self.trigger_train_n = int(trigger_train_n)
            self.honest_rounds = int(honest_rounds)
            self.calib_rounds = int(calib_rounds)
            self._prepared = False
            self._orig_loader = self.loader
            self.trace = []

        # override the base class's produce_update to switch to the reduced loader after W rounds
        def produce_update(self, global_state, prev_global_state, round_idx):
            phase = self._phase_action(round_idx) # honest | calib (last K warmup rounds) | freeride
            if phase == "freeride":
                if self.common_per_class < 0:
                    # FULL SHARD: training exactly like honest clients
                    submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                    self.trace.append({"round": round_idx, "action": "tap",
                                       "eta_frozen": None, "reduced_n": self.num_samples,
                                       "common_per_class": -1})
                    return submit, n
                # REDUCED SHARD: switch to the reduced shard and keep training like an honest client on less data
                self._prepare(self.common_per_class,
                              n_common_classes=self.n_common_classes,
                              trigger_train_n=self.trigger_train_n) # build the reduced loader once
                self.loader = self._reduced_loader # switch to the reduced loader
                submit, n = super().produce_update(global_state, prev_global_state, round_idx) # train on the reduced loader
                self.trace.append({"round": round_idx, "action": "tap",
                                   "eta_frozen": None, "reduced_n": self._reduced_n,
                                   "common_per_class": self.common_per_class,
                                   "n_common_classes": self.n_common_classes,
                                   "trigger_train_n": self.trigger_train_n,
                                   "common_classes_used": getattr(self, "_common_classes_used", None)}) # re-embeds every round
                return submit, n
            # warmup / calibration window: pure honest client on the original shard
            submit, n = super().produce_update(global_state, prev_global_state, round_idx)
            self.trace.append({"round": round_idx, "action": phase, "eta_frozen": None}) 
            return submit, n

    return ReducedDataFreeRider


# --------------------------------------------------------------------------- #
# Oracle Tap Attack: honest, then oracle-threshold tap/coast                 #
# --------------------------------------------------------------------------- #
def make_tap_attack(base_cls):

    class OracleTapFreeRider(_SimpleFRMixin, base_cls):
        is_free_rider = True
        attack_name = "tap_oracle"

        def __init__(self, *a, oracle_eta: float, honest_rounds: int = 12,
                     calib_rounds: int = 4, common_per_class: int = 5,
                     margin: float = 0.02, **kw):
            super().__init__(*a, **kw)
            self.oracle_eta = float(oracle_eta)          # the true server threshold
            self.honest_rounds = int(honest_rounds)
            self.calib_rounds = int(calib_rounds)
            self.common_per_class = int(common_per_class)
            self.margin = float(margin)                  # stay this far under eta
            self._prepared = False
            self._orig_loader = self.loader
            self.trace = []

        def produce_update(self, global_state, prev_global_state, round_idx):
            phase = self._phase_action(round_idx)

            if phase != "freeride":
                # honest warmup / calibration on the original shard
                submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                # expose the oracle on the last calib round so the timeline can draw it
                eta = self.oracle_eta if phase == "calib" else None
                self.trace.append({"round": round_idx, "action": phase, "eta_frozen": eta})
                return submit, n

            # ---- free-ride: coast if the mark is safely present, else tap ----
            self._prepare(self.common_per_class, n_probe_holdout=16)
            target = max(0.0, self.oracle_eta - self.margin)
            ber_now = self._probe_ber(global_state)      # is my mark still in the model?

            if ber_now is not None and ber_now <= target:
                # COAST: submit the global unchanged -> zero training compute
                self.meter.start_round(round_idx); self.meter.end_round(trained=False)
                self.trace.append({"round": round_idx, "action": "coast",
                                   "eta_frozen": self.oracle_eta,
                                   "ber_after": round(ber_now, 4)})
                return {k: v.clone() for k, v in global_state.items()}, self.num_samples

            # TAP: one honest-style pass on the reduced shard to refresh the mark
            self.loader = self._reduced_loader
            submit, n = super().produce_update(global_state, prev_global_state, round_idx)
            self.loader = self._orig_loader
            self.trace.append({"round": round_idx, "action": "tap",
                               "eta_frozen": self.oracle_eta,
                               "ber_after": None if self._probe_ber(submit) is None
                                            else round(self._probe_ber(submit), 4)})
            return submit, n

    return OracleTapFreeRider


# ------------------------------------------------------------------------------ #
#  Adaptive Tap Attack: the submarine (attack="adaptive_tap")                    #  
#  honest, then train (full or reduced) when nearing (estimated) threshold       #
#    - threshold estimation ...... tap_eta_source ("oracle" | "self"), tap_eta_k
#    - when to tap ............... tap_when ("threshold" | "always" | "every_k"),
#                                  tap_period, tap_max_coast, tap_margin
#    - how much data on a tap .... tap_data_cpc (-1 full | 0 trigger-only | N)
#    - how much MODEL on a tap ... tap_scope ("full" | "block2" | "block" | "head")
#    - free-riding between taps .. tap_coast_mode ("resend" global -> mark decays
#                                  fully in 1 round | "decay" own last-tapped
#                                  weights -> mark frozen flat, but a replay |
#                                  "graft" global body + frozen mark head -> mark
#                                  decays gradually = the sawtooth, submission still
#                                  tracks the global. Pair graft with scope="head".)
# ------------------------------------------------------------------------------ #
def make_adaptive_tap_attack(base_cls):

    class AdaptiveTapFreeRider(_SimpleFRMixin, base_cls):
        is_free_rider = True
        attack_name = "adaptive_tap"

        # full => everything trainable (identical to the honest path).
        _SCOPE_KEEP = {"full": None, "block2": 20, "block": 8, "head": 2}

        def __init__(self, *a, oracle_eta: float = 0.0, honest_rounds: int = 12,
                     calib_rounds: int = 4, eta_source: str = "oracle",
                     eta_k: float = 3.0, margin: float = 0.02,
                     when: str = "threshold", period: int = 1, max_coast: int = 999,
                     data_cpc: int = 5, scope: str = "full",
                     coast_mode: str = "resend", probe_holdout: int = 16,
                     trigger_train_n: int = -1, **kw):
            super().__init__(*a, **kw)
            self.oracle_eta = float(oracle_eta)
            self.honest_rounds = int(honest_rounds)
            self.calib_rounds = int(calib_rounds)
            self.eta_source = str(eta_source)
            self.eta_k = float(eta_k)
            self.margin = float(margin)
            self.when = str(when)
            self.period = max(1, int(period))
            self.max_coast = int(max_coast)
            self.data_cpc = int(data_cpc)
            self.scope = str(scope)
            self.coast_mode = str(coast_mode)
            self.probe_holdout = int(probe_holdout)
            self.trigger_train_n = int(trigger_train_n)
            self._prepared = False
            self._orig_loader = self.loader
            self._calib_bers = []       # own probe BER over the calib window -> "self" eta
            self._eta_frozen = None     # frozen once, at defection
            self._coast_streak = 0
            self._last_submit = None    # for coast_mode="decay"
            self._ber_before = None
            self.trace = []

        # ---- scope freeze/restore --------------------------------------------
        def _freeze_scope(self):
            keep = self._SCOPE_KEEP.get(self.scope)
            named = list(self.model.named_parameters())
            if keep is None:
                for _, p in named:
                    p.requires_grad_(True)
                return
            cut = len(named) - int(keep)
            for i, (_, p) in enumerate(named):
                p.requires_grad_(i >= cut)

        def _restore_scope(self):
            for p in self.model.parameters():
                p.requires_grad_(True)

        # ---- eta to use ------------------------------------------
        def _resolve_eta(self):
            if self.eta_source == "self" and self._calib_bers:
                import statistics as _st
                mu = _st.mean(self._calib_bers)
                sd = _st.pstdev(self._calib_bers) if len(self._calib_bers) > 1 else 0.0
                return max(0.0, mu + self.eta_k * sd)
            return self.oracle_eta                     # oracle / fallback

        # ---- one tap (scope-limited, reduced- or full-shard) -----------------
        def _do_tap(self, global_state, prev_global_state, round_idx, eta, target):
            self.loader = (self._reduced_loader if (self.data_cpc >= 0
                           and getattr(self, "_reduced_loader", None) is not None)
                           else self._orig_loader)
            self._freeze_scope()
            try:
                submit, n = super().produce_update(global_state, prev_global_state, round_idx) # train on the reduced loader or full loader
            finally:
                self._restore_scope() 
                self.loader = self._orig_loader
            self._last_submit = {k: v.clone() for k, v in submit.items()} 
            self._coast_streak = 0
            ba = self._probe_ber(submit) # probe the BER after the tap
            self.trace.append({"round": round_idx, "action": "tap",
                               "eta_frozen": round(eta, 4), "target": round(target, 4),
                               "scope": self.scope, "data_cpc": self.data_cpc,
                               "reduced_n": getattr(self, "_reduced_n", self.num_samples),
                               # sanity: n_trigger_train should be ~tens, NOT ~1 
                               "n_trigger_train": getattr(self, "_trigger_train_n", None),
                               "n_probe_holdout": getattr(self, "_n_probe", None),
                               "ber_before": self._ber_before,
                               "ber_after": None if ba is None else round(ba, 4)})
            return submit, n

        # ---- graft support (coast_mode="graft") ------------------------------
        def _graft_keys(self):
            """State-dict keys of the watermark-carrying output layer to keep
            frozen during a graft coast. Uses the tap scope's kept params; if the
            tap trains 'full', fall back to the head (last 2 params) so graft
            still leaves the body free to follow the global."""
            keep = self._SCOPE_KEEP.get(self.scope) or 2
            named = list(self.model.named_parameters())
            return [name for name, _ in named[len(named) - int(keep):]]

        def _coast_candidate(self, global_state):
            """The model the FR would submit if it coasts this round, per coast_mode:
              resend -> the received global (mark fully decays to ~0.5 in 1 round)
              decay  -> the FR's own last-tapped weights verbatim (mark frozen flat,
                        but a replay: identical submissions round over round)
              graft  -> global body + FR's frozen last-tapped mark head. The body
                        tracks the global while the frozen head's projected bits 
                        degrade gradually as the drifting features flow through them 
            The threshold probes this before deciding tap/coast"""
            if self.coast_mode == "resend" or self._last_submit is None:
                return global_state
            if self.coast_mode == "decay":
                return self._last_submit
            # graft: start from the fresh global, overwrite only the mark-head params
            out = {k: v.clone() for k, v in global_state.items()}
            for k in self._graft_keys():
                if k in self._last_submit:
                    out[k] = self._last_submit[k].clone()
            return out

        # ---- one coast (no training) -----------------------------------------
        def _do_coast(self, global_state, round_idx, eta, target, ber_now):
            self._coast_streak += 1 # increment the coast streak
            # submit exactly what the threshold probed: the coast candidate for this mode
            out = {k: v.clone() for k, v in self._coast_candidate(global_state).items()}
            self.meter.start_round(round_idx); self.meter.end_round(trained=False)
            self.trace.append({"round": round_idx, "action": "coast",
                               "eta_frozen": round(eta, 4), "target": round(target, 4),
                               "coast_mode": self.coast_mode,
                               "coast_streak": self._coast_streak,
                               "ber_before": None if ber_now is None else round(ber_now, 4),
                               "ber_after": None if ber_now is None else round(ber_now, 4)})
            return out, self.num_samples

        # ---- self-probe -----------------
        # Only threshold-tapping and self-eta need the probe to decide, but always recorded for (fade/recovery measurement)
        def _probe_needed(self):
            return (self.when == "threshold") or (self.eta_source == "self")

        # ---- main ------------------------------------------------------------
        def produce_update(self, global_state, prev_global_state, round_idx):
            phase = self._phase_action(round_idx)
            # Always request the probe holdout so ber_before/ber_after are recorded
            holdout = self.probe_holdout

            # warmup + calibration window: pure honest client on the full shard
            if phase != "freeride":
                submit, n = super().produce_update(global_state, prev_global_state, round_idx)
                if phase == "calib":
                    # prepare the probe holdout and record own BER for "self" eta
                    self._prepare(max(0, self.data_cpc), n_probe_holdout=holdout,
                                  trigger_train_n=self.trigger_train_n)
                    b = self._probe_ber(submit)
                    if b is not None:
                        self._calib_bers.append(b)
                self.trace.append({"round": round_idx, "action": phase, "eta_frozen": None})
                return submit, n

            # first free-ride round: build loaders + freeze eta once
            self._prepare(max(0, self.data_cpc), n_probe_holdout=holdout,
                          trigger_train_n=self.trigger_train_n)
            if self._eta_frozen is None:
                self._eta_frozen = self._resolve_eta()
            eta = self._eta_frozen
            target = max(0.0, eta - self.margin)

            # Probe the model the FR would submit if it coasts this round (per coast_mode),
            # not the raw received global. Under resend the candidate is the global (mark
            # ~0.5 -> correctly taps); under decay/graft the candidate holds the mark, so
            # threshold coasts while BER < target instead of degenerating to always-tap.
            ber_now = self._probe_ber(self._coast_candidate(global_state))
            self._ber_before = None if ber_now is None else round(ber_now, 4)

            # decide: tap or coast
            force = self._coast_streak >= self.max_coast
            if self.when == "always":
                tap = True
            elif self.when == "every_k":
                tap = (round_idx % self.period == 0)
            else:  # "threshold"
                tap = (ber_now is None) or (ber_now > target)
            if force:
                tap = True

            if tap:
                return self._do_tap(global_state, prev_global_state, round_idx, eta, target) # tap on the reduced shard (or full shard if data_cpc < 0)
            return self._do_coast(global_state, round_idx, eta, target, ber_now)

    return AdaptiveTapFreeRider


# =============================================================================
# SUBMARINE FREE-RIDER -- disabled for now
# =============================================================================
# DISABLED:
#   _AdaptiveMixin          self-probe, _embed_loop (scope-limited taps), own-eta
#                           estimation (_freeze_own_eta), coast/tap bookkeeping
#   make_submarine_attack   the SubmarineFreeRider factory (attack="submarine"
#                           and its back-compat alias "autopilot")
#
# STILL LIVE ABOVE THIS LINE (do not touch):
#   _SimpleFRMixin          shared reduced-loader + self-probe helper
#   make_reduced_attack     attack="reduced"    <- the attacker the thesis uses
#   make_tap_attack         attack="tap_oracle" <- still TODO but independent
#
# TO REVIVE
#   1. uncomment this whole block (strip the leading "# " from each line)
#   2. uncomment the dispatch branch in SECTION 2 of this file
#      (build_watermarked_clients) 
#   3. uncomment the 16 autop_* fields in faremark/config.py
#   4. uncomment the 16 --autop_* flags + _OVERRIDABLE entries in
#      scripts/run_experiment.py
#   5. uncomment the 16 AUTOP_* env hooks in infra/submit_experiment.sh
#   6. TODO: fix the warmup-loader bug first
# =============================================================================

# """Advanced Adaptive free-rider (submarine and reduced submarine) 
# NOTE: - LEGACY put aside for now - see the above simpler attacks
#
# Threat model:
#   * The free-rider is an honest client with an assigned trigger class + key + wm
#     bits. It estimates (using its own BER) the eta to stay undetected - (or given
#     the oracle eta for controlled testing). 
#
# Submarine behaviour:
#   1. Uses the honest client's modules (key/bits/lambda/alpha/beta/memory/
#      _local_train_wm) -- subclasse of WatermarkClient
#   2. Estimates eta (or uses the oracle) -- _eta_est().
#   3. Behaves honestly until warmup and convergence window have passed.
#      On dynamic mode: it waits until its own watermark BER has converged (same window
#      the server calibrates eta on): it watches its probe BER flatten, then observes
#      K more honest rounds as the calibration window, freezes eta, and defects.
#      On 'fixed' mode, follows deterministic schedule: 
#      warmup = [1 .. W-1], calib window = [W-K .. W-1], free-ride >= W, with W = autop_honest_until.
#   4. After warmup + convergence window it re-embeds ("taps") to hold its mark under eta. 
#      A tap trains on trigger-only / +N-per-common-class / the full shard (autop_common_per_class)
#      with scope full|block2|block|head (autop_scope) -- so a tap's cost = the data
#      and params it uses. With autop_stay_min it coasts (no training) while safely
#      under target and taps only when needed; otherwise it taps every round (honest-style)
#
# Per-round decisions are recorded in self.trace for plotting.
# """
# # FIXED (pre-existing SyntaxError -- this file did not compile as uploaded):
# #   `from __future__ import annotations` appeared a SECOND time here, at line 245.
# #   Python requires __future__ imports to be the first statement in a file, so the
# #   whole module raised:
# #       SyntaxError: from __future__ imports must occur at the beginning of the file
# #   That makes `import faremark.attacks_adaptive` fail, and wm_client's
# #   build_watermarked_clients imports make_reduced_attack / make_submarine_attack /
# #   make_tap_attack from it -- so EVERY watermarked run would crash at client build.
# #   The cause looks like two files concatenated: the header block below (future
# #   import + imports) is a duplicate of the one at the top of the file. The line is
# #   commented out rather than deleted so the history stays visible; the duplicate
# #   imports beneath it are legal mid-file and left alone.
# #   NOTE: if your cluster copy runs fine, it differs from what was uploaded --
# #   diff it before deploying this file.
# # from __future__ import annotations
#
# import statistics          # (only import here not already at the top of the file)
#
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
#
# # (_to_cpu_state is defined in PART 1 of this file)
# from . import watermark as wm
#
#
# class _AdaptiveMixin:
#     """Host class is a WatermarkClient (has key, target_bits, trigger_class,
#     wm_kind, wm_alpha, exclude, model, loader, device, meter, lr, momentum,
#     weight_decay, wm_lambda, label_smoothing, local_epochs, memory,
#     _local_train_wm, _memory_update)."""
#
#     def _ensure_triggers(self, n_probe: int = 64):   # TODO hardcoded: probe-image count (steadier self-BER); tie to N_T?
#         """trigger samples, probe samples, and reduced loader for a tap (data-ablation)"""
#         if getattr(self, "_prepared", False): 
#             return
#         self._prepared = True 
#         self._enr_loader = None                     
#         orig_bs = getattr(self.loader, "batch_size", 16) or 16
#         trig, comm_x, comm_y = [], [], [] 
#         # gather trigger-class samples and common-class samples for the reduced loader
#         for x, y in self.loader: 
#             tm = (y == self.trigger_class)
#             if tm.any():
#                 trig.append(x[tm])
#             om = ~tm
#             if om.any():
#                 comm_x.append(x[om]); comm_y.append(y[om])
#         # concatenate all trigger samples and reserve a probe slice for the FR's own BER probing
#         if not trig:
#             self._probe_x = None
#             self._reduced_loader = None
#             return
#         allt = torch.cat(trig) # all trigger samples in this shard
#         hr = getattr(self, "autop_holdout_ratio", 0.5) # fraction of trigger samples to hold out for probing
#         k = min(n_probe, max(1, int(len(allt) * hr))) # number of probe samples
#         # probe holdout: first k trigger images are probe - not trained on
#         self._probe_x = allt[:k].clone()            # held out for probing ONLY
#         trig_train = allt[k:]                        # train the mark on the REST
#         if len(trig_train) == 0:                     # tiny shard: keep >=1 for training
#             trig_train = allt[-1:].clone(); self._probe_x = allt[:-1].clone()
#
#         # FR training loader = whole shard - held-out probe images
#         # NOTE: rebuilt from in-memory tensors -> data augmentation is frozen for the FR.
#         self._full_loader = self.loader
#         if comm_x:
#             cx_all = torch.cat(comm_x); cy_all = torch.cat(comm_y)
#             X_tr = torch.cat([trig_train, cx_all])
#             Y_tr = torch.cat([torch.full((len(trig_train),), self.trigger_class,
#                                           dtype=torch.long), cy_all])
#         else:
#             X_tr = trig_train
#             Y_tr = torch.full((len(trig_train),), self.trigger_class, dtype=torch.long)
#         self.loader = DataLoader(TensorDataset(X_tr, Y_tr),
#                                  batch_size=min(orig_bs, len(X_tr)), shuffle=True)
#
#         # reduced shard for a tap (data-ablation): trigger-TRAIN samples + N images from each common class
#         # autop_common_per_class = -1 -> use full shard instead
#         ncpc = getattr(self, "autop_common_per_class", -1)
#         self._reduced_loader = None
#         if ncpc >= 0: # reduced loader: (held-out) trigger + N common-class samples
#             xs = [trig_train]
#             ys = [torch.full((len(trig_train),), self.trigger_class, dtype=torch.long)]
#             if ncpc > 0 and comm_x:
#                 cx = torch.cat(comm_x); cy = torch.cat(comm_y)
#                 for cls in cy.unique():
#                     m = (cy == cls).nonzero(as_tuple=True)[0]
#                     take = m[torch.randperm(len(m))[:ncpc]]
#                     xs.append(cx[take]); ys.append(cy[take])
#             X, Y = torch.cat(xs), torch.cat(ys)
#             self._reduced_data_n = len(X)
#             self._reduced_loader = DataLoader(
#                 TensorDataset(X, Y), batch_size=min(32, len(X)), shuffle=True) # TODO hardcoded batch=32 for reduced loader
#
#     # Probe the FR's own watermark BER on its held-out trigger samples (self._probe_x).
#     @torch.no_grad()
#     def _probe_ber_current_model(self):
#         if self._probe_x is None:
#             return None
#         self.model.eval()
#         x = self._probe_x.to(self.device)
#         probs = F.softmax(self.model(x), dim=1) # get the predicted probabilities for the held-out trigger samples
#         bits = wm.extract_bits(probs, self.key.to(self.device), self.wm_kind,
#                                self.wm_alpha, exclude=self.exclude) # extract the watermark bits from the model's predictions
#         if self.meter is not None and self.meter._cur is not None: 
#             self.meter.record_forward_only(len(x)) # record the number of probe samples processed
#         return wm.bit_error_rate(bits, self.target_bits)
#
#     # Probe the FR's own watermark BER on its held-out trigger samples after loading a given model state
#     @torch.no_grad()
#     def _probe_ber_state(self, state):
#         if self._probe_x is None:
#             return None
#         self.model.load_state_dict(state)
#         return self._probe_ber_current_model()
#
#     _PROBE_EVERY = 3      # TODO hardcoded: probe cadence (batches) when early-stop is active (warmup only)
#
#     def _embed_loop(self, global_state, max_batches, floor, scope=None,
#                     early_stop=True, use_full=False, round_idx=None):
#         """Load global, train the watermark until probe BER <= floor. Returns #batches.
#
#         scope: 
#         None/"full" -> whole model; 
#         "block2" -> last 20 tensors; 
#         "block" -> last 8; 
#         "head" -> last 2 (backbone frozen => cheaper backward)
#
#         Loader priority: if use_full (the forced-honest warmup) -> the full shard,
#         so the free-rider is same as honest client while eta is being calibrated; 
#         else reduced (data-ablation, cpc>=0) else full shard
#         """
#         self.model.load_state_dict(global_state)
#         self.model.train()
#         named = list(self.model.named_parameters())
#         # freeze all but the last few layers according to scope, so that only those layers are updated during training
#         if scope in ("head", "block", "block2"):
#             keep = {"head": 2, "block": 8, "block2": 20}[scope]
#             for i, (_, pp) in enumerate(named):
#                 pp.requires_grad_(i >= len(named) - keep)
#             train_params = [pp for pp in self.model.parameters() if pp.requires_grad]
#         # if scope is None or "full", train all parameters (full scope like honest client)
#         else:
#             for _, pp in named:
#                 pp.requires_grad_(True)
#             train_params = list(self.model.parameters())
#         opt = torch.optim.SGD(train_params, lr=self.lr,
#                               momentum=self.momentum, weight_decay=self.weight_decay) # optimizer for training the model
#         key = self.key.to(self.device) # move the watermark key to the device (GPU/CPU)
#         bits = self.target_bits.to(self.device) # move the target watermark bits to the device (GPU/CPU)
#         # select the appropriate data loader based on the use_full flag and the autop_common_per_class setting
#         if use_full:
#             loader = self.loader                                  # honest warmup: full shard
#         elif getattr(self, "autop_common_per_class", -1) >= 0 and self._reduced_loader is not None: # reduced loader: trigger + N common-class samples
#             loader = self._reduced_loader
#         else: # default to the full shard if no reduced loader is available
#             loader = self.loader
#         steps, passes = 0, 0 # initialize counters for the number of training steps and passes through the data loader
#         cl_sum = wm_sum = tot_sum = 0.0 # initialize accumulators for the cross-entropy loss, watermark loss, and total loss
#         n_wm = 0; tc_correct = tc_total = 0 
#         try:
#             # Train the model in a loop until the early stopping condition is met or the maximum number of batches is reached
#             while True:
#                 for x, y in loader:
#                     x, y = x.to(self.device), y.to(self.device)
#                     opt.zero_grad()
#                     logits = self.model(x) # forward pass through the model to get the logits (predicted class scores)
#                     cl = F.cross_entropy(logits, y,
#                                          label_smoothing=self.label_smoothing) # compute the cross-entropy loss with optional label smoothing
#                     loss = cl # initialize the total loss with the cross-entropy loss
#                     tmask = (y == self.trigger_class) # create a mask for the trigger class samples in the batch
#                     # if there are any trigger class samples in the batch, compute the watermark loss and add it to the total loss
#                     if tmask.any():
#                         probs = F.softmax(logits[tmask], dim=1) # compute the predicted probabilities for the trigger class samples
#                         wml = wm.watermark_loss(probs, key, bits, self.wm_kind,
#                                                 self.wm_alpha, exclude=self.exclude) # compute the watermark loss for the trigger class samples
#                         loss = loss + self.wm_lambda * wml # add the weighted watermark loss to the total loss
#                         wm_sum += float(wml.detach()); n_wm += 1 # accumulate the watermark loss and increment the watermark sample counter
#                         with torch.no_grad():
#                             tc_correct += int((logits[tmask].argmax(1) == self.trigger_class).sum()) # count the number of correctly classified trigger class samples
#                             tc_total += int(tmask.sum()) # count the total number of trigger class samples in the batch
#                     loss.backward() # backpropagate the total loss to compute gradients for the model parameters
#                     opt.step() # update the model parameters
#                     self.meter.record_batch(len(x))     # image-passes (scope-blind)
#                     cl_sum += float(cl.detach()); tot_sum += float(loss.detach()) # accumulate the cross-entropy loss and total loss
#                     steps += 1
#                     # Check for early stopping conditions: 
#                     # if early stopping is enabled and the number of steps is a multiple of the probe cadence, probe the current model's watermark BER. 
#                     # If the BER is below the specified floor, log the training statistics and return the number of steps taken. 
#                     # If a maximum number of batches is specified and reached, log the statistics and return. 
#                     # If no maximum is specified and the number of passes through the data loader exceeds the local epochs, log the statistics and return.
#                     if early_stop and steps % self._PROBE_EVERY == 0:
#                         b = self._probe_ber_current_model()
#                         self.model.train()
#                         if b is not None and b <= floor:
#                             self._log_tap_stats(round_idx, cl_sum, wm_sum, tot_sum,
#                                                 steps, n_wm, tc_correct, tc_total)
#                             return steps
#                     if max_batches is not None and steps >= max_batches:
#                         self._log_tap_stats(round_idx, cl_sum, wm_sum, tot_sum,
#                                             steps, n_wm, tc_correct, tc_total)
#                         return steps
#                 passes += 1
#                 if max_batches is None and passes >= self.local_epochs:
#                     self._log_tap_stats(round_idx, cl_sum, wm_sum, tot_sum,
#                                         steps, n_wm, tc_correct, tc_total)
#                     return steps
#         finally: # ensure that all model parameters are set to require gradients after training, regardless of the scope used during training
#             for _, pp in named:
#                 pp.requires_grad_(True)
#
#     # Log the average losses and accuracy for the current round of training, storing them in the wm_stats dictionary
#     def _log_tap_stats(self, round_idx, cl_sum, wm_sum, tot_sum, steps, n_wm,
#                        tc_correct, tc_total):
#         if round_idx is None:
#             return
#         if not hasattr(self, "wm_stats"):
#             self.wm_stats = {}
#         self.wm_stats[int(round_idx)] = {
#             "cls_loss": round(cl_sum / max(steps, 1), 5),
#             "wm_loss": round(wm_sum / max(n_wm, 1), 5) if n_wm else None,
#             "total_loss": round(tot_sum / max(steps, 1), 5),
#             "trig_train_acc": round(tc_correct / tc_total, 4) if tc_total else None,
#             "trigger_class": int(self.trigger_class),
#             "phase": "tap",
#         }
#
#
# def make_submarine_attack(base_cls):
#     """submarine adaptive free-rider factory. `base_cls` is WatermarkClient
#
#     schedule:
#       rounds  warmup       forced honest (full shard, exactly like an honest client
#                            and pays the honest warmup cost).
#                            Ends dynamically when the FR's own probe BER converges
#                            (autop_warmup_mode="dynamic"), or at a fixed round W
#                            (autop_warmup_mode="fixed", W=autop_honest_until).
#       calibration window   the K (=autop_calib_rounds) converged honest rounds: the
#                            server freezes eta here on all clients; the free-rider
#                            freezes its own eta estimate here too (only sees own BER)
#       free-ride            tap (reduced data x scope) or coast (stay_min).
#     """
#     _ETA_FALLBACK = 0.35 # TODO adjust the fallback eta
#
#     class SubmarineFreeRider(_AdaptiveMixin, base_cls):
#         is_free_rider = True
#         attack_name = "submarine"
#
#         def __init__(self, *a,
#                      autop_oracle_eta: float = 0.0,
#                      autop_honest_until: int = 12,   # fixed-mode W / dynamic fallback
#                      autop_calib_rounds: int = 4,    # K: converged rounds that calibrate eta
#                      autop_eta_k: float = 3.0,
#                      autop_eta_mode: str = "tight",  # "tight" | "loose" | "cumulative" (tight by default - strongest)
#                      autop_num_clients_est: int = 10,  # N used for the sqrt(N) shrink in "tight"
#                      autop_margin0: float = 0.06,    # headroom below eta the FR aims for
#                      autop_safety: float = 0.02,     # extra guard for probe/test mismatch
#                      autop_max_coast: int = 4,       # force a re-tap after this many coasts
#                      autop_floor: float = 0.05,
#                      autop_common_per_class: int = -1,
#                      autop_scope: str = "full",
#                      autop_stay_min: bool = False,
#                      autop_holdout_ratio: float = 0.5,
#                      autop_honest_clone: bool = False,
#                      autop_warmup_mode: str = "dynamic",   # "dynamic" | "fixed"
#                      autop_honest_min: int = 6,            # never defect before this round
#                      autop_warmup_cap: int = 15,           # hard stop if never converges
#                      autop_conv_eps: float = 0.03,         # flatness tolerance on probe BER
#                      autop_conv_patience: int = 2,         # consecutive flat rounds required
#                      **kw):
#             super().__init__(*a, **kw)
#             self.autop_oracle_eta = autop_oracle_eta
#             self.autop_honest_until = int(autop_honest_until)
#             self.autop_calib_rounds = int(autop_calib_rounds)
#             self.autop_eta_k = autop_eta_k
#             self.autop_eta_mode = autop_eta_mode
#             self.autop_num_clients_est = int(autop_num_clients_est)
#             self.autop_margin0 = autop_margin0
#             self.autop_safety = float(autop_safety)
#             self.autop_max_coast = int(autop_max_coast)
#             self.autop_floor = autop_floor
#             self.autop_common_per_class = autop_common_per_class
#             self.autop_scope = autop_scope
#             self.autop_stay_min = autop_stay_min
#             self.autop_holdout_ratio = autop_holdout_ratio
#             self.autop_honest_clone = autop_honest_clone
#             self.autop_warmup_mode = autop_warmup_mode
#             self.autop_honest_min = int(autop_honest_min)
#             self.autop_warmup_cap = int(autop_warmup_cap)
#             self.autop_conv_eps = float(autop_conv_eps)
#             self.autop_conv_patience = int(autop_conv_patience)
#             # ---- schedule state ----
#             self._phase = "warmup"        # "warmup" -> "calib" -> "freeride"
#             self._honest_ber_hist = []    # probe BER each honest round (convergence test)
#             self._calib_start = None      # first calibration round (set at convergence)
#             # 'fixed' mode reproduces the old [W-K, W-1] window by forcing the
#             # convergence transition to fire exactly at round W-K.
#             if self.autop_warmup_mode == "fixed":
#                 W, K = self.autop_honest_until, self.autop_calib_rounds
#                 self._eff_honest_min = W - K
#                 self._eff_warmup_cap = W - K
#                 self._force_conv = True
#             else:
#                 self._eff_honest_min = self.autop_honest_min
#                 self._eff_warmup_cap = self.autop_warmup_cap
#                 self._force_conv = False
#             # ---- estimate state ----
#             self._prepared = False
#             self._probe_x = None
#             self._enr_loader = None
#             self._reduced_loader = None
#             self._own_calib_bers = []     # this FR's own BER in the calibration window
#             self._eta_frozen = None       # estimated once, at the end of the calib window
#             self._mark_delta = None
#             self._coast_streak = 0        # consecutive coasts (bounded by autop_max_coast)
#             self.wm_stats = {}            # per-round loss/acc during honest + tap rounds
#             self.trace = []
#
#         # ---- convergence test on the FR's own (coarse) probe BER ----
#         def _converged(self):
#             if self._force_conv:
#                 return True
#             h = self._honest_ber_hist
#             need = self.autop_conv_patience + 1
#             if len(h) < need:
#                 return False
#             recent = h[-need:]
#             return (max(recent) - min(recent)) <= self.autop_conv_eps
#
#         def _eta_target(self):
#             if self.autop_oracle_eta and self.autop_oracle_eta > 0:
#                 return self.autop_oracle_eta            # ORACLE (testing)
#             return self._eta_frozen if self._eta_frozen is not None else _ETA_FALLBACK
#
#         def _freeze_own_eta(self):
#             """Estimate the server's detection threshold from the FR's own honest BER stream
#             TODO: set the mode to the tighest (most conservative) threshold so that FR can stay under.
#             - server eta is calibrated over honest clients on previous runs using mu+3sigma where
#             mean taken over all clients in a round and then mean taken over those K rounds.
#             Uses the K calibration-window BERs for tight/loose"""
#             calib = [b for b in self._own_calib_bers if b is not None] # calibration window BERs
#             allh = [b for b in self._honest_ber_hist if b is not None] # all honest BERs (warmup + calib)   
#             src = allh if self.autop_eta_mode == "cumulative" else calib 
#             if len(src) >= 2: # enough data to compute mean and stddev
#                 mu = statistics.mean(src)
#                 sd = statistics.pstdev(src)
#             elif src:
#                 mu, sd = src[-1], 0.0 # only one sample, no stddev
#             else:
#                 self._eta_frozen = _ETA_FALLBACK
#                 return
#             # compute the target eta based on the mode (tight, loose, or cumulative)
#             if self.autop_eta_mode == "tight":
#                 shrink = max(1.0, self.autop_num_clients_est) ** 0.5
#                 eta = mu + self.autop_eta_k * sd / shrink
#             else:  # loose or cumulative
#                 eta = mu + self.autop_eta_k * sd
#             self._eta_frozen = max(eta, self.autop_floor) # ensure eta is not below the specified floor
#
#         # --- coast state: what the FR would submit if it coasts (no training) ---
#         def _coast_state(self, global_state):
#             if self._mark_delta is not None: # if we have a mark delta, apply it to the global state to simulate coasting
#                 return {k: (g + self._mark_delta[k]) if (k in self._mark_delta and torch.is_floating_point(g))
#                         else g.clone() for k, g in global_state.items()}
#             if self.memory is not None: # if we have memory, return a clone of the memory state
#                 return {k: v.clone() for k, v in self.memory.items()}
#             return {k: v.clone() for k, v in global_state.items()}
#
#         # --- update the mark delta based on the difference between the FR's memory and the global state ---
#         def _update_mark_delta(self, global_state):
#             if self.memory is None:
#                 return
#             self._mark_delta = {k: (self.memory[k] - global_state[k])
#                                 for k, v in self.memory.items()
#                                 if torch.is_floating_point(v) and k in global_state}
#
#         # --- main per-round update: warmup -> calib -> free-ride ---
#         def produce_update(self, global_state, prev_global_state, round_idx):
#             self._ensure_triggers()
#
#             # honest mode: purely honest (FULL shard incl. probe imgs, full scope).
#             # Uses _full_loader so this floor-control is pixel-exact honest.
#             if self.autop_honest_clone:
#                 saved = self.loader
#                 self.loader = getattr(self, "_full_loader", self.loader)
#                 submit, n = super().produce_update(global_state, prev_global_state, round_idx)
#                 self.loader = saved
#                 self._update_mark_delta(global_state)
#                 self.trace.append({"round": round_idx, "action": "honest_clone",
#                                    "ber_after": self._probe_ber_state(submit)})
#                 return submit, n
#
#             # ---- HONEST PHASES (warmup -> calibration): train exactly like an
#             #      honest client. Warmup ends dynamically when the FR's own BER has
#             #      converged (or the hard cap is hit); the next K rounds are the
#             #      calibration window over which eta is frozen; then it defects ----
#             if self._phase in ("warmup", "calib"):
#                 submit, n = super().produce_update(global_state, prev_global_state, round_idx)
#                 self._update_mark_delta(global_state)
#                 ber = self._probe_ber_state(submit)
#                 if ber is not None:
#                     self._honest_ber_hist.append(ber)
#
#                 # warmup -> calib transition (dynamic convergence, or the hard cap)
#                 if self._phase == "warmup":
#                     past_min = round_idx >= self._eff_honest_min
#                     hit_cap = round_idx >= self._eff_warmup_cap
#                     if (past_min and self._converged()) or hit_cap:
#                         self._phase = "calib"
#                         self._calib_start = round_idx        # first calibration round
#
#                 action = "honest"
#                 if self._phase == "calib":
#                     action = "calib"
#                     if ber is not None:
#                         self._own_calib_bers.append(ber)
#                     # freeze eta and end the calibration window after K rounds
#                     if round_idx - self._calib_start + 1 >= self.autop_calib_rounds:
#                         self._freeze_own_eta()
#                         self._phase = "freeride"
#
#                 self.trace.append({"round": round_idx, "action": action,
#                                    "ber_after": None if ber is None else round(ber, 4),
#                                    "eta_frozen": self._eta_frozen})
#                 return submit, n
#
#             # ---- FREE-RIDE: rounds after the calibration window ----
#             self.meter.start_round(round_idx)
#             eta = self._eta_target()
#             # aim a safety gap below eta. margin0 = deliberate headroom; safety =
#             # guard for the probe/test mismatch (the FR probes its own held-out
#             # trigger images; the server measures on the test trigger bank, so the
#             # FR's estimate is noisy and can under-read -- stay conservative).
#             target = max(self.autop_floor, eta - self.autop_margin0 - self.autop_safety)
#
#             coast_reason = None
#             if self.autop_stay_min:                      # coast only when provably safe
#                 coast = self._coast_state(global_state)
#                 cref = self._probe_ber_state(coast)      # predicted BER IF we coast
#                 forced = self._coast_streak >= self.autop_max_coast
#                 safe = (cref is not None) and (cref <= target)
#                 if safe and not forced:
#                     self._coast_streak += 1
#                     self.meter.end_round(trained=False)
#                     self.trace.append({"round": round_idx, "action": "coast",
#                                        "eta": round(eta, 4), "target": round(target, 4),
#                                        "ber_after": None if cref is None else round(cref, 4),
#                                        "coast_streak": self._coast_streak})
#                     return coast, self.num_samples
#                 # tap when over target (cref > target) or to break a long
#                 # coast streak (forced) -- prevents silent drift past the server's BER.
#                 coast_reason = "forced_retap" if (safe and forced) else "over_target"
#                 self._coast_streak = 0
#
#             nb = self._embed_loop(global_state, None, floor=self.autop_floor,
#                                   scope=self.autop_scope, early_stop=False,
#                                   round_idx=round_idx)   # TAP (logs wm_stats)
#             w = _to_cpu_state(self.model)
#             submit = self._memory_update(global_state, w)
#             self._update_mark_delta(global_state)
#             ber = self._probe_ber_state(submit)
#             self.meter.end_round(trained=True)
#             self.trace.append({"round": round_idx, "action": "tap",
#                                "eta": round(eta, 4), "target": round(target, 4),
#                                "tap_batches": nb, "tap_reason": coast_reason,
#                                "ber_after": None if ber is None else round(ber, 4)})
#             return submit, self.num_samples
#
#     return SubmarineFreeRider