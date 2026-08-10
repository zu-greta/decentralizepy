"""clients -- honest and free-rider clients

SECTION 1  HONEST      Client, _to_cpu_state        
SECTION 2  WATERMARK   WatermarkClient (Eq.11-12 + Eq.14 memory)
                        build_watermarked_clients      
SECTION 3  ATTACKERS   _SimpleFRMixin, make_reduced_attack,
                        make_adaptive_tap_attack (the adaptive "submarine")

inheritance chain:
    Client                          honest FedAvg: load global -> local SGD -> return
      +-- WatermarkClient           ... + L_wm on trigger-class samples + Eq.14 memory update
        +-- ReducedFreeRider        ... but trains on a reduced shard after round W
        +-- AdaptiveTapFreeRider    ... the submarine: estimates eta, warms up
                                        (fixed or dynamic), then coasts/taps to hold
                                        its mark. graft coast -> the sawtooth.

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


def _client_class_counts(client_loaders, num_classes):
    """Per-client class histogram [N_clients][num_classes], read cheaply from the
    Subset's underlying targets (no data iteration). Falls back to a single loader
    pass if targets aren't exposed. The server legitimately knows these counts in FL
    accounting terms (shard sizes / class mix), so using them to assign trigger
    classes is within the server's power."""
    counts = [[0] * num_classes for _ in range(len(client_loaders))]
    for cid, loader in enumerate(client_loaders):
        ds = getattr(loader, "dataset", None)
        base = getattr(ds, "dataset", None)          # torch Subset -> underlying dataset
        indices = getattr(ds, "indices", None)
        labels = None
        if base is not None and indices is not None:
            for attr in ("targets", "labels"):
                if hasattr(base, attr):
                    import numpy as _np
                    labels = _np.asarray(getattr(base, attr))[list(indices)]
                    break
        if labels is not None:
            for y in labels:
                yy = int(y)
                if 0 <= yy < num_classes:
                    counts[cid][yy] += 1
        else:                                        # fallback: one pass over the loader
            for _x, y in loader:
                for yy in y.tolist():
                    if 0 <= int(yy) < num_classes:
                        counts[cid][int(yy)] += 1
    return counts


def _assign_triggers_by_distribution(client_loaders, num_classes, reserve=None):
    """Server-side, distribution-aware trigger assignment (fairness fix for non-IID).

    Instead of the blind round-robin `cid % n`, give each client a trigger class it
    actually holds a lot of, so it is NOT data-starved on the very class it must
    watermark (the Group-E starvation mechanism). Greedy max-weight matching on the
    per-client class-count matrix: repeatedly take the (client, class) pair with the
    highest count among still-unassigned clients/classes. Guarantees a UNIQUE class
    per client (needed for detection) while maximising the trigger images each client
    holds. `reserve` = {cid: class} forced assignments (e.g. pinned free-riders) that
    are honoured first. Returns {cid: trigger_class}.

    Only sensible when num_clients <= num_classes (one class per client). For
    oversubscription fall back to round-robin at the call site.
    """
    counts = _client_class_counts(client_loaders, num_classes)
    N = len(client_loaders)
    assign = {}
    used_classes = set()
    if reserve:
        for cid, c in reserve.items():
            assign[cid] = int(c) % num_classes
            used_classes.add(assign[cid])
    # candidate (count, cid, class) for every unassigned client/class
    cand = []
    for cid in range(N):
        if cid in assign:
            continue
        for c in range(num_classes):
            cand.append((counts[cid][c], cid, c))
    cand.sort(reverse=True)                          # highest count first
    for cnt, cid, c in cand:
        if cid in assign or c in used_classes:
            continue
        assign[cid] = c
        used_classes.add(c)
        if len(assign) == N:
            break
    # any client that never matched (all its top classes taken) -> lowest free class
    for cid in range(N):
        if cid not in assign:
            free = next((c for c in range(num_classes) if c not in used_classes), cid % num_classes)
            assign[cid] = free
            used_classes.add(free)
    return assign


def build_watermarked_clients(cfg, client_loaders, model, device, seed,
                              num_classes, registry):
    """Factory: Each client gets a unique trigger class + secret key + bits.
    Honest slots embed; free-rider slots are the submarine attack (watermark-capable) or
    a baseline. Returns (clients, free_rider_indices)."""

    # random (unbalanced) keys, full softmax (no trigger-class exclusion), m = n//10
    PF_GROUP = 10                                  # TODO hardcoded: bits-per-class divisor (m = num_classes // 10)
    m = cfg.wm_bits or max(2, num_classes // PF_GROUP)
    # exclude_col controls whether the trigger-class column is dropped from the
    # watermark projection. DEFAULT None = full softmax = paper-faithful 
    if bool(getattr(cfg, "wm_exclude_trigger", False)):
        exclude_col = "trigger"                    # per-client -> its own trigger_class
        l = wm.grouping(num_classes - 1, m)        # ablation: fit projection into n-1 columns
    else:
        exclude_col = None                         # full softmax (no trigger-class exclusion)
        l = wm.grouping(num_classes, m)

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

    # ---- trigger-class assignment policy -----------------------------------
    #   roundrobin (default): cid % n  -- blind, ignores what the client holds.
    #   distribution: the server assigns each client a class it holds a lot of
    #     (greedy max-count matching), so a non-IID client is not starved on its
    #     own trigger class (the Group-E fairness fix). Explicit trigger_class_map
    #     entries are honoured as forced assignments. Only used when
    #     num_clients <= num_classes; otherwise round-robin (sharing is required).
    assign_mode = str(getattr(cfg, "wm_trigger_assign", "roundrobin"))
    dist_assign = {}
    if assign_mode == "distribution" and len(client_loaders) <= num_classes:
        dist_assign = _assign_triggers_by_distribution(
            client_loaders, num_classes, reserve=tmap)
        registry.trigger_assign = "distribution"
    else:
        if assign_mode == "distribution":
            import warnings
            warnings.warn("[watermark] wm_trigger_assign='distribution' needs "
                          "num_clients <= num_classes; falling back to round-robin.")
        registry.trigger_assign = "roundrobin"

    # Per-client class histogram (server-known) -> record how many trigger-class
    # images each client actually holds, so BER-vs-trigger-samples plots (the non-IID
    # fairness check) can read it straight from result.json.
    _counts = _client_class_counts(client_loaders, num_classes)
    registry.trigger_holdings = {}      # cid -> #images of its trigger class in its shard
    registry.shard_sizes = {}           # cid -> total shard size

    clients, unembed = [], []
    # build each client with its trigger class, key, and target bits
    for cid, loader in enumerate(client_loaders):
        # priority: explicit map > distribution assignment > round-robin
        if cid in tmap:
            trigger_class = tmap[cid]
        elif cid in dist_assign:
            trigger_class = dist_assign[cid]
        else:
            trigger_class = cid % num_classes
        # key balance config: balanced=True removes structurally-unembeddable same-sign rows 
        bal = bool(getattr(cfg, "wm_balanced_keys", False))
        key = wm.make_key(m, l, seed=seed + 1000 * cid + 1, balanced=bal)
        unembed.append(wm.unembeddable_fraction(key)) # compute the fraction of same-sign rows (structurally unembeddable)
        bits = wm.make_bits(m, seed=seed + 1000 * cid + 1) # random target bits for the watermark
        reg_exclude = exclude_col              # None = full softmax (paper); "trigger" = ablation
        registry.register(cid, trigger_class, key, bits,
                          kind=cfg.wm_f, alpha=cfg.wm_alpha, exclude=reg_exclude) # register the client's watermark parameters in the registry
        # record how many trigger-class images this client actually holds (fairness signal)
        registry.trigger_holdings[cid] = int(_counts[cid][trigger_class]) if cid < len(_counts) else None
        registry.shard_sizes[cid] = int(sum(_counts[cid])) if cid < len(_counts) else None

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
            # The adaptive submarine is attack="adaptive_tap" (AdaptiveTapFreeRider),
            # dispatched below. attack="submarine"/"autopilot" are accepted as aliases
            # for it for back-compat.
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
            elif attack in ("adaptive_tap", "submarine", "autopilot"):
                # "submarine"/"autopilot" are aliases for the adaptive tap free-rider.
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
                    graft_decay=getattr(cfg, "tap_graft_decay", 0.0),
                    probe_holdout=getattr(cfg, "tap_probe_holdout", 16),
                    trigger_train_n=int(getattr(cfg, "autop_trigger_train_n", -1)),
                    # dynamic knobs (default to fixed behaviour)
                    margin_mode=getattr(cfg, "tap_margin_mode", "fixed"),
                    margin_k=getattr(cfg, "tap_margin_k", 1.0),
                    warmup_mode=getattr(cfg, "tap_warmup_mode", "fixed"),
                    conv_eps=getattr(cfg, "tap_conv_eps", 0.03),
                    conv_patience=getattr(cfg, "tap_conv_patience", 2),
                    honest_min=getattr(cfg, "tap_honest_min", 6),
                    warmup_cap=getattr(cfg, "tap_warmup_cap", 15),
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
                raise ValueError(
                    f"attack='{attack}' not supported in the watermark path "
                    f"(use 'reduced', 'adaptive_tap' (aka 'submarine'), "
                    f"'previous_models', 'gaussian', or 'none').")
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
        used by the adaptive-tap free-rider to decide whether to coast or tap."""
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
                     graft_decay: float = 0.0,
                     trigger_train_n: int = -1,
                     # ---- DYNAMIC knobs (all default to the previous FIXED behaviour) ----
                     margin_mode: str = "fixed", margin_k: float = 1.0,
                     warmup_mode: str = "fixed", conv_eps: float = 0.03,
                     conv_patience: int = 2, honest_min: int = 6,
                     warmup_cap: int = 15, **kw):
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
            self.graft_decay = float(graft_decay)
            self.probe_holdout = int(probe_holdout)
            self.trigger_train_n = int(trigger_train_n)
            # ---- DYNAMIC config (opt-in; "fixed" reproduces the prior behaviour) ----
            #  margin_mode="derived": target = eta_hat - margin_k * sigma(calib probe BER),
            #      so the safety gap scales with how noisy the FR's own eta estimate is
            #  warmup_mode="dynamic": defect when the FR's OWN probe BER has converged
            #      (flat within conv_eps for conv_patience+1 rounds), bounded to
            #      [honest_min, warmup_cap]; the K calib rounds then run after convergence
            self.margin_mode = str(margin_mode)
            self.margin_k = float(margin_k)
            self.warmup_mode = str(warmup_mode)
            self.conv_eps = float(conv_eps)
            self.conv_patience = int(conv_patience)
            self.honest_min = int(honest_min)
            self.warmup_cap = int(warmup_cap)
            self._prepared = False
            self._orig_loader = self.loader
            self._calib_bers = []       # own probe BER over the calib window -> "self" eta
            self._eta_frozen = None     # frozen once, at defection
            self._target_frozen = None  # margin resolved once, at defection (derived mode)
            self._coast_streak = 0
            self._last_submit = None    # for coast_mode="decay"
            self._ber_before = None
            # dynamic-warmup bookkeeping
            self._probe_hist = []       # own probe BER each honest/calib round (dynamic warmup)
            self._converged_at = None   # round the probe first converged (dynamic warmup)
            self._defect_round = None   # resolved first free-ride round (== honest_rounds in fixed mode)
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

        # ---- dynamic warmup: when to defect ----------------------------------
        def _probe_converged(self):
            """True once the FR's own probe BER has been flat (within conv_eps)
            for conv_patience+1 consecutive rounds."""
            need = self.conv_patience + 1
            if len(self._probe_hist) < need:
                return False
            window = self._probe_hist[-need:]
            return (max(window) - min(window)) <= self.conv_eps

        def _phase_action(self, round_idx: int) -> str:
            """honest | calib | freeride.
            fixed   : the original schedule -- warmup [1, W-1], calib [W-K, W-1],
                      freeride >= W (W=honest_rounds, K=calib_rounds). 
            dynamic : defect after the probe converges. Watch the probe flatten
                      (>= honest_min rounds, at most warmup_cap), then run K calib
                      rounds, then free-ride. 
            """
            if self.warmup_mode != "dynamic":
                return super()._phase_action(round_idx)
            # dynamic: resolve the defect round lazily as the probe history grows
            if self._defect_round is not None:
                if round_idx >= self._defect_round:
                    return "freeride"
                return "calib" if round_idx >= (self._defect_round - self.calib_rounds) else "honest"
            # not yet converged: honest until we either converge (+K) or hit the cap
            if self._converged_at is None:
                if round_idx >= self.honest_min and self._probe_converged():
                    self._converged_at = round_idx
                    self._defect_round = min(self.warmup_cap,
                                             self._converged_at + self.calib_rounds)
                elif round_idx >= self.warmup_cap - self.calib_rounds:
                    # never converged -> force the calib window to end at the cap
                    self._converged_at = self.warmup_cap - self.calib_rounds
                    self._defect_round = self.warmup_cap
            if self._defect_round is not None:
                return "calib" if round_idx >= (self._defect_round - self.calib_rounds) else "honest"
            return "honest"

        # ---- eta to use ------------------------------------------
        def _resolve_eta(self):
            if self.eta_source == "self" and self._calib_bers:
                import statistics as _st
                mu = _st.mean(self._calib_bers)
                sd = _st.pstdev(self._calib_bers) if len(self._calib_bers) > 1 else 0.0
                self._eta_self_est = max(0.0, mu + self.eta_k * sd)
                return self._eta_self_est
            return self.oracle_eta                     # oracle / fallback

        # ---- target (eta - margin), fixed or derived from estimation noise ----
        def _resolve_target(self, eta):
            """target = eta - margin.

            fixed   : margin is the hand-tuned constant `self.margin`.
            derived : margin = margin_k * sigma(own calib-window probe BER), so the
                      safety gap widens exactly when the FR's own eta estimate is
                      noisy (STATUS_AND_PLAN 10.4 #2). Falls back to the fixed margin
                      if the calib window is empty (no probe -> starved trigger class).
            """
            if self.margin_mode == "derived" and len(self._calib_bers) > 1:
                import statistics as _st
                sd = _st.pstdev(self._calib_bers)
                m = max(self.margin, self.margin_k * sd)   # never tighter than the floor
            else:
                m = self.margin
            self._margin_used = m
            return max(0.0, eta - m)

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
                               "ber_after": None if ba is None else round(ba, 4),
                               # dynamic diagnostics (None/constant in fixed mode)
                               "eta_source": self.eta_source,
                               "eta_self_est": (round(self._eta_self_est, 4)
                                                if getattr(self, "_eta_self_est", None) is not None else None),
                               "margin_used": (round(self._margin_used, 4)
                                               if getattr(self, "_margin_used", None) is not None else None),
                               "defect_round": self._defect_round})
            return submit, n

        # ---- graft support (coast_mode="graft") ------------------------------
        def _graft_keys(self):
            """State-dict keys of the watermark-carrying output layer to keep
            frozen during a graft coast. Uses the tap scope's kept params"""
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
            d = float(getattr(self, "graft_decay", 0.0))
            for k in self._graft_keys():
                if k in self._last_submit:
                    if d > 0.0 and k in global_state:
                        out[k] = ((1.0 - d) * self._last_submit[k] + d * global_state[k]).clone()
                    else:
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
                # dynamic warmup needs a probe every warmup round to detect convergence;
                # fixed warmup only needs it during the calib window
                if self.warmup_mode == "dynamic" or phase == "calib":
                    self._prepare(max(0, self.data_cpc), n_probe_holdout=holdout,
                                  trigger_train_n=self.trigger_train_n)
                    b = self._probe_ber(submit)
                    if b is not None:
                        self._probe_hist.append(b)      # dynamic-warmup convergence signal
                    if phase == "calib" and b is not None:
                        self._calib_bers.append(b)      # calibrates the "self" eta + derived margin
                self.trace.append({"round": round_idx, "action": phase, "eta_frozen": None,
                                   "probe_ber": None if (self.warmup_mode != "dynamic")
                                   else (round(self._probe_hist[-1], 4) if self._probe_hist else None)})
                return submit, n

            # first free-ride round: build loaders + freeze eta + target once
            self._prepare(max(0, self.data_cpc), n_probe_holdout=holdout,
                          trigger_train_n=self.trigger_train_n)
            if self._eta_frozen is None:
                self._eta_frozen = self._resolve_eta()
                self._target_frozen = self._resolve_target(self._eta_frozen)
                if self._defect_round is None:
                    self._defect_round = round_idx      # record where we actually defected
            eta = self._eta_frozen
            target = self._target_frozen

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
# NOTE ON THE "SUBMARINE"
# =============================================================================
# The adaptive submarine is `AdaptiveTapFreeRider` above (attack="adaptive_tap").
# It subsumes what an earlier, separate `SubmarineFreeRider` prototype was meant to
# do -- estimate its own eta, warm up until convergence, then coast/tap to hold its
# mark -- via the tap_* knobs:
#   * self-estimated eta ....... eta_source="self", eta_k
#   * dynamic warmup ........... warmup_mode="dynamic", conv_eps, conv_patience,
#                                honest_min, warmup_cap
#   * uncertainty-scaled margin  margin_mode="derived", margin_k
#   * coast without a replay ... coast_mode="graft" (+ scope) -> the gradual-fade sawtooth