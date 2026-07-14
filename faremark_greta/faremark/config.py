"""Experiment configs

`config_idx` selects an experiment; `repeat` selects a seed.

`expected_acc` is a loose [low, high] band on final FedAvg test accuracy for reference
"""
from dataclasses import dataclass, field, asdict


@dataclass
class ExpConfig:
    name: str
    model: str
    dataset: str
    num_clients: int
    rounds: int = 50
    local_epochs: int = 5
    lr: float = 0.01
    batch_size: int = 16
    momentum: float = 0.9
    weight_decay: float = 5e-4
    base_seed: int = 1000
    expected_acc: tuple = (0.0, 100.0)      # correctness band

    # ---- free-rider selection / paper baselines ----
    attack: str = "none"                    # "none" | "previous_models" | "gaussian" | "autopilot"
    num_free_riders: int = 0                # how many of num_clients are free-riders
    free_rider_ids: str = ""                # NEW: "3,6" pins which cids free-ride (overrides the
                                            # seeded choice). Empty => choose_free_riders(seed).
    noise_sigma: float = 0.1                # GaussianNoiseFreeRider std
    noise_decay: float = 0.0                # >0 -> sigma_t = sigma0 * t^(-decay)
    partition: str = "iid"                  # 'iid' or 'dirichlet' (non-IID)
    dirichlet_alpha: float = 0.5            # dirichlet skew; small=severe non-IID, large~=IID

    # ---- autopilot adaptive free-rider ----
    # Acts exactly like an honest client, except: 
    # (1) it estimates the detection threshold eta (or is GIVEN it for testing), 
    # (2) it behaves honestly until the server's eta is calibrated (the forced-honest window), then defects, 
    # (3) it can train on trigger+N/common or the full shard, and 
    # (4) after warmup it coasts and taps to hold its mark; a tap's cost = the data it trains on.
    autop_oracle_eta: float = 0.0           # >0 => FR is GIVEN the true eta (testing). 0 => estimate.
    autop_honest_until: int = 12            # behave honestly until BER converges or this round (the
                                            # forced-honest window the server calibrates eta on). 0=off.
    autop_conv_eps: float = 0.02            # "converged" = honest BER improves < this for 2 rounds
    autop_honest_extra: int = 3             # stay honest N rounds AFTER convergence (better frozen eta)
    autop_eta_k: float = 3.0                # k in the frozen estimate mu + k*sigma over converged honest
    autop_protect_until: int = 8            # never defect before this round
    autop_warmup_cap: int = 15              # hard cap so warmup can't run forever
    autop_max_batches: int = 250            # batch budget for the (rare) non-honest warmup transition
    autop_margin0: float = 0.06             # safety gap: target BER = eta - margin
    autop_floor: float = 0.05               # "mark is good" bar
    autop_common_per_class: int = -1        # DATA per tap: -1=full shard; 0=triggers-only; N=+N/common-class
    autop_scope: str = "full"              # PARAMS per tap: full | block2 | block | head
    autop_stay_min: bool = False            # coast (no training) while safely under target, tap only when needed.
                                            # False => tap EVERY post-warmup round (honest-style, for the data sweep).
    autop_holdout_ratio: float = 0.5        # fraction of trigger imgs reserved for the self-probe
    autop_honest_clone: bool = False        # DIAGNOSTIC: embed via the EXACT honest path every round
                                            # (control: shows the ~0.11 floor is the position, not the embedder)

    # ---- watermarking ----
    watermark: bool = False
    wm_bits: int = 0                        # m; 0 -> auto
    wm_lambda: float = 5.0                  # weight of L_wm (Eq. 11)
    wm_alpha: float = 0.4                   # smoothing exponent (Eq. 8)
    wm_f: str = "power"                     # smoothing kind: "power" | "sin"
    wm_beta: float = 0.6                    # memory coefficient (Eq. 14)
    wm_label_smoothing: float = 0.1
    wm_num_triggers: int = 50               # N_T trigger samples for extraction (Eq. 15)
    wm_eta: float = 0.25                    # detection-threshold floor (Eq. 16)
    wm_verify_every: int = 1
    paper_faithful: bool = True             # random keys, no trigger-class exclusion, cumulative mu+3sigma
    calib_on_all: bool = False              # calibrate eta over ALL clients (exposes circularity) vs benign-only

    def to_dict(self):
        return asdict(self)


CONFIGS = [
    # 0: fast smoke test to prove the pipeline learns
    ExpConfig("smoke_mnist_smallcnn", "smallcnn", "mnist", num_clients=5,
              rounds=5, local_epochs=1, batch_size=64, expected_acc=(95.0, 100.0)),

    # ---- Table I reproduction (FedAvg baseline) ----
    ExpConfig("resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              expected_acc=(88.0, 94.0)),
    ExpConfig("resnet18_mnist", "resnet18", "mnist", num_clients=10,
              expected_acc=(98.0, 99.7)),
    ExpConfig("resnet18_cifar100", "resnet18", "cifar100", num_clients=100,
              expected_acc=(70.0, 80.0)),
    ExpConfig("alexnet_cifar10", "alexnet", "cifar10", num_clients=10,
              expected_acc=(82.0, 90.0)),
    ExpConfig("alexnet_mnist", "alexnet", "mnist", num_clients=10,
              expected_acc=(88.0, 99.5)),
    ExpConfig("alexnet_cifar100", "alexnet", "cifar100", num_clients=10,
              expected_acc=(62.0, 74.0)),

    # 7: fast free-rider smoke (Fig. 7 trend)
    ExpConfig("fr_smoke_mnist", "smallcnn", "mnist", num_clients=10,
              rounds=10, local_epochs=1, batch_size=64,
              attack="previous_models", num_free_riders=0,
              expected_acc=(0.0, 100.0)),
    # 8: previous-models free-rider (Fig. 7a)
    ExpConfig("fr_prev_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              attack="previous_models", num_free_riders=2,
              expected_acc=(0.0, 100.0)),
    # 9: Gaussian-noise free-rider (Fig. 7c)
    ExpConfig("fr_gauss_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              attack="gaussian", num_free_riders=2, noise_sigma=0.1,
              expected_acc=(0.0, 100.0)),

    # ---- watermarking ----
    # 10: fast watermark smoke
    ExpConfig("wm_smoke_mnist", "smallcnn", "mnist", num_clients=10,
              rounds=10, local_epochs=1, batch_size=64,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              expected_acc=(0.0, 100.0)),
    # 11: fidelity, all honest + watermarked
    ExpConfig("wm_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              expected_acc=(86.0, 94.0)),
    # 12: detection, watermark + crude free-riders (Tables III-V)
    ExpConfig("wm_fr_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="previous_models", num_free_riders=2,
              expected_acc=(0.0, 100.0)),
    # 13: paper-faithful detection target, CIFAR-100
    ExpConfig("paper_faithful_resnet18_cifar100", "resnet18", "cifar100",
              num_clients=10, watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="previous_models", num_free_riders=2,
              paper_faithful=True, expected_acc=(0.0, 100.0)),

    # 14: autopilot submarine free-rider
    #     Override autop_* / free_rider_ids / attack via CLI (see run_tests.sh)
    ExpConfig("autopilot_paper_faithful_resnet18_cifar100", "resnet18", "cifar100",
              num_clients=10, watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="autopilot", num_free_riders=2, paper_faithful=True,
              expected_acc=(0.0, 100.0)),
]

AUTOPILOT_IDX = 14   # convenience for scripts


def get_config(idx: int) -> ExpConfig:
    if idx < 0 or idx >= len(CONFIGS):
        raise IndexError(
            f"config_idx {idx} out of range (have {len(CONFIGS)}): "
            + ", ".join(f"{i}:{c.name}" for i, c in enumerate(CONFIGS)))
    return CONFIGS[idx]


def seed_for(cfg: ExpConfig, repeat: int) -> int:
    return cfg.base_seed + repeat