"""Experiment registry.

`config_idx` selects an experiment; `repeat` selects a seed (the paper averages
over 10 repeats). This matches your existing submit script's
`--config_idx N --repeat M` interface.

`expected_acc` is a loose [low, high] band on final FedAvg test accuracy used as
the Stage-1 correctness gate. The reference points are the FedAvg(%) column of
Table I in the paper:

    ResNet-18  CIFAR-10  (10 clients)  91.85
    ResNet-18  MNIST     (10 clients)  98.75
    ResNet-18  CIFAR-100 (100 clients) 76.54
    AlexNet    CIFAR-10  (10 clients)  86.35
    AlexNet    MNIST     (10 clients)  91.54
    AlexNet    CIFAR-100 (10 clients)  68.45

Paper training budget for the fidelity table: 50 rounds x 5 local epochs,
lr=0.01, batch=16 (the experimental-settings section says local_epoch=2 /
global=100 — the paper is internally inconsistent; we follow the fidelity
section since Table I is the fidelity result). All of these are overridable on
the CLI.
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
    expected_acc: tuple = (0.0, 100.0)  # correctness band for Stage 1

    # ---- Stage 2: free-rider attacks ----
    attack: str = "none"        # "none" | "previous_models" | "gaussian"
    num_free_riders: int = 0    # how many of num_clients are free-riders
    noise_sigma: float = 0.1    # GaussianNoiseFreeRider std
    noise_decay: float = 0.0    # >0 -> sigma_t = sigma0 * t^(-decay)
    attack_round: int = 50      # train_then_attack: round at which the FR defects (Table IV)
    n_trigger_samples: int = 8  # trigger_only: # trigger samples the FR overfits (Table V)

    # ---- Stage 3: watermarking ----
    watermark: bool = False     # honest clients embed an output-space watermark
    wm_bits: int = 0            # m; 0 -> auto = (num_classes - 1)//2 (l=2)
    wm_lambda: float = 5.0      # weight of L_wm in L = L_cl + lambda L_wm (Eq.11)
    wm_alpha: float = 0.4       # smoothing f() exponent (Eq. 8)
    wm_f: str = "power"         # smoothing kind: "power" | "sin" (Eq. 7-9)
    wm_beta: float = 0.6        # memory coefficient in the Eq. 14 update
    wm_label_smoothing: float = 0.1  # keeps the softmax tail movable
    wm_num_triggers: int = 50   # N_T trigger samples used for extraction (Eq.15)
    wm_eta: float = 0.25        # detection threshold on BER (Eq. 16)
    wm_verify_every: int = 1    # run verification every k rounds (cost control)

    def to_dict(self):
        return asdict(self)


# Index 0 is a fast smoke test (minutes on a GPU) to prove the pipeline learns
# before committing to a multi-hour Table I run.
CONFIGS = [
    ExpConfig("smoke_mnist_smallcnn", "smallcnn", "mnist", num_clients=5,
              rounds=5, local_epochs=1, batch_size=64, expected_acc=(95.0, 100.0)),

    # ---- Table I reproduction configs (FedAvg baseline) ----
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

    # ---- Stage 2: free-rider attacks ----
    # The Stage-2 gate is the TREND (main-task accuracy falls as the free-rider
    # fraction rises, cf. Fig. 7), not a single accuracy band. Sweep the number
    # of free-riders with the --num_free_riders override (or NUM_FREE_RIDERS in
    # submit_experiment.sh) and watch accuracy drop.

    # idx 7: fast Stage-2 smoke so you can see the Fig.7 trend in minutes.
    ExpConfig("fr_smoke_mnist", "smallcnn", "mnist", num_clients=10,
              rounds=10, local_epochs=1, batch_size=64,
              attack="previous_models", num_free_riders=0,
              expected_acc=(0.0, 100.0)),

    # idx 8: free-riding with previous models, ResNet-18 / CIFAR-10 (Fig. 7a).
    ExpConfig("fr_prev_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              attack="previous_models", num_free_riders=2,
              expected_acc=(0.0, 100.0)),

    # idx 9: free-riding with Gaussian noise, ResNet-18 / CIFAR-10 (Fig. 7c).
    ExpConfig("fr_gauss_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              attack="gaussian", num_free_riders=2, noise_sigma=0.1,
              expected_acc=(0.0, 100.0)),

    # ---- Stage 3: watermarking ----
    # The Stage-3 honest-only gate is (a) FIDELITY: final acc within ~2 pts of
    # the Stage-1 FedAvg baseline, AND (b) EMBEDDING: mean benign BER ~ 0 so the
    # watermark is recovered (Tables II / VII). Free-rider DETECTION (idx 12)
    # checks that fabricated updates fail extraction (high BER) -> Stage 4.

    # idx 10: fast watermark smoke so you can see embed+extract in minutes.
    ExpConfig("wm_smoke_mnist", "smallcnn", "mnist", num_clients=10,
              rounds=10, local_epochs=1, batch_size=64,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              expected_acc=(0.0, 100.0)),

    # idx 11: fidelity run, ResNet-18 / CIFAR-10, all honest + watermarked
    # (compare final acc to the 93.22% Stage-1 baseline; Table I "Ours").
    ExpConfig("wm_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              expected_acc=(86.0, 94.0)),

    # idx 12: detection run, watermark + free-riders (Tables III-V / Stage 4).
    ExpConfig("wm_fr_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="previous_models", num_free_riders=2,
              expected_acc=(0.0, 100.0)),
]


def get_config(idx: int) -> ExpConfig:
    if idx < 0 or idx >= len(CONFIGS):
        raise IndexError(
            f"config_idx {idx} out of range (have {len(CONFIGS)}): "
            + ", ".join(f"{i}:{c.name}" for i, c in enumerate(CONFIGS))
        )
    return CONFIGS[idx]


def seed_for(cfg: ExpConfig, repeat: int) -> int:
    return cfg.base_seed + repeat