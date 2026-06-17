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
