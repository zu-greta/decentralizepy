"""
FareMark: Configuration.

All hyperparameters match Section V-A of the paper unless otherwise noted.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FareMarkConfig:
    # -----------------------------------------------------------------------
    # Federated learning setup
    # -----------------------------------------------------------------------
    num_clients: int = 10           # N — number of participating clients
    num_free_riders: int = 0        # how many clients are free-riders
    free_rider_type: str = "previous_models"  # 'previous_models' or 'gaussian_noise'

    global_rounds: int = 100        # communication rounds
    local_epochs: int = 2           # E — local epochs per round (paper: 2)
    batch_size: int = 16            # B — local batch size (paper: 16)
    lr: float = 0.01                # learning rate (paper: SGD lr=0.01)
    momentum: float = 0.9

    # -----------------------------------------------------------------------
    # Watermarking
    # -----------------------------------------------------------------------
    wm_bits: int = 8                # m — watermark bit length per client
    lam: float = 1.0                # lambda — L_cl + lambda * L_wm
    beta: float = 0.9               # memory-enhanced blend factor
    smooth_fn: str = "frac_power"   # 'neg_power', 'frac_power', or 'sin'
    alpha_smooth: float = 0.5       # smoothing parameter alpha
    n_triggers: int = 100           # N_T — trigger samples for verification
    eta: Optional[float] = None     # detection threshold; None = auto (mu+3sigma)

    # -----------------------------------------------------------------------
    # Model & dataset
    # -----------------------------------------------------------------------
    model_name: str = "resnet18"    # 'alexnet', 'resnet18', 'shufflenet', 'googlenet'
    dataset_name: str = "cifar10"   # 'mnist', 'cifar10', 'cifar100', 'food100'
    data_root: str = "./data"

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    eval_every: int = 10            # evaluate every N rounds
    save_every: int = 50            # checkpoint every N rounds

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    seed: int = 42

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    output_dir: str = "./results"
    exp_name: str = "default"

    # -----------------------------------------------------------------------
    # Hardware
    # -----------------------------------------------------------------------
    device: str = "cuda"            # 'cuda' or 'cpu'
    num_workers: int = 2


# ---------------------------------------------------------------------------
# Preset configurations matching paper experiments
# ---------------------------------------------------------------------------

def config_table1(model: str, dataset: str, num_clients: int = 10) -> FareMarkConfig:
    """
    Table I — Fidelity: model accuracy with and without watermarking.
    All clients are benign; compare FedAvg baseline with FareMark.
    """
    return FareMarkConfig(
        model_name=model,
        dataset_name=dataset,
        num_clients=num_clients,
        num_free_riders=0,
        global_rounds=50,
        local_epochs=5,        # paper: "each round, client update for 5 epoch"
        exp_name=f"table1_{model}_{dataset}_n{num_clients}",
    )


def config_table2(model: str, dataset: str, num_clients: int = 10) -> FareMarkConfig:
    """
    Table II — Watermark detection accuracy comparison with FedIPR.
    N_T=100 triggers, 50 training rounds.
    """
    return FareMarkConfig(
        model_name=model,
        dataset_name=dataset,
        num_clients=num_clients,
        num_free_riders=0,
        global_rounds=50,
        local_epochs=5,
        n_triggers=100,
        exp_name=f"table2_{model}_{dataset}",
    )


def config_fig7(
    model: str,
    dataset: str,
    num_free_riders: int,
    free_rider_type: str = "previous_models",
) -> FareMarkConfig:
    """
    Figure 7 — Main task accuracy vs number of free-riders.
    """
    return FareMarkConfig(
        model_name=model,
        dataset_name=dataset,
        num_clients=10,
        num_free_riders=num_free_riders,
        free_rider_type=free_rider_type,
        global_rounds=100,
        local_epochs=2,
        exp_name=f"fig7_{model}_{dataset}_fr{num_free_riders}_{free_rider_type}",
    )


def config_fig8(model: str, dataset: str) -> FareMarkConfig:
    """
    Figure 8 — Watermark detection rate over training rounds.
    Local models uploaded every 10 epochs = 1 communication round.
    """
    return FareMarkConfig(
        model_name=model,
        dataset_name=dataset,
        num_clients=10,
        num_free_riders=1,
        global_rounds=100,
        local_epochs=10,      # 10 training epochs per round
        eval_every=1,         # evaluate every round for the curve
        exp_name=f"fig8_{model}_{dataset}",
    )


def config_table3(
    model: str,
    dataset: str,
    fr_ratio: float = 0.2,
) -> FareMarkConfig:
    """
    Table III — Free-rider detection with varying free-rider ratios.
    """
    num_clients = 10
    num_fr = max(1, int(num_clients * fr_ratio))
    return FareMarkConfig(
        model_name=model,
        dataset_name=dataset,
        num_clients=num_clients,
        num_free_riders=num_fr,
        global_rounds=100,
        local_epochs=2,
        n_triggers=50,
        exp_name=f"table3_{model}_{dataset}_fr{int(fr_ratio*100)}pct",
    )


def config_robustness_dp(model: str = "resnet18", dataset: str = "cifar10") -> FareMarkConfig:
    """Table VI — Robustness against differential privacy."""
    return FareMarkConfig(
        model_name=model,
        dataset_name=dataset,
        num_clients=10,
        num_free_riders=0,
        global_rounds=100,
        local_epochs=2,
        exp_name=f"table6_dp_{model}_{dataset}",
    )


def config_table7(model: str, dataset: str, n_triggers: int) -> FareMarkConfig:
    """Table VII — Effect of number of trigger samples."""
    return FareMarkConfig(
        model_name=model,
        dataset_name=dataset,
        num_clients=10,
        num_free_riders=0,
        global_rounds=50,
        local_epochs=5,
        n_triggers=n_triggers,
        exp_name=f"table7_{model}_{dataset}_nt{n_triggers}",
    )