"""FareMark reproduction — a minimal, modular federated-learning simulator

Honest FedAvg with no free-riders and no watermarking

  * Client.produce_update(global_state, prev_global_state, round_idx)
        -> the single hook a free-rider or watermark-embedding client overrides
  * Aggregator.aggregate(updates)
        -> FedAvg by default; can be swapped/extended
  * Server.verify_hook
        -> no-op by default; becomes watermark extraction + detection
"""

from .config import CONFIGS, get_config, seed_for
from .clients import Client
from .server import Server, Aggregator

__all__ = [
    "CONFIGS",
    "get_config",
    "seed_for",
    "Client",
    "Server",
    "Aggregator",
]
