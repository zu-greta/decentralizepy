"""FareMark reproduction — a minimal, modular federated-learning simulator.

Stage 1 (this module set) implements honest FedAvg with no free-riders and no
watermarking. The design leaves explicit seams for the later stages:

  * Client.produce_update(global_state, prev_global_state, round_idx)
        -> the single hook a free-rider (Stage 2) or watermark-embedding client
           (Stage 3) overrides.
  * Aggregator.aggregate(updates)
        -> FedAvg now; can be swapped/extended later.
  * Server.verify_hook
        -> no-op now; becomes watermark extraction + detection in Stage 3/4.
"""

from .config import CONFIGS, get_config, seed_for
from .client import Client
from .server import Server, Aggregator

__all__ = [
    "CONFIGS",
    "get_config",
    "seed_for",
    "Client",
    "Server",
    "Aggregator",
]
