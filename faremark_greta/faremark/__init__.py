"""federated-learning simulator - FedAvg 

  * Client.produce_update(global_state, prev_global_state, round_idx)
        -> method to override for custom clients
  * Aggregator.aggregate(updates)
        -> FedAvg by default; can be swapped/extended
  * Server.verify_hook
        -> no-op by default; or watermark extraction + detection
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
