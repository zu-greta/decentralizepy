"""Server side: FedAvg aggregation + round orchestration.

The server keeps the last two global states so Stage 2 free-riders (which need
W_t and W_{t-1}) work without changing the loop. `verify_hook` is a no-op now;
in Stage 3/4 it becomes per-client watermark extraction + detection.
"""
import copy

import torch

from .utils import evaluate_accuracy


class Aggregator:
    """Weighted FedAvg. With equal IID shards this equals the simple mean the
    paper uses (W_g = 1/N * sum W_i)."""

    @staticmethod
    def aggregate(updates: list) -> dict:
        # updates: list of (state_dict_on_cpu, num_samples)
        total = sum(n for _, n in updates)
        agg = {}
        first = updates[0][0]
        for key, ref in first.items():
            if ref.is_floating_point():
                acc = torch.zeros_like(ref, dtype=torch.float64)
                for state, n in updates:
                    acc += state[key].double() * (n / total)
                agg[key] = acc.to(ref.dtype)
            else:
                # Integer buffers (e.g. num_batches_tracked): copy, don't average.
                agg[key] = ref.clone()
        return agg


class Server:
    def __init__(self, model, clients, test_loader, device, logger,
                 verify_hook=None):
        self.model = model
        self.clients = clients
        self.test_loader = test_loader
        self.device = device
        self.logger = logger
        self.aggregator = Aggregator()
        self.verify_hook = verify_hook  # callable(server, round, updates) or None

        self.global_state = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
        self.prev_global_state = None
        self.history = []  # list of dicts: {round, test_acc, ...}

    def run(self, rounds: int):
        for r in range(1, rounds + 1):
            updates = []
            for client in self.clients:
                state, n = client.produce_update(
                    copy.deepcopy(self.global_state),
                    copy.deepcopy(self.prev_global_state),
                    r,
                )
                updates.append((state, n))

            # Stage 3/4 will plug watermark verification in here.
            verify_info = {}
            if self.verify_hook is not None:
                verify_info = self.verify_hook(self, r, updates) or {}

            self.prev_global_state = self.global_state
            self.global_state = self.aggregator.aggregate(updates)

            acc = self._evaluate()
            record = {"round": r, "test_acc": acc, **verify_info}
            self.history.append(record)
            msg = f"round {r:3d}/{rounds}  test_acc={acc:6.2f}%"
            # Surface watermark metrics so embedding/detection can be watched
            # live (Fig. 8) instead of only via result.json.
            if "wm_benign_ber" in verify_info:
                msg += f"  benign_BER={verify_info['wm_benign_ber']:.3f}"
                if verify_info.get("wm_fr_ber") is not None:
                    msg += f"  fr_BER={verify_info['wm_fr_ber']:.3f}"
                if verify_info.get("wm_detect_acc") is not None:
                    msg += f"  det_acc={verify_info['wm_detect_acc']:.2f}"
            self.logger.info(msg)
        return self.history

    def _evaluate(self) -> float:
        self.model.load_state_dict(self.global_state)
        self.model.to(self.device)
        return evaluate_accuracy(self.model, self.test_loader, self.device)