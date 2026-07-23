"""Server side: FedAvg aggregation + round orchestration.

The server keeps the last two global states so free-riders (which need
W_t and W_{t-1}) work without changing the loop. `verify_hook` is per-client 
watermark extraction + detection.
"""
import copy

import torch

from .utils import evaluate_accuracy
from .runlog import RoundTable


class Aggregator:
    """Weighted FedAvg. With equal IID shards this equals the simple mean the
    paper uses (W_g = 1/N * sum W_i)."""

    @staticmethod
    def aggregate(updates: list) -> dict:
        # updates: list of (state_dict_on_cpu, num_samples)
        total = sum(n for _, n in updates)
        agg = {}
        first = updates[0][0]
        # weighted mean per param tensor
        for key, ref in first.items():
            if ref.is_floating_point():
                acc = torch.zeros_like(ref, dtype=torch.float64)
                for state, n in updates:
                    acc += state[key].double() * (n / total)
                agg[key] = acc.to(ref.dtype)
            else:
                # integer buffers (e.g. num_batches_tracked): copy, don't average
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
        table = RoundTable(self.logger, rounds,
                           watermarked=self.verify_hook is not None)
        seen_action = {}          # cid -> last trace action, for phase-change notes

        for r in range(1, rounds + 1):
            updates = []
            for client in self.clients:
                state, n = client.produce_update( 
                    copy.deepcopy(self.global_state),
                    copy.deepcopy(self.prev_global_state),
                    r,
                )
                updates.append((state, n))

            # watermark verification
            verify_info = {}
            if self.verify_hook is not None:
                verify_info = self.verify_hook(self, r, updates) or {}

            self.prev_global_state = self.global_state
            self.global_state = self.aggregator.aggregate(updates) # FedAvg aggregation

            acc = self._evaluate()
            record = {"round": r, "test_acc": acc, **verify_info}
            self.history.append(record)

            # announce free-rider phase transitions (honest -> calib -> tap / coast).
            changes = []
            for cid, c in enumerate(self.clients):
                tr = getattr(c, "trace", None)
                if not tr or tr[-1].get("round") != r:
                    continue
                act = tr[-1].get("action")
                if act and seen_action.get(cid) != act:
                    seen_action[cid] = act
                    changes.append((cid, act))
            table.row(r, acc, verify_info)
            if changes:
                table.phase_note(r, changes)
        return self.history

    def _evaluate(self) -> float:
        self.model.load_state_dict(self.global_state)
        self.model.to(self.device)
        return evaluate_accuracy(self.model, self.test_loader, self.device)