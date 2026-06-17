"""Client side of the simulation.

`produce_update` is the ONE method later stages override:

  * Stage 2 free-rider: ignore the data, fabricate weights from the global
    history (`global_state`, `prev_global_state`) per Eq. 17 / Eq. 18.
  * Stage 3 watermark client: split data into trigger / common classes, add the
    L_wm regularizer (Eq. 11-12), and apply the memory-enhanced update (Eq. 14).

The honest client below just does standard local SGD starting from the current
global weights — i.e. textbook FedAvg.
"""
from __future__ import annotations  

import copy

import torch
import torch.nn as nn


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
        self._local_train()
        return _to_cpu_state(self.model), self.num_samples

    # ---- honest local training --------------------------------------------
    def _local_train(self):
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr,
            momentum=self.momentum, weight_decay=self.weight_decay,
        )
        for _ in range(self.local_epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                optimizer.step()