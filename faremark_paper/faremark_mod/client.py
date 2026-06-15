"""
FareMark: FL Client.

Implements:
  - Client-side watermark embedding via multi-task loss (Eq. 11):
        L = L_cl + lambda * L_wm
    where L_cl is cross-entropy on all classes and L_wm only applies
    to samples of the trigger class.

  - Memory-enhanced local gradient update rule (Eq. 14):
        W_{i+1} = beta * (W_i + lr * dL/dW_g) + (1 - beta) * W_g
    This helps the watermark survive FedAvg aggregation by blending
    local updates with the previous global model.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Optional

from .watermark import WatermarkKey, watermark_loss


class FLClient:
    """
    One participating client in the FareMark federated learning system.

    Args:
        client_id (int): Unique client identifier.
        model (nn.Module): Local model (deep copy of global model).
        trigger_class (int): Class index assigned by the server as watermark trigger.
        key (WatermarkKey): Client's private watermark key.
        train_dataset: Full local training dataset.
        device (torch.device): Computation device.
        lr (float): SGD learning rate (default 0.01 per paper).
        local_epochs (int): Local training epochs per round (default 2 per paper).
        batch_size (int): Local batch size (default 16 per paper).
        lam (float): lambda — trade-off between L_cl and L_wm (Eq. 11).
        beta (float): Memory-enhancement blend factor (Eq. 14).
        smooth_fn (str): Smoothing function for watermark projection.
        alpha (float): Smoothing parameter.
        is_free_rider (bool): If True, client behaves as a free-rider.
        free_rider_type (str): 'previous_models' or 'gaussian_noise'.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        trigger_class: int,
        key: WatermarkKey,
        train_dataset,
        device: torch.device,
        lr: float = 0.01,
        local_epochs: int = 2,
        batch_size: int = 16,
        lam: float = 1.0,
        beta: float = 0.9,
        smooth_fn: str = "frac_power",
        alpha: float = 0.5,
        is_free_rider: bool = False,
        free_rider_type: str = "previous_models",
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.trigger_class = trigger_class
        self.key = key.to(device)
        self.train_dataset = train_dataset
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lam = lam
        self.beta = beta
        self.smooth_fn = smooth_fn
        self.alpha = alpha
        self.is_free_rider = is_free_rider
        self.free_rider_type = free_rider_type

        # For free-rider 'previous_models' strategy
        self._prev_global: Optional[dict] = None
        self._prev_prev_global: Optional[dict] = None

        # Split dataset into trigger-class and common-class subsets
        self._split_dataset()

    def _split_dataset(self):
        """Separate trigger-class samples from common-class samples."""
        trigger_indices = []
        common_indices = []
        for i in range(len(self.train_dataset)):
            _, label = self.train_dataset[i]
            if label == self.trigger_class:
                trigger_indices.append(i)
            else:
                common_indices.append(i)

        self.trigger_subset = Subset(self.train_dataset, trigger_indices)
        self.common_subset = Subset(self.train_dataset, common_indices)

        self.trigger_loader = DataLoader(
            self.trigger_subset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        ) if trigger_indices else None

        self.common_loader = DataLoader(
            self.common_subset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        ) if common_indices else None

    def set_model(self, global_state_dict: dict):
        """Load global model parameters into local model."""
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def train(self, global_state_dict: dict) -> dict:
        """
        Run one round of local training.

        Returns the updated local model state dict.
        For free-riders, returns a fake model without real training.
        """
        if self.is_free_rider:
            return self._free_rider_update(global_state_dict)

        # Load current global model
        self.set_model(global_state_dict)
        global_params = copy.deepcopy(global_state_dict)

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.local_epochs):
            # --- Common class: standard cross-entropy ---
            if self.common_loader:
                for images, labels in self.common_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    # Memory-enhanced update (Eq. 14) — applied after backward
                    self._memory_enhanced_step(optimizer, global_params)

            # --- Trigger class: cross-entropy + watermark loss ---
            if self.trigger_loader:
                for images, labels in self.trigger_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(images)
                    l_cl = criterion(logits, labels)
                    l_wm = watermark_loss(logits, self.key, self.smooth_fn, self.alpha)
                    loss = l_cl + self.lam * l_wm
                    loss.backward()
                    self._memory_enhanced_step(optimizer, global_params)

        return copy.deepcopy(self.model.state_dict())

    def _memory_enhanced_step(self, optimizer: optim.SGD, global_params: dict):
        """
        Apply memory-enhanced update rule instead of plain SGD step.

        Eq. 14:  W_{i+1} = beta * (W_i + lr * grad) + (1-beta) * W_g

        We implement this by:
          1. Performing a standard SGD step (W_i + lr * grad)
          2. Blending the result with the global model
        """
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_params:
                    g = global_params[name].to(self.device)
                    param.data = self.beta * param.data + (1.0 - self.beta) * g

    def _free_rider_update(self, global_state_dict: dict) -> dict:
        """
        Produce a fake local model without real training.

        'previous_models':  W_free = W_t - W_{t-1}  (Eq. 17)
        'gaussian_noise':   W_free = W_t + N(0, sigma)  (Eq. 18)
        """
        if self.free_rider_type == "previous_models":
            if self._prev_global is None:
                # First round — just submit global model as-is
                fake = copy.deepcopy(global_state_dict)
                self._prev_global = copy.deepcopy(global_state_dict)
                return fake
            fake = {}
            for k in global_state_dict:
                fake[k] = global_state_dict[k] - self._prev_global[k]
            self._prev_global = copy.deepcopy(global_state_dict)
            return fake

        elif self.free_rider_type == "gaussian_noise":
            sigma = 0.01
            fake = {}
            for k, v in global_state_dict.items():
                noise = torch.randn_like(v.float()) * sigma
                fake[k] = v + noise.to(v.dtype)
            return fake

        else:
            raise ValueError(f"Unknown free_rider_type: {self.free_rider_type}")