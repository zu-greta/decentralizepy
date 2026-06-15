"""
FareMark: FL Server.

Responsibilities:
  1. Trigger assignment  — assign each client a unique trigger class (Stage I).
  2. FedAvg aggregation  — average all received local models (Stage II).
  3. Free-rider detection — extract watermarks from submitted models and
                            compare against registered keys (Eq. 15-16).
  4. IPR verification    — post-deployment ownership check.

Detection threshold eta is set to mu + 3*sigma over legitimate client
error rates observed during training (Section IV-D-3).
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np

from .watermark import WatermarkKey, extract_watermark, watermark_similarity, bit_accuracy


class FLServer:
    """
    Central server for the FareMark federated learning system.

    Args:
        global_model (nn.Module): Initial global model.
        num_clients (int): Total number of participating clients.
        num_classes (int): Number of output classes.
        wm_bits (int): Watermark bit-length per client.
        device (torch.device): Computation device.
        smooth_fn (str): Smoothing function used by all clients.
        alpha (float): Smoothing parameter.
        eta (float | None): Detection threshold.  If None, estimated
                             automatically from training statistics.
    """

    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        num_classes: int,
        wm_bits: int,
        device: torch.device,
        smooth_fn: str = "frac_power",
        alpha: float = 0.5,
        eta: Optional[float] = None,
    ):
        self.global_model = global_model.to(device)
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.wm_bits = wm_bits
        self.device = device
        self.smooth_fn = smooth_fn
        self.alpha = alpha
        self.eta = eta  # detection threshold

        # Populated by register_clients()
        self.client_keys: Dict[int, WatermarkKey] = {}
        self.trigger_classes: Dict[int, int] = {}

        # Running stats for threshold estimation
        self._benign_errors: List[float] = []

    # ------------------------------------------------------------------
    # Stage I: trigger assignment & key registration
    # ------------------------------------------------------------------

    def assign_triggers(self, client_ids: List[int]) -> Dict[int, int]:
        """
        Assign each client a unique trigger class (round-robin).
        If more clients than classes, multiple clients share a class
        but use distinct watermark keys (Section V-A-3 capacity analysis).

        Returns dict {client_id: trigger_class}.
        """
        assignment = {}
        for i, cid in enumerate(client_ids):
            assignment[cid] = i % self.num_classes
        self.trigger_classes = assignment
        return assignment

    def register_client(self, client_id: int, key: WatermarkKey):
        """Store a client's watermark key in the server database."""
        self.client_keys[client_id] = key

    # ------------------------------------------------------------------
    # Stage II: FedAvg aggregation
    # ------------------------------------------------------------------

    def aggregate(self, local_state_dicts: List[dict]) -> dict:
        """
        FedAvg: average all submitted local model parameter dicts.

        Returns the new global model state dict.
        """
        assert local_state_dicts, "No models received"
        avg = copy.deepcopy(local_state_dicts[0])
        for key in avg:
            avg[key] = avg[key].float()

        for sd in local_state_dicts[1:]:
            for key in avg:
                avg[key] += sd[key].float()

        n = len(local_state_dicts)
        for key in avg:
            avg[key] /= n

        # Cast back to original dtype
        ref = local_state_dicts[0]
        for key in avg:
            avg[key] = avg[key].to(ref[key].dtype)

        self.global_model.load_state_dict(avg)
        return copy.deepcopy(avg)

    # ------------------------------------------------------------------
    # Stage II: watermark verification per received local model
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verify_watermark(
        self,
        model: nn.Module,
        client_id: int,
        trigger_loader: DataLoader,
        n_triggers: int = 50,
    ) -> Tuple[float, torch.Tensor, bool]:
        """
        Extract the watermark from a submitted model and compare with
        the registered key.

        Args:
            model: The submitted local model.
            client_id: Which client submitted it.
            trigger_loader: DataLoader of trigger-class samples.
            n_triggers: How many trigger samples to use (N_T).

        Returns:
            (error, b_hat, is_free_rider)
            error: (1/m) sum |b_hat_k - b_k|  — lower is better match
            b_hat: recovered watermark bits
            is_free_rider: True if error > eta
        """
        key = self.client_keys[client_id]
        model.eval()
        model.to(self.device)

        # Collect up to n_triggers logits
        all_logits = []
        count = 0
        for images, _ in trigger_loader:
            images = images.to(self.device)
            logits = model(images)
            all_logits.append(logits)
            count += images.size(0)
            if count >= n_triggers:
                break

        all_logits = torch.cat(all_logits, dim=0)[:n_triggers]
        b_hat = extract_watermark(all_logits, key, self.smooth_fn, self.alpha)
        error = watermark_similarity(b_hat, key.B.to(self.device))

        # Auto-estimate threshold from benign history
        eta = self.eta if self.eta is not None else self._auto_eta()
        is_free_rider = error > eta

        return error, b_hat, is_free_rider

    def record_benign_error(self, error: float):
        """Call after verifying a known-benign client to update threshold stats."""
        self._benign_errors.append(error)

    def _auto_eta(self) -> float:
        """eta = mu + 3*sigma of observed benign errors (paper Section IV-D-3)."""
        if len(self._benign_errors) < 5:
            return 0.3  # fallback before enough data
        mu = float(np.mean(self._benign_errors))
        sigma = float(np.std(self._benign_errors))
        return mu + 3 * sigma

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_global_state_dict(self) -> dict:
        return copy.deepcopy(self.global_model.state_dict())

    def get_global_model(self) -> nn.Module:
        return self.global_model