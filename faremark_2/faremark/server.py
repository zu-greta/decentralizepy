import torch
import copy
import numpy as np
from collections import defaultdict

class Server:
    def __init__(self, global_model, watermark_manager, device='cpu',
                 threshold_factor=3.0, warmup_rounds=5, trigger_count=50):
        self.global_model = global_model.to(device)
        self.watermark_manager = watermark_manager
        self.device = device
        self.threshold_factor = threshold_factor
        self.warmup_rounds = warmup_rounds
        self.trigger_count = trigger_count  # number of trigger samples used for extraction (from validation set)
        self.round = 0
        self.free_rider_flags = {}
        self.watermark_error_history = defaultdict(list)  # client_id -> list of errors
        self.thresholds = {}  # client_id -> threshold

    def aggregate(self, client_states, client_sizes):
        """FedAvg aggregation."""
        avg_state = {}
        total_size = sum(client_sizes)
        for key in client_states[0].keys():
            avg_state[key] = torch.zeros_like(client_states[0][key])
            for state, size in zip(client_states, client_sizes):
                avg_state[key] += (size / total_size) * state[key]
        self.global_model.load_state_dict(avg_state)
        return self.global_model

    def detect_free_riders(self, client_states, client_ids, trigger_data_loader):
        """
        Extract watermark from each client's model and compare with registered bit.
        Returns dict of detected free riders (client_id: True/False).
        """
        self.round += 1
        free_riders = {}

        # For each client, extract watermark using trigger samples from a held-out dataset.
        # We need a loader that provides only trigger-class samples for each client.
        # For simplicity, we assume trigger_data_loader is a list of loaders per client (or a single loader with known trigger class).
        # We'll extract using a fixed set of trigger samples from the test set (we can pre-allocate).
        # In this implementation, we pass a list of (data, targets) for each client's trigger class.
        # We'll have a utility to get trigger samples.

        for client_id, state in zip(client_ids, client_states):
            # Load state into a temporary model
            temp_model = copy.deepcopy(self.global_model)
            temp_model.load_state_dict(state)
            temp_model.eval()

            # Get trigger samples for this client (class = trigger_class)
            trigger_class = self.watermark_manager.trigger_classes[client_id]
            # We'll use the trigger_data_loader (which should be a DataLoader of test set)
            # Filter samples of trigger class
            trigger_logits = []
            with torch.no_grad():
                for data, targets in trigger_data_loader:
                    # Select samples with target == trigger_class
                    mask = (targets == trigger_class)
                    if mask.any():
                        data_sel = data[mask].to(self.device)
                        logits = temp_model(data_sel)
                        trigger_logits.append(logits)
                    if sum(len(l) for l in trigger_logits) >= self.trigger_count:
                        break
            if not trigger_logits:
                # Not enough samples, skip or assume free rider
                free_riders[client_id] = True
                continue

            extracted_bit, avg_z = self.watermark_manager.extract_watermark(
                trigger_logits, client_id, transform='power'
            )
            registered_bit = self.watermark_manager.watermark_bits[client_id].item()
            error = abs(extracted_bit - registered_bit)

            # Record error for threshold learning (only during warmup)
            if self.round <= self.warmup_rounds:
                self.watermark_error_history[client_id].append(error)

            # Determine threshold if warmup done
            if self.round > self.warmup_rounds:
                if client_id not in self.thresholds:
                    # Compute threshold from history (μ + 3σ)
                    hist = self.watermark_error_history.get(client_id, [0])
                    mu = np.mean(hist)
                    sigma = np.std(hist) if len(hist) > 1 else 0.1
                    self.thresholds[client_id] = mu + self.threshold_factor * sigma
                # Detection: if error > threshold, free rider
                free_riders[client_id] = (error > self.thresholds[client_id])
            else:
                # During warmup, treat all as benign (or mark based on error=0)
                free_riders[client_id] = False

        return free_riders