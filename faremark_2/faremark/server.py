import torch
import copy
import numpy as np
from collections import defaultdict
import logging

class Server:
    def __init__(self, global_model, watermark_manager, device='cpu',
                 threshold_factor=3.0, warmup_rounds=5, trigger_count=50):
        self.global_model = global_model.to(device)
        self.watermark_manager = watermark_manager
        self.device = device
        self.threshold_factor = threshold_factor
        self.warmup_rounds = warmup_rounds
        self.trigger_count = trigger_count
        self.round = 0
        self.free_rider_flags = {}
        # Store errors for all clients during warmup
        self.error_history = []  # list of (client_id, error)
        self.global_threshold = None

    def aggregate(self, client_states, client_sizes):
        """FedAvg aggregation with support for integer tensors."""
        avg_state = {}
        total_size = sum(client_sizes)
        for key in client_states[0].keys():
            orig_dtype = client_states[0][key].dtype
            avg_state[key] = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for state, size in zip(client_states, client_sizes):
                avg_state[key] += (size / total_size) * state[key].float()
            if not torch.is_floating_point(client_states[0][key]):
                avg_state[key] = torch.round(avg_state[key]).to(orig_dtype)
        self.global_model.load_state_dict(avg_state)
        return self.global_model

    def detect_free_riders(self, client_states, client_ids, trigger_data_loader):
        self.round += 1
        free_riders = {}
        errors = {}  # store error per client for this round

        for client_id, state in zip(client_ids, client_states):
            temp_model = copy.deepcopy(self.global_model)
            temp_model.load_state_dict(state)
            temp_model.eval()

            trigger_class = self.watermark_manager.trigger_classes[client_id]
            trigger_logits = []
            with torch.no_grad():
                for data, targets in trigger_data_loader:
                    mask = (targets == trigger_class)
                    if mask.any():
                        data_sel = data[mask].to(self.device)
                        logits = temp_model(data_sel)
                        trigger_logits.append(logits)
                    if sum(len(l) for l in trigger_logits) >= self.trigger_count:
                        break

            if not trigger_logits:
                # Not enough trigger samples – treat as free rider
                free_riders[client_id] = True
                errors[client_id] = 1.0
                continue

            extracted_bit, avg_z = self.watermark_manager.extract_watermark(
                trigger_logits, client_id, transform='power'
            )
            registered_bit = self.watermark_manager.watermark_bits[client_id].item()
            error = abs(extracted_bit - registered_bit)
            errors[client_id] = error

            # During warmup, collect errors
            if self.round <= self.warmup_rounds:
                self.error_history.append((client_id, error))
                # Mark all as benign during warmup
                free_riders[client_id] = False
            else:
                # After warmup, use global threshold
                if self.global_threshold is None:
                    # Compute global threshold from all errors collected during warmup
                    if self.error_history:
                        all_errors = [e for _, e in self.error_history]
                        mu = np.mean(all_errors)
                        sigma = np.std(all_errors)
                        self.global_threshold = mu + self.threshold_factor * sigma
                        logging.info(f"Global threshold computed: μ={mu:.4f}, σ={sigma:.4f}, threshold={self.global_threshold:.4f}")
                    else:
                        self.global_threshold = 0.5  # fallback

                # Flag as free rider if error exceeds global threshold
                is_free_rider = (error > self.global_threshold)
                free_riders[client_id] = is_free_rider

                # Debug logging
                logging.debug(f"Client {client_id}: registered={registered_bit}, extracted={extracted_bit}, "
                              f"error={error:.2f}, threshold={self.global_threshold:.2f}, free_rider={is_free_rider}")

        return free_riders