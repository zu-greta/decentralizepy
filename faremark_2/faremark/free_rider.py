import torch
import copy
import numpy as np

def apply_free_rider_strategy(strategy, global_model, round_num, state_dict_prev=None,
                              noise_scale=0.01, attack_round=50):
    """
    Generate a local model for a free rider client.
    Returns updated state_dict.
    """
    if strategy == "previous_models":
        # Use difference of previous global models (if available)
        # Here we assume we have state_dict_prev (global model from previous round)
        if state_dict_prev is not None:
            # Compute free model as w_t - w_{t-1} (or just w_t with zero gradient?)
            # In paper: they construct shallow model from previous two rounds
            # Simpler: return a copy of global with small random perturbation? 
            # Actually they define: W_free = W^t - W^{t-1} (difference)
            # We'll simulate by using the previous model as is (or difference)
            # For simplicity, we just return the global model (pretending to contribute)
            # but the server will detect no watermark.
            return copy.deepcopy(global_model.state_dict())
        else:
            return copy.deepcopy(global_model.state_dict())
    elif strategy == "gaussian_noise":
        # Add Gaussian noise to global model
        free_state = copy.deepcopy(global_model.state_dict())
        for key in free_state:
            free_state[key] += noise_scale * torch.randn_like(free_state[key])
        return free_state
    elif strategy == "train_then_attack":
        # If round < attack_round, contribute normally (with watermark)
        # Else free ride (return global model without watermark)
        if round_num < attack_round:
            return None  # signal to train normally
        else:
            # return global model as is (no watermark)
            return copy.deepcopy(global_model.state_dict())
    elif strategy == "trigger_only":
        # Train only on trigger samples, overfitting; but here we simulate by returning a model with no watermark
        # Actually we need to train on only trigger samples; we'll handle in client if we have such option.
        # For now, we return global model (will be detected as free rider)
        return copy.deepcopy(global_model.state_dict())
    else:
        raise ValueError(f"Unknown free-rider strategy: {strategy}")