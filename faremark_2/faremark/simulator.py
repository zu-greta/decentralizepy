import torch
import copy
import numpy as np
import logging
import time
import os
from .client import Client
from .server import Server
from .watermark import WatermarkManager
from .free_rider import apply_free_rider_strategy
from .datasets import partition_data, get_dataloaders, get_dataset
from .models import get_model
from .utils import set_seed, setup_logging

def run_simulation(config):
    # Setup
    set_seed(config.get('seed', 42))
    log_dir = setup_logging(config['experiment_name'])
    logging.info(f"Starting experiment: {config['experiment_name']}")
    logging.info(f"Config: {config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Dataset
    train_set, test_set, num_classes = get_dataset(config['dataset'])
    client_indices = partition_data(train_set, config['num_clients'], iid=True)
    client_loaders, test_loader = get_dataloaders(client_indices, train_set, test_set, config['batch_size'])

    # Model
    global_model = get_model(config['model'], num_classes)
    global_model.to(device)

    # Watermark Manager
    wm_manager = WatermarkManager(
        num_clients=config['num_clients'],
        num_classes=num_classes,
        trigger_sample_count=config['trigger_sample_count'],
        alpha=config.get('alpha', 0.5),
        device=device
    )

    # Server
    server = Server(global_model, wm_manager, device=device, trigger_count=config['trigger_sample_count'])

    # Clients
    clients = []
    for i in range(config['num_clients']):
        client = Client(
            client_id=i,
            model=copy.deepcopy(global_model),
            train_loader=client_loaders[i],
            watermark_manager=wm_manager,
            device=device,
            lr=config['learning_rate'],
            watermark_lambda=config['watermark_lambda'],
            memory_mu=config['memory_mu'],
            alpha=config['alpha']
        )
        clients.append(client)

    # Free rider configuration
    free_rider_ratio = config.get('free_rider_ratio', 0.0)
    free_rider_strategy = config.get('free_rider_strategy', 'previous_models')
    attack_round = config.get('attack_round', 50)
    noise_scale = config.get('noise_scale', 0.01)

    # Determine which clients are free riders
    num_free_riders = int(config['num_clients'] * free_rider_ratio)
    free_rider_ids = np.random.choice(config['num_clients'], num_free_riders, replace=False).tolist()
    logging.info(f"Free riders: {free_rider_ids}")

    # Training loop
    client_sizes = [len(indices) for indices in client_indices]
    round_accuracies = []
    round_watermark_errors = {i: [] for i in range(config['num_clients'])}
    free_rider_detection_rates = []

    # For previous_models strategy, we need to store previous global state
    prev_global_state = None

    for round_num in range(1, config['num_rounds']+1):
        logging.info(f"Round {round_num}")
        client_states = []
        client_ids = []
        selected_clients = np.random.choice(config['num_clients'], config['num_clients'], replace=False).tolist()

        for client_id in selected_clients:
            is_free_rider = (client_id in free_rider_ids)
            # Check train-then-attack: if free rider and round >= attack_round, apply attack
            if is_free_rider and free_rider_strategy == 'train_then_attack' and round_num >= attack_round:
                # Return global model without training
                state = copy.deepcopy(server.global_model.state_dict())
            else:
                if is_free_rider and free_rider_strategy == 'gaussian_noise':
                    # Apply noise directly
                    state = apply_free_rider_strategy(
                        'gaussian_noise', server.global_model, round_num,
                        noise_scale=noise_scale
                    )
                elif is_free_rider and free_rider_strategy == 'previous_models':
                    # Use difference (simplified: use global model as is)
                    state = apply_free_rider_strategy(
                        'previous_models', server.global_model, round_num,
                        state_dict_prev=prev_global_state
                    )
                else:
                    # Normal client training
                    client = clients[client_id]
                    # Pass previous global model for memory-enhanced update
                    prev_model_state = prev_global_state if config['memory_mu'] > 0 else None
                    # For memory term, we need the previous global model parameters
                    prev_global_for_prox = None
                    if prev_global_state is not None and config['memory_mu'] > 0:
                        # Create a temporary model with previous global state
                        prev_model = copy.deepcopy(server.global_model)
                        prev_model.load_state_dict(prev_global_state)
                        prev_global_for_prox = prev_model
                    state = client.local_train(
                        server.global_model,
                        config['local_epochs'],
                        previous_global_model=prev_global_for_prox
                    )
            client_states.append(state)
            client_ids.append(client_id)

        # Aggregate
        sizes = [client_sizes[ci] for ci in client_ids]
        server.aggregate(client_states, sizes)
        prev_global_state = copy.deepcopy(server.global_model.state_dict())

        # Evaluate global model accuracy
        global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = server.global_model(data)
                _, pred = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (pred == targets).sum().item()
        acc = correct / total
        round_accuracies.append(acc)
        logging.info(f"Round {round_num} Global accuracy: {acc:.4f}")

        # Detect free riders (using test_loader for trigger samples)
        # We need to pass a data loader that yields trigger samples. We'll use test_loader.
        free_riders = server.detect_free_riders(client_states, client_ids, test_loader)
        # Log detection
        detected = [cid for cid, flag in free_riders.items() if flag]
        logging.info(f"Detected free riders: {detected}")

        # Compute detection rate (among actual free riders)
        if free_rider_ids:
            detected_free = [cid for cid in detected if cid in free_rider_ids]
            detection_rate = len(detected_free) / len(free_rider_ids)
        else:
            detection_rate = 0.0
        free_rider_detection_rates.append(detection_rate)
        logging.info(f"Detection rate: {detection_rate:.4f}")

    # Final results
    final_acc = round_accuracies[-1]
    avg_detection_rate = np.mean(free_rider_detection_rates[-10:]) if len(free_rider_detection_rates) >= 10 else np.mean(free_rider_detection_rates)

    # Save metrics
    results = {
        'final_accuracy': final_acc,
        'detection_rate': avg_detection_rate,
        'round_accuracies': round_accuracies,
        'detection_rates': free_rider_detection_rates,
    }
    # Save to file
    np.savez(os.path.join(log_dir, 'results.npz'), **results)

    logging.info(f"Experiment finished. Final acc: {final_acc:.4f}, Detection rate: {avg_detection_rate:.4f}")
    return results