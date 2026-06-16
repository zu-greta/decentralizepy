import random
import numpy as np
import torch
import yaml
import os
import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(experiment_name):
    log_dir = os.path.join("logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_dir

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config