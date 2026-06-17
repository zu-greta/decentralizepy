"""Small helpers shared across the simulator."""
import logging
import os
import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG we touch so a (config, repeat) pair is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN determinism trades a little speed for reproducibility. Keep it on
    # while we are validating correctness; you can flip it off for big sweeps.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "faremark", logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on re-entry
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@torch.no_grad()
def evaluate_accuracy(model, loader, device) -> float:
    """Top-1 accuracy (%) over a data loader."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)
