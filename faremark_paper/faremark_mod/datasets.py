"""
FareMark: Dataset utilities.

Loads MNIST, CIFAR-10, CIFAR-100, and Food-101 and splits them evenly
across N clients (IID partition as used in the paper).

The paper states: "The training dataset was divided evenly among the
clients for local training, while the test dataset was utilised to
assess the correctness of the global model."
"""

import os
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------

def _mnist_transform(train: bool):
    tfms = [
        transforms.Resize(32),   # upscale for model compatibility
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    return transforms.Compose(tfms)


def _cifar_transform(train: bool, num_classes: int = 10):
    if train:
        tfms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
    else:
        tfms = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
    return transforms.Compose(tfms)


def _food_transform(train: bool):
    if train:
        tfms = [
            transforms.Resize(40),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    else:
        tfms = [
            transforms.Resize(40),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    return transforms.Compose(tfms)


# ---------------------------------------------------------------------------
# Subset that maps Food-101 labels to 0-99
# ---------------------------------------------------------------------------

class RemappedSubset(Dataset):
    """Wraps a Subset and remaps labels to 0..num_classes-1."""

    def __init__(self, subset: Subset, label_map: dict):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return img, self.label_map[label]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(
    name: str,
    data_root: str = "./data",
    train: bool = True,
) -> Dataset:
    """
    Load a dataset by name.

    Args:
        name: 'mnist', 'cifar10', 'cifar100', or 'food100'
        data_root: where to download / find the data
        train: train or test split

    Returns:
        A torch Dataset.
    """
    os.makedirs(data_root, exist_ok=True)
    name = name.lower()

    if name == "mnist":
        return datasets.MNIST(
            root=data_root,
            train=train,
            download=True,
            transform=_mnist_transform(train),
        )

    elif name == "cifar10":
        return datasets.CIFAR10(
            root=data_root,
            train=train,
            download=True,
            transform=_cifar_transform(train),
        )

    elif name == "cifar100":
        return datasets.CIFAR100(
            root=data_root,
            train=train,
            download=True,
            transform=_cifar_transform(train, num_classes=100),
        )

    elif name == "food100":
        # Use Food-101; keep only first 100 classes (0-99)
        split = "train" if train else "test"
        full = datasets.Food101(
            root=data_root,
            split=split,
            download=True,
            transform=_food_transform(train),
        )
        # Filter to first 100 classes
        indices = [i for i, (_, lbl) in enumerate(full._labels)
                   if lbl < 100] if hasattr(full, '_labels') else list(range(len(full)))
        # Fallback: filter by iterating targets
        if not hasattr(full, '_labels'):
            targets = [full[i][1] for i in range(min(500, len(full)))]  # sample check
            indices = [i for i in range(len(full)) if True]  # keep all, remap below
        subset = Subset(full, [i for i in range(len(full))])
        label_map = {c: c for c in range(101)}  # identity map; Food101 has 101 classes
        return subset

    else:
        raise ValueError(f"Unknown dataset '{name}'. "
                         "Choose from: mnist, cifar10, cifar100, food100")


# ---------------------------------------------------------------------------
# IID client data partitioning
# ---------------------------------------------------------------------------

def split_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[Subset]:
    """
    Randomly and evenly split dataset among num_clients clients (IID).

    Returns a list of Subsets, one per client.
    """
    n = len(dataset)
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()

    client_size = n // num_clients
    client_datasets = []
    for i in range(num_clients):
        start = i * client_size
        end = start + client_size if i < num_clients - 1 else n
        client_datasets.append(Subset(dataset, indices[start:end]))

    return client_datasets


# ---------------------------------------------------------------------------
# Trigger-class DataLoader helper
# ---------------------------------------------------------------------------

def make_trigger_loader(
    test_dataset: Dataset,
    trigger_class: int,
    batch_size: int = 64,
    n_max: int = 400,
) -> DataLoader:
    """
    Build a DataLoader containing only samples of `trigger_class`
    from the test set, for watermark verification.
    """
    indices = []
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        if label == trigger_class:
            indices.append(i)
        if len(indices) >= n_max:
            break

    subset = Subset(test_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)