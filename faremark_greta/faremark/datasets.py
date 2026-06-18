"""Dataset loading and the IID partition across clients.

The paper divides the training set *evenly* among clients (IID). We implement
that here. A `partition` field is left in the signature so a later stage can add
non-IID (Dirichlet) splits without changing call sites.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_NORM = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


@dataclass
class DataBundle:
    client_loaders: list          # one DataLoader per client (train)
    test_loader: DataLoader       # global test set
    num_classes: int
    in_channels: int
    test_dataset: object = None   # raw test set (for Stage-3 trigger sampling)


def _build_transforms(name: str, train: bool):
    mean, std = _NORM[name]
    tfms = []
    if name in ("cifar10", "cifar100") and train:
        tfms += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    tfms += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(tfms)


def _load_raw(name: str, data_root: str):
    name = name.lower()
    if name == "mnist":
        train = datasets.MNIST(data_root, train=True, download=True,
                               transform=_build_transforms(name, True))
        test = datasets.MNIST(data_root, train=False, download=True,
                              transform=_build_transforms(name, False))
        return train, test, 10, 1
    if name == "cifar10":
        train = datasets.CIFAR10(data_root, train=True, download=True,
                                 transform=_build_transforms(name, True))
        test = datasets.CIFAR10(data_root, train=False, download=True,
                                transform=_build_transforms(name, False))
        return train, test, 10, 3
    if name == "cifar100":
        train = datasets.CIFAR100(data_root, train=True, download=True,
                                  transform=_build_transforms(name, True))
        test = datasets.CIFAR100(data_root, train=False, download=True,
                                 transform=_build_transforms(name, False))
        return train, test, 100, 3
    raise ValueError(f"Unknown dataset '{name}'.")


def iid_partition(num_samples: int, num_clients: int, seed: int) -> list:
    """Shuffle all indices and split into `num_clients` near-equal shards."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_samples)
    return [list(shard) for shard in np.array_split(idx, num_clients)]


def build_data(name: str, data_root: str, num_clients: int, batch_size: int,
               seed: int, num_workers: int = 2, partition: str = "iid") -> DataBundle:
    train, test, num_classes, in_channels = _load_raw(name, data_root)

    if partition != "iid":
        raise NotImplementedError("Only IID is implemented in Stage 1.")
    shards = iid_partition(len(train), num_clients, seed)

    client_loaders = [
        DataLoader(Subset(train, shard), batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=False,
                   generator=torch.Generator().manual_seed(seed + cid))
        for cid, shard in enumerate(shards)
    ]
    test_loader = DataLoader(test, batch_size=256, shuffle=False,
                             num_workers=num_workers)
    return DataBundle(client_loaders, test_loader, num_classes, in_channels,
                      test_dataset=test)
