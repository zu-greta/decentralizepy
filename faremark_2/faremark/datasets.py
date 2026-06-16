import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataset(dataset_name, data_dir="./data"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # For MNIST (simple)
    transform_mnist_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_mnist_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset_name.lower() == "cifar10":
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
    elif dataset_name.lower() == "mnist":
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_mnist_train)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_mnist_test)
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return train_set, test_set, num_classes

def partition_data(train_set, num_clients, iid=True):
    """
    Partition training data among clients.
    Returns list of subsets (or indices) for each client.
    """
    targets = np.array(train_set.targets) if hasattr(train_set, 'targets') else np.array(train_set.labels)
    indices = np.arange(len(train_set))
    if iid:
        # Shuffle and split evenly
        shuffled = np.random.permutation(indices)
        chunks = np.array_split(shuffled, num_clients)
        client_indices = [chunk.tolist() for chunk in chunks]
    else:
        # Non-IID: use Dirichlet (optional)
        # Simple label-based partitioning for demonstration
        label_indices = [np.where(targets == c)[0] for c in range(max(targets)+1)]
        client_indices = [[] for _ in range(num_clients)]
        # Assign each label to clients in round-robin
        for c, idxs in enumerate(label_indices):
            np.random.shuffle(idxs)
            splits = np.array_split(idxs, num_clients)
            for i, split in enumerate(splits):
                client_indices[i].extend(split.tolist())
    return client_indices

def get_dataloaders(client_indices, train_set, test_set, batch_size):
    # Create data loaders for each client
    client_loaders = []
    for idxs in client_indices:
        subset = Subset(train_set, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
        client_loaders.append(loader)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return client_loaders, test_loader