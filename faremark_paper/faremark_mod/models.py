"""
FareMark: Model definitions.

The paper uses AlexNet, ShuffleNet, ResNet-18, and GoogleNet — all
available in torchvision.models.  We wrap them to accept a configurable
num_classes argument so they work on MNIST (10), CIFAR-10 (10),
CIFAR-100 (100), and Food-101 (101 → treated as 100 in paper).
"""

import torch.nn as nn
import torchvision.models as tvm


def _replace_head(model: nn.Module, num_classes: int, model_name: str) -> nn.Module:
    """Replace the final classification layer for the given num_classes."""
    if model_name == "alexnet":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "shufflenet":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "googlenet":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # Also fix auxiliary classifiers if present
        if model.aux_logits:
            model.aux1.fc2 = nn.Linear(1024, num_classes)
            model.aux2.fc2 = nn.Linear(1024, num_classes)

    return model


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    Build a model by name.

    Args:
        model_name: One of 'alexnet', 'resnet18', 'shufflenet', 'googlenet'.
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet weights (False for paper's
                    from-scratch training on CIFAR/MNIST).

    Returns:
        nn.Module with the correct output dimension.
    """
    weights = None  # torchvision v0.13+ API

    if model_name == "alexnet":
        model = tvm.alexnet(weights=weights)
        # Adapt first conv for small images (MNIST/CIFAR are 32x32 or 28x28)
        model.features[0] = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1
        )
        model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        model.classifier[1] = nn.Linear(64 * 2 * 2, 4096)

    elif model_name == "resnet18":
        model = tvm.resnet18(weights=weights)
        # Adapt for 32x32 inputs
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    elif model_name == "shufflenet":
        model = tvm.shufflenet_v2_x1_0(weights=weights)

    elif model_name == "googlenet":
        model = tvm.googlenet(weights=weights, aux_logits=True, init_weights=True)
        # Adapt for 32x32 inputs
        model.conv1 = tvm.googlenet.BasicConv2d(3, 64, kernel_size=3, stride=1, padding=1)
        model.maxpool1 = nn.Identity()
        model.maxpool2 = nn.Identity()

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose from: alexnet, resnet18, shufflenet, googlenet"
        )

    model = _replace_head(model, num_classes, model_name)
    return model


# Adapter for 1-channel (MNIST) inputs
class GrayscaleToRGB(nn.Module):
    """Expand 1-channel inputs to 3 channels by repeating."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.base(x)


def build_model_for_dataset(
    model_name: str, dataset_name: str
) -> nn.Module:
    """
    Convenience wrapper: pick num_classes from dataset name and wrap
    for grayscale if needed.
    """
    num_classes_map = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "food100": 100,   # Food-101 truncated to 100 classes as in paper
    }
    num_classes = num_classes_map[dataset_name.lower()]
    model = build_model(model_name, num_classes)
    if dataset_name.lower() == "mnist":
        model = GrayscaleToRGB(model)
    return model