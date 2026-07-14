"""Models

FareMark: AlexNet, ShuffleNet, ResNet-18 and GoogleNet on MNIST / CIFAR-10 / CIFAR-100. 
Implemented: ResNet-18 and AlexNet + tiny SmallCNN for fast smoke tests 
ShuffleNet / GoogleNet are to be added later via `build_model`

NOTE: Both nets are adapted for small images (28x28 / 32x32). The stock torchvision
ResNet-18 is built for 224x224 ImageNet inputs; on CIFAR you must shrink the
stem (3x3 stride-1 conv, drop the max-pool) or the feature maps collapse and
accuracy stalls in the 80s. This adaptation is the standard "CIFAR ResNet".
"""
import torch.nn as nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        net = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        # CIFAR/MNIST stem adaptation.
        net.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        net.maxpool = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


class AlexNetSmall(nn.Module):
    """A compact AlexNet-style CNN that works for both 28x28 and 32x32 inputs."""

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        # Adaptive pool makes the classifier input size independent of 28 vs 32.
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 2 * 2, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


class SmallCNN(nn.Module):
    """Tiny net for fast pipeline smoke tests (a few rounds, high MNIST acc)."""

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


def build_model(name: str, num_classes: int, in_channels: int) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return ResNet18(num_classes, in_channels)
    if name == "alexnet":
        return AlexNetSmall(num_classes, in_channels)
    if name == "smallcnn":
        return SmallCNN(num_classes, in_channels)
    raise ValueError(f"Unknown model '{name}'. Add it to build_model().")
