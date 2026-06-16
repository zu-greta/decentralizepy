import torch
import torch.nn as nn
import torchvision.models as tv_models

def get_model(model_name, num_classes, **kwargs):
    if model_name == "resnet18":
        model = tv_models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "alexnet":
        model = tv_models.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "shufflenet":
        model = tv_models.shufflenet_v2_x1_0(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "googlenet":
        model = tv_models.googlenet(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model