import torch
import torch.nn.utils.prune as prune
import numpy as np
from opacus import PrivacyEngine

def fine_tune(model, train_loader, test_loader, epochs, lr=0.001, device='cpu'):
    """Fine-tune model on original task without watermark loss."""
    model = copy.deepcopy(model)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
    acc = correct / total
    return model, acc

def prune_model(model, amount=0.3):
    """Apply global magnitude pruning to model."""
    model = copy.deepcopy(model)
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    # Remove pruning masks (make permanent)
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    return model

def quantize_model(model, bits=8):
    """Quantize model weights to lower precision (simulated)."""
    model = copy.deepcopy(model)
    # For demonstration, we use dynamic quantization (torch.quantization.quantize_dynamic)
    # But that only works for some layers. We'll simulate by reducing precision via clamping.
    # Simpler: we just convert to half precision (FP16) and back to simulate quantization.
    # For actual quantization, we could use torch.quantization.quantize_dynamic.
    try:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except:
        # Fallback: just return model
        pass
    return model

def apply_differential_privacy(model, train_loader, epochs, lr, device, noise_multiplier=0.1, max_grad_norm=1.0):
    """Train model with DP-SGD using Opacus."""
    model = copy.deepcopy(model)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model