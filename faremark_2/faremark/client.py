import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

class Client:
    def __init__(self, client_id, model, train_loader, watermark_manager, device='cpu',
                 lr=0.01, watermark_lambda=0.1, memory_mu=0.01, alpha=0.5):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.watermark_manager = watermark_manager
        self.device = device
        self.lr = lr
        self.watermark_lambda = watermark_lambda
        self.memory_mu = memory_mu
        self.alpha = alpha
        self.trigger_class, self.watermark_bit, self.M = watermark_manager.get_watermark_info(client_id)
        self.optimizer = None

    def local_train(self, global_model, num_epochs, previous_global_model=None):
        """
        Perform local training with watermark loss and memory-enhanced update.
        previous_global_model: model from previous round (for proximal term)
        """
        self.model.load_state_dict(global_model.state_dict())
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        # Convert previous global model parameters to tensor for proximal term
        if previous_global_model is not None:
            prev_params = [p.detach().clone() for p in previous_global_model.parameters()]
        else:
            prev_params = None

        self.model.train()
        for epoch in range(num_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                # Classification loss (cross-entropy) on all samples
                loss_ce = nn.CrossEntropyLoss()(outputs, targets)

                # Watermark loss: apply only to trigger class samples
                trigger_mask = (targets == self.trigger_class)
                if trigger_mask.any():
                    trigger_outputs = outputs[trigger_mask]
                    loss_wm = self.watermark_manager.compute_watermark_loss(
                        trigger_outputs, self.client_id, transform='power'
                    )
                else:
                    loss_wm = torch.tensor(0.0, device=self.device)

                # Memory-enhanced proximal term
                loss_prox = 0.0
                if prev_params is not None and self.memory_mu > 0:
                    for param, prev_param in zip(self.model.parameters(), prev_params):
                        loss_prox += torch.norm(param - prev_param, p=2) ** 2
                    loss_prox = (self.memory_mu / 2) * loss_prox

                loss = loss_ce + self.watermark_lambda * loss_wm + loss_prox
                loss.backward()
                self.optimizer.step()

        # Return updated model parameters (as state dict)
        return self.model.state_dict()

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = correct / total
        return acc