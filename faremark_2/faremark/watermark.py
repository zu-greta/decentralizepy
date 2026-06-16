import torch
import torch.nn.functional as F
import numpy as np

class WatermarkManager:
    def __init__(self, num_clients, num_classes, trigger_sample_count, alpha=0.5, device='cpu'):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.trigger_sample_count = trigger_sample_count
        self.alpha = alpha
        self.device = device

        # Assign each client a unique trigger class (modulo num_classes)
        self.trigger_classes = [i % num_classes for i in range(num_clients)]
        # For each client, generate a random bit and a random projection vector M (length num_classes)
        self.watermark_bits = torch.randint(0, 2, (num_clients,), device=device)
        self.M = torch.randn(num_clients, num_classes, device=device)  # each row is M_i
        # Normalize M to unit norm (optional)
        self.M = self.M / torch.norm(self.M, dim=1, keepdim=True)

    def get_watermark_info(self, client_id):
        return self.trigger_classes[client_id], self.watermark_bits[client_id], self.M[client_id]

    def compute_watermark_loss(self, logits, client_id, transform='power'):
        """
        Compute the watermark loss for a batch of logits (from trigger-class samples).
        logits: [B, num_classes]
        """
        softmax = F.softmax(logits, dim=1)  # [B, C]
        # Apply smoothing function
        if transform == 'power':
            # f(x) = x^alpha, alpha < 1 to flatten
            softmax = torch.pow(softmax, self.alpha)
        elif transform == 'sin':
            softmax = torch.sin(softmax * self.alpha)
        else:
            raise ValueError("Transform must be 'power' or 'sin'")

        bit = self.watermark_bits[client_id]
        M_i = self.M[client_id]  # [C]
        # Compute z = sum_j f(p_j) * M_i[j]
        z = torch.matmul(softmax, M_i)  # [B]
        # We want sign(z) == (2*bit - 1) i.e. +1 for bit=1, -1 for bit=0
        target = 2 * bit - 1  # +1 or -1
        # Hinge loss: max(0, margin - target * z)
        margin = 1.0
        loss = torch.mean(torch.relu(margin - target * z))
        return loss

    def extract_watermark(self, logits_list, client_id, transform='power'):
        """
        Extract the watermark bit from a list of logits (multiple trigger samples).
        logits_list: list of [B, C] tensors (or single tensor)
        Returns: extracted bit (0 or 1)
        """
        softmax_list = []
        for logits in logits_list:
            softmax = F.softmax(logits, dim=1)
            if transform == 'power':
                softmax = torch.pow(softmax, self.alpha)
            elif transform == 'sin':
                softmax = torch.sin(softmax * self.alpha)
            softmax_list.append(softmax)
        all_softmax = torch.cat(softmax_list, dim=0)  # [total_samples, C]
        M_i = self.M[client_id]  # [C]
        z = torch.matmul(all_softmax, M_i)  # [total_samples]
        avg_z = torch.mean(z)
        bit = 1 if avg_z >= 0 else 0
        return bit, avg_z.item()