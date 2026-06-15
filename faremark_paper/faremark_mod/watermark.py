"""
FareMark: Watermark representation and extraction.

Implements the box-free watermarking scheme from Section IV-A of the paper.
Each client's watermark is encoded in the model's softmax output using a
pseudorandom projection matrix M and a smoothing function f(x).

Key equations:
    z_k = sum_j f(p_j^k) * M_{i,k,j}          (Eq. 1 / 13)
    b_k = delta(z_k)                             (Eq. 2)
    L_wm = BCE(z, b)                             (Eq. 12)

Three smoothing functions (Eq. 7-9):
    f(x) = x^alpha,  alpha < 0
    f(x) = x^alpha,  0 < alpha < 1
    f(x) = sin(alpha * x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Smoothing functions
# ---------------------------------------------------------------------------

def smooth_neg_power(logits: torch.Tensor, alpha: float = -1.0) -> torch.Tensor:
    """f(x) = x^alpha, alpha < 0.  Clamps to avoid division by zero."""
    assert alpha < 0, "alpha must be negative for this branch"
    probs = F.softmax(logits, dim=-1)
    probs = probs.clamp(min=1e-6)
    return probs ** alpha


def smooth_frac_power(logits: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """f(x) = x^alpha, 0 < alpha < 1."""
    assert 0 < alpha < 1, "alpha must be in (0,1) for this branch"
    probs = F.softmax(logits, dim=-1)
    return probs ** alpha


def smooth_sin(logits: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """f(x) = sin(alpha * x)."""
    probs = F.softmax(logits, dim=-1)
    return torch.sin(alpha * probs)


SMOOTH_FNS = {
    "neg_power": smooth_neg_power,
    "frac_power": smooth_frac_power,
    "sin": smooth_sin,
}


# ---------------------------------------------------------------------------
# Watermark key generation
# ---------------------------------------------------------------------------

class WatermarkKey:
    """
    Holds the secret key M (pseudorandom projection matrix) and the
    binary watermark message B for one client.

    Args:
        num_classes (int): Number of output classes n.
        wm_bits (int): Watermark bit-length m.  Must satisfy m <= n.
        client_id (int): Used to seed the RNG so each client gets a unique M.
        device (str | torch.device): Where to place tensors.
    """

    def __init__(
        self,
        num_classes: int,
        wm_bits: int,
        client_id: int,
        device: torch.device = torch.device("cpu"),
    ):
        assert wm_bits <= num_classes, "wm_bits must be <= num_classes"
        self.num_classes = num_classes
        self.wm_bits = wm_bits
        self.client_id = client_id
        self.device = device

        # l = n // m  (dimension of each group P^k)
        self.group_size = num_classes // wm_bits  # l

        rng = np.random.RandomState(seed=client_id + 42)

        # M: shape (wm_bits, group_size) — values in {-1, +1}
        M_np = rng.choice([-1.0, 1.0], size=(wm_bits, self.group_size))
        self.M = torch.tensor(M_np, dtype=torch.float32, device=device)

        # Random binary watermark message B: shape (wm_bits,)
        B_np = rng.randint(0, 2, size=(wm_bits,)).astype(np.float32)
        self.B = torch.tensor(B_np, dtype=torch.float32, device=device)

    def to(self, device):
        self.device = device
        self.M = self.M.to(device)
        self.B = self.B.to(device)
        return self


# ---------------------------------------------------------------------------
# Core projection: logits → z values
# ---------------------------------------------------------------------------

def project_logits(
    logits: torch.Tensor,
    key: WatermarkKey,
    smooth_fn: str = "frac_power",
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Compute z_k = sum_j f(p_j^k) * M_{k,j}  for k=1..m.

    Args:
        logits: shape (batch, num_classes)
        key: WatermarkKey for the client
        smooth_fn: one of 'neg_power', 'frac_power', 'sin'
        alpha: smoothing parameter

    Returns:
        z: shape (batch, wm_bits)
    """
    fn = SMOOTH_FNS[smooth_fn]
    f_probs = fn(logits, alpha)  # (batch, num_classes)

    m = key.wm_bits
    l = key.group_size
    used = m * l  # only use first m*l classes

    f_used = f_probs[:, :used]                    # (batch, m*l)
    f_grouped = f_used.view(-1, m, l)             # (batch, m, l)

    # M: (m, l) → broadcast over batch
    z = (f_grouped * key.M.unsqueeze(0)).sum(dim=-1)  # (batch, m)
    return z


# ---------------------------------------------------------------------------
# Watermark extraction (inference — no gradients needed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_watermark(
    logits: torch.Tensor,
    key: WatermarkKey,
    smooth_fn: str = "frac_power",
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Extract watermark bits from a batch of trigger-class logits.

    Averages z over all trigger samples then thresholds at 0 (Eq. 15).

    Args:
        logits: shape (N_T, num_classes) — outputs on trigger samples
        key: WatermarkKey
        smooth_fn / alpha: smoothing params (must match training)

    Returns:
        b_hat: shape (wm_bits,) — recovered binary watermark
    """
    z = project_logits(logits, key, smooth_fn, alpha)  # (N_T, m)
    z_mean = z.mean(dim=0)                              # (m,)
    b_hat = (z_mean >= 0).float()
    return b_hat


# ---------------------------------------------------------------------------
# Watermarking loss  (Eq. 12)
# ---------------------------------------------------------------------------

def watermark_loss(
    logits: torch.Tensor,
    key: WatermarkKey,
    smooth_fn: str = "frac_power",
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Binary-cross-entropy loss that drives z_k toward key.B.

    L_wm = sum_k [ b_k * log sigmoid(z_k) + (1-b_k) * log(1 - sigmoid(z_k)) ]

    Args:
        logits: shape (batch, num_classes)
        key: WatermarkKey
        smooth_fn / alpha: smoothing params

    Returns:
        scalar loss (averaged over batch and bits)
    """
    z = project_logits(logits, key, smooth_fn, alpha)  # (batch, m)
    # BCEWithLogitsLoss expects targets in [0,1]
    target = key.B.unsqueeze(0).expand_as(z)           # (batch, m)
    return F.binary_cross_entropy_with_logits(z, target)


# ---------------------------------------------------------------------------
# Similarity / detection metric  (Eq. 16)
# ---------------------------------------------------------------------------

def watermark_similarity(b_hat: torch.Tensor, b_true: torch.Tensor) -> float:
    """
    Returns (1/m) * sum |b_hat_k - b_k|.
    Lower = better match.  Threshold eta typically set to mu + 3*sigma.
    """
    assert b_hat.shape == b_true.shape
    return (b_hat - b_true).abs().mean().item()


def bit_accuracy(b_hat: torch.Tensor, b_true: torch.Tensor) -> float:
    """Returns fraction of bits correctly recovered."""
    return (b_hat == b_true).float().mean().item()