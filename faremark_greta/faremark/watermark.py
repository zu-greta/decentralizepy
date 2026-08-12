"""Watermarking: box-free, output-space watermarking scheme 

The watermark of client i is an m-bit string B^i embedded into the model's
softmax output on inputs of that client's trigger class. Nothing is read from
the weights, verification only needs model outputs ("box-free")

Pipeline (equation numbers refer to the paper):
  1. Split the n-dim softmax P into m groups of size l = n // m.            (section IV-A)
  2. Smooth each probability with f() so the argmax doesn't dominate.    (Eq. 7-9)
  3. Project each group onto a per-client pseudo-random +/-1 key row M.  (Eq. 1/13)
        z_k = sum_j f(p_{k,j}) * M_{k,j}
  4. Bit k is sign(z_k): >=0 -> 1, <0 -> 0.                                 (Eq. 2)
  5. Embed by adding a BCE term that drives sign(z_k) -> target bit b_k. (Eq. 11-12)
  6. Extract by averaging z over N_T trigger samples, then sign.          (Eq. 15)
  7. Detect via bit-error-rate vs the registered bits, threshold eta.     (Eq. 16)

Smoothing: cross-entropy makes the softmax steep (one class ~1, the rest ~0), 
so the projection would be decided by the argmax alone and the
watermark couldn't be shaped without hurting accuracy. f(x)=x^a with 0<a<1
amplifies the small tail probabilities so they can carry the bits while the
argmax (the true class) is preserved. (section IV-A, Fig. 6.)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ============================================================================
# FAREMARK PAPER MAPPING 
#   Eq. 1/13  z_k = sum_j f(p_{k,j}) * M_{i,k,j}        -> project_logits()
#   Eq. 2     b_k = 1 if z_k >= 0 else 0                -> extract_bits() / sign
#   Eq. 4-6   within-group anti-dominance (p_max<=0.5)  -> dominance_ratio()
#   Eq. 7-9   smoothing f(): x^a (a<0 or 0<a<1), sin    -> smooth()
#   Eq. 10    f(max)/sum f < 0.5 constraint             -> dominance_ratio()
#   Eq. 11-12 L = L_cl + lambda*L_wm, L_wm = BCE        -> watermark_loss()
#   Eq. 14    memory-enhanced update                    -> wm_client._memory_update
#   Eq. 15    avg over N_T trigger samples, then sign   -> extract_bits()
#   Eq. 16    BER < eta ; eta = mu + 3*sigma            -> bit_error_rate()/detected()
#   Grouping  "the (l*(k-1)+j)-th element" => consecutive blocks of size l = n//m,
#             using the first m*l softmax outputs (paper: only {p_1..p_{m*l}}).
# ============================================================================


# ----------------------------------------------------------------------------
# Smoothing function f()  (Eq. 7-9)
# ----------------------------------------------------------------------------
import math as _math
import os as _os

# --- smoothing epsilon -------------------------------------------------------
SMOOTH_EPS = float(_os.environ.get("SMOOTH_EPS", "1e-3"))

SIN_ALPHA_MAX = _math.pi / 2      # sin(a*x) is monotone on x in [0,1] only for a <= pi/2


def smoothing_gain(kind: str, alpha: float, lo: float = 0.05, hi: float = 0.70,
                   eps: float = 1e-8) -> float:
    """How much f() amplifies a tail probability relative to a peak one.

    gain = [f(lo)/f(hi)] / (lo/hi).  gain=1 means f does nothing and the projection 
    stays dominated by p_max:

        power alpha=0.4   -> 4.91   (working default)
        sin   alpha=0.4   -> 1.01   <-- the bug: no smoothing whatsoever
        sin   alpha=pi/2  -> 1.23   (best sin can do while staying monotone)
    """
    f = (lambda x: (x + eps) ** alpha) if kind == "power" else (lambda x: _math.sin(alpha * x))
    return (f(lo) / f(hi)) / (lo / hi)


def smooth(p: torch.Tensor, kind: str = "power", alpha: float = 0.4,
           eps: float | None = None) -> torch.Tensor:
    """f(p) applied elementwise to probabilities p >= 0.

    kind="power", 0<alpha<1  -> Eq. 8  (default; amplifies small probabilities)
    kind="power", alpha<0    -> Eq. 7
    kind="sin"               -> Eq. 9  f(x)=sin(alpha*x)
    Smaller alpha => more smoothing (flatter distribution)
    """
    if eps is None:
        eps = SMOOTH_EPS
    if kind == "power":
        return (p.clamp(min=0.0) + eps) ** alpha
    if kind == "sin":
        if not (0.0 < alpha <= SIN_ALPHA_MAX + 1e-9):
            raise ValueError(
                f"wm_f='sin' needs 0 < alpha <= pi/2 ({SIN_ALPHA_MAX:.4f}); got {alpha}. ")
        if smoothing_gain("sin", alpha) < 1.10:
            raise ValueError(
                f"wm_f='sin' with alpha={alpha} gives smoothing gain "
                f"{smoothing_gain('sin', alpha):.2f} (1.0 = no smoothing at all). ")
        return torch.sin(alpha * p)
    raise ValueError(f"unknown smoothing kind '{kind}'")


# ----------------------------------------------------------------------------
# Per-client secret key M  (the +/-1 projection matrix, section IV-A)
# ----------------------------------------------------------------------------
def make_key(num_bits: int, group_size: int, seed: int,
             balanced: bool = False) -> torch.Tensor:
    """Per-client secret projection matrix M, shape [m, l], entries +/-1.

    balanced=True: each row is sign-balanced (equal +1/-1, shuffled)
    Balanced rows make z_k = sum_j f(p_j) M_{k,j} shapeable to either sign

    balanced=False (default, matches config.wm_balanced_keys=False): paper-exact
    pseudo-random +/-1 entries (the paper's M is drawn at random, Fig. 5). 
    """
    g = torch.Generator().manual_seed(seed)
    if not balanced:
        return (torch.randint(0, 2, (num_bits, group_size), generator=g)
                .float() * 2 - 1)                       # +/-1, fully random
    half = group_size // 2
    base = torch.tensor([1.0] * half + [-1.0] * (group_size - half))
    rows = [base[torch.randperm(group_size, generator=g)] for _ in range(num_bits)]
    return torch.stack(rows)


def unembeddable_fraction(key: torch.Tensor) -> float:
    """Fraction of key rows that are same-sign (all +1 or all -1).

    A same-sign row forces z_k = sum_j f(p_j) M_{k,j} to a fixed sign for every
    input (because f(p) >= 0), so that bit cannot be embedded 
    """
    same = ((key > 0).all(dim=1) | (key < 0).all(dim=1)).float().mean().item()
    return same


def make_bits(num_bits: int, seed: int, balanced: bool = True) -> torch.Tensor:
    """Target watermark B^i in {0,1}^m.

    balanced=True (default): equal 0s and 1s, shuffled 
    balanced=False for the paper-exact uniform draw
    """
    g = torch.Generator().manual_seed(seed + 7919)
    if not balanced or num_bits < 4:
        return torch.randint(0, 2, (num_bits,), generator=g).long()
    half = num_bits // 2
    base = torch.tensor([1] * half + [0] * (num_bits - half))
    return base[torch.randperm(num_bits, generator=g)].long()


def grouping(num_classes: int, num_bits: int) -> int:
    """l = n // m, the size of each softmax group. Requires l >= 1."""
    l = num_classes // num_bits
    if l < 1:
        raise ValueError(f"num_bits={num_bits} too large for n={num_classes} "
                         f"(need num_bits <= num_classes).")
    return l


# ----------------------------------------------------------------------------
# Projection: probabilities -> per-bit logits z  (Eq. 1/13)
# ----------------------------------------------------------------------------
def project_logits(probs: torch.Tensor, key: torch.Tensor,
                   kind: str = "power", alpha: float = 0.4,
                   exclude: int | None = None) -> torch.Tensor:
    """probs [B, n] -> z [B, m].  z_k = sum_j f(p_{k,j}) * M_{k,j}  (Eq. 13).

    If `exclude` is given (the client's trigger class), that column is dropped
    """
    if exclude is not None:
        keep = [c for c in range(probs.shape[1]) if c != exclude]
        probs = probs[:, keep]
    m, l = key.shape
    # m = key.shape[0]                     # number of bits
    # C_rem = probs.shape[1]               # classes left after exclusion
    # l = C_rem // m                       # new group size
    used = m * l
    p = probs[:, :used].reshape(probs.shape[0], m, l)   # [B, m, l]
    fp = smooth(p, kind, alpha)                          # [B, m, l]
    z = (fp * key.unsqueeze(0)).sum(dim=2)               # [B, m]
    return z


# ----------------------------------------------------------------------------
# Embedding loss  (Eq. 11-12):  L_wm = BCE(sign-logit z, target bits)
# ----------------------------------------------------------------------------
def watermark_loss(probs: torch.Tensor, key: torch.Tensor,
                   target_bits: torch.Tensor, kind: str = "power",
                   alpha: float = 0.4, exclude: int | None = None) -> torch.Tensor:
    """Per-sample BCE driving sign(z_k) -> b_k. Minimizing it embeds B^i."""
    z = project_logits(probs, key, kind, alpha, exclude)   # [B, m]
    t = target_bits.to(z.device).float().unsqueeze(0).expand_as(z)
    return F.binary_cross_entropy_with_logits(z, t)


# ----------------------------------------------------------------------------
# Extraction (Eq. 15) and detection (Eq. 16)
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_bits(probs: torch.Tensor, key: torch.Tensor, kind: str = "power",
                 alpha: float = 0.4, exclude: int | None = None) -> torch.Tensor:
    """Average z over the N_T trigger samples, then take the sign (Eq. 15)."""
    z = project_logits(probs, key, kind, alpha, exclude)  # [N_T, m]
    zbar = z.mean(dim=0)                                  # [m]
    return (zbar >= 0).long()                             # [m] bits


def bit_error_rate(bits: torch.Tensor, target: torch.Tensor) -> float:
    """(1/m) sum |b_hat_k - b_k|  (Eq. 16, left-hand side)."""
    return (bits.cpu() != target.cpu()).float().mean().item()


def detected(ber: float, eta: float) -> bool:
    """Watermark considered present (benign client) iff BER < eta (Eq. 16).
    """
    return ber < eta


def calibrate_eta(benign_bers, floor: float = 0.05) -> float:
    """Paper's detection threshold (Eq. 16): eta = mu + 3*sigma of the benign BER"""
    import statistics
    vals = [b for b in benign_bers if b is not None]
    if not vals:
        return floor
    mu = statistics.mean(vals)
    sigma = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return max(mu + 3.0 * sigma, floor)


@torch.no_grad()
def dominance_ratio(probs: torch.Tensor, kind: str = "power", alpha: float = 0.4,
                    exclude: int | None = None) -> float:
    """Eq. 6/10 diagnostic: mean over samples of f(p_max) / sum_j f(p_j).

    The paper requires this to stay below 0.5 so the watermark is not dominated
    by the single largest probability. 
    """
    if exclude is not None:
        keep = [c for c in range(probs.shape[1]) if c != exclude]
        probs = probs[:, keep]
    fp = smooth(probs, kind, alpha)
    ratio = fp.max(dim=1).values / fp.sum(dim=1).clamp(min=1e-9)
    return ratio.mean().item()