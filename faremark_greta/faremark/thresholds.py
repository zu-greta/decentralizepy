"""Threshold (eta) variants 

eta only decides the detection label (flag / don't flag) from BERs 

------------------------------------------------------------------------------

Variants (all selectable as flags in the plotters via --eta):
  cumulative : mu+3sigma over b[0..t]      
  ->frozen     : mu+3sigma over a stable post-convergence window, then held fixed.
               window = b[warmup : warmup+converged]  (skip the noisy pre-embed
               rounds; calibrate on the first converged block; freeze). 
               [params: warmup, converged, or an explicit calib_start]
  windowed   : mu+3sigma over the last K rounds b[t-K..t] (adaptive, no memory)
  converged  : mu+3sigma over the last C rounds, held fixed (fair, but tail can be
               poisoned)
  fixed      : a constant (e.g. 0.25) for reference
"""
from __future__ import annotations
import statistics as st

# The variant to headline in text/plots 
HEADLINE = "frozen"


def _mu3s(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return 0.5
    mu = st.mean(xs)
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    return mu + 3.0 * sd


def eta_series(benign, variant="cumulative", floor=0.05, warmup=10, window=10,
               converged=10, fixed=0.25, calib_start=None):
    """Return a per-round eta list the same length as `benign` (list of per-round
    mean benign BERs).

    `frozen` calibrates on a stable post-convergence window and freezes it:
        start = calib_start if given else `warmup`   (skip the pre-embed rounds)
        win   = benign[start : start+converged]
    This is not the first `warmup` rounds: honest clients have not embedded yet, 
    so benign BER is high and mu+3sigma would be a too-loose threshold. 
    Calibrating on the first converged block matches the
    paper's "typical error rate" and is computed before any defection.
    """
    T = len(benign)
    out = []
    if variant == "fixed":
        return [fixed] * T
    if variant == "frozen":
        start = warmup if calib_start is None else calib_start
        start = max(0, min(start, max(0, T - 1)))
        win = [b for b in benign[start:start + max(1, converged)] if b is not None]
        if not win:                                  # series shorter than the skip
            win = [b for b in benign[:max(1, warmup)] if b is not None]
        val = max(floor, _mu3s(win))
        return [val] * T
    if variant == "converged":
        val = max(floor, _mu3s(benign[-max(1, converged):]))
        return [val] * T
    for t in range(T):
        if variant == "windowed":
            lo = max(0, t - window + 1)
            out.append(max(floor, _mu3s(benign[lo:t + 1])))
        else:  # cumulative (default / old paper_faithful)
            out.append(max(floor, _mu3s(benign[:t + 1])))
    return out


ALL_VARIANTS = ["cumulative", "frozen", "windowed", "converged", "fixed"]

# fixed styles so the same variant looks the same on every figure
STYLE = {
    "cumulative": dict(color="#c0392b", ls="--", label="η cumulative (old paper_faithful; swings)"),
    "frozen":     dict(color="#1e7a46", ls="-",  label="η frozen (post-convergence, fair — HEADLINE)"),
    "windowed":   dict(color="#e08e0b", ls=":",  label="η windowed"),
    "converged":  dict(color="#1f77b4", ls="-.", label="η converged (last-C, fair backup)"),
    "fixed":      dict(color="#7f7f7f", ls=(0, (1, 1)), label="η fixed=0.25"),
}


def evades_under(fr_ber, benign, variant, tail=10, **kw):
    """Fraction of the last `tail` rounds where the free-rider is under eta
    (i.e. evades) under a given variant. 1.0 = fully evades; 0.0 = always caught.
    """
    et = eta_series(benign, variant, **kw)
    pairs = [(f, e) for f, e in zip(fr_ber[-tail:], et[-tail:]) if f is not None]
    if not pairs:
        return None
    return sum(1 for f, e in pairs if f < e) / len(pairs)


def summary_evasion(fr_ber, benign, tail=10, **kw):
    """{variant: evade_fraction} across all variants"""
    return {v: evades_under(fr_ber, benign, v, tail=tail, **kw) for v in ALL_VARIANTS}