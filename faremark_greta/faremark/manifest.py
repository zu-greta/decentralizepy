"""Self-describing run metadata.

The manifest is descriptive only. Values come from CLI flags 
(forwarded by submit_experiment.sh as FAMILY/NOTE/SWEEP_VAR/
SWEEP_LEVEL) 
"""
from __future__ import annotations

# Per-metric reading guide, embedded in every run. Keep in sync with wm_verify.py's emitted keys.
INTERPRETATION = {
    "wm_benign_ber": "honest bit-error-rate; LOWER = watermark embeds well (defender-good).",
    "wm_fr_ber": "free-rider bit-error-rate; HIGHER = free-rider looks un-watermarked "
                 "(defender-good). LOW fr_ber = evasion (attacker-good).",
    "wm_fpr": "fraction of honest clients wrongly flagged; LOWER = defender-good.",
    "wm_fr_recall": "fraction of free-riders caught; HIGHER = defender-good. "
                    "recall->0 = attack succeeds.",
    "wm_detect_acc": "overall honest-vs-FR classification accuracy of the detector.",
    "wm_eta_used": "detection threshold actually applied (BER>=eta => flagged).",
    "final_acc": "global model test accuracy (fidelity).",
    "compute.effort_ratio_gpu": "free-rider GPU-time / honest GPU-time. <<1 with "
                                "recall->0 = 'cheap evasion' => scheme broken.",
}


def build_manifest(cfg, args=None) -> dict:
    """Return the manifest block to embed in result.json.

    args (argparse.Namespace) may carry --manifest_family/--manifest_note/
    --sweep_var/--sweep_level. Anything missing is inferred from cfg.
    """
    def _get(name, default=None):
        return getattr(args, name, None) if args is not None else default

    family = _get("manifest_family") or _infer_family(cfg)
    sweep_var = _get("sweep_var")
    sweep_level = _get("sweep_level")
    # If a sweep var was named but no level given, read the level off the config.
    if sweep_var and sweep_level is None:
        sweep_level = getattr(cfg, sweep_var, None)

    return {
        "family": family,                        # e.g. "A2_train_then_attack"
        "note": _get("manifest_note") or "",     # one-line human hypothesis
        "sweep_var": sweep_var,                  # config field being swept, e.g. "attack_round"
        "sweep_level": sweep_level,              # this run's value of that field
        "attack": getattr(cfg, "attack", "none"),
        "partition": getattr(cfg, "partition", "iid"),
        "dirichlet_alpha": getattr(cfg, "dirichlet_alpha", None),
        "num_free_riders": getattr(cfg, "num_free_riders", 0),
        "paper_faithful": getattr(cfg, "paper_faithful", False),
        "calib_on_all": getattr(cfg, "calib_on_all", False),
        "outcome_keys": ["wm_fr_recall", "wm_fpr", "wm_benign_ber", "wm_fr_ber",
                         "final_acc", "compute.effort_ratio_gpu"],
        "interpretation": INTERPRETATION,
    }


def _infer_family(cfg) -> str:
    """Best-effort family name when --manifest_family is not passed."""
    atk = getattr(cfg, "attack", "none")
    part = getattr(cfg, "partition", "iid")
    if getattr(cfg, "num_free_riders", 0) == 0:
        base = "fidelity" if getattr(cfg, "watermark", False) else "baseline"
    else:
        base = atk
    return f"{base}__{part}" if part != "iid" else base
