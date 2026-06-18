"""Stage 3/4 server side: registration, extraction and detection.

The verification center registers every client's (trigger class, secret key,
watermark bits). Each round it extracts the watermark from each submitted model
using N_T trigger samples (Eq. 15) and computes the bit-error-rate (Eq. 16):

  * benign client   -> trained with L_wm  -> BER ~ 0          (watermark present)
  * free-rider      -> fabricated update  -> BER ~ 0.5         (no watermark)

A client is flagged as a free-rider when BER >= eta. This module is the
mechanism behind Tables II-V; the honest-only run (idx 11) exercises the
extraction/fidelity half, the free-rider run (idx 12) the detection half.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import watermark as wm


class WatermarkRegistry:
    """cid -> (trigger_class, key, target_bits, kind, alpha). One entry per
    client slot, including slots that turn out to be free-riders (they have a
    registered watermark they simply cannot reproduce)."""

    def __init__(self):
        self.entries: dict[int, dict] = {}

    def register(self, cid, trigger_class, key, target_bits, kind="power", alpha=0.4):
        self.entries[cid] = dict(trigger_class=trigger_class, key=key,
                                 target_bits=target_bits, kind=kind, alpha=alpha)

    def __len__(self):
        return len(self.entries)


def build_trigger_bank(test_dataset, classes, n_triggers, seed=0):
    """Collect up to n_triggers samples per trigger class from the test set."""
    g = torch.Generator().manual_seed(seed)
    by_class = {c: [] for c in classes}
    order = torch.randperm(len(test_dataset), generator=g).tolist()
    need = set(classes)
    for i in order:
        if not need:
            break
        x, y = test_dataset[i]
        y = int(y)
        if y in by_class and len(by_class[y]) < n_triggers:
            by_class[y].append(x)
            if len(by_class[y]) >= n_triggers:
                need.discard(y)
    return {c: torch.stack(v) for c, v in by_class.items() if v}


def make_verifier(registry, trigger_bank, verify_model, device,
                  free_rider_indices, eta=0.25, verify_every=1):
    """Return a verify_hook(server, round, updates) for Server.

    Records per-round: mean benign BER, mean free-rider BER, detection accuracy
    (benign kept + free-riders flagged), and false-positive rate.
    """
    fr_set = set(free_rider_indices)

    @torch.no_grad()
    def verify_hook(server, rnd, updates):
        if rnd % verify_every != 0:
            return {}
        verify_model.to(device).eval()
        benign_bers, fr_bers = [], []
        benign_flagged = fr_flagged = 0
        for cid, (state, _n) in enumerate(updates):
            entry = registry.entries.get(cid)
            if entry is None:
                continue
            tc = entry["trigger_class"]
            if tc not in trigger_bank:
                continue
            verify_model.load_state_dict(state)
            x = trigger_bank[tc].to(device)
            probs = F.softmax(verify_model(x), dim=1)
            bits = wm.extract_bits(probs, entry["key"].to(device),
                                   entry["kind"], entry["alpha"], exclude=tc)
            ber = wm.bit_error_rate(bits, entry["target_bits"])
            flagged = not wm.detected(ber, eta)          # BER >= eta -> free-rider
            if cid in fr_set:
                fr_bers.append(ber); fr_flagged += int(flagged)
            else:
                benign_bers.append(ber); benign_flagged += int(flagged)

        n_benign = max(len(benign_bers), 1)
        n_fr = len(fr_bers)
        info = {
            "wm_benign_ber": round(sum(benign_bers) / n_benign, 4),
            "wm_fr_ber": round(sum(fr_bers) / n_fr, 4) if n_fr else None,
            "wm_fpr": round(benign_flagged / n_benign, 4),
            "wm_fr_recall": round(fr_flagged / n_fr, 4) if n_fr else None,
        }
        # detection accuracy = correctly-classified clients / total registered
        total = len(benign_bers) + n_fr
        correct = (len(benign_bers) - benign_flagged) + fr_flagged
        info["wm_detect_acc"] = round(correct / max(total, 1), 4)
        return info

    return verify_hook
