"""Server side: registration, extraction and detection

The verification center registers every client's (trigger class, secret key,
watermark bits). Each round it extracts the watermark from each submitted model
using N_T trigger samples (Eq. 15) and computes the bit-error-rate (Eq. 16):

  * benign client   -> trained with L_wm  -> BER ~ 0          (watermark present)
  * free-rider      -> fabricated update  -> BER ~ 0.5         (no watermark)

A client is flagged as a free-rider when BER >= eta
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import watermark as wm


class WatermarkRegistry:
    """cid -> (trigger_class, key, target_bits, kind, alpha). One entry per
    client slot, including slots that turn out to be free-riders"""

    def __init__(self):
        self.entries: dict[int, dict] = {}
        # filled in by build_watermarked_clients for self-documenting results:
        self.m = None                 # number of watermark bits per client
        self.l = None                 # group size (n//m or (n-1)//m)
        self.unembeddable_frac = 0.0  # mean fraction of same-sign (stuck) key rows

    def register(self, cid, trigger_class, key, target_bits, kind="power",
                 alpha=0.4, exclude="trigger"):
        # exclude: which projection column the verifier drops. "trigger" sentinel
        # -> use trigger_class (our mode); None -> paper-faithful full softmax.
        exc = trigger_class if exclude == "trigger" else exclude
        self.entries[cid] = dict(trigger_class=trigger_class, key=key,
                                 target_bits=target_bits, kind=kind, alpha=alpha,
                                 exclude=exc)

    def __len__(self):
        return len(self.entries)


def build_trigger_bank(test_dataset, classes, n_triggers, seed=0):
    """Collect up to n_triggers samples per trigger class from the test set"""
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


# TODO: verify the threshold calibration 
def make_verifier(registry, trigger_bank, verify_model, device,
                  free_rider_indices, eta_floor=0.05, verify_every=1,
                  paper_faithful=False, calib_on_all=False):
    """Return a verify_hook(server, round, updates) for Server

    The threshold is always the computed eta = mu + 3*sigma over the benign BER
    distribution (Eq. 16). `eta_floor` is a guard: if every benign BER is ~0, 
    mu+3sigma collapses to 0 and the rule "flag iff BER >= eta" would flag every 
    honest client (BER 0 >= 0); the floor keeps eta strictly positive. 
    It sits well below the honest hard-position band (~0.10-0.20) and the tight 
    round-mean eta (~0.09), so it never binds

    paper_faithful=True: cumulative mu+3sigma over all rounds 
    else: a sliding window of recent rounds so eta can recover after a transient 
    benign-BER spike; a cumulative mean would stay poisoned forever.
    calib_on_all=True: calibrate eta over every client's BER (server cannot tell
    honest from free-rider), exposing the paper's circularity -- free-rider BER
    ~0.5 poisons mu+3sigma. Default False matches the paper's 'observe legitimate
    clients' (a trusted pool ?)
    """
    fr_set = set(free_rider_indices)
    benign_history = []          # per-round BER means used to calibrate eta
    CAL_WINDOW = 15              # rounds of recent BER used for mu+3sigma

    @torch.no_grad()
    def verify_hook(server, rnd, updates):
        if rnd % verify_every != 0:
            return {}
        verify_model.to(device).eval()
        # Pass 1: extract every client's watermark and measure BER (no flagging yet)
        measured = []            # (cid, ber, is_free_rider)
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
                                   entry["kind"], entry["alpha"],
                                   exclude=entry.get("exclude", tc))
            ber = wm.bit_error_rate(bits, entry["target_bits"])
            measured.append((cid, ber, cid in fr_set, tc))  # +trigger class

        # Calibrate the threshold from the benign BER distribution (Eq. 16):
        #   eta = mu + 3*sigma   over the per-round MEAN benign BER.
        # This is ALWAYS the computed value (no hardcoded floor/cap); `eta_floor`
        # is only the tiny degenerate guard described in make_verifier's docstring.
        # paper_faithful -> cumulative over all rounds; else a sliding window so eta
        # can recover after a transient benign-BER spike.
        benign_now = [b for _, b, isfr, _ in measured if not isfr]
        calib_now = [b for _, b, _, _ in measured] if calib_on_all else benign_now
        if calib_now:
            benign_history.append(sum(calib_now) / len(calib_now))
        if paper_faithful:
            eta_round = (wm.calibrate_eta(benign_history, floor=eta_floor)
                         if benign_history else eta_floor)
        else:
            recent = benign_history[-CAL_WINDOW:]
            eta_round = (wm.calibrate_eta(recent, floor=eta_floor)
                         if recent else eta_floor)

        benign_bers, fr_bers = [], []
        benign_flagged = fr_flagged = 0
        # per-client: records so we can see the BER distribution (not just the mean).
        # This is what exposes a false-positive: an honest client at a hard trigger
        # class can sit as high as a re-embedding free-rider, so a tight mu+3sigma eta
        # flags honest clients too. (analysis of the 0.11 floor)
        per_client = []
        for cid, ber, is_fr, tc in measured:
            flagged = not wm.detected(ber, eta_round)    # BER >= eta_round -> free-rider
            per_client.append({"cid": cid, "trigger_class": int(tc),
                               "ber": round(ber, 4), "is_free_rider": bool(is_fr),
                               "flagged": bool(flagged)})
            if is_fr:
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
            "wm_eta_round": round(eta_round, 4),
            # distributions (for the false-positive / per-class analysis)
            "wm_benign_ber_list": [round(b, 4) for b in benign_bers],
            "wm_fr_ber_list": [round(b, 4) for b in fr_bers],
            "wm_per_client": per_client,
        }
        total = len(benign_bers) + n_fr
        correct = (len(benign_bers) - benign_flagged) + fr_flagged
        info["wm_detect_acc"] = round(correct / max(total, 1), 4)
        return info

    return verify_hook