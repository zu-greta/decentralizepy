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
    """Collect up to n_triggers samples per trigger class from the test set.

    Keyed by class: every client whose trigger class is c is verified on the same
    held-out images. Used whenever one class == one client. 
    Under oversubscription (clients sharing a class) two clients differ ONLY by 
    their key M^i and bits B^i 
    """
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


def build_trigger_bank_per_client(test_dataset, registry, n_triggers, seed=0):
    """Table IX from FareMark, held-out variant. Keyed by CID.

    "While clients sharing the same trigger class utilize identical class labels,
    their watermarks remain distinguishable through client-specific trigger
    variations." -> each client gets its own disjoint slice of that class's images,
    so two clients on class c are verified on different images (plus their own M^i,
    B^i). Images still come from the held-out test set, so the mark must generalise.
    """
    # cid -> class, and class -> [cids] (stable order so slices are reproducible)
    cls_of = {cid: e["trigger_class"] for cid, e in registry.entries.items()}
    per_cls = {}
    for cid in sorted(cls_of):
        per_cls.setdefault(cls_of[cid], []).append(cid)

    need_per_cls = {c: n_triggers * len(cids) for c, cids in per_cls.items()}
    pool = {c: [] for c in per_cls}
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(test_dataset), generator=g).tolist()
    remaining = set(per_cls)
    for i in order:
        if not remaining:
            break
        x, y = test_dataset[i]
        y = int(y)
        if y in pool and len(pool[y]) < need_per_cls[y]:
            pool[y].append(x)
            if len(pool[y]) >= need_per_cls[y]:
                remaining.discard(y)

    bank = {}
    for c, cids in per_cls.items():
        imgs = pool.get(c, [])
        if not imgs:
            continue
        # deal out disjoint slices
        for j, cid in enumerate(cids):
            lo = j * n_triggers
            sl = imgs[lo: lo + n_triggers]
            if not sl:                      # class exhausted -> reuse from the top
                sl = imgs[(lo % max(len(imgs), 1)): (lo % max(len(imgs), 1)) + n_triggers]
            if sl:
                bank[cid] = torch.stack(sl)
    return bank


def build_trigger_bank_from_train(client_loaders, registry, n_triggers):
    """Table IX from FareMark, trigger sample consistency variant. Keyed by CID.

    "we enforce trigger sample consistency: the trigger samples used during testing
    are identical to those employed in training." -> each client's verification images
    are drawn from its own training shard, i.e. images it actually trained on. This is
    the paper-exact capacity protocol and it makes the mark trivially separable per
    client -- but it is pure memorisation: the paper itself notes (Table V) that a mark
    fitted to specific samples "cannot be generalized to other trigger-class samples".
    Use it to reproduce the paper's capacity numbers; use the held-out banks to test
    whether the mark means anything beyond those exact images.
    """
    bank = {}
    for cid, e in registry.entries.items():
        if cid >= len(client_loaders):
            continue
        tc = e["trigger_class"]
        got = []
        for x, y in client_loaders[cid]:
            m = (y == tc)
            if m.any():
                got.append(x[m])
                if sum(len(t) for t in got) >= n_triggers:
                    break
        if got:
            bank[cid] = torch.cat(got)[:n_triggers]
    return bank


def make_verifier(registry, trigger_bank, verify_model, device,
                  free_rider_indices, eta_floor=0.05, verify_every=1,
                  calib_on_all=False, eta_fixed=0.0, per_client_bank=False):
    """Return a verify_hook(server, round, updates) for Server

    THRESHOLD: eta is a pre-calibrated passed in as `eta_fixed` 
    (from calibrate_eta.py: mu+3sigma over per-round mean-over-clients
    benign BER, pooled over honest-only seeds, frozen)

    `eta_floor` stays a tiny degenerate guard (keeps a >0 threshold if eta_fixed
    is somehow 0). calib_on_all is kept only for the circularity demo.

    NOTE: the old LIVE threshold calculations (paper_faithful cumulative mu+3sigma,
    and the sliding-window variant) are commented out below -- kept for reference
    but no longer used. Set eta_fixed=0 to fall back to them if ever needed.
    """
    fr_set = set(free_rider_indices)
    benign_history = []          # per-round BER means (only used by the commented-out live calc)
    CAL_WINDOW = 15              # TODO hardcoded (dead): sliding-window length for the commented-out live calc

    @torch.no_grad()
    def verify_hook(server, rnd, updates):
        if rnd % verify_every != 0:
            return {}
        verify_model.to(device).eval()
        # Pass 1: extract every client's watermark and measure BER (no flagging yet)
        measured = []            # (cid, ber, is_free_rider)
        diag = {}                # cid -> per-class difficulty diagnostics 
        for cid, (state, _n) in enumerate(updates):
            entry = registry.entries.get(cid)
            if entry is None:
                continue
            tc = entry["trigger_class"]
            # per_client_bank: each client has its own trigger images else the bank is shared per trigger class.
            bkey = cid if per_client_bank else tc
            if bkey not in trigger_bank:
                continue
            verify_model.load_state_dict(state)
            x = trigger_bank[bkey].to(device)
            probs = F.softmax(verify_model(x), dim=1)
            bits = wm.extract_bits(probs, entry["key"].to(device),
                                   entry["kind"], entry["alpha"],
                                   exclude=entry.get("exclude", tc))
            ber = wm.bit_error_rate(bits, entry["target_bits"])

            # diagnostics: how hard is this trigger class to classify? (for analysis of BER floors)
            pmax, pred = probs.max(dim=1)
            trig_acc = (pred == tc).float().mean().item()      # is the class classified correctly
            ent = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=1).mean().item()
            dom = wm.dominance_ratio(probs, entry["kind"], entry["alpha"],
                                     exclude=entry.get("exclude"))   # Eq. 6/10: <0.5 wanted
            diag[cid] = {"trig_acc": round(trig_acc, 4),
                         "pmax": round(pmax.mean().item(), 4),
                         "entropy": round(ent, 4),
                         "dominance": round(dom, 4)}
            measured.append((cid, ber, cid in fr_set, tc))  # +trigger class

        # ================= THRESHOLD =================
        # a pre-calibrated constant (calibrate_eta.py)
        benign_now = [b for _, b, isfr, _ in measured if not isfr]
        if eta_fixed and eta_fixed > 0:
            eta_round = float(eta_fixed) # when eta_fixed > 0 it is used for every round
        else:
            eta_round = eta_floor        # degenerate fallback only

        # ---- Legacy threhsolds (for reference) ----------
        # calib_now = [b for _, b, _, _ in measured] if calib_on_all else benign_now
        # if calib_now:
        #     benign_history.append(sum(calib_now) / len(calib_now))
        # if paper_faithful:
        #     eta_round = (wm.calibrate_eta(benign_history, floor=eta_floor)
        #                  if benign_history else eta_floor)          # cumulative mu+3sigma
        # else:
        #     recent = benign_history[-CAL_WINDOW:]
        #     eta_round = (wm.calibrate_eta(recent, floor=eta_floor)
        #                  if recent else eta_floor)                  # sliding-window mu+3sigma
        # -------------------------------------------------------------------------

        benign_bers, fr_bers = [], []
        benign_flagged = fr_flagged = 0
        per_client = []
        # ================== FLAGGING =================
        for cid, ber, is_fr, tc in measured:
            flagged = not wm.detected(ber, eta_round)    # BER >= eta_round -> free-rider
            d = diag.get(cid, {})
            per_client.append({"cid": cid, "trigger_class": int(tc),
                               "ber": round(ber, 4), "is_free_rider": bool(is_fr),
                               "flagged": bool(flagged),
                               # per-class difficulty diagnostics 
                               "trig_acc": d.get("trig_acc"),
                               "pmax": d.get("pmax"),
                               "entropy": d.get("entropy"),
                               "dominance": d.get("dominance")})
            if is_fr:
                fr_bers.append(ber); fr_flagged += int(flagged) # count FRs that were flagged
            else:
                benign_bers.append(ber); benign_flagged += int(flagged) # count honest clients that were flagged (false positives)

        # ================= SUMMARY =================
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
        # diagnostics: average trigger-class difficulty for honest clients (for analysis of BER floors)
        hon_diag = [diag[cid] for cid, _, is_fr, _ in measured
                    if not is_fr and cid in diag]
        if hon_diag:
            def _m(k):
                vs = [d[k] for d in hon_diag if d.get(k) is not None]
                return round(sum(vs) / len(vs), 4) if vs else None
            info["wm_benign_trig_acc"] = _m("trig_acc")
            info["wm_benign_pmax"] = _m("pmax")
            info["wm_benign_entropy"] = _m("entropy")
            info["wm_benign_dominance"] = _m("dominance")
        total = len(benign_bers) + n_fr
        correct = (len(benign_bers) - benign_flagged) + fr_flagged
        info["wm_detect_acc"] = round(correct / max(total, 1), 4)
        return info

    return verify_hook