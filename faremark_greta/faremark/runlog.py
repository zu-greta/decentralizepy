"""runlog -- all human-readable run.log formatting 

DESIGN RULES
------------
1. run.log is a log file to read live and see if run is healthy or not. 
   Anything an analysis script needs lives in result.json. run.log exists so a human
2. Fixed-width columns with one header, so `grep`/`awk`/`column -t` all work:
       awk '$1=="R"' run.log | awk '{print $2, $4}'      # round, honest BER
3. Every block is delimited by a `== NAME ==` line so `sed -n '/== SETUP/,/== ROUNDS/p'`
   pulls a section out.
4. Non-default config only. The full ExpConfig snapshot is in result.json["config"];
   repeating all 60 fields in the log hides the 3 that were actually set.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import MISSING as _MISSING_SENTINEL, fields as _dc_fields


class _MISSING:
    """Marker type for dataclass fields that have no default (required fields)."""


def _is_missing(v):
    return v is _MISSING_SENTINEL

# Width of the divider lines. 92 keeps the round table inside a standard terminal.
_W = 92


def _rule(ch="-"):
    return ch * _W


def _kv(k, v, pad=26):
    return f"  {k:<{pad}} {v}"


# --------------------------------------------------------------------- banner
def banner(log, *, config_idx, cfg, repeat, seed, device, gpu_name=None,
           gpu_count=0, family=None, note=None, output_dir=None):
    """Run identity. First thing in run.log; answers 'what am I looking at?'."""
    log.info(_rule("="))
    log.info(f"  FareMark run | {cfg.name}  (config {config_idx})")
    log.info(_rule("="))
    log.info(_kv("family", family or "(none -- set FAMILY= to group runs for plots)"))
    log.info(_kv("model / dataset", f"{cfg.model} / {cfg.dataset}"))
    log.info(_kv("clients / rounds / epochs", f"{cfg.num_clients} / {cfg.rounds} / {cfg.local_epochs}"))
    log.info(_kv("partition", cfg.partition + (f" (alpha={cfg.dirichlet_alpha})"
                                               if cfg.partition != "iid" else "")))
    log.info(_kv("attack", cfg.attack))
    log.info(_kv("repeat / seed", f"{repeat} / {seed}"))
    log.info(_kv("device", f"{device}"
                 + (f"  [{gpu_name} x{gpu_count}]" if gpu_name else "")))
    if output_dir:
        log.info(_kv("output_dir", output_dir))
    if note:
        log.info(_kv("note", note))


# ---------------------------------------------------------------------- setup
def config_block(log, cfg, cfg_cls):
    """Print ONLY the fields that differ from the ExpConfig defaults.

    Replaces the old `logger.info(json.dumps(cfg.to_dict()))` one-liner. The full
    snapshot is still written to result.json["config"], so nothing is lost -- this
    just stops 60 defaults from burying the 3 knobs the run actually changed.
    """
    defaults = {f.name: f.default for f in _dc_fields(cfg_cls)}
    diffs, required = [], []
    for f in _dc_fields(cfg_cls):
        cur = getattr(cfg, f.name, None)
        dflt = defaults.get(f.name)
        if f.name == "name":
            continue
        # model/dataset/num_clients have no dataclass default (they are required
        # positional fields), so there is nothing to diff against -- list them
        # separately rather than printing a _MISSING_TYPE repr.
        if _is_missing(dflt):
            required.append((f.name, cur))
        elif cur != dflt:
            diffs.append((f.name, dflt, cur))
    log.info("")
    log.info("== SETUP: config overrides (vs ExpConfig defaults) ==")
    for name, cur in required:
        log.info(f"  {name:<26} {'(required)':<18}    {cur}")
    if not diffs:
        log.info("  (no optional field differs from its default)")
    for name, dflt, cur in diffs:
        log.info(f"  {name:<26} {str(dflt):<18} -> {cur}")
    log.info(_kv("(full config)", "result.json[\"config\"]"))


def data_block(log, *, dataset, num_classes, num_clients, shard_sizes,
               partition, alpha, batch_size, test_n):
    """What the data actually looks like after partitioning.

    a non-IID shard can hold 0 images of a client's own trigger class, 
    which floors that client's BER at ~0.5 for reasons that have nothing to do with the watermark. 
    min/max shard size and the empty-trigger-class warning are printed up front.
    """
    log.info("")
    log.info("== SETUP: data ==")
    log.info(_kv("dataset / classes", f"{dataset} / {num_classes}"))
    log.info(_kv("partition", partition + (f" (dirichlet alpha={alpha})"
                                           if partition != "iid" else "")))
    log.info(_kv("test set", f"{test_n} images"))
    if shard_sizes:
        lo, hi = min(shard_sizes), max(shard_sizes)
        mean = sum(shard_sizes) / len(shard_sizes)
        log.info(_kv("client shards",
                     f"{num_clients} shards, {lo}-{hi} imgs (mean {mean:.0f}), batch {batch_size}"))
        if hi > 3 * max(lo, 1):
            log.info("  NOTE  shard sizes vary >3x -- expected under Dirichlet, "
                     "but it also skews FedAvg weighting.")


def watermark_block(log, *, m, l, num_classes, unembeddable_frac, n_triggers,
                    trigger_mode, n_banks, n_clients, balanced_keys,
                    wm_lambda, wm_beta, wm_alpha, wm_f, eta_fixed,
                    clients_per_class=None):
    """Watermark geometry + the two structural warnings that explain BER floors.

    the 'structural BER ceiling' line. With random (unbalanced) keys a fraction
    of key rows come out all-same-sign and can never be embedded, so honest BER can
    not go below 0.5*unembeddable_frac no matter how long you train. Printing the
    implied ceiling here means a 75%-instead-of-96% watermark accuracy at round 0
    """
    log.info("")
    log.info("== SETUP: watermark ==")
    log.info(_kv("bits m / group l", f"{m} / {l}   (n={num_classes}, uses first m*l={m * l} softmax outputs)"))
    log.info(_kv("keys", "sign-balanced" if balanced_keys else "random +/-1 (paper-faithful)"))
    log.info(_kv("smoothing f / alpha", f"{wm_f} / {wm_alpha}"))
    log.info(_kv("lambda / beta", f"{wm_lambda} / {wm_beta}"))
    log.info(_kv("N_T / trigger mode", f"{n_triggers} / {trigger_mode}"))
    log.info(_kv("trigger banks", f"{n_banks} "
                 + ("(per client)" if trigger_mode != "class" else "(per class)")))
    if clients_per_class:
        lo, hi = min(clients_per_class.values()), max(clients_per_class.values())
        share = f"{lo}" if lo == hi else f"{lo}-{hi}"
        log.info(_kv("clients per trigger class", f"{share}"
                     + ("   (oversubscribed: clients SHARE a class)" if hi > 1 else "")))
    log.info(_kv("eta (detection threshold)",
                 f"{eta_fixed:.5f} (frozen, WM_ETA_FIXED)" if eta_fixed and eta_fixed > 0
                 else "NOT SET -- falling back to eta_floor; flags are meaningless"))
    if unembeddable_frac and unembeddable_frac > 0:
        ceil = 100.0 * (1.0 - 0.5 * unembeddable_frac)
        log.info(f"  WARNING  {unembeddable_frac:.1%} of key rows are same-sign and "
                 f"structurally unembeddable.")
        log.info(f"           -> honest BER floors near {0.5 * unembeddable_frac:.3f}; "
                 f"watermark-accuracy ceiling ~{ceil:.2f}%.")
        log.info(f"           -> P(row same-sign) = 2^(1-l) = {2.0 ** (1 - l):.3f} at l={l}. "
                 f"Use --wm_balanced_keys to remove this artifact.")
    if n_banks < n_clients and trigger_mode != "class":
        log.info(f"  WARNING  only {n_banks}/{n_clients} clients got a trigger bank "
                 f"-- the rest are never verified.")


def free_rider_block(log, *, attack, indices, trigger_class_of=None, knobs=None):
    """Who free-rides, on which trigger class, with which knobs."""
    log.info("")
    log.info("== SETUP: free-riders ==")
    if not indices:
        log.info("  (none -- all clients honest; this is a calibration/sanity run)")
        return
    log.info(_kv("attack", attack))
    log.info(_kv("free-rider cids", indices))
    if trigger_class_of:
        pairs = ", ".join(f"cid{c}->cls{trigger_class_of.get(c)}" for c in indices)
        log.info(_kv("trigger classes", pairs))
    for k, v in (knobs or {}).items():
        log.info(_kv(f"  {k}", v))


# ----------------------------------------------------------------- round table
class RoundTable:
    """Per-round progress as a fixed-width table with a single header.

    Columns, and why each earns its place:
      R          row marker, so `awk '$1=="R"'` extracts the table cleanly
      round      round index
      acc%       global test accuracy (main task)
      ber_h      MEAN honest BER              
      p90        90th percentile honest BER   
      max        max honest BER               
      ber_fr     mean free-rider BER          
      eta        the live threshold           
      flag       n_flagged / n_clients        
      fpr        honest false-positive rate   
      rec        free-rider recall            
      s/r        seconds for this round       
    """

    HDR = (f"{'R':<2}{'round':>6}{'acc%':>8}{'ber_h':>8}{'p90':>7}{'max':>7}"
           f"{'ber_fr':>8}{'eta':>8}{'flag':>8}{'fpr':>7}{'rec':>7}{'s/r':>7}")

    def __init__(self, log, total_rounds, watermarked=True):
        self.log = log
        self.total = total_rounds
        self.wm = watermarked
        self._t = time.perf_counter()
        self._printed = False

    def header(self):
        self.log.info("")
        self.log.info(f"== ROUNDS (1..{self.total}) ==")
        if self.wm:
            self.log.info(self.HDR)
            self.log.info(_rule())
        self._printed = True

    def row(self, rnd, acc, info):
        if not self._printed:
            self.header()
        dt = time.perf_counter() - self._t
        self._t = time.perf_counter()
        if not self.wm or "wm_benign_ber" not in info:
            self.log.info(f"R {rnd:>4}/{self.total}  test_acc={acc:6.2f}%  ({dt:5.1f}s)")
            return

        def _f(v, w, p=3):
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

        n_flag = len(info.get("wm_flagged_cids") or [])
        n_cl = len(info.get("wm_per_client") or [])
        self.log.info(
            f"R {rnd:>6}{acc:>8.2f}"
            f"{_f(info.get('wm_benign_ber'), 8)}"
            f"{_f(info.get('wm_benign_ber_p90'), 7)}"
            f"{_f(info.get('wm_benign_ber_max'), 7)}"
            f"{_f(info.get('wm_fr_ber'), 8)}"
            f"{_f(info.get('wm_eta_round'), 8, 4)}"
            f"{(str(n_flag) + '/' + str(n_cl)):>8}"
            f"{_f(info.get('wm_fpr'), 7)}"
            f"{_f(info.get('wm_fr_recall'), 7)}"
            f"{dt:>7.1f}")

    def phase_note(self, rnd, changes):
        """Log free-rider phase transitions (honest -> calib -> tap/coast)"""
        for cid, action in changes:
            self.log.info(f"   * round {rnd}: free-rider cid{cid} -> {action.upper()}")


# ---------------------------------------------------------------------- report
def report(log, *, final_acc, best_acc, expected, passed, elapsed_sec,
           wm_summary=None, compute_summary=None, per_class=None, out_path=None,
           eta_used=None, tail=10):
    """Closing block: the numbers + PASS/FAIL + provenance."""
    log.info("")
    log.info("== REPORT ==")
    lo, hi = expected
    log.info(_kv("final / best test acc", f"{final_acc:.2f}% / {best_acc:.2f}%"))
    log.info(_kv("expected band", f"{lo}-{hi}%   -> {'PASS' if passed else 'FAIL'}"))
    if not passed:
        log.info("  NOTE  attack runs legitimately land below the band "
                 "(free-riders drag accuracy down). Exit code 2 is EXPECTED there; "
                 "result.json is written before the exit.")

    if wm_summary:
        log.info(f"  -- watermark (mean over last {wm_summary.get('wm_detect_window', tail)} rounds) --")
        log.info(_kv("honest BER", wm_summary.get("wm_benign_ber")))
        log.info(_kv("free-rider BER", wm_summary.get("wm_fr_ber")))
        log.info(_kv("watermark accuracy %",
                     None if wm_summary.get("wm_benign_ber") is None
                     else f"{100.0 * (1 - wm_summary['wm_benign_ber']):.2f}"))
        log.info(_kv("eta used", eta_used))
        log.info(_kv("FPR / recall / det acc",
                     f"{wm_summary.get('wm_fpr')} / {wm_summary.get('wm_fr_recall')} "
                     f"/ {wm_summary.get('wm_detect_acc')}"))
        log.info(_kv("bits m / group l",
                     f"{wm_summary.get('wm_bits_m')} / {wm_summary.get('wm_group_size_l')} "
                     f"(unembeddable {wm_summary.get('wm_unembeddable_frac')})"))

    if per_class and per_class.get("by_class"):
        bc = per_class["by_class"]
        ranked = sorted(bc.items(), key=lambda kv: kv[1]["acc"])
        worst = ", ".join(f"c{k}:{v['acc']:.0f}%" for k, v in ranked[:5])
        best = ", ".join(f"c{k}:{v['acc']:.0f}%" for k, v in ranked[-3:])
        log.info("  -- per-class test accuracy (final model) --")
        log.info(_kv("hardest 5 classes", worst))
        log.info(_kv("easiest 3 classes", best))

    if compute_summary:
        cs = compute_summary
        log.info("  -- effort --")
        log.info(_kv("honest mean",
                     f"{cs.get('honest_mean_gpu_ms', 0):.0f} gpu-ms, "
                     f"{cs.get('honest_mean_samples', 0):.0f} samples"))
        log.info(_kv("free-rider mean",
                     f"{cs.get('fr_mean_gpu_ms', 0):.0f} gpu-ms, "
                     f"{cs.get('fr_mean_samples', 0):.0f} samples"))
        log.info(_kv("effort ratio (gpu/samples)",
                     f"{cs.get('effort_ratio_gpu')} / {cs.get('effort_ratio_samples')}"))

    log.info(_kv("elapsed", f"{elapsed_sec / 60.0:.1f} min"))
    if out_path:
        log.info(_kv("wrote", out_path))
        log.info(_kv("inspect",
                     f"python scripts/resultio.py digest --in {os.path.dirname(out_path)}/result.json"))
    log.info(_rule("="))