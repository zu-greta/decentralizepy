# FareMark — reproduction + limitations study

Re-implementation of **FareMark: Model-Watermark-Driven Free-Rider Detection in
Federated Learning** (Li et al., IEEE IoT-J 2025), extended into a limitations study.
Centralized FedAvg simulated on one GPU + a per-client watermark loss, a
memory-enhanced update, and server-side verification.

**Headline finding (a negative result):** the honest watermark BER floor is
**position-dependent** (which trigger class a client is assigned). The detection
threshold `eta` is calibrated on the per-round MEAN honest BER (tight) but applied to
INDIVIDUAL clients (wide). No single scalar `eta` both catches an embedding free-rider
AND spares honest hard-position clients — a false-positive / threshold-calibration
limitation, demonstrated with an adaptive **submarine** free-rider.

---

## Current layout (what actually exists)

```
faremark/
  client.py            honest FedAvg client (base)
  server.py            FedAvg loop + verify hook
  datasets.py          IID / Dirichlet shards + trigger test set
  models.py            resnet18 / alexnet / smallcnn
  watermark.py         the math (Eq.1-16): smooth/key/bits/project/embed/extract/BER
  wm_client.py         WatermarkClient (embed + Eq.14 memory) + client factory
  attacks.py           crude baselines (previous_models, gaussian) + FR selection
  attacks_adaptive.py  SUBMARINE adaptive free-rider (make_submarine_attack)
  wm_verify.py         server: extract -> BER -> FROZEN eta -> flag + diagnostics
  compute_meter.py     per-client effort (samples/gpu_ms/flops/duty)
  manifest.py, utils.py, robustness.py, plotstyle.py
scripts/
  run_experiment.py    one (config, repeat) -> result.json
  threshold.py         ALL threshold code: canonical eta + `calibrate` CLI
  plots.py             ALL plotting (thresholds, class_dynamics, positions, timeline, ...)
  run_all.sh           honest -> calibrate -> attacks -> PLOTALL
  submit_experiment.sh one RunAI job (env -> CLI flags)
```

> **Deleted/renamed:** `eta_calib.py`+`calibrate_eta.py` -> `threshold.py`;
> `plot_diag.py`+`plot_analysis.py`+`plot_tests.py` -> `plots.py`; the `autopilot`
> attack is now **`submarine`** ("autopilot" kept as a back-compat alias); the
> `paper_faithful` flag is **removed** (its True-behaviour — random keys, full softmax,
> m=n//10 — is now the only mode).

See **CODE_MAP.md** for the full technical reference (every module, formula, method).

---

## The threshold (do this first)

One canonical, pre-calibrated constant:
`eta = mu + 3*sigma` over per-round (mean-over-clients) honest BER, on the converged
tail, pooled over honest-only seeds. Frozen to `eta_calibrated.json`, reused everywhere.

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_all.sh honest    # all-honest, multi-seed
# ...wait for jobs...
./run_all.sh calibrate                              # -> $RES/eta_calibrated.json
python scripts/plots.py thresholds --in "$RES/*/result.json" --family honest_iid \
    --out "$RES/figs"                               # prove the line is right
```
Attack runs read it automatically (`WM_ETA_FIXED`); no hardcoding.

---

## Experiments (one knob at a time)

```bash
./run_all.sh attacks     # tap_every (+5/common, full) + tap_stay (coast-to-stay)
./run_all.sh PLOTALL     # timeline + class_dynamics + positions + thresholds + fidelity + honest_fpr
```
Vary ONE knob per batch — position (`POS=3,6` vs `1,7`), data-per-tap
(`AUTOP_COMMON_PER_CLASS`), coast (`AUTOP_STAY_MIN`). Full plan + expected results in
**STATUS.md -> EXPERIMENTS TO RUN**. Story + pseudocode in **STORYLINE.md**.

---

## Config knobs

Every tunable is a field on `ExpConfig` (`config.py`), overridable via `--flag`
(run_experiment) or `ENV=val` (submit_experiment.sh). Key ones:
`attack` (none/previous_models/gaussian/submarine), `free_rider_ids`, `partition`/
`dirichlet_alpha`, the `autop_*` submarine knobs (warmup, calib, eta_mode, scope,
common_per_class, stay_min, margin0/safety), and `wm_eta_fixed`. Full table in CODE_MAP §8.

---

## Reproduction (stages 1-4, still valid)

Configs 0-13 reproduce the paper (FedAvg baselines, crude free-riders, watermark
fidelity/detection). `python scripts/run_experiment.py --list_configs`. Stage-1
accuracy gates live in each config's `expected_acc`.