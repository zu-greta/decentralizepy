# FareMark — reproduction + limitations study

Re-implementation of FareMark: Model-Watermark-Driven Free-Rider Detection in
Federated Learning (Li et al., IEEE IoT-J 2025), extended into a limitations study.
Centralized FedAvg simulated on one GPU + a per-client watermark loss, a
memory-enhanced update, and server-side verification.

---

## Current layout 

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

---

## The threshold 

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
Attack runs read it automatically (`WM_ETA_FIXED`)

### Where files live
- **per run:** `$MOUNT/home/zu/results/<RUN_TAG>/result.json` (+ `run.log`, `pod.log`),
  `RUN_TAG = cfg14_rep<seed>_<timestamp>`.
- **calibration:** `$RES/eta_calibrated.json`.
- **REQUIRED:** set `RES = $MOUNT/home/zu/results` (submit writes under `$MOUNT/...`,
  run_all/calibrate read from `$RES/...`). With `.env` `MOUNT=/mnt/nfs` the default
  `RES=/mnt/nfs/home/zu/results` already matches.

### Double-check the numbers
```bash
python scripts/threshold.py verify --in "$RES/*/result.json" \
    --honest-family honest_iid --eta-file "$RES/eta_calibrated.json"
```
PASS = recomputed eta matches the file and every attack run used the frozen constant
(flat `wm_eta_round` == eta). Also inspect `eta_calibrated.json` (per-seed etas should
agree) and `result["per_class"]` (per-class acc/loss).

---

## Experiments 

```bash
./run_all.sh attacks     # tap_every (+5/common, full) + tap_stay (coast-to-stay)
./run_all.sh PLOTALL     # timeline + class_dynamics + positions + thresholds + fidelity + honest_fpr
```
Vary one knob per batch — class index (`POS=3,6` vs `1,7`), data-per-tap
(`AUTOP_COMMON_PER_CLASS`), coast (`AUTOP_STAY_MIN`)

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