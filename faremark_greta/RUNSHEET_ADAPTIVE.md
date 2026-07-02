# RUNSHEET — adaptive attacks (A7 submarine, A8 memory-exploit)

Extends `RUNSHEET_CLUSTER.md` with the effort-minimizing free-riders. Same rules
(flags → UPPERCASE env vars, `config_idx`/`repeat` positional, `WAIT=0` to queue,
`TAG=` to make the dir findable). Two additions specific to this work:

- **Self-describing runs.** Add `FAMILY=…`, `SWEEP_VAR=…`, optional `NOTE="…"`.
  These are stamped into `result.json.manifest` and are what `plot_adaptive.py`
  groups on. `SWEEP_LEVEL` is inferred from the config if you name a `SWEEP_VAR`.
- **New env vars:** `SUB_MARGIN`, `SUB_FLOOR`, `SUB_ETA_MODE` (`adaptive|fixed`),
  `SUB_ETA_FIXED`, `SUB_MAX_BURST_BATCHES`, `SUB_PROBE_EVERY`, `WARMUP_ROUNDS`,
  `MEM_BLEND_GLOBAL`. All optional; defaults live in `config.py`.

Configs: **14** = submarine (paper-faithful CIFAR-100, 2 FR); **15** =
memory-exploit (same). Both already set `PAPER_FAITHFUL=1` and `WATERMARK` via the
config, so you do not need to pass those.

`$RES` = your results dir on the PVC. Plotting is local/interactive, not a job.

------------------------------------------------------------------------
## A7 — Submarine (adaptive threshold-tracking)

Run **both** threshold options × 3 seeds. Option 1 = attacker guesses η
(`CALIB_ON_ALL=0`); option 2 = η poisoned by the attacker (`CALIB_ON_ALL=1`).

```bash
for CAL in 0 1; do for R in 0 1 2; do
  ATTACK=submarine CALIB_ON_ALL=$CAL \
    FAMILY=A7_submarine SWEEP_VAR=calib_on_all NOTE="submarine, opt$((CAL+1))" \
    TAG=a7-cal$CAL WAIT=0 ./submit_experiment.sh 14 $R
done; done
```

Freshness/staleness trade-off sweep (how much global to blend into a coast):
```bash
for MBG in 0.0 0.3 0.6; do for R in 0 1 2; do
  ATTACK=submarine CALIB_ON_ALL=1 MEM_BLEND_GLOBAL=$MBG \
    FAMILY=A7_submarine SWEEP_VAR=mem_blend_global \
    TAG=a7-mbg$MBG WAIT=0 ./submit_experiment.sh 14 $R
done; done
```

Aggressiveness sweep (how close to η it sails — smaller margin = cheaper, riskier):
```bash
for M in 0.02 0.05 0.10; do for R in 0 1 2; do
  ATTACK=submarine CALIB_ON_ALL=1 SUB_MARGIN=$M \
    FAMILY=A7_submarine SWEEP_VAR=sub_margin \
    TAG=a7-m$M WAIT=0 ./submit_experiment.sh 14 $R
done; done
```

η-guess mode (does the attacker even need to model η well?):
```bash
for MODE in adaptive fixed; do for R in 0 1 2; do
  ATTACK=submarine CALIB_ON_ALL=0 SUB_ETA_MODE=$MODE SUB_ETA_FIXED=0.25 \
    FAMILY=A7_submarine SWEEP_VAR=sub_eta_mode \
    TAG=a7-eta$MODE WAIT=0 ./submit_experiment.sh 14 $R
done; done
```

## A8 — Memory-exploit (train once, replay frozen mark)

```bash
for W in 1 3 5; do for R in 0 1 2; do
  ATTACK=memory_exploit WARMUP_ROUNDS=$W \
    FAMILY=A8_memory_exploit SWEEP_VAR=warmup_rounds NOTE="replay frozen mark" \
    TAG=a8-w$W WAIT=0 ./submit_experiment.sh 15 $R
done; done
```

Freshness of the replay (0 = pure frozen memory → max staleness; >0 = blend global):
```bash
for MBG in 0.0 0.2 0.5; do for R in 0 1 2; do
  ATTACK=memory_exploit MEM_BLEND_GLOBAL=$MBG \
    FAMILY=A8_memory_exploit SWEEP_VAR=mem_blend_global \
    TAG=a8-mbg$MBG WAIT=0 ./submit_experiment.sh 15 $R
done; done
```

## Anchor: static free-rider (expensive-to-catch = never trains, always caught)
```bash
for R in 0 1 2; do
  ATTACK=previous_models FAMILY=previous_models NOTE="static strawman" \
    TAG=anchor-prev WAIT=0 ./submit_experiment.sh 13 $R
done
```

------------------------------------------------------------------------
## Plots (local, after syncing `$RES`)

The **effort plane** is the headline figure — attacker effort vs detection,
std over seeds, single axis:
```bash
python scripts/plot_adaptive.py effort \
  --in "$RES/*a7-cal1*" "$RES/*a8-w1*" "$RES/*anchor-prev*" \
  --out figs/effort_plane --effort gpu --metric wm_fr_recall
```

Squeezing (honest vs FR BER bands + η, and the shrinking margin):
```bash
python scripts/plot_adaptive.py squeezing --in "$RES/*a7-cal1*" --out figs/a7_squeeze
```

Sweeps (recall or FPR vs the swept knob):
```bash
python scripts/plot_adaptive.py sweep --in "$RES/*a7-mbg*" \
  --sweep_var mem_blend_global --metric wm_fr_recall --out figs/a7_mbg
python scripts/plot_adaptive.py sweep --in "$RES/*a8-w*" \
  --sweep_var warmup_rounds --metric wm_fr_recall --out figs/a8_warmup
```

Submarine duty cycle (when it taps vs coasts, BER vs η over rounds; use one seed):
```bash
python scripts/plot_adaptive.py duty --in "$RES/*a7-cal1*rep0*" --out figs/a7_duty
```

## What to look for in `result.json`
- `compute.summary.effort_ratio_gpu` — free-rider GPU-time ÷ honest GPU-time.
  **≪ 1** is the claim.
- `compute.summary.effort_ratio_samples` — same, device-independent (use this on
  the effort-plane x-axis when comparing across machines).
- `wm_fr_recall` — should collapse toward **0** for A7/A8 while the anchor stays
  near 1.
- `wm_fpr` — watch honest clients are not collaterally flagged (esp. option 2,
  where a poisoned η can *lower* FPR as a side effect).
- per-free-rider `compute.per_client[cid].trace` — the tap/coast log (A7) or
  embed/replay log (A8), and `duty_cycle` in its `total`.

## Notes / caveats
- If BER columns come back empty, the watermark path was off — configs 13–15
  turn it on, but if you switch to config 12 add `WATERMARK=1`.
- `effort_ratio_*` divides by the **honest** mean; it is only meaningful when at
  least one honest watermark client exists in the run (true for all these).
- FLOPs (`flops`, and thus a FLOPs-based effort axis) are populated only if
  `fvcore`, `thop`, or `ptflops` is installed in the image; otherwise use `gpu`
  or `samples` for `--effort`. GPU-ms and samples are always populated.
