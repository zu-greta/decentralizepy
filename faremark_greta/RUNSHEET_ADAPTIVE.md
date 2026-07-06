# RUNSHEET 

Rules:
- Flags map to UPPERCASE env vars: `--attack x` -> `ATTACK=x`,
  `--num_free_riders N` -> `NUM_FREE_RIDERS=N`, `--attack_round R` -> `ATTACK_ROUND=R`,
  `--n_trigger_samples NS` -> `N_TRIGGER_SAMPLES=NS`, `--honest_prob HP` -> `HONEST_PROB=HP`,
  `--blend B` -> `BLEND=B`, `--full_trigger_class` -> `FULL_TRIGGER_CLASS=1`,
  `--n_common_samples NC` -> `N_COMMON_SAMPLES=NC`, `--partition p` -> `PARTITION=p`,
  `--dirichlet_alpha A` -> `DIRICHLET_ALPHA=A`, `--dataset d` -> `DATASET=d`,
  `--wm_bits m` -> `WM_BITS=m`, `--calib_on_all` -> `CALIB_ON_ALL=1`.
- `--config_idx` and `--repeat` are the two positional args
- `--output_dir` is auto on the cluster
- `TAG=<slot>` to suffix the run dir so it is findable later (dir becomes `cfg12_rep0-<TAG>_<timestamp>`)
- `WAIT=0` = fire-and-forget (queue many jobs). Omit for a single blocking run.
- If any run comes back with EMPTY BER columns, the watermark path was off — add
  `WATERMARK=1` (the make_* attacks need it).
- self-describing runs: Add `FAMILY=…`, `SWEEP_VAR=…`, optional `NOTE="…"`.
  These are stamped into `result.json.manifest` and are what `plot_adaptive.py`
  groups on. `SWEEP_LEVEL` is inferred from the config if you name a `SWEEP_VAR`.

`$RES` below = your results dir on the PVC (…/home/zu/results). 
Plotting is done locally — run `plot_results.py` locally after syncing `$RES`

------------------------------------------------------------------------
## CURRENT: one sweep for every attack's weak point (prefer this)

Read STATUS.md first. The single command that runs the whole weak-point sweep
(submarine warmup / samples / coast-type, memory warmup, reembed frontier),
priority-ordered, WAIT=0, ~23 runs at 1 seed:

```bash
./scripts/run_full_sweep.sh              # SEEDS="0 1" for tighter bands
RES=/mnt/nfs/home/zu/results ./scripts/make_sweep_figs.sh   # all figures
```

Individual pieces if you want to run just one:

```bash
# REEMBED frontier (the theoretically-motivated attack): scope × steps
for SC in head block full; do for ST in 10 40 100; do
  ROUNDS=50 ATTACK=reembed REEMBED_SCOPE=$SC REEMBED_STEPS=$ST \
    FAMILY=R_frontier SWEEP_VAR=reembed_effort NOTE="reembed $SC×$ST" \
    WAIT=0 ./submit_experiment.sh 16 0
done; done

# MEMORY warmup (Q: good point + effort for evasion)
for W in 2 5 8 12; do
  ROUNDS=50 ATTACK=memory_exploit WARMUP_ROUNDS=$W \
    FAMILY=M_warmup SWEEP_VAR=warmup_rounds NOTE="memory warmup=$W" \
    WAIT=0 ./submit_experiment.sh 15 0
done

# SUBMARINE warmup (Q: rounds to fall under η)
for W in 3 8 12; do
  ROUNDS=50 ATTACK=submarine SUB_WARMUP=$W SUB_COAST_MODE=blend \
    FAMILY=S_warmup SWEEP_VAR=sub_warmup NOTE="submarine warmup=$W" \
    WAIT=0 ./submit_experiment.sh 14 0
done

# SUBMARINE coast type (Q: doing-nothing/noise/replay/blend/transplant)
for CM in replay blend transplant noise global; do
  ROUNDS=50 ATTACK=submarine SUB_WARMUP=8 SUB_COAST_MODE=$CM \
    FAMILY=S_coast SWEEP_VAR=sub_coast_mode NOTE="coast=$CM" \
    WAIT=0 ./submit_experiment.sh 14 0
done
```

Read the results by **`wm_fr_ber`** (below η = evades) and **`final_acc`** (~72 =
healthy, low = poisoned), not `wm_fr_recall`. The money figure is
`figs/weakpoint_all.png`.

------------------------------------------------------------------------
## OLDER per-family commands (kept for reference; the sweep above supersedes them)
------------------------------------------------------------------------
## 7.0  Robustness (finish baseline)
```bash
SCRIPT=scripts/run_robustness.py TAG=robust WAIT=0 ./submit_experiment.sh 11 0
```

## A1 — Threshold fragility (previous-model fraction)
```bash
for N in 2 4 6 8; do
  NUM_FREE_RIDERS=$N ATTACK=previous_models TAG=a1-frac$N WAIT=0 ./submit_experiment.sh 12 0
done
# plot (local): 
python scripts/plot_results.py --in $RES/*a1-frac2* $RES/*a1-frac4* $RES/*a1-frac6* $RES/*a1-frac8* --out figs/a1_prevmodel_frac
cp figs/a1_prevmodel_frac/sweep.png UPLOAD__a1_prevmodel_frac__effort.png
```

## A2 — Train-then-attack (attack_round)
```bash
for R in 0 10 20 30 40; do
  ATTACK=train_then_attack ATTACK_ROUND=$R TAG=a2-ar$R WAIT=0 ./submit_experiment.sh 12 0
done
python scripts/plot_results.py --in $RES/*a2-ar0* $RES/*a2-ar10* $RES/*a2-ar20* $RES/*a2-ar30* $RES/*a2-ar40* --out figs/a2_ttattack_round
cp figs/a2_ttattack_round/sweep.png UPLOAD__a2_ttattack_round__effort.png
```

## A3 — Trigger-only (n_trigger_samples)
```bash
for NS in 2 8 32 128; do
  ATTACK=trigger_only N_TRIGGER_SAMPLES=$NS TAG=a3-ns$NS WAIT=0 ./submit_experiment.sh 12 0
done
python scripts/plot_results.py --in $RES/*a3-ns2* $RES/*a3-ns8* $RES/*a3-ns32* $RES/*a3-ns128* --out figs/a3_triggeronly_ns
cp figs/a3_triggeronly_ns/sweep.png UPLOAD__a3_triggeronly_ns__effort.png
```

## A4 — Random-round (honest_prob)   [needs SWEEP_KEYS patch]
```bash
for HP in 0.2 0.4 0.6 0.8; do
  ATTACK=random_round HONEST_PROB=$HP NUM_FREE_RIDERS=2 TAG=a4-hp$HP WAIT=0 ./submit_experiment.sh 12 0
done
python scripts/plot_results.py --in $RES/*a4-hp0.2* $RES/*a4-hp0.4* $RES/*a4-hp0.6* $RES/*a4-hp0.8* --out figs/a4_randomround_hp
cp figs/a4_randomround_hp/sweep.png UPLOAD__a4_randomround_hp__effort.png
# single run (just to run random-round once):
ATTACK=random_round HONEST_PROB=0.5 NUM_FREE_RIDERS=2 TAG=a4-hp0.5 WAIT=0 ./submit_experiment.sh 12 0
```

## A5 — Mixed forgery (disguise effort)   [needs SWEEP_KEYS patch for the curve]
```bash
for NC in 0 20 50 100; do
  ATTACK=mixed FULL_TRIGGER_CLASS=1 N_COMMON_SAMPLES=$NC BLEND=0.5 TAG=a5-nc$NC WAIT=0 ./submit_experiment.sh 12 0
done
python scripts/plot_results.py --in $RES/*a5-nc0* $RES/*a5-nc20* $RES/*a5-nc50* $RES/*a5-nc100* --out figs/a5_mixed_effort
cp figs/a5_mixed_effort/sweep.png UPLOAD__a5_mixed_effort__effort.png
```

## A6 — Collusion (colluder count)   [implement collusion_attack_scaffold.py first]
```bash
for K in 2 3 4 6; do
  ATTACK=collusion NUM_FREE_RIDERS=$K TAG=a6-k$K WAIT=0 ./submit_experiment.sh 12 0
done
python scripts/plot_results.py --in $RES/*a6-k2* $RES/*a6-k3* $RES/*a6-k4* $RES/*a6-k6* --out figs/a6_collusion_frac
cp figs/a6_collusion_frac/sweep.png UPLOAD__a6_collusion_frac__effort.png
```

## C1 — Bit-count ceiling (CIFAR-10 vs CIFAR-100)
```bash
ATTACK=mixed FULL_TRIGGER_CLASS=1 N_COMMON_SAMPLES=50 DATASET=cifar10  TAG=c1-c10  WAIT=0 ./submit_experiment.sh 12 0
ATTACK=mixed FULL_TRIGGER_CLASS=1 N_COMMON_SAMPLES=50 DATASET=cifar100 TAG=c1-c100 WAIT=0 ./submit_experiment.sh 12 0
python scripts/plot_results.py --in $RES/*c1-c10* $RES/*c1-c100* --out figs/c1_bitcount_ds
cp figs/c1_bitcount_ds/sweep.png UPLOAD__c1_bitcount_ds__effort.png
```

## L1 — Non-IID false positives (Dirichlet, all honest)   [needs SWEEP_KEYS patch]
```bash
for A in 0.1 0.5 1 100; do
  NUM_FREE_RIDERS=0 PARTITION=dirichlet DIRICHLET_ALPHA=$A TAG=l1-a$A WAIT=0 ./submit_experiment.sh 12 0
done
python scripts/plot_results.py --in $RES/*l1-a0.1* $RES/*l1-a0.5* $RES/*l1-a1* $RES/*l1-a100* --out figs/l1_noniid_alpha
cp figs/l1_noniid_alpha/sweep.png UPLOAD__l1_noniid_alpha__effort.png
```

## Bit-budget knob sweep (optional, for the S22 argument directly)
```bash
for M in 2 4 8 16 49; do
  WM_BITS=$M ATTACK=mixed FULL_TRIGGER_CLASS=1 N_COMMON_SAMPLES=50 TAG=bits-m$M WAIT=0 ./submit_experiment.sh 12 0
done
python scripts/plot_results.py --in $RES/*bits-m2* $RES/*bits-m4* $RES/*bits-m8* $RES/*bits-m16* $RES/*bits-m49* --out figs/bits_sweep
```

## Repeats (10 seeds) — credibility
```bash
for R in $(seq 0 9); do
  ATTACK=previous_models NUM_FREE_RIDERS=2 TAG=rep WAIT=0 ./submit_experiment.sh 12 $R
done
# then locally: python scripts/aggregate_results.py $RES
```

------------------------------------------------------------------------
## A7 — Submarine (adaptive threshold-tracking)

> STATUS: preliminary runs show the submarine is caught (blend coast → mark
> decays) or poisons (replay coast → stale). The `mem_blend_global` sweep below
> is now the *freshness-vs-decay* diagnostic (E5), not a "cheap evasion" win. The
> coast-type comparison lives in the CURRENT sweep (`S_coast`). Read `fr_ber` +
> `final_acc`, not recall.

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

> STATUS: preliminary runs show memory-exploit evades η (fr_ber ~0.15 at
> warmup≥5) BUT poisons the global (acc 72→55, honest BER→0.5). The warmup sweep
> below is the effort-vs-poisoning curve. Read `fr_ber` + `final_acc`.

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