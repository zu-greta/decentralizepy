#!/usr/bin/env bash
# =====================================================================
# run_weekend_batch.sh — queue the whole adaptive-attack matrix and walk away.
#
# Every job is submitted with WAIT=0 (fire-and-forget) so they QUEUE on RunAI.
# Jobs are emitted in PRIORITY ORDER: if the scheduler only gets through the
# first N on one GPU, you still come back to the most important results.
#
# Usage:
#   ./run_weekend_batch.sh            # submit everything
#   TIER=1 ./run_weekend_batch.sh     # only tier-1 (headline; fits ~1 weekend/1 GPU)
#   DRY=1  ./run_weekend_batch.sh     # print the commands, submit nothing
#
# Rough cost: one 50-round CIFAR-100 ResNet-18 run (10 clients) ~= 2.5-3 h on an
# A100 (honest clients dominate runtime regardless of the attack). Tier 1 is
# ~15 runs; tiers 1-3 together are ~80 runs. If you have >1 GPU of quota, submit
# all tiers — RunAI will parallelize. On a single GPU, run TIER=1 (and maybe 2).
#
# Prereq: COMMIT + PUSH the fixed faremark_greta/ to the branch first (the pod
# clones fresh). Then from the repo root: ./run_weekend_batch.sh
# =====================================================================
set -euo pipefail
TIER="${TIER:-9}"          # submit tiers <= this number
DRY="${DRY:-0}"
SEEDS_HEADLINE="0 1 2"
SEEDS_SWEEP="0 1"          # 2 seeds for sweeps to save GPU; 3 for headline plots

sub () {  # sub <cfg> <repeat> ENV=VAL ENV=VAL ...
  local cfg="$1" rep="$2"; shift 2
  if [ "$DRY" = "1" ]; then echo "[DRY] $* ROUNDS=50 WAIT=0 ./submit_experiment.sh $cfg $rep";
  else env "$@" ROUNDS=50 WAIT=0 ./submit_experiment.sh "$cfg" "$rep"; fi
}

echo "############ TIER 1 — headline: does the fixed attack evade, and how cheap? ############"
# V — validation of the fixed submarine (full-shard warmup), option 1 (paper's assumption)
for R in $SEEDS_HEADLINE; do
  sub 14 $R ATTACK=submarine CALIB_ON_ALL=0 SUB_WARMUP=8 \
      FAMILY=A7_submarine SWEEP_VAR=calib_on_all NOTE="validation 50r opt1 fixed-warmup"
done
# E2a — the MONEY PLOT: memory-exploit warmup sweep = amortized-cost knee.
#       (memory-exploit is the clean lower bound: embed W rounds, replay forever.)
for W in 2 5 8 12; do for R in $SEEDS_HEADLINE; do
  sub 15 $R ATTACK=memory_exploit WARMUP_ROUNDS=$W \
      FAMILY=E2_effort SWEEP_VAR=warmup_rounds NOTE="memexploit warmup=$W (effort knee)"
done; done

if [ "$TIER" -ge 2 ]; then
echo "############ TIER 2 — the comparisons you asked for ############"
# E1 — coast mechanism: only the memory-borne coast should evade; the static
#       fabricators (previous_models, gaussian) coast on NO mark -> caught.
for R in $SEEDS_SWEEP; do
  sub 14 $R ATTACK=submarine        CALIB_ON_ALL=0 SUB_WARMUP=8 FAMILY=E1_coast SWEEP_VAR=attack NOTE="coast=submarine"
  sub 15 $R ATTACK=memory_exploit   WARMUP_ROUNDS=8            FAMILY=E1_coast SWEEP_VAR=attack NOTE="coast=frozen-replay"
  sub 13 $R ATTACK=previous_models                            FAMILY=E1_coast SWEEP_VAR=attack NOTE="coast=extrapolated-global (no mark)"
  sub 13 $R ATTACK=gaussian                                   FAMILY=E1_coast SWEEP_VAR=attack NOTE="coast=noise (no mark)"
done
# E2b — submarine warmup sweep (its own effort dial, complements E2a)
for W in 3 5 8; do for R in $SEEDS_SWEEP; do
  sub 14 $R ATTACK=submarine CALIB_ON_ALL=0 SUB_WARMUP=$W \
      FAMILY=E2_effort SWEEP_VAR=sub_warmup NOTE="submarine warmup=$W"
done; done
# E3 — threshold option 1 vs 2 (does poisoning eta help the attacker?)
for CAL in 0 1; do for R in $SEEDS_SWEEP; do
  sub 14 $R ATTACK=submarine CALIB_ON_ALL=$CAL SUB_WARMUP=8 \
      FAMILY=A7_submarine SWEEP_VAR=calib_on_all NOTE="option $((CAL+1))"
done; done
# E7 — CONTROLLED PROOF: "you need the whole shard to embed a generalizing mark".
#      Vary ONLY the attacker's training data composition, measure server BER
#      (on the server's held-out TEST triggers). Few trigger samples overfit
#      (BER stays ~0.5); full shard generalizes (BER -> ~0.05). This is the
#      rigorous, non-confounded version of figs/proof_fullshard_vs_triggeronly.png
#      and reproduces the paper's Table V on CIFAR-100.
for NS in 2 8 32; do for R in $SEEDS_SWEEP; do
  sub 13 $R ATTACK=trigger_only N_TRIGGER_SAMPLES=$NS \
      FAMILY=E7_embed_composition SWEEP_VAR=n_trigger_samples NOTE="trigger-only, $NS samples"
done; done
for R in $SEEDS_SWEEP; do
  sub 13 $R ATTACK=mixed FULL_TRIGGER_CLASS=1 N_COMMON_SAMPLES=50 BLEND=1.0 \
      FAMILY=E7_embed_composition SWEEP_VAR=n_trigger_samples NOTE="all trigger + 50 common"
  sub 13 $R ATTACK=none \
      FAMILY=E7_embed_composition SWEEP_VAR=n_trigger_samples NOTE="full shard (honest benign BER = the reference)"
done
fi

if [ "$TIER" -ge 3 ]; then
echo "############ TIER 3 — environment sweeps + forgery baseline ############"
# E4 — IID vs non-IID (opposing effects: skew inflates eta but starves the trigger class)
for A in 0.1 0.5 100; do for R in $SEEDS_SWEEP; do
  sub 14 $R ATTACK=submarine PARTITION=dirichlet DIRICHLET_ALPHA=$A CALIB_ON_ALL=1 SUB_WARMUP=8 \
      FAMILY=E4_noniid SWEEP_VAR=dirichlet_alpha NOTE="dirichlet alpha=$A"
done; done
# E5 — freshness/staleness dial
for MBG in 0.0 0.2 0.5; do for R in $SEEDS_SWEEP; do
  sub 14 $R ATTACK=submarine MEM_BLEND_GLOBAL=$MBG CALIB_ON_ALL=1 SUB_WARMUP=8 \
      FAMILY=A7_submarine SWEEP_VAR=mem_blend_global NOTE="mem_blend=$MBG"
done; done
# E6 — bit budget (cheaper evasion where embedding is easy)
for DS in cifar10 cifar100; do for R in $SEEDS_SWEEP; do
  sub 14 $R ATTACK=submarine DATASET=$DS CALIB_ON_ALL=0 SUB_WARMUP=8 \
      FAMILY=E6_bitbudget SWEEP_VAR=dataset NOTE="dataset=$DS"
done; done
# MIXED — forgery baseline for comparison (a different cheap route: blend, not embed)
for B in 0.2 0.5 0.8; do for R in $SEEDS_SWEEP; do
  sub 13 $R ATTACK=mixed FULL_TRIGGER_CLASS=1 N_COMMON_SAMPLES=50 BLEND=$B \
      FAMILY=A5_mixed SWEEP_VAR=blend NOTE="forgery blend=$B"
done; done
# Static anchors + all-honest fidelity (for the effort plane's reference points)
for R in $SEEDS_SWEEP; do
  sub 13 $R ATTACK=previous_models FAMILY=baseline_prev SWEEP_VAR=none NOTE="static anchor"
  sub 13 $R ATTACK=none            FAMILY=fidelity      SWEEP_VAR=none NOTE="all-honest fidelity"
done
fi

echo "############ submitted (TIER<=$TIER). Check: runai list jobs ############"
echo "On Monday, sync \$RES and run the plot commands in ADAPTIVE_ATTACKS.md 'Making the figures'."