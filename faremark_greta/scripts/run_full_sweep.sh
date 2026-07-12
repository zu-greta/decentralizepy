#!/usr/bin/env bash
# run_full_sweep.sh — STAY-UNDER test: given a threshold, CAN the free-rider stay
# under it with taps, and how cheaply?  (replaces the effort-minimising sweep)
#
# ─────────────────────────────────────────────────────────────────────────────
# NAMING (read this — the old names confused us):
#   Each arm is  <eta-source>_<scope>[_<data>]  where
#     eta-source : oracle = FR is GIVEN the true fair η (~0.09);  est = FR ESTIMATES η.
#     scope      : which PARAMETERS a tap re-trains — the compute lever:
#                    full   = whole model (like honest; strongest, most backprop)
#                    block2 = last two ResNet stages + head (backbone frozen; less GPU)
#     data       : which IMAGES a tap trains on — the other compute lever
#                    (only on the data-ablation arms; default = full shard).
#   So `oracle_full` = given η, retrain the WHOLE model on the FULL shard each tap.
#      `oracle_block2` = given η, retrain only the last two stages.
#   (There is no "full_block2"/"block2_block2" — the two tokens are eta-source and
#    scope, not two scopes. That earlier double-"full" naming was the bug.)
#
# THE FIX under test (attacks_adaptive.py, AUTOP_STAY_UNDER / auto-on under oracle):
#   PRIORITISE STAYING UNDER η, THEN be cheap. Every post-warmup round the FR
#   re-embeds on the FRESH global with a FIXED honest-style budget (local_epochs
#   passes over the selected shard; NO probe early-stop; NO dynamic tap sizing).
#   The probe (which overfits the FR's own shard and reads ~0 while the server reads
#   ~0.1) is NOT allowed to gate training. Cost is then a clean function of SCOPE and
#   DATA — not the probe. Taps are FIXED-size, so "samples per tap" is constant
#   (= local_epochs × |shard| image-passes), which is the honest per-round cost.
#
# WHY oracle arms need NO new flag: stay-under auto-enables whenever AUTOP_ORACLE_ETA>0.
# The est_* arms DO pass AUTOP_STAY_UNDER=1 → they need the one-line run_experiment.py
# addition (see run_experiment_ADDITIONS.md). Without it the est_* jobs error (the
# oracle jobs still run fine, since WAIT=0 fire-and-forget).
#
#   ./run_full_sweep.sh              # submit (SEEDS="0 1 2")
#   RES=/path ./run_full_sweep.sh PLOT
set -uo pipefail
SEEDS="${SEEDS:-0 1 2}"; RES="${RES:-/mnt/nfs/home/zu/results}"
PT="python scripts/plot_thresholds.py"; SB="python scripts/seedband.py"; ORACLE=0.09
# data-ablation hops on the x-axis: triggers-only(0) → +N/common-class → full shard(-1)
CPC_HOPS="${CPC_HOPS:-0 5 10 20 50 -1}"
# non-IID Dirichlet skews to test (smaller = more skew; >=100 ≈ IID)
ALPHAS="${ALPHAS:-0.1 0.5 1.0}"

if [ "${1:-}" = "PLOT" ]; then
  OUT="${OUT:-figs}"; mkdir -p "$OUT"; ALL="$RES/*/result.json"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

  # ============ GO/NO-GO: does the FR stay UNDER the fair η now? (all arms) ============
  ARMS="oracle_full oracle_block2 est_full est_block2"
  run $PT evade_bars --in "'$ALL'" --family $ARMS --out "$OUT/final_evade"
  run $PT worth      --in "'$ALL'" --family $ARMS --out "$OUT/final_worth"
  run $PT meters     --in "'$ALL'" --family $ARMS --out "$OUT/meters"

  # ============ PER-ARM: BER vs rounds + forced/coast-tap markers + effort ============
  for FAM in $ARMS; do
    run $PT timeline  --in "'$ALL'" --family $FAM --out "$OUT/timeline_$FAM"
    run $PT submarine --in "'$ALL'" --family $FAM --out "$OUT/submarine_$FAM"
  done
  # seed-bands (mean±std; shows FR BER vs the fair η and the attacker's η line)
  run $SB --in "'$ALL'" --note "'oracle full stayunder 3seed'"   --title "'oracle · full scope · stay-under (3 seeds)'"   --out "$OUT/seedband_oracle_full"
  run $SB --in "'$ALL'" --note "'oracle block2 stayunder 3seed'" --title "'oracle · block2 scope · stay-under (3 seeds)'" --out "$OUT/seedband_oracle_block2"
  run $SB --in "'$ALL'" --note "'est full stayunder 3seed'"      --title "'estimated η · full scope · stay-under (3 seeds)'"   --out "$OUT/seedband_est_full"
  run $SB --in "'$ALL'" --note "'est block2 stayunder 3seed'"    --title "'estimated η · block2 scope · stay-under (3 seeds)'" --out "$OUT/seedband_est_block2"

  # ============ DATA ABLATION: triggers-only → +N/class → full shard (x-axis) ============
  # BER (does it stay under η?) on top, effort (does more data cost more?) on the bottom.
  run $PT knob --in "'$ALL'" --family data_oracle_full   --sweep_var autop_common_per_class --out "$OUT/data_oracle_full"
  run $PT knob --in "'$ALL'" --family data_oracle_block2 --sweep_var autop_common_per_class --out "$OUT/data_oracle_block2"

  # ============ DIAGNOSTIC: honest-clone vs the autopilot embedder ============
  # If clone_full's FR band sits at the honest floor (~0.04) while oracle_full sits at
  # ~0.11, the gap was the autopilot embedder (attack revivable). If clone_full ALSO
  # plateaus ~0.11, the gap is fundamental (late-join / dynamics).
  run $SB --in "'$ALL'" --note "'clone full stayunder 3seed'" --title "'DIAGNOSTIC: honest-clone embed · full scope (3 seeds)'" --out "$OUT/seedband_clone_full"
  run $PT evade_bars --in "'$ALL'" --family oracle_full clone_full --out "$OUT/diag_clone_vs_autopilot"

  # ============ NON-IID: one seedband per α, plus an IID-vs-α evasion comparison ============
  for A in $ALPHAS; do
    run $SB --in "'$ALL'" --note "'noniid full a=$A'" --title "'non-IID (Dirichlet α=$A) · full scope · stay-under'" --out "$OUT/seedband_noniid_a$A"
  done
  # IID (oracle_full) vs each non-IID α, side by side: does looser η let the FR under?
  run $PT evade_bars --in "'$ALL'" --family oracle_full $(for A in $ALPHAS; do echo -n "noniid_full_a$A "; done) --out "$OUT/noniid_vs_iid_evade"
  run $PT knob --in "'$ALL'" --family $(for A in $ALPHAS; do echo -n "noniid_full_a$A "; done) --sweep_var dirichlet_alpha --out "$OUT/noniid_alpha_sweep"

  # ============ FALSE-POSITIVE: per-client BER distribution vs eta ============
  # KEY finding: is the FR's ~0.11 floor INSIDE the honest spread? If honest clients
  # at hard trigger classes also exceed eta, the detector false-positives on honest.
  run $PT fpr --in "'$ALL'" --family oracle_full clone_full --out "$OUT/fpr_iid_full"
  for A in $ALPHAS; do
    run $PT fpr --in "'$ALL'" --family noniid_full_a$A --out "$OUT/fpr_noniid_a$A"
  done

  # ============ KEEP-TRYING arms: min-effort + dynamic low-cost ============
  run $SB --in "'$ALL'" --note "'oracle full min-effort 3seed'" --title "'oracle · full · MIN-EFFORT (coast when safe)'" --out "$OUT/seedband_oracle_full_min"
  run $SB --in "'$ALL'" --note "'dynamic lowcost margin12 3seed'" --title "'dynamic low-cost (holdout .25, margin .12)'" --out "$OUT/seedband_dyn_lowcost"
  for FAM in oracle_full_min dyn_lowcost; do
    run $PT submarine --in "'$ALL'" --family $FAM --out "$OUT/submarine_$FAM"
    run $PT timeline  --in "'$ALL'" --family $FAM --out "$OUT/timeline_$FAM"
  done
  run $PT worth --in "'$ALL'" --family oracle_full oracle_full_min dyn_lowcost clone_full --out "$OUT/keeptrying_worth"

  # ============ CONTROLS: all-honest FPR + collusion/poisoning ============
  run $PT fpr --in "'$ALL'" --family all_honest --out "$OUT/fpr_all_honest"
  run $SB --in "'$ALL'" --note "'poison eta calibonall fr5'" --title "'η-poisoning (calib on all, 5 free-riders)'" --out "$OUT/seedband_poison_eta"
  run $PT evade_bars --in "'$ALL'" --family oracle_full collude_fr3 collude_fr5 poison_eta --out "$OUT/collusion_evade"
  run $PT fpr --in "'$ALL'" --family poison_eta --out "$OUT/fpr_poison_eta"

  echo
  echo "READ ORDER:"
  echo "  1) final_evade.png — under the fair (frozen/converged) η, are the bars LOW now"
  echo "                       (FR caught) or does stay-under push them under? decides the fix."
  echo "  2) seedband_oracle_full.png — the money plot: FR BER band should sit BELOW the"
  echo "                       green fair η the whole run (feasibility proven)."
  echo "  3) data_oracle_full.png — how little data still stays under (triggers-only should"
  echo "                       FAIL per paper Table V; +N/class and full shard should pass)."
  echo "  4) final_worth.png / meters.png — how cheap: is it a bit under honest (block2/data)?"
  echo "  5) seedband_est_*.png — is the ESTIMATED η now ON the fair η (~0.09), not ~0.18?"
  exit 0
fi

# ---- submit helper. stay-under taps = local_epochs passes over the shard (fixed) ----
# AUTOP_MAX_BATCHES only bounds the brief warmup transition; stay-under ignores it.
sub(){ local r="$1"; shift; env "$@" ROUNDS=50 CALIB_ON_ALL=0 \
        AUTOP_HONEST_UNTIL=12 AUTOP_HONEST_EXTRA=3 AUTOP_MARGIN0=0.06 AUTOP_MAX_BATCHES=250 \
        WAIT=0 ./submit_experiment.sh 17 "$r"; }

for R in $SEEDS; do
  # ── ORACLE (given true η): stay-under AUTO-ON. full shard, both scopes. ──
  sub $R ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=full   \
        FAMILY=oracle_full   SWEEP_VAR=none NOTE="oracle full stayunder 3seed"
  sub $R ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=block2 \
        FAMILY=oracle_block2 SWEEP_VAR=none NOTE="oracle block2 stayunder 3seed"

  # ── ESTIMATED η + stay-under (validates the estimator fix; needs the flag). ──
  sub $R ATTACK=autopilot AUTOP_STAY_UNDER=1 AUTOP_SCOPE=full   \
        FAMILY=est_full   SWEEP_VAR=none NOTE="est full stayunder 3seed"
  sub $R ATTACK=autopilot AUTOP_STAY_UNDER=1 AUTOP_SCOPE=block2 \
        FAMILY=est_block2 SWEEP_VAR=none NOTE="est block2 stayunder 3seed"

  # ── DATA ABLATION (oracle, stay-under): triggers-only → +N/class → full shard. ──
  for NC in $CPC_HOPS; do
    sub $R ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=full   AUTOP_COMMON_PER_CLASS=$NC \
          FAMILY=data_oracle_full   SWEEP_VAR=autop_common_per_class SWEEP_LEVEL=$NC NOTE="data oracle full cpc=$NC"
    sub $R ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=block2 AUTOP_COMMON_PER_CLASS=$NC \
          FAMILY=data_oracle_block2 SWEEP_VAR=autop_common_per_class SWEEP_LEVEL=$NC NOTE="data oracle block2 cpc=$NC"
  done

  # ── DIAGNOSTIC: honest-clone (embed via the EXACT honest path). Settles Q2:
  #    does the FR reach the honest BER floor (~0.04) or plateau (~0.11)? ──
  sub $R ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=full AUTOP_HONEST_CLONE=1 \
        FAMILY=clone_full SWEEP_VAR=none NOTE="clone full stayunder 3seed"

  # ── KEEP-TRYING #1: STAY-UNDER MIN-EFFORT (coast when safely under; tap only when
  #    needed). Now that taps make a deep, reliable-probe mark, this should stay under
  #    with effort < 1.0 — IF the floor is under eta. Conservative margin 0.02. ──
  sub $R ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=full AUTOP_STAY_MIN=1 AUTOP_MARGIN0=0.02 \
        FAMILY=oracle_full_min SWEEP_VAR=none NOTE="oracle full min-effort 3seed"

  # ── KEEP-TRYING #2: the suggestions.txt DYNAMIC low-cost attack (no stay_under):
  #    smaller holdout (0.25) + wide margin (0.12) + bounded taps. Tests whether the
  #    dynamic controller can ride under cheaply. (Expected to hit the same ~0.11 floor.) ──
  sub $R ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=full \
        AUTOP_HOLDOUT_RATIO=0.25 AUTOP_MARGIN0=0.12 AUTOP_MIN_BATCHES=30 AUTOP_MAX_BATCHES=300 \
        FAMILY=dyn_lowcost SWEEP_VAR=none NOTE="dynamic lowcost margin12 3seed"

  # ── NON-IID (Dirichlet): the promising angle. Higher honest-BER variance → looser
  #    η → the FR's ~0.11 may slip under. One arm per α, plus IID (oracle_full above). ──
  for A in $ALPHAS; do
    sub $R ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=full \
          PARTITION=dirichlet DIRICHLET_ALPHA=$A \
          FAMILY=noniid_full_a$A SWEEP_VAR=dirichlet_alpha SWEEP_LEVEL=$A NOTE="noniid full a=$A"
  done

  # ── CONTROL: ALL-HONEST (attack=none). The definitive "can honest sit under η?"
  #    test — records every honest client's per-class BER with NO free-rider. If some
  #    honest classes read ~0.11, the FR's floor is just a hard position (→ false
  #    positive), not an attack weakness. ──
  sub $R ATTACK=none FAMILY=all_honest SWEEP_VAR=none NOTE="all honest control"

  # ── COLLUSION / RATIO: does the attack work with MORE free-riders? ──
  for NF in 3 5; do
    env NUM_FREE_RIDERS=$NF ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=full \
        ROUNDS=50 CALIB_ON_ALL=0 AUTOP_HONEST_UNTIL=12 AUTOP_HONEST_EXTRA=3 AUTOP_MARGIN0=0.06 \
        AUTOP_MAX_BATCHES=250 WAIT=0 FAMILY=collude_fr$NF SWEEP_VAR=num_free_riders \
        SWEEP_LEVEL=$NF NOTE="collude fr=$NF" ./submit_experiment.sh 17 $R
  done
  # ── η-POISONING: server calibrates η on ALL clients (can't tell honest from FR).
  #    5 free-riders at BER~0.11+ inflate μ+3σ → η rises → they pass. Exploits the
  #    paper's calibration circularity directly. ──
  env NUM_FREE_RIDERS=5 CALIB_ON_ALL=1 ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_SCOPE=full \
      ROUNDS=50 AUTOP_HONEST_UNTIL=12 AUTOP_HONEST_EXTRA=3 AUTOP_MARGIN0=0.06 \
      AUTOP_MAX_BATCHES=250 WAIT=0 FAMILY=poison_eta SWEEP_VAR=none \
      NOTE="poison eta calibonall fr5" ./submit_experiment.sh 17 $R
done
echo "submitted. When done: RES=$RES ./run_full_sweep.sh PLOT"