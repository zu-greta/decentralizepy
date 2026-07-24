#!/usr/bin/env bash
# =============================================================================
# run_now.sh -- Builds jobs.tsv for the current batch.
#
#   ./run_now.sh                        # writes jobs.tsv (submits nothing)
#   POOLS="a100-80 a100-40" WORKERS_LIST="6 3" PODS=2 ./submit_pool.sh
#
# 20 runs, 3 seeds each unless noted. See README PERFORMANCE for timing --
# the 50/100/200-client runs are dominated by DataLoader worker churn, not compute.
# GOAL: show that no threshold separates honest clients from free-riders, under
#       the paper's own setup and under three extensions (non-IID, adaptive
#       free-rider, more clients than classes).
#
# Every run is labelled R1..R8. The label is the family prefix, so a figure can
# always be traced back to the command that made it.
# =============================================================================
set -uo pipefail
export DRYRUN=1 JOBS_FILE="${JOBS_FILE:-jobs.tsv}"

# ---------------------------------------------------------------------------
# NUM_WORKERS -- read before changing.
# The DataLoaders use num_workers=2 with persistent_workers unset, so 2 worker
# processes are forked and killed on EVERY iterator: num_clients x local_epochs
# x rounds times per run. For this batch that is ~432,000 fork/teardown cycles;
# at even 0.5 s each that is 60 GPU-hours spent on nothing but process creation.
# It gets WORSE under pod packing: 6 concurrent runs = 12 worker processes
# churning on one node's cores. At 32x32 with batch 16 the actual loading work
# is trivial, so 0 is normally faster.
# TIME ONE RUN BOTH WAYS before trusting this default (see README PERFORMANCE).
# ---------------------------------------------------------------------------
export NUM_WORKERS="${NUM_WORKERS:-0}"

rm -f "$JOBS_FILE"
echo "== building $JOBS_FILE (nothing submitted, NUM_WORKERS=$NUM_WORKERS) =="

# -----------------------------------------------------------------------------
# GROUP B -- does my implementation match the paper?      [plot request (b)]
# -----------------------------------------------------------------------------
# R1  CIFAR-100 / 100 clients / all honest / m=10 / N_T=100
#     Paper Table I+II row: wm 99.71 %, acc 75.31 %.  N_T=100 per Sec. V-C.
#     m=10 -> l=10 -> stuck-bit ceiling 99.90 %, so the paper's number IS
#     reachable with the paper's own random keys. This is the faithful config.
#     100 clients on CIFAR-100 also means cid%100 covers EVERY class, so this
#     single run is also the class-difficulty experiment.   [plot request (c)]
for s in 0 1 2; do
  env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=100 WM_NUM_TRIGGERS=100 BALANCED=0 ROUNDS=50 \
      FAMILY="R1_paper_c100_nc100" \
      NOTE="R1 paper Table I+II CIFAR-100 100 clients, m=10 unbalanced, N_T=100" \
      ./submit_experiment.sh 14 "$s"
done

# R2  CIFAR-10 / 10 clients / all honest / m=1 / N_T=100
#     Paper Table I+II row: wm 99.72 %, acc 90.78 %.
#     m MUST be 1 here. The code default m=max(2,n//10)=2 gives l=5, 6.25 % of
#     key rows all-same-sign, ceiling 96.88 % -- 3pp BELOW the paper, unreachable
#     no matter how long you train. m=1 -> l=10 -> ceiling 99.90 %. So this run
#     also demonstrates that the paper's CIFAR-10 number forces a 1-BIT
#     watermark, i.e. BER is a single coin flip.
for s in 0 1 2; do
  env ATTACK=none NUM_FREE_RIDERS=0 WM_BITS=1 WM_NUM_TRIGGERS=100 BALANCED=0 ROUNDS=50 \
      FAMILY="R2_paper_c10_m1" \
      NOTE="R2 paper Table I+II CIFAR-10 10 clients, m=1 unbalanced, N_T=100" \
      ./submit_experiment.sh 11 "$s"
done

# R3/R4  the paper's own free-riders (Eq. 17 / Eq. 18), 1 seed each.
#        Pure sanity: if these are not cleanly caught, the detector is broken in
#        our code and every negative result below is worthless.
#        Expect FR BER ~0.5, honest ~0.03, recall ~1.0, OVL ~0.
env ATTACK=previous_models NUM_FREE_RIDERS=2 FREE_RIDER_IDS=3,6 ROUNDS=50 \
    FAMILY="R3_crude_prevmodels_c100" NOTE="R3 paper Eq.17 crude free-rider" \
    ./submit_experiment.sh 14 0
env ATTACK=gaussian NUM_FREE_RIDERS=2 FREE_RIDER_IDS=3,6 NOISE_SIGMA=0.1 ROUNDS=50 \
    FAMILY="R4_crude_gaussian_c100" NOTE="R4 paper Eq.18 crude free-rider" \
    ./submit_experiment.sh 14 0

# -----------------------------------------------------------------------------
# GROUP C -- more clients than classes (paper Table IX regime)  [plot (d)]
# -----------------------------------------------------------------------------
# R0  Table IX: CIFAR-10 / 50 clients / client_train / N_T=50 / m=1.
#     Paper row: wm 95.78 %, acc 88.42 %. 5 clients forced onto every trigger class.
#     client_train = "trigger sample consistency" (paper Sec. V-F3): the verifier uses
#     the client's OWN training images. That is memorisation -- exactly what Table V
#     calls a failure mode. The contradiction is the point (README F14).
#     WARNING: if your standalone Table IX job is still running under a DIFFERENT
#     family name, this will NOT be deduplicated and you will run it twice.
for s in 0 1 2; do
  env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=50 ROUNDS=50 \
      WM_TRIGGER_MODE=client_train WM_NUM_TRIGGERS=50 WM_BITS=1 BALANCED=0 \
      FAMILY="R0_paper_t9_nc50" \
      NOTE="R0 paper Table IX CIFAR-10 50 clients, client_train, m=1, N_T=50" \
      ./submit_experiment.sh 11 "$s"
done

# R5  200 clients on CIFAR-100 -> 2 clients forced onto every trigger class.
#     Honest top-up to 6 seeds (you only have 3; eta's own seed-std is ~40 %).
for s in 3 4 5; do
  env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=200 BALANCED=0 ROUNDS=50 \
      FAMILY="R5_honest_nc200" NOTE="R5 honest 200 clients (oversubscription)" \
      ./submit_experiment.sh 14 "$s"
done

# R6  the attack in that regime: free-riders at cid 106,107 land on classes 6,7
#     where honest cid 6,7 already sit. Systemic same-trigger-class overlap.
#     eta only drives live flagging; detection.py recomputes offline.
for s in 0 1 2; do
  env ATTACK=reduced NUM_CLIENTS=200 FREE_RIDER_IDS=106,107 \
      AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
      WM_ETA_FIXED=0.384 BALANCED=0 ROUNDS=50 \
      FAMILY="R6_reduced_nc200_c67" \
      NOTE="R6 reduced +5 at cids 106,107 sharing classes 6,7 with honest" \
      ./submit_experiment.sh 14 "$s"
done

# -----------------------------------------------------------------------------
# GROUP D -- adaptive free-rider                                 [plot (d)]
# -----------------------------------------------------------------------------
# R7  tap_oracle: given the true eta, it COASTS (submits the model untouched,
#     zero compute) whenever its mark is safely under eta and only trains when
#     the mark decays. This is the adaptive attacker; `reduced` is the static one.
# for s in 0 1 2; do
#   env ATTACK=tap_oracle NUM_FREE_RIDERS=2 FREE_RIDER_IDS=3,6 \
#       AUTOP_ORACLE_ETA=0.063 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
#       AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=0.063 BALANCED=0 ROUNDS=50 \
#       FAMILY="R7_tap_c100_c36" NOTE="R7 adaptive (coast/tap) free-rider, oracle eta" \
#       ./submit_experiment.sh 14 "$s"
# done

# -----------------------------------------------------------------------------
# GROUP E -- the "even in your best case" control
# -----------------------------------------------------------------------------
# R8  balanced keys + same trigger class. Balanced removes stuck bits, so honest
#     BER -> 0, i.e. the paper's own claimed operating point (99.7 %+). If the
#     free-rider ALSO reaches 0 there, non-separability is not an artefact of a
#     weak watermark -- it holds exactly where the scheme works best.
#     You already have honest_c100_bdef_bal_iid (6 seeds); this adds the
#     same-class attacker at 3 seeds.
for s in 0 1 2; do
  env ATTACK=reduced FREE_RIDER_IDS=0 TRIGGER_CLASS_MAP="0:6" \
      AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
      WM_ETA_FIXED=0.001 BALANCED=1 ROUNDS=50 \
      FAMILY="R8_sameclass_bal_c6" \
      NOTE="R8 balanced keys, FR cid0 pinned to class 6 alongside honest cid6" \
      ./submit_experiment.sh 14 "$s"
done

# scp -r <cluster>:$MOUNT/home/zu/results ~/local/results
# RES=~/local/results ./plot_now.sh

echo
printf "== %s runs queued ==\n" "$(grep -c . "$JOBS_FILE")"
cut -f1 "$JOBS_FILE" | sed 's/^/   /'
cat <<'NEXT'

Next:
    unset DRYRUN
    POOLS="a100-80 a100-40" WORKERS_LIST="6 3" PODS=2 ./submit_pool.sh
    runai list jobs          # must show exactly 2
Then leave. Progress: $MOUNT/home/zu/results/.poollogs/pool_w{0,1}.log
NEXT