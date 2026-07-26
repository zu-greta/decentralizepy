#!/usr/bin/env bash
# =============================================================================
# run_now.sh -- builds jobs.tsv for the experiment plan (see STATUS_AND_PLAN.md).
#
#   ./run_now.sh              # build manifest (submits nothing)
#   ./run_now.sh A            # build ONLY group A (the safe proven baseline)
#   unset DRYRUN; WORKERS=6 PODS=2 ./submit_pool.sh
#
# Groups: A=proven baseline  C=smoothing  D=+N spectrum  E=non-iid  F=capacity
#         P=paper reproduction (ONLY after the embedding probe passes)
# All families use the KNOWN-GOOD cifar100/10-client base unless the point is to
# vary it. NUM_WORKERS=0 (DataLoader worker churn, see prior analysis).
# =============================================================================
set -uo pipefail
export DRYRUN=1 JOBS_FILE="${JOBS_FILE:-jobs.tsv}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
WANT="${1:-ACDEF}"                       # which groups to build; default = the safe set
rm -f "$JOBS_FILE"
echo "== building $JOBS_FILE  groups=[$WANT]  NUM_WORKERS=$NUM_WORKERS =="
has(){ [[ "$WANT" == *"$1"* ]]; }

# ---------------------------------------------------------------------------
# GROUP A -- proven baseline (cifar100, 10 clients). THIS CONFIG IS KNOWN GOOD.
#   A1 honest x6  A2 reduced easy  A3 reduced hard  A4 sameclass
# ---------------------------------------------------------------------------
if has A; then
  for s in 0 1 2 3 4 5; do
    env ATTACK=none NUM_FREE_RIDERS=0 DS=c100 NUM_CLIENTS=10 ROUNDS=50 \
        FAMILY="A1_honest_c100" NOTE="A1 honest baseline (known-good config)" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=reduced FREE_RIDER_IDS=1,7 AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 \
        AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
        FAMILY="A2_reduced_c100_c17" NOTE="A2 reduced +5 easy classes 1,7" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=reduced FREE_RIDER_IDS=3,6 AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 \
        AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
        FAMILY="A3_reduced_c100_c36" NOTE="A3 reduced +5 hard classes 3,6" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=reduced FREE_RIDER_IDS=0 TRIGGER_CLASS_MAP="0:6" AUTOP_COMMON_PER_CLASS=5 \
        AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
        FAMILY="A4_sameclass_c100_c6" NOTE="A4 FR shares class 6 with honest" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP C -- smoothing function (does a different f() move the floors?)
# ---------------------------------------------------------------------------
if has C; then
  for s in 0 1 2; do
    env ATTACK=none NUM_FREE_RIDERS=0 WM_F=sin WM_ALPHA=1.5708 ROUNDS=50 \
        FAMILY="C1_honest_sin_c100" NOTE="C1 sin smoothing alpha=pi/2" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP D -- +N free-riding spectrum at the hard classes
#   N = images kept per common class; -1 = full shard (still a free-rider)
# ---------------------------------------------------------------------------
if has D; then
  for N in -1 0 1 2 5 10 25 50; do
    for s in 0 1 2; do
      env ATTACK=reduced FREE_RIDER_IDS=3,6 AUTOP_COMMON_PER_CLASS=$N \
          AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
          FAMILY="D1_reduced_c100_c36_n${N}" NOTE="D1 +N spectrum N=$N" \
          ./submit_experiment.sh 14 "$s"
    done
  done
fi

# ---------------------------------------------------------------------------
# GROUP E -- non-IID (Dirichlet alpha=0.5)
# ---------------------------------------------------------------------------
if has E; then
  for s in 0 1 2; do
    env ATTACK=none NUM_FREE_RIDERS=0 PART=niid DIRICHLET_ALPHA=0.5 ROUNDS=50 \
        FAMILY="E1_honest_niid_c100" NOTE="E1 non-iid honest a=0.5" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=reduced FREE_RIDER_IDS=3,6 PART=niid DIRICHLET_ALPHA=0.5 \
        AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
        WM_ETA_FIXED=0.161 ROUNDS=50 \
        FAMILY="E2_reduced_niid_c36" NOTE="E2 non-iid reduced hard" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP F -- more clients than classes (200 clients). NEEDS MORE ROUNDS to train.
# ---------------------------------------------------------------------------
if has F; then
  for s in 0 1 2; do
    env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=200 ROUNDS=100 \
        FAMILY="F1_honest_nc200" NOTE="F1 200 clients honest, 100 rounds" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=reduced NUM_CLIENTS=200 FREE_RIDER_IDS=106,107 AUTOP_COMMON_PER_CLASS=5 \
        AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.384 ROUNDS=100 \
        FAMILY="F2_reduced_nc200_c67" NOTE="F2 200cl reduced, shared classes 6,7" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP P -- paper reproduction. RUN ONLY AFTER the embedding probe passes.
#   (kept commented so a stray ./run_now.sh P cannot burn the night on nan)
# ---------------------------------------------------------------------------
# if has P; then
#   ROW=c10  SEEDS='0 1 2' BALANCED=0 FAM=P_paper_c10  ./paper_check.sh submit
#   ROW=c100 SEEDS='0 1 2' BALANCED=0 FAM=P_paper_c100 ./paper_check.sh submit
#   ROW=t9   SEEDS='0 1 2' BALANCED=0 HELDOUT=1 FAM=P_paper_t9 ./paper_check.sh submit
# fi

N=$(grep -c . "$JOBS_FILE" 2>/dev/null || echo 0)
echo
echo "== $N runs queued  (groups: $WANT) =="
cut -f1 "$JOBS_FILE" | sed 's/_rep[0-9]*$//' | sort | uniq -c | sed 's/^/   /'
cat <<NEXT

Next:
    unset DRYRUN
    WORKERS=6 PODS=2 ./submit_pool.sh
    runai list jobs

FIRST TIME after the code fix, run the probe instead:
    NUM_WORKERS=0 FAMILY=probe_fix WM_BITS=2 WM_NUM_TRIGGERS=50 ROUNDS=25 ./submit_experiment.sh 11 0
    # confirm BER falls and pmax is not nan BEFORE launching the batch
NEXT