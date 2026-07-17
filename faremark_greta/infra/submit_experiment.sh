#!/usr/bin/env bash
set -euo pipefail

# ===================================================
#  Usage:
#     ./submit_experiment.sh [CONFIG_IDX] [REPEAT]
#     ./submit_experiment.sh 14 0                       # submarine, seed 0
#     ATTACK=none FAMILY=t1_all_honest ./submit_experiment.sh 14 0
#  Set DEBUG_HOLD=1 to keep the pod alive 1h after the run for inspection.
# ===================================================
CONFIG_IDX="${1:-0}"
REPEAT="${2:-0}"
DEBUG_HOLD="${DEBUG_HOLD:-0}"

if [ -f .env ]; then set -a; source .env; set +a
else echo "Error: .env file not found!"; exit 1; fi

echo "=== env: PROJECT=$PROJECT IMAGE=$IMAGE PVC=$PVC MOUNT=$MOUNT NAMESPACE=$NAMESPACE ==="

GIT_REPO="https://github.com/zu-greta/decentralizepy.git"
GIT_BRANCH="main"
PKG_SUBDIR="faremark_greta"
SCRIPT="${SCRIPT:-scripts/run_experiment.py}"

# ---- Python overrides assembled from env vars (only the ones you set) ----
PY_EXTRA=""
# general
[ -n "${MODEL:-}" ]            && PY_EXTRA="$PY_EXTRA --model ${MODEL}"
[ -n "${DATASET:-}" ]          && PY_EXTRA="$PY_EXTRA --dataset ${DATASET}"
[ -n "${ROUNDS:-}" ]          && PY_EXTRA="$PY_EXTRA --rounds ${ROUNDS}"
[ -n "${LOCAL_EPOCHS:-}" ]     && PY_EXTRA="$PY_EXTRA --local_epochs ${LOCAL_EPOCHS}"
[ -n "${BATCH_SIZE:-}" ]      && PY_EXTRA="$PY_EXTRA --batch_size ${BATCH_SIZE}"
[ -n "${LR:-}" ]              && PY_EXTRA="$PY_EXTRA --lr ${LR}"
[ -n "${PARTITION:-}" ]        && PY_EXTRA="$PY_EXTRA --partition ${PARTITION}"
[ -n "${DIRICHLET_ALPHA:-}" ]  && PY_EXTRA="$PY_EXTRA --dirichlet_alpha ${DIRICHLET_ALPHA}"
# free-rider selection
[ -n "${ATTACK:-}" ]          && PY_EXTRA="$PY_EXTRA --attack ${ATTACK}"
[ -n "${NUM_FREE_RIDERS:-}" ] && PY_EXTRA="$PY_EXTRA --num_free_riders ${NUM_FREE_RIDERS}"
[ -n "${FREE_RIDER_IDS:-}" ]  && PY_EXTRA="$PY_EXTRA --free_rider_ids ${FREE_RIDER_IDS}"
[ -n "${NOISE_SIGMA:-}" ]     && PY_EXTRA="$PY_EXTRA --noise_sigma ${NOISE_SIGMA}"
[ -n "${NOISE_DECAY:-}" ]     && PY_EXTRA="$PY_EXTRA --noise_decay ${NOISE_DECAY}"
# submarine / autopilot
[ -n "${AUTOP_ORACLE_ETA:-}" ]      && PY_EXTRA="$PY_EXTRA --autop_oracle_eta ${AUTOP_ORACLE_ETA}"
[ -n "${AUTOP_WARMUP_MODE:-}" ]     && PY_EXTRA="$PY_EXTRA --autop_warmup_mode ${AUTOP_WARMUP_MODE}"
[ -n "${AUTOP_HONEST_MIN:-}" ]      && PY_EXTRA="$PY_EXTRA --autop_honest_min ${AUTOP_HONEST_MIN}"
[ -n "${AUTOP_WARMUP_CAP:-}" ]      && PY_EXTRA="$PY_EXTRA --autop_warmup_cap ${AUTOP_WARMUP_CAP}"
[ -n "${AUTOP_CONV_EPS:-}" ]        && PY_EXTRA="$PY_EXTRA --autop_conv_eps ${AUTOP_CONV_EPS}"
[ -n "${AUTOP_CONV_PATIENCE:-}" ]   && PY_EXTRA="$PY_EXTRA --autop_conv_patience ${AUTOP_CONV_PATIENCE}"
[ -n "${AUTOP_HONEST_UNTIL:-}" ]    && PY_EXTRA="$PY_EXTRA --autop_honest_until ${AUTOP_HONEST_UNTIL}"
[ -n "${AUTOP_CALIB_ROUNDS:-}" ]  && PY_EXTRA="$PY_EXTRA --autop_calib_rounds ${AUTOP_CALIB_ROUNDS}"
[ -n "${AUTOP_ETA_K:-}" ]           && PY_EXTRA="$PY_EXTRA --autop_eta_k ${AUTOP_ETA_K}"
[ -n "${AUTOP_ETA_MODE:-}" ]        && PY_EXTRA="$PY_EXTRA --autop_eta_mode ${AUTOP_ETA_MODE}"
[ -n "${AUTOP_NUM_CLIENTS_EST:-}" ] && PY_EXTRA="$PY_EXTRA --autop_num_clients_est ${AUTOP_NUM_CLIENTS_EST}"
[ -n "${AUTOP_MARGIN0:-}" ]         && PY_EXTRA="$PY_EXTRA --autop_margin0 ${AUTOP_MARGIN0}"
[ -n "${AUTOP_SAFETY:-}" ]          && PY_EXTRA="$PY_EXTRA --autop_safety ${AUTOP_SAFETY}"
[ -n "${AUTOP_MAX_COAST:-}" ]       && PY_EXTRA="$PY_EXTRA --autop_max_coast ${AUTOP_MAX_COAST}"
[ -n "${AUTOP_FLOOR:-}" ]           && PY_EXTRA="$PY_EXTRA --autop_floor ${AUTOP_FLOOR}"
[ -n "${AUTOP_COMMON_PER_CLASS:-}" ] && PY_EXTRA="$PY_EXTRA --autop_common_per_class ${AUTOP_COMMON_PER_CLASS}"
[ -n "${AUTOP_SCOPE:-}" ]           && PY_EXTRA="$PY_EXTRA --autop_scope ${AUTOP_SCOPE}"
[ "${AUTOP_STAY_MIN:-}" = "1" ]     && PY_EXTRA="$PY_EXTRA --autop_stay_min"
[ -n "${AUTOP_HOLDOUT_RATIO:-}" ]   && PY_EXTRA="$PY_EXTRA --autop_holdout_ratio ${AUTOP_HOLDOUT_RATIO}"
[ "${AUTOP_HONEST_CLONE:-}" = "1" ] && PY_EXTRA="$PY_EXTRA --autop_honest_clone"
# watermarking
[ -n "${WATERMARK:-}" ]        && PY_EXTRA="$PY_EXTRA --watermark"
[ -n "${WM_BITS:-}" ]          && PY_EXTRA="$PY_EXTRA --wm_bits ${WM_BITS}"
[ -n "${WM_NUM_TRIGGERS:-}" ]  && PY_EXTRA="$PY_EXTRA --wm_num_triggers ${WM_NUM_TRIGGERS}"
[ -n "${WM_LAMBDA:-}" ]        && PY_EXTRA="$PY_EXTRA --wm_lambda ${WM_LAMBDA}"
[ -n "${WM_BETA:-}" ]          && PY_EXTRA="$PY_EXTRA --wm_beta ${WM_BETA}"
[ -n "${WM_ETA_FLOOR:-}" ]     && PY_EXTRA="$PY_EXTRA --wm_eta_floor ${WM_ETA_FLOOR}"
[ -n "${WM_ETA_FIXED:-}" ]     && PY_EXTRA="$PY_EXTRA --wm_eta_fixed ${WM_ETA_FIXED}"
[ -n "${PAPER_FAITHFUL:-}" ]   && PY_EXTRA="$PY_EXTRA --paper_faithful"
[ "${CALIB_ON_ALL:-0}" = "1" ] && PY_EXTRA="$PY_EXTRA --calib_on_all"
# manifest (descriptive)
[ -n "${FAMILY:-}" ]      && PY_EXTRA="$PY_EXTRA --manifest_family ${FAMILY}"
[ -n "${SWEEP_VAR:-}" ]   && PY_EXTRA="$PY_EXTRA --sweep_var ${SWEEP_VAR}"
[ -n "${SWEEP_LEVEL:-}" ] && PY_EXTRA="$PY_EXTRA --sweep_level ${SWEEP_LEVEL}"

# Tag results/job uniquely
FR_TAG=""; [ -n "${NUM_FREE_RIDERS:-}" ] && FR_TAG="-fr${NUM_FREE_RIDERS}"
USER_TAG="${TAG:+-${TAG}}"
RUN_TAG="cfg${CONFIG_IDX}_rep${REPEAT}${FR_TAG}${USER_TAG}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${MOUNT}/home/zu/results/${RUN_TAG}"
DATA_ROOT="${MOUNT}/home/zu/data"
JOB_NAME="faremark-c${CONFIG_IDX}-r${REPEAT}${FR_TAG}${USER_TAG}-$(date +%H%M%S)"

echo "=== Submitting $JOB_NAME (config_idx=$CONFIG_IDX repeat=$REPEAT) ==="

runai submit "$JOB_NAME" \
  --project "$PROJECT" -g 1 --image "$IMAGE" --pvc "$PVC:$MOUNT" \
  --run-as-uid "$USER_UID" --run-as-gid "$USER_GID" --memory "$MEMORY" \
  -e "CONFIG_IDX=$CONFIG_IDX" -e "REPEAT=$REPEAT" -e "OUTPUT_DIR=$OUTPUT_DIR" \
  -e "DATA_ROOT=$DATA_ROOT" -e "GIT_REPO=$GIT_REPO" -e "GIT_BRANCH=$GIT_BRANCH" \
  -e "PKG_SUBDIR=$PKG_SUBDIR" -e "SCRIPT=$SCRIPT" -e "PY_EXTRA=$PY_EXTRA" \
  -e "NOTE=${NOTE:-}" -e "DEBUG_HOLD=$DEBUG_HOLD" \
  --command -- bash -c '
    set -euo pipefail
    export USER=zu
    mkdir -p "$OUTPUT_DIR" "$DATA_ROOT"
    exec > >(tee "$OUTPUT_DIR/pod.log") 2>&1
    echo "=== pod start: $(date) ==="
    rm -rf /tmp/decentralizepy
    git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" /tmp/decentralizepy
    if [ ! -d "/tmp/decentralizepy/$PKG_SUBDIR" ]; then
      echo "ERROR: $PKG_SUBDIR/ not found in the repo."; sync; sleep 2; exit 3
    fi
    export PYTHONPATH="/tmp/decentralizepy/$PKG_SUBDIR"
    cd "/tmp/decentralizepy/$PKG_SUBDIR"
    set +e
    EXTRA_ARR=($PY_EXTRA)
    [ -n "${NOTE:-}" ] && EXTRA_ARR+=(--manifest_note "$NOTE")
    set +u
    python -u "$SCRIPT" --config_idx "$CONFIG_IDX" --repeat "$REPEAT" --device cuda --output_dir "$OUTPUT_DIR" --data_root "$DATA_ROOT" "${EXTRA_ARR[@]}"
    EXIT=$?
    set -u; set -e
    echo "experiment exit code: $EXIT"
    if [ "$DEBUG_HOLD" = "1" ]; then echo "DEBUG_HOLD: sleeping 1h"; sleep 3600; fi
    sync; sleep 2
    exit $EXIT
  '

if [ "${WAIT:-1}" = "0" ]; then
  echo "Submitted (fire-and-forget): $JOB_NAME  ->  $OUTPUT_DIR"
  exit 0
fi

POD_NAME=""
for i in $(seq 1 60); do
  POD_NAME=$(kubectl get pods -n "$NAMESPACE" --no-headers \
    -o custom-columns=":metadata.name" 2>/dev/null | grep "^${JOB_NAME}-" | head -1 || true)
  [ -n "$POD_NAME" ] && break; sleep 5
done
if [ -z "$POD_NAME" ]; then
  echo "Pod not created after ~5 min — likely queued for a GPU. Check:"
  echo "   runai describe job $JOB_NAME -p $PROJECT"; exit 0
fi
echo "Pod: $POD_NAME | logs: kubectl logs -n $NAMESPACE $POD_NAME -f | results -> $OUTPUT_DIR"
while true; do
  PHASE=$(kubectl get pod -n "$NAMESPACE" "$POD_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo Unknown)
  case "$PHASE" in
    Succeeded) echo "Succeeded."; runai delete job "$JOB_NAME" --project "$PROJECT" || true; break ;;
    Failed)    echo "Failed. Inspect: kubectl logs -n $NAMESPACE $POD_NAME"; break ;;
    *)         sleep 30 ;;
  esac
done