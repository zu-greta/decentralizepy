#!/usr/bin/env bash
set -euo pipefail

# ===================================================
#  Usage:
#     ./submit_experiment.sh [CONFIG_IDX] [REPEAT]
#     ./submit_experiment.sh 14 0                       # submarine, seed 0
#     ATTACK=none FAMILY=t1_all_honest ./submit_experiment.sh 14 0
#  Set DEBUG_HOLD=1 to keep the pod alive 1h after the run for inspection.
# ===================================================
# RUNAI_EXTRA: extra flags appended verbatim to `runai submit`. Use it to pin a GPU
# type on a heterogeneous cluster (RCP has V100 / A100-40 / A100-80 / H100 / H200, so
# wall-clock and gpu_ms are not comparable across jobs unless you pin), e.g.
#   RUNAI_EXTRA="--node-pools a100-80" ./submit_experiment.sh 14 0
# (check the pool names with: runai list node-pools)
# NOTE: so far A100-80 have been used for experiments

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

# ---- Python overrides assembled from env vars ----
PY_EXTRA=""
# general
[ -n "${MODEL:-}" ]            && PY_EXTRA="$PY_EXTRA --model ${MODEL}"
[ -n "${DATASET:-}" ]          && PY_EXTRA="$PY_EXTRA --dataset ${DATASET}"
[ -n "${ROUNDS:-}" ]          && PY_EXTRA="$PY_EXTRA --rounds ${ROUNDS}"
[ -n "${NUM_CLIENTS:-}" ]     && PY_EXTRA="$PY_EXTRA --num_clients ${NUM_CLIENTS}"
[ -n "${LOCAL_EPOCHS:-}" ]     && PY_EXTRA="$PY_EXTRA --local_epochs ${LOCAL_EPOCHS}"
[ -n "${BATCH_SIZE:-}" ]      && PY_EXTRA="$PY_EXTRA --batch_size ${BATCH_SIZE}"
[ -n "${LR:-}" ]              && PY_EXTRA="$PY_EXTRA --lr ${LR}"
[ -n "${PARTITION:-}" ]        && PY_EXTRA="$PY_EXTRA --partition ${PARTITION}"
[ -n "${DIRICHLET_ALPHA:-}" ]  && PY_EXTRA="$PY_EXTRA --dirichlet_alpha ${DIRICHLET_ALPHA}"
[ -n "${TRIGGER_CLASS_MAP:-}" ] && PY_EXTRA="$PY_EXTRA --trigger_class_map ${TRIGGER_CLASS_MAP}"
# free-rider selection
[ -n "${ATTACK:-}" ]          && PY_EXTRA="$PY_EXTRA --attack ${ATTACK}"
[ -n "${NUM_FREE_RIDERS:-}" ] && PY_EXTRA="$PY_EXTRA --num_free_riders ${NUM_FREE_RIDERS}"
[ -n "${FREE_RIDER_IDS:-}" ]  && PY_EXTRA="$PY_EXTRA --free_rider_ids ${FREE_RIDER_IDS}"
[ -n "${NOISE_SIGMA:-}" ]     && PY_EXTRA="$PY_EXTRA --noise_sigma ${NOISE_SIGMA}"
[ -n "${NOISE_DECAY:-}" ]     && PY_EXTRA="$PY_EXTRA --noise_decay ${NOISE_DECAY}"
# submarine / autopilot
# 16 AUTOP_* hooks are commented out with the submarine attacker 
[ -n "${AUTOP_ORACLE_ETA:-}" ]      && PY_EXTRA="$PY_EXTRA --autop_oracle_eta ${AUTOP_ORACLE_ETA}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_WARMUP_MODE:-}" ]     && PY_EXTRA="$PY_EXTRA --autop_warmup_mode ${AUTOP_WARMUP_MODE}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_HONEST_MIN:-}" ]      && PY_EXTRA="$PY_EXTRA --autop_honest_min ${AUTOP_HONEST_MIN}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_WARMUP_CAP:-}" ]      && PY_EXTRA="$PY_EXTRA --autop_warmup_cap ${AUTOP_WARMUP_CAP}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_CONV_EPS:-}" ]        && PY_EXTRA="$PY_EXTRA --autop_conv_eps ${AUTOP_CONV_EPS}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_CONV_PATIENCE:-}" ]   && PY_EXTRA="$PY_EXTRA --autop_conv_patience ${AUTOP_CONV_PATIENCE}"
[ -n "${AUTOP_HONEST_UNTIL:-}" ]    && PY_EXTRA="$PY_EXTRA --autop_honest_until ${AUTOP_HONEST_UNTIL}"
[ -n "${AUTOP_CALIB_ROUNDS:-}" ]  && PY_EXTRA="$PY_EXTRA --autop_calib_rounds ${AUTOP_CALIB_ROUNDS}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_ETA_K:-}" ]           && PY_EXTRA="$PY_EXTRA --autop_eta_k ${AUTOP_ETA_K}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_ETA_MODE:-}" ]        && PY_EXTRA="$PY_EXTRA --autop_eta_mode ${AUTOP_ETA_MODE}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_NUM_CLIENTS_EST:-}" ] && PY_EXTRA="$PY_EXTRA --autop_num_clients_est ${AUTOP_NUM_CLIENTS_EST}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_MARGIN0:-}" ]         && PY_EXTRA="$PY_EXTRA --autop_margin0 ${AUTOP_MARGIN0}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_SAFETY:-}" ]          && PY_EXTRA="$PY_EXTRA --autop_safety ${AUTOP_SAFETY}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_MAX_COAST:-}" ]       && PY_EXTRA="$PY_EXTRA --autop_max_coast ${AUTOP_MAX_COAST}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_FLOOR:-}" ]           && PY_EXTRA="$PY_EXTRA --autop_floor ${AUTOP_FLOOR}"
[ -n "${AUTOP_COMMON_PER_CLASS:-}" ] && PY_EXTRA="$PY_EXTRA --autop_common_per_class ${AUTOP_COMMON_PER_CLASS}"
[ -n "${AUTOP_N_COMMON_CLASSES:-}" ] && PY_EXTRA="$PY_EXTRA --autop_n_common_classes ${AUTOP_N_COMMON_CLASSES}"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_SCOPE:-}" ]           && PY_EXTRA="$PY_EXTRA --autop_scope ${AUTOP_SCOPE}"
# [SUBMARINE-ONLY, DISABLED] [ "${AUTOP_STAY_MIN:-}" = "1" ]     && PY_EXTRA="$PY_EXTRA --autop_stay_min"
# [SUBMARINE-ONLY, DISABLED] [ -n "${AUTOP_HOLDOUT_RATIO:-}" ]   && PY_EXTRA="$PY_EXTRA --autop_holdout_ratio ${AUTOP_HOLDOUT_RATIO}"
# [SUBMARINE-ONLY, DISABLED] [ "${AUTOP_HONEST_CLONE:-}" = "1" ] && PY_EXTRA="$PY_EXTRA --autop_honest_clone"
# watermarking
[ -n "${WATERMARK:-}" ]        && PY_EXTRA="$PY_EXTRA --watermark"
[ -n "${WM_BITS:-}" ]          && PY_EXTRA="$PY_EXTRA --wm_bits ${WM_BITS}"
[ "${BALANCED:-}" = "1" ]      && PY_EXTRA="$PY_EXTRA --wm_balanced_keys"
[ -n "${WM_F:-}" ]             && PY_EXTRA="$PY_EXTRA --wm_f ${WM_F}"
[ -n "${WM_ALPHA:-}" ]         && PY_EXTRA="$PY_EXTRA --wm_alpha ${WM_ALPHA}" # tuning non-iid alpha
[ -n "${WM_NUM_TRIGGERS:-}" ]  && PY_EXTRA="$PY_EXTRA --wm_num_triggers ${WM_NUM_TRIGGERS}"
[ -n "${WM_TRIGGER_MODE:-}" ]  && PY_EXTRA="$PY_EXTRA --wm_trigger_mode ${WM_TRIGGER_MODE}"
[ -n "${WM_LAMBDA:-}" ]        && PY_EXTRA="$PY_EXTRA --wm_lambda ${WM_LAMBDA}"
[ -n "${WM_BETA:-}" ]          && PY_EXTRA="$PY_EXTRA --wm_beta ${WM_BETA}"
[ -n "${WM_ETA_FLOOR:-}" ]     && PY_EXTRA="$PY_EXTRA --wm_eta_floor ${WM_ETA_FLOOR}"
[ -n "${WM_ETA_FIXED:-}" ]     && PY_EXTRA="$PY_EXTRA --wm_eta_fixed ${WM_ETA_FIXED}"
# [ -n "${PAPER_FAITHFUL:-}" ]   && PY_EXTRA="$PY_EXTRA --paper_faithful"
[ "${CALIB_ON_ALL:-0}" = "1" ] && PY_EXTRA="$PY_EXTRA --calib_on_all"
# manifest (descriptive)
[ -n "${FAMILY:-}" ]      && PY_EXTRA="$PY_EXTRA --manifest_family ${FAMILY}"
[ -n "${SWEEP_VAR:-}" ]   && PY_EXTRA="$PY_EXTRA --sweep_var ${SWEEP_VAR}"
[ -n "${SWEEP_LEVEL:-}" ] && PY_EXTRA="$PY_EXTRA --sweep_level ${SWEEP_LEVEL}"

# Tag results/job uniquely.
# RUN_TAG (the output-dir name) 
#   * via run_all.sh: FAMILY encodes dataset+bits+attack+partition+positions,
#     the dir is exactly "<FAMILY>_rep<seed>_<ts>"
#   * bare submit_experiment.sh (no FAMILY): assemble the tag from the knobs 
USER_TAG="${TAG:+_${TAG}}"
FR_TAG=""                                 # always defined (JOB_NAME uses it under set -u)
if [ -n "${FAMILY:-}" ]; then
  RUN_TAG="${FAMILY}${USER_TAG}_rep${REPEAT}_$(date +%Y%m%d_%H%M%S)"
else
  CORE="cfg${CONFIG_IDX}"
  BITS_TAG="";  [ -n "${WM_BITS:-}" ]           && BITS_TAG="_b${WM_BITS}"
  POS_TAG="";   [ -n "${FREE_RIDER_IDS:-}" ]    && POS_TAG="_c${FREE_RIDER_IDS//,/}"
  MAP_TAG="";   [ -n "${TRIGGER_CLASS_MAP:-}" ] && MAP_TAG="_map$(printf '%s' "${TRIGGER_CLASS_MAP}" | tr -d ':,' )"
  ETA_TAG="";   [ -n "${WM_ETA_FIXED:-}" ]      && ETA_TAG="_eta$(printf '%s' "${WM_ETA_FIXED}" | tr -d '.')"
  F_TAG="";     [ -n "${WM_F:-}" ]              && F_TAG="_${WM_F}"
  FR_TAG="";    [ -n "${NUM_FREE_RIDERS:-}" ]   && FR_TAG="_fr${NUM_FREE_RIDERS}"
  RUN_TAG="${CORE}${BITS_TAG}${POS_TAG}${MAP_TAG}${FR_TAG}${ETA_TAG}${F_TAG}${USER_TAG}_rep${REPEAT}_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_DIR="${MOUNT}/home/zu/results/${RUN_TAG}"
DATA_ROOT="${MOUNT}/home/zu/data"
JOB_NAME="faremark-c${CONFIG_IDX}-r${REPEAT}${FR_TAG}${USER_TAG}-$(date +%H%M%S)"

echo "=== Submitting $JOB_NAME (config_idx=$CONFIG_IDX repeat=$REPEAT) ==="

runai submit "$JOB_NAME" \
  --project "$PROJECT" -g 1 --image "$IMAGE" --pvc "$PVC:$MOUNT" \
  ${RUNAI_EXTRA:-} \
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
    # ---- pod.log structure -------------------------------------------------
    # pod.log is the ENVIRONMENT record (what machine, what code, what flags);
    # run.log is the EXPERIMENT record; result.json is the DATA. Previously
    # pod.log was an undelimited tee of git-clone noise + run.log, so you could
    # not answer "which commit produced this number?" -- the pod clones a moving
    # branch, so two runs a week apart can be different code, identical config.
    echo "================================================================"
    echo "== POD =="
    echo "================================================================"
    printf "  %-22s %s\n" "started (UTC)"  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf "  %-22s %s\n" "job"            "${JOB_NAME:-?}"
    printf "  %-22s %s\n" "node"           "${NODE_NAME:-unknown}"
    printf "  %-22s %s\n" "output_dir"     "$OUTPUT_DIR"
    printf "  %-22s %s\n" "config_idx/rep" "$CONFIG_IDX / $REPEAT"
    echo "== GPU =="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed "s/^/  /" || echo "  nvidia-smi unavailable"

    echo "== CODE =="
    rm -rf /tmp/decentralizepy
    # NOTE: still --depth 1 (fast). rev-parse below pins the exact commit anyway.
    git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" /tmp/decentralizepy 2>&1 | sed "s/^/  /"
    if [ ! -d "/tmp/decentralizepy/$PKG_SUBDIR" ]; then
      echo "ERROR: $PKG_SUBDIR/ not found in the repo."; sync; sleep 2; exit 3
    fi
    # ADDED: exact commit SHA. Exported so run_experiment.py records it in
    # result.json["env"]["git_commit"] -- the run becomes self-identifying code-wise.
    GIT_COMMIT="$(git -C /tmp/decentralizepy rev-parse HEAD 2>/dev/null || echo unknown)"
    export GIT_COMMIT GIT_BRANCH
    printf "  %-22s %s\n" "repo"    "$GIT_REPO"
    printf "  %-22s %s\n" "branch"  "$GIT_BRANCH"
    printf "  %-22s %s\n" "commit"  "$GIT_COMMIT"
    printf "  %-22s %s\n" "python"  "$(python -V 2>&1)"
    printf "  %-22s %s\n" "torch"   "$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo n/a)"

    echo "== ARGS =="
    printf "  %-22s %s\n" "script"  "$SCRIPT"
    # one flag per line: PY_EXTRA used to be one long unreadable string
    echo "$PY_EXTRA" | tr " " "\n" | grep -v "^$" | paste - - 2>/dev/null | sed "s/^/  /" || echo "  $PY_EXTRA"
    [ -n "${NOTE:-}" ] && printf "  %-22s %s\n" "note" "$NOTE"
    echo "================================================================"

    export PYTHONPATH="/tmp/decentralizepy/$PKG_SUBDIR"
    cd "/tmp/decentralizepy/$PKG_SUBDIR"
    set +e
    EXTRA_ARR=($PY_EXTRA)
    [ -n "${NOTE:-}" ] && EXTRA_ARR+=(--manifest_note "$NOTE")
    set +u
    python -u "$SCRIPT" --config_idx "$CONFIG_IDX" --repeat "$REPEAT" --device cuda --output_dir "$OUTPUT_DIR" --data_root "$DATA_ROOT" "${EXTRA_ARR[@]}"
    EXIT=$?
    set -u; set -e
    echo "================================================================"
    echo "== EXIT =="
    # Exit 2 = accuracy outside the expected_acc band of the config. EXPECTED for
    # attack runs (free-riders drag accuracy down) and result.json is already
    # written. Only 1/3/>=4 are real failures. Spelling this out in pod.log stops
    # `runai` job-failure noise from being mistaken for lost data.
    case "$EXIT" in
      0) echo "  exit 0  OK (accuracy inside expected band)" ;;
      2) echo "  exit 2  accuracy outside expected band -- NORMAL for attack runs;"
         echo "          result.json was written before exit, data is intact." ;;
      3) echo "  exit 3  repo layout error (PKG_SUBDIR missing)" ;;
      *) echo "  exit $EXIT  FAILED -- inspect run.log above" ;;
    esac
    printf "  %-22s %s\n" "finished (UTC)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf "  %-22s %s\n" "result" "$OUTPUT_DIR/result.json"
    echo "================================================================"
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