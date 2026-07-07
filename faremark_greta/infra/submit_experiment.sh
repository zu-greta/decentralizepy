#!/usr/bin/env bash
set -euo pipefail

# ===================================================
#  Usage:
#     ./submit_experiment.sh [CONFIG_IDX] [REPEAT]
#  Example (smoke test):       ./submit_experiment.sh 0 0
#  Example (Table I RN18/C10): ./submit_experiment.sh 1 0
#
#  Optional env overrides (see the script for the full list):
#     NUM_FREE_RIDERS=4 ATTACK=previous_models ./submit_experiment.sh 8 0
#     ROUNDS=10 BATCH_SIZE=64 ./submit_experiment.sh 7 0
#     # submarine sweep with self-describing manifest:
#     ATTACK=submarine CALIB_ON_ALL=1 MEM_BLEND_GLOBAL=0.3 \
#       FAMILY=A7_submarine SWEEP_VAR=mem_blend_global \
#       TAG=a7-mbg03 WAIT=0 ./submit_experiment.sh 14 0
#  Set DEBUG_HOLD=1 to keep the pod alive 1h after the run for inspection.
# ===================================================
CONFIG_IDX="${1:-0}"
REPEAT="${2:-0}"
DEBUG_HOLD="${DEBUG_HOLD:-0}"

# Setup .env file based on .env_example
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Error: .env file not found!"
    exit 1
fi

# check variables imported
echo "=== Checking env variables ==="
echo "PROJECT=$PROJECT"
echo "IMAGE=$IMAGE"
echo "PVC=$PVC"
echo "MOUNT=$MOUNT"
echo "USER_UID=$USER_UID"
echo "USER_GID=$USER_GID"
echo "MEMORY=$MEMORY"
echo "NAMESPACE=$NAMESPACE"

# ---- Cluster / account config ----
PROJECT="$PROJECT"
IMAGE="$IMAGE" # Note: must match infra/build.sh IMAGE_NAME
PVC="$PVC"
MOUNT="$MOUNT"
USER_UID="$USER_UID"
USER_GID="$USER_GID"
MEMORY="$MEMORY"
NAMESPACE="$NAMESPACE"

# ---- Code + paths ----
GIT_REPO="https://github.com/zu-greta/decentralizepy.git"
GIT_BRANCH="main"
PKG_SUBDIR="faremark_greta"
SCRIPT="${SCRIPT:-scripts/run_experiment.py}"   # override: SCRIPT=scripts/run_robustness.py

# ---- Optional Python overrides assembled from env vars ----
# Only the ones you set get forwarded; everything else uses the config defaults
PY_EXTRA=""
[ -n "${MODEL:-}" ]            && PY_EXTRA="$PY_EXTRA --model ${MODEL}"
[ -n "${DATASET:-}" ]          && PY_EXTRA="$PY_EXTRA --dataset ${DATASET}"
[ -n "${WM_NUM_TRIGGERS:-}" ]  && PY_EXTRA="$PY_EXTRA --wm_num_triggers ${WM_NUM_TRIGGERS}"
[ -n "${WM_BITS:-}" ]          && PY_EXTRA="$PY_EXTRA --wm_bits ${WM_BITS}"
[ -n "${WM_LAMBDA:-}" ]        && PY_EXTRA="$PY_EXTRA --wm_lambda ${WM_LAMBDA}"
[ -n "${WM_BETA:-}" ]          && PY_EXTRA="$PY_EXTRA --wm_beta ${WM_BETA}"
[ -n "${ATTACK_ROUND:-}" ]     && PY_EXTRA="$PY_EXTRA --attack_round ${ATTACK_ROUND}"
[ -n "${N_TRIGGER_SAMPLES:-}" ] && PY_EXTRA="$PY_EXTRA --n_trigger_samples ${N_TRIGGER_SAMPLES}"
[ -n "${HONEST_PROB:-}" ]      && PY_EXTRA="$PY_EXTRA --honest_prob ${HONEST_PROB}"
[ -n "${BLEND:-}" ]            && PY_EXTRA="$PY_EXTRA --blend ${BLEND}"
[ "${FULL_TRIGGER_CLASS:-}" = "1" ] && PY_EXTRA="$PY_EXTRA --full_trigger_class"
[ -n "${N_COMMON_SAMPLES:-}" ] && PY_EXTRA="$PY_EXTRA --n_common_samples ${N_COMMON_SAMPLES}"
[ -n "${PARTITION:-}" ]        && PY_EXTRA="$PY_EXTRA --partition ${PARTITION}"
[ -n "${DIRICHLET_ALPHA:-}" ]  && PY_EXTRA="$PY_EXTRA --dirichlet_alpha ${DIRICHLET_ALPHA}"
[ -n "${LOCAL_EPOCHS:-}" ]     && PY_EXTRA="$PY_EXTRA --local_epochs ${LOCAL_EPOCHS}"
[ -n "${LR:-}" ]              && PY_EXTRA="$PY_EXTRA --lr ${LR}"
[ -n "${WATERMARK:-}" ]        && PY_EXTRA="$PY_EXTRA --watermark"
[ -n "${PAPER_FAITHFUL:-}" ]   && PY_EXTRA="$PY_EXTRA --paper_faithful"
[ -n "${CALIB_ON_ALL:-}" ]     && PY_EXTRA="$PY_EXTRA --calib_on_all"
[ -n "${NUM_FREE_RIDERS:-}" ] && PY_EXTRA="$PY_EXTRA --num_free_riders ${NUM_FREE_RIDERS}"
[ -n "${ATTACK:-}" ]          && PY_EXTRA="$PY_EXTRA --attack ${ATTACK}"
[ -n "${NOISE_SIGMA:-}" ]     && PY_EXTRA="$PY_EXTRA --noise_sigma ${NOISE_SIGMA}"
[ -n "${NOISE_DECAY:-}" ]     && PY_EXTRA="$PY_EXTRA --noise_decay ${NOISE_DECAY}"
[ -n "${ROUNDS:-}" ]          && PY_EXTRA="$PY_EXTRA --rounds ${ROUNDS}"
[ -n "${BATCH_SIZE:-}" ]      && PY_EXTRA="$PY_EXTRA --batch_size ${BATCH_SIZE}"

# ---- adaptive-attack overrides (submarine / memory_exploit) ----
[ -n "${SUB_WARMUP:-}" ]           && PY_EXTRA="$PY_EXTRA --sub_warmup ${SUB_WARMUP}"
[ -n "${SUB_COAST_MODE:-}" ]       && PY_EXTRA="$PY_EXTRA --sub_coast_mode ${SUB_COAST_MODE}"
[ -n "${REEMBED_SCOPE:-}" ]        && PY_EXTRA="$PY_EXTRA --reembed_scope ${REEMBED_SCOPE}"
[ -n "${REEMBED_STEPS:-}" ]        && PY_EXTRA="$PY_EXTRA --reembed_steps ${REEMBED_STEPS}"
[ -n "${AUTOP_MAX_BATCHES:-}" ]    && PY_EXTRA="$PY_EXTRA --autop_max_batches ${AUTOP_MAX_BATCHES}"
[ -n "${AUTOP_MIN_BATCHES:-}" ]    && PY_EXTRA="$PY_EXTRA --autop_min_batches ${AUTOP_MIN_BATCHES}"
[ -n "${AUTOP_MARGIN0:-}" ]        && PY_EXTRA="$PY_EXTRA --autop_margin0 ${AUTOP_MARGIN0}"
[ -n "${AUTOP_WARMUP_CAP:-}" ]     && PY_EXTRA="$PY_EXTRA --autop_warmup_cap ${AUTOP_WARMUP_CAP}"
[ -n "${AUTOP_SCOPE:-}" ]          && PY_EXTRA="$PY_EXTRA --autop_scope ${AUTOP_SCOPE}"
[ -n "${AUTOP_PROTECT_UNTIL:-}" ]  && PY_EXTRA="$PY_EXTRA --autop_protect_until ${AUTOP_PROTECT_UNTIL}"
[ -n "${AUTOP_LOOKAHEAD:-}" ]      && PY_EXTRA="$PY_EXTRA --autop_lookahead ${AUTOP_LOOKAHEAD}"
[ "${AUTOP_ENRICHED:-}" = "1" ]    && PY_EXTRA="$PY_EXTRA --autop_enriched"
[ -n "${SUB_WARMUP_BATCHES:-}" ]   && PY_EXTRA="$PY_EXTRA --sub_warmup_batches ${SUB_WARMUP_BATCHES}"
[ -n "${SUB_COMMON_SAMPLES:-}" ]   && PY_EXTRA="$PY_EXTRA --sub_common_samples ${SUB_COMMON_SAMPLES}"
[ -n "${SUB_MARGIN:-}" ]           && PY_EXTRA="$PY_EXTRA --sub_margin ${SUB_MARGIN}"
[ -n "${SUB_FLOOR:-}" ]            && PY_EXTRA="$PY_EXTRA --sub_floor ${SUB_FLOOR}"
[ -n "${SUB_ETA_MODE:-}" ]        && PY_EXTRA="$PY_EXTRA --sub_eta_mode ${SUB_ETA_MODE}"
[ -n "${SUB_ETA_FIXED:-}" ]       && PY_EXTRA="$PY_EXTRA --sub_eta_fixed ${SUB_ETA_FIXED}"
[ -n "${SUB_MAX_BURST_BATCHES:-}" ] && PY_EXTRA="$PY_EXTRA --sub_max_burst_batches ${SUB_MAX_BURST_BATCHES}"
[ -n "${SUB_PROBE_EVERY:-}" ]     && PY_EXTRA="$PY_EXTRA --sub_probe_every ${SUB_PROBE_EVERY}"
[ -n "${WARMUP_ROUNDS:-}" ]       && PY_EXTRA="$PY_EXTRA --warmup_rounds ${WARMUP_ROUNDS}"
[ -n "${MEM_BLEND_GLOBAL:-}" ]    && PY_EXTRA="$PY_EXTRA --mem_blend_global ${MEM_BLEND_GLOBAL}"

# ---- self-describing manifest (descriptive only) ----
[ -n "${FAMILY:-}" ]      && PY_EXTRA="$PY_EXTRA --manifest_family ${FAMILY}"
# NOTE is free-text (may contain spaces) -> passed as its own env var and
# appended as a single quoted arg inside the container (see the array below).
[ -n "${SWEEP_VAR:-}" ]   && PY_EXTRA="$PY_EXTRA --sweep_var ${SWEEP_VAR}"
[ -n "${SWEEP_LEVEL:-}" ] && PY_EXTRA="$PY_EXTRA --sweep_level ${SWEEP_LEVEL}"

# Tag results/job uniquely
FR_TAG=""
[ -n "${NUM_FREE_RIDERS:-}" ] && FR_TAG="-fr${NUM_FREE_RIDERS}"
USER_TAG="${TAG:+-${TAG}}"     # optional: TAG=mixblend03 -> dir/job suffixed with it
RUN_TAG="cfg${CONFIG_IDX}_rep${REPEAT}${FR_TAG}${USER_TAG}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${MOUNT}/home/zu/results/${RUN_TAG}"
DATA_ROOT="${MOUNT}/home/zu/data"
JOB_NAME="faremark-c${CONFIG_IDX}-r${REPEAT}${FR_TAG}${USER_TAG}-$(date +%H%M%S)"
# =====================================================

echo "=== Submitting $JOB_NAME (config_idx=$CONFIG_IDX repeat=$REPEAT) ==="

# Pass all paths/values as ENV VARS (-e), expanded here by the outer shell into
# simple KEY=VALUE flags (safe — no nested quoting). The command script below is
# wrapped in SINGLE quotes so the outer shell does NOT touch it; every $VAR in it
# is expanded by the CONTAINER's bash from the env we injected
runai submit "$JOB_NAME" \
  --project "$PROJECT" \
  -g 1 \
  --image "$IMAGE" \
  --pvc "$PVC:$MOUNT" \
  --run-as-uid "$USER_UID" \
  --run-as-gid "$USER_GID" \
  --memory "$MEMORY" \
  -e "CONFIG_IDX=$CONFIG_IDX" \
  -e "REPEAT=$REPEAT" \
  -e "OUTPUT_DIR=$OUTPUT_DIR" \
  -e "DATA_ROOT=$DATA_ROOT" \
  -e "GIT_REPO=$GIT_REPO" \
  -e "GIT_BRANCH=$GIT_BRANCH" \
  -e "PKG_SUBDIR=$PKG_SUBDIR" \
  -e "SCRIPT=$SCRIPT" \
  -e "PY_EXTRA=$PY_EXTRA" \
  -e "NOTE=${NOTE:-}" \
  -e "DEBUG_HOLD=$DEBUG_HOLD" \
  --command -- bash -c '
    set -euo pipefail
    export USER=zu
    mkdir -p "$OUTPUT_DIR" "$DATA_ROOT"
    # Mirror all output to a log on the PVC so the run is debuggable even if the pod dies
    exec > >(tee "$OUTPUT_DIR/pod.log") 2>&1
    echo "=== pod start: $(date) ==="
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "DATA_ROOT=$DATA_ROOT"
    echo "python: $(which python || echo MISSING)  $(python --version 2>&1 || true)"
    echo "nvidia-smi:"; nvidia-smi -L || echo "(no GPU visible)"
    rm -rf /tmp/decentralizepy
    echo "cloning $GIT_REPO (branch $GIT_BRANCH) ..."
    git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" /tmp/decentralizepy
    echo "repo top-level:"; ls -la /tmp/decentralizepy
    if [ ! -d "/tmp/decentralizepy/$PKG_SUBDIR" ]; then
      echo "ERROR: $PKG_SUBDIR/ not found in the repo."
      echo "Did you commit+push faremark_greta/ to branch $GIT_BRANCH of $GIT_REPO?"
      sync; sleep 2; exit 3
    fi
    export PYTHONPATH="/tmp/decentralizepy/$PKG_SUBDIR"
    cd "/tmp/decentralizepy/$PKG_SUBDIR"
    echo "package dir:"; ls -la
    set +e
    # PY_EXTRA holds only space-free flags -> word-split into an array; append the
    # free-text NOTE as a single quoted arg so spaces survive. set +u guards the
    # empty-array expansion on older bash.
    EXTRA_ARR=($PY_EXTRA)
    [ -n "${NOTE:-}" ] && EXTRA_ARR+=(--manifest_note "$NOTE")
    set +u
    python -u "$SCRIPT" --config_idx "$CONFIG_IDX" --repeat "$REPEAT" --device cuda --output_dir "$OUTPUT_DIR" --data_root "$DATA_ROOT" "${EXTRA_ARR[@]}"
    EXIT=$?
    set -u
    set -e
    echo "experiment exit code: $EXIT"
    if [ "$DEBUG_HOLD" = "1" ]; then echo "DEBUG_HOLD: sleeping 1h"; sleep 3600; fi
    sync; sleep 2   # let tee flush to NFS before the pod exits
    exit $EXIT
  '


# ---- fire-and-forget mode (WAIT=0): sweep scripts ----
if [ "${WAIT:-1}" = "0" ]; then
  echo "Submitted (fire-and-forget): $JOB_NAME"
  echo "Results -> $OUTPUT_DIR"
  echo "Check:  runai describe job $JOB_NAME -p $PROJECT"
  exit 0
fi

# ---- wait for the pod to be created (can be slow if queued for a free GPU) ----
POD_NAME=""
for i in $(seq 1 60); do
  POD_NAME=$(kubectl get pods -n "$NAMESPACE" --no-headers \
    -o custom-columns=":metadata.name" 2>/dev/null | grep "^${JOB_NAME}-" | head -1 || true)
  [ -n "$POD_NAME" ] && break
  sleep 5
done
if [ -z "$POD_NAME" ]; then
  echo ""
  echo "Pod not created after ~5 min — the job is most likely queued for a free GPU."
  echo "It is still submitted and will run when a GPU frees up. Check later with:"
  echo "   runai describe job $JOB_NAME -p $PROJECT"
  echo "   kubectl get pods -n $NAMESPACE | grep $JOB_NAME"
  echo "Then tail it:  kubectl logs -n $NAMESPACE <pod-name> -f"
  exit 0
fi

echo "Pod: $POD_NAME"
echo "Live logs:  kubectl logs -n $NAMESPACE $POD_NAME -f"
echo "Results ->  $OUTPUT_DIR"
echo "Waiting for completion (poll loop)..."

FINAL_PHASE=""
while true; do
  PHASE=$(kubectl get pod -n "$NAMESPACE" "$POD_NAME" \
            -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
  case "$PHASE" in
    Succeeded) echo "Pod Succeeded."; FINAL_PHASE="Succeeded"; break ;;
    Failed)    echo "Pod Failed."; FINAL_PHASE="Failed"; break ;;
    *)         sleep 30 ;;
  esac
done

if [ "$FINAL_PHASE" = "Succeeded" ]; then
  echo "Deleting job $JOB_NAME..."
  runai delete job "$JOB_NAME" --project "$PROJECT" || true
  echo "Done. Inspect $OUTPUT_DIR/result.json and stdout.log on the PVC."
else
  echo ""
  echo "Job FAILED. The pod is kept for inspection. Look at:"
  echo "   kubectl logs -n $NAMESPACE $POD_NAME"
  echo "   $OUTPUT_DIR/stdout.log   (on the PVC, if the run got that far)"
  echo "When done, clean up with: runai delete job $JOB_NAME --project $PROJECT"
fi