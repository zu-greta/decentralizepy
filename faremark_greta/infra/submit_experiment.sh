#!/usr/bin/env bash
set -euo pipefail

# ===================================================
#  Usage:
#     ./submit_experiment.sh [CONFIG_IDX] [REPEAT]
#  Example (smoke test):       ./submit_experiment.sh 0 0
#  Example (Table I RN18/C10): ./submit_experiment.sh 1 0
#
#  Optional env overrides (handy for Stage-2 Fig.7 sweeps):
#     NUM_FREE_RIDERS=4 ATTACK=previous_models ./submit_experiment.sh 8 0
#     ROUNDS=10 BATCH_SIZE=64 ./submit_experiment.sh 7 0
#  Set DEBUG_HOLD=1 to keep the pod alive 1h after the run for inspection.
# ===================================================
CONFIG_IDX="${1:-0}"
REPEAT="${2:-0}"
DEBUG_HOLD="${DEBUG_HOLD:-0}"

# ---- Cluster / account config ----
PROJECT="sacs-zu"
# Must match infra/build.sh IMAGE_NAME.
IMAGE="registry.rcp.epfl.ch/sacs-zu/faremark:latest"
PVC="sacs-scratch"
MOUNT="/mnt/nfs"
USER_UID=325874
USER_GID=11259
MEMORY="32Gi"
NAMESPACE="runai-sacs-zu"

# ---- Code + paths ----
GIT_REPO="https://github.com/zu-greta/decentralizepy.git"
GIT_BRANCH="main"
PKG_SUBDIR="faremark_greta"
SCRIPT="scripts/run_experiment.py"

# ---- Optional Python overrides assembled from env vars ----
# Only the ones you set get forwarded; everything else uses the config defaults.
PY_EXTRA=""
[ -n "${NUM_FREE_RIDERS:-}" ] && PY_EXTRA="$PY_EXTRA --num_free_riders ${NUM_FREE_RIDERS}"
[ -n "${ATTACK:-}" ]          && PY_EXTRA="$PY_EXTRA --attack ${ATTACK}"
[ -n "${NOISE_SIGMA:-}" ]     && PY_EXTRA="$PY_EXTRA --noise_sigma ${NOISE_SIGMA}"
[ -n "${ROUNDS:-}" ]          && PY_EXTRA="$PY_EXTRA --rounds ${ROUNDS}"
[ -n "${BATCH_SIZE:-}" ]      && PY_EXTRA="$PY_EXTRA --batch_size ${BATCH_SIZE}"

# Tag results/job uniquely so parallel sweeps never collide.
FR_TAG=""
[ -n "${NUM_FREE_RIDERS:-}" ] && FR_TAG="-fr${NUM_FREE_RIDERS}"
RUN_TAG="cfg${CONFIG_IDX}_rep${REPEAT}${FR_TAG}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${MOUNT}/home/zu/results/${RUN_TAG}"
DATA_ROOT="${MOUNT}/home/zu/data"
JOB_NAME="faremark-c${CONFIG_IDX}-r${REPEAT}${FR_TAG}-$(date +%H%M%S)"
# =====================================================

echo "=== Submitting $JOB_NAME (config_idx=$CONFIG_IDX repeat=$REPEAT) ==="

# Pass all paths/values as ENV VARS (-e), expanded here by the outer shell into
# simple KEY=VALUE flags (safe — no nested quoting). The command script below is
# wrapped in SINGLE quotes so the outer shell does NOT touch it; every $VAR in it
# is expanded by the CONTAINER's bash from the env we injected. This avoids the
# quoting trap where variables vanish when runai re-parses the command string.
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
  -e "DEBUG_HOLD=$DEBUG_HOLD" \
  --command -- bash -c '
    set -euo pipefail
    export USER=zu
    mkdir -p "$OUTPUT_DIR" "$DATA_ROOT"
    # Mirror ALL output to a log on the PVC so the run is debuggable even if the
    # pod dies abruptly (kubectl logs may be gone; NFS stdout buffers can be lost).
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
      echo "Did you commit+push faremark_paper/ to branch $GIT_BRANCH of $GIT_REPO?"
      sync; sleep 2; exit 3
    fi
    export PYTHONPATH="/tmp/decentralizepy/$PKG_SUBDIR"
    cd "/tmp/decentralizepy/$PKG_SUBDIR"
    echo "package dir:"; ls -la
    set +e
    python -u "$SCRIPT" --config_idx "$CONFIG_IDX" --repeat "$REPEAT" --device cuda --output_dir "$OUTPUT_DIR" --data_root "$DATA_ROOT" $PY_EXTRA
    EXIT=$?
    set -e
    echo "experiment exit code: $EXIT"
    if [ "$DEBUG_HOLD" = "1" ]; then echo "DEBUG_HOLD: sleeping 1h"; sleep 3600; fi
    sync; sleep 2   # let tee flush to NFS before the pod exits
    exit $EXIT
  '


# ---- fire-and-forget mode (WAIT=0): used by the sweep scripts ----
# Submit the job and return immediately so many jobs can be queued at once and
# the cluster runs them as GPUs free up. Default WAIT=1 keeps the old blocking
# behaviour (wait for completion + auto-cleanup) for single interactive runs.
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
  exit 0   # not a failure — the job lives on the cluster independently
fi

echo "Pod: $POD_NAME"
echo "Live logs:  kubectl logs -n $NAMESPACE $POD_NAME -f"
echo "Results ->  $OUTPUT_DIR"
echo "Waiting for completion (poll loop)..."

# Poll the pod phase instead of `kubectl wait --for=condition=complete`
# (that condition is for Jobs, not bare pods, and silently times out).
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
  # Do NOT delete on failure — keep the pod so its logs stay readable.
  echo ""
  echo "Job FAILED. The pod is kept for inspection. Look at:"
  echo "   kubectl logs -n $NAMESPACE $POD_NAME"
  echo "   $OUTPUT_DIR/stdout.log   (on the PVC, if the run got that far)"
  echo "When done, clean up with: runai delete job $JOB_NAME --project $PROJECT"
fi