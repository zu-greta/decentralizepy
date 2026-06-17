#!/usr/bin/env bash
set -euo pipefail

# ===================================================
#  Usage:
#     ./submit_experiment.sh [CONFIG_IDX] [REPEAT]
#  Example (smoke test):
#     ./submit_experiment.sh 0 0
#  Example (Table I: ResNet-18 / CIFAR-10):
#     ./submit_experiment.sh 1 0
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
# Where the faremark_greta package lives inside the cloned repo.
PKG_SUBDIR="faremark_greta"
SCRIPT="scripts/run_experiment.py"

RUN_TAG="cfg${CONFIG_IDX}_rep${REPEAT}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${MOUNT}/home/zu/results/${RUN_TAG}"
DATA_ROOT="${MOUNT}/home/zu/data"
JOB_NAME="faremark-cfg${CONFIG_IDX}-rep${REPEAT}-$(date +%H%M%S)"
# =====================================================

echo "=== Submitting $JOB_NAME (config_idx=$CONFIG_IDX repeat=$REPEAT) ==="

HOLD_CMD=""
if [ "$DEBUG_HOLD" = "1" ]; then
  HOLD_CMD="echo 'DEBUG_HOLD: sleeping 1h for inspection'; sleep 3600"
fi

runai submit "$JOB_NAME" \
  --project "$PROJECT" \
  -g 1 \
  --image "$IMAGE" \
  --pvc "$PVC:$MOUNT" \
  --run-as-uid "$USER_UID" \
  --run-as-gid "$USER_GID" \
  --memory "$MEMORY" \
  --command -- bash -c "
    set -euo pipefail
    export USER=zu
    mkdir -p '$OUTPUT_DIR' '$DATA_ROOT'
    rm -rf /tmp/decentralizepy
    git clone --depth 1 --branch '$GIT_BRANCH' '$GIT_REPO' /tmp/decentralizepy
    export PYTHONPATH=/tmp/decentralizepy/${PKG_SUBDIR}
    cd /tmp/decentralizepy/${PKG_SUBDIR}
    python -u '$SCRIPT' \
      --config_idx $CONFIG_IDX --repeat $REPEAT --device cuda \
      --output_dir '$OUTPUT_DIR' --data_root '$DATA_ROOT' \
      2>&1 | tee '$OUTPUT_DIR/stdout.log'
    EXIT=\${PIPESTATUS[0]}
    echo \"experiment exit code: \$EXIT\"
    $HOLD_CMD
    exit \$EXIT
  "

# ---- wait for the pod, then report + clean up ----
sleep 5
POD_NAME=$(kubectl get pods -n "$NAMESPACE" --no-headers \
  -o custom-columns=":metadata.name" | grep "^${JOB_NAME}-" | head -1 || true)
if [ -z "$POD_NAME" ]; then
  echo "ERROR: could not find pod for job $JOB_NAME"; exit 1
fi

echo "Pod: $POD_NAME"
echo "Live logs:  kubectl logs -n $NAMESPACE $POD_NAME -f"
echo "Results ->  $OUTPUT_DIR"
echo "Waiting for completion (poll loop)..."

# Poll the pod phase instead of `kubectl wait --for=condition=complete`
# (that condition is for Jobs, not bare pods, and silently times out).
while true; do
  PHASE=$(kubectl get pod -n "$NAMESPACE" "$POD_NAME" \
            -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
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