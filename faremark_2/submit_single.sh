#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CONFIGURATION – Edit these values
# ============================================================
PROJECT="sacs-zu"
IMAGE="registry.rcp.epfl.ch/sacs-zu/faremark-custom:latest"
PVC="sacs-scratch"
MOUNT="/mnt/nfs"
USER_UID=325874          
USER_GID=11259           
MEMORY="32Gi"
CPUS=4

# Code repository (or local path if already on NFS)
GIT_REPO="https://github.com/zu-greta/decentralizepy.git"
REPO_BRANCH="main"       # optional

# Experiment settings
SCRIPT="scripts/exp_table1.py"   # relative to repo root
SCRIPT_ARGS="--config_idx 0 --repeat 0 --device cuda"
OUTPUT_DIR="${MOUNT}/home/zu/results_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="${MOUNT}/home/zu/data"

# Job name
JOB_NAME="faremark-$(date +%Y%m%d-%H%M%S)"
# ============================================================

echo "=== Submitting job: $JOB_NAME ==="

runai submit "$JOB_NAME" \
  --project "$PROJECT" \
  -g 1 \
  --cpu "$CPUS" \
  --image "$IMAGE" \
  --pvc "$PVC:$MOUNT" \
  --run-as-uid "$USER_UID" \
  --run-as-gid "$USER_GID" \
  --memory "$MEMORY" \
  --command -- bash -c "
    set -euo pipefail
    export USER=zu   # adjust to your username on NFS
    mkdir -p $OUTPUT_DIR $DATA_ROOT

    # Clone or update the repository inside the container
    cd /tmp
    if [ -d /tmp/decentralizepy ]; then
      cd /tmp/decentralizepy && git pull
    else
      git clone --branch $REPO_BRANCH $GIT_REPO /tmp/decentralizepy
    fi

    # Set PYTHONPATH to include the package
    export PYTHONPATH=/tmp/decentralizepy/faremark_paper:\$PYTHONPATH
    cd /tmp/decentralizepy/faremark_paper

    # Run the experiment
    python -u $SCRIPT $SCRIPT_ARGS \
      --output_dir $OUTPUT_DIR \
      --data_root $DATA_ROOT \
      2>&1 | tee $OUTPUT_DIR/stdout.log

    # Keep pod alive for 1 hour after completion for manual inspection
    echo 'Experiment finished. Sleeping 3600s for manual inspection...'
    sleep 3600
  "

# Wait for pod to be created
sleep 10

# Get pod name
POD_NAME=$(kubectl get pods -n runai-"$PROJECT" --no-headers -o custom-columns=":metadata.name" | grep "^${JOB_NAME}-" | head -1)
if [ -z "$POD_NAME" ]; then
  echo "ERROR: Could not find pod for job $JOB_NAME"
  exit 1
fi

echo ""
echo "✅ Job submitted successfully!"
echo "📊 Monitor progress with:"
echo "   kubectl logs -n runai-$PROJECT $POD_NAME -f"
echo ""
echo "📁 Results will be saved to: $OUTPUT_DIR"
echo ""

# Wait for completion
echo "⏳ Waiting for job to finish (this may take hours)..."
kubectl wait --for=condition=complete pod/"$POD_NAME" -n runai-"$PROJECT" --timeout=36000s 2>/dev/null || {
  echo "Job may have failed. Checking status..."
}

# Final status
POD_STATUS=$(kubectl get pod -n runai-"$PROJECT" "$POD_NAME" -o jsonpath='{.status.phase}')
if [ "$POD_STATUS" == "Succeeded" ]; then
  echo "✅ Job completed successfully."
else
  echo "⚠️  Job finished with status: $POD_STATUS. Check logs for errors."
fi

# Cleanup
echo "🧹 Deleting job $JOB_NAME..."
runai delete job "$JOB_NAME" --project "$PROJECT"

echo "Done. Results are in $OUTPUT_DIR on the persistent storage."