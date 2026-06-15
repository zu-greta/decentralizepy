#!/bin/bash
PROJECT="sacs-zu"
IMAGE="registry.rcp.epfl.ch/sacs-zu/faremark-custom:latest"
PVC="sacs-scratch"
MOUNT="/mnt/nfs"
PERSISTENT_ROOT="$MOUNT/home/zu"
CODE_DIR="$PERSISTENT_ROOT/code"
RESULTS_DIR="$PERSISTENT_ROOT/results"
DATA_DIR="$PERSISTENT_ROOT/data"

JOB_NAME="fm-final-$(date +%Y%m%d-%H%M%S)"
LOCAL_RESULTS="./results_$(date +%Y%m%d_%H%M%S)"

runai submit "$JOB_NAME" \
  --project "$PROJECT" \
  -g 1 \
  --image "$IMAGE" \
  --pvc "$PVC:$MOUNT" \
  --run-as-uid 325874 \
  --run-as-gid 11259 \
  --command -- bash -c "
    set -e
    echo '=== Ensuring directories exist ==='
    mkdir -p $DATA_DIR $RESULTS_DIR

    echo '=== Using cached datasets (if any) ==='
    if [ ! -f $DATA_DIR/.downloaded ]; then
        python -c '
import torchvision, os
os.makedirs(\"$DATA_DIR\", exist_ok=True)
for ds in [\"MNIST\",\"CIFAR10\",\"CIFAR100\"]:
    print(f\"Downloading {ds}\")
    getattr(torchvision.datasets, ds)(root=\"$DATA_DIR\", download=True)
'
        touch $DATA_DIR/.downloaded
    fi

    echo '=== Running experiment ==='
    export PYTHONPATH=$CODE_DIR/faremark_paper
    cd $CODE_DIR/faremark_paper
    python faremark_mod/scripts/exp_table1.py --config_idx 0 --repeat 0 \
      --output_dir $RESULTS_DIR \
      --data_root $DATA_DIR \
      --device cuda \
      > $RESULTS_DIR/experiment.log 2>&1

    echo '=== Experiment finished, sleeping ==='
    sleep 7200
  "

# Wait and download results (same as before)
echo "Waiting for pod..."
sleep 10
POD_NAME=$(kubectl get pods -n runai-sacs-zu --no-headers -o custom-columns=":metadata.name" | grep "^${JOB_NAME}-" | head -1)
echo "Pod: $POD_NAME"

while ! kubectl exec -n runai-sacs-zu "$POD_NAME" -- test -f "$RESULTS_DIR/experiment.log" 2>/dev/null; do
  sleep 10
done

echo "Results ready. Starting downloader pod..."
DOWNLOADER="fm-dl-$(date +%Y%m%d-%H%M%S)"
runai submit "$DOWNLOADER" --project "$PROJECT" -g 0 --image alpine:latest \
  --pvc "$PVC:$MOUNT" --run-as-uid 325874 --run-as-gid 11259 --command -- sleep 3600
sleep 10
DL_POD=$(kubectl get pods -n runai-sacs-zu -o name | grep "$DOWNLOADER" | cut -d/ -f2)
mkdir -p "$LOCAL_RESULTS"
kubectl cp "runai-sacs-zu/$DL_POD:$RESULTS_DIR" "$LOCAL_RESULTS"
runai delete job "$DOWNLOADER" --project "$PROJECT"

read -p "Delete main job $JOB_NAME? (y/n) " -n 1 -r; echo
if [[ $REPLY =~ ^[Yy]$ ]]; then runai delete job "$JOB_NAME" --project "$PROJECT"; fi

echo "Results saved to $LOCAL_RESULTS"