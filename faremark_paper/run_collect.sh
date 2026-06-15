#!/bin/bash

PROJECT="sacs-zu"
BASE_IMAGE="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
PVC_NAME="sacs-scratch"
MOUNT_PATH="/mnt/nfs"

PERSISTENT_ROOT="$MOUNT_PATH/home/zu"
GIT_REPO="https://github.com/zu-greta/decentralizepy.git"  

JOB_NAME="fm-run-$(date +%Y%m%d-%H%M%S)"
LOCAL_RESULTS_DIR="./results_$(date +%Y%m%d_%H%M%S)"

echo "=== Submitting job: $JOB_NAME ==="

runai submit "$JOB_NAME" \
  --project "$PROJECT" \
  -g 1 \
  --node-pool default \
  --image "$BASE_IMAGE" \
  --pvc "$PVC_NAME:$MOUNT_PATH" \
  --command -- bash -c "
    set -e

    # Create persistent directories (if not exist)
    mkdir -p $PERSISTENT_ROOT/data $PERSISTENT_ROOT/results

    # Install git and other required packages
    apt-get update && apt-get install -y git

    # Install Python dependencies (torch is already present)
    pip install --no-cache-dir torchvision opacus

    echo '=== Cloning repository ==='
    git clone $GIT_REPO /tmp/decentralizepy
    export PYTHONPATH=/tmp/decentralizepy/faremark_paper

    echo '=== Downloading datasets to persistent cache ==='
    DATASET_CACHE=$PERSISTENT_ROOT/data
    if [ ! -f \$DATASET_CACHE/.downloaded ]; then
        python -c '
import torchvision
import os
data_root = \"\$DATASET_CACHE\"
os.makedirs(data_root, exist_ok=True)
for ds in [\"MNIST\", \"CIFAR10\", \"CIFAR100\"]:
    print(f\"Downloading {ds}...\")
    getattr(torchvision.datasets, ds)(root=data_root, download=True)
print(\"All datasets downloaded.\")
        '
        touch \$DATASET_CACHE/.downloaded
    else
        echo 'Datasets already cached.'
    fi

    echo '=== Running experiment ==='
    cd /tmp/decentralizepy/faremark_paper
    python faremark_mod/scripts/exp_table1.py --config_idx 0 --repeat 0 \
      --output_dir $PERSISTENT_ROOT/results \
      --data_root \$DATASET_CACHE \
      --device cuda \
      > $PERSISTENT_ROOT/results/experiment.log 2>&1

    echo '=== Experiment finished ==='
    sleep 7200
  "

echo "Waiting for pod to be created..."
sleep 5

POD_NAME=""
while [ -z "$POD_NAME" ]; do
  POD_NAME=$(kubectl get pods -n runai-sacs-zu --no-headers -o custom-columns=":metadata.name" | grep "^${JOB_NAME}-" | head -1)
  sleep 2
done

echo "Pod: $POD_NAME"

echo "Waiting for experiment.log ..."
while true; do
  if kubectl exec -n runai-sacs-zu "$POD_NAME" -- test -f "$PERSISTENT_ROOT/results/experiment.log" 2>/dev/null; then
    echo "Experiment finished."
    break
  fi
  sleep 10
done

sleep 5

echo "Starting downloader pod ..."
DOWNLOADER_JOB="fm-downloader-$(date +%Y%m%d-%H%M%S)"
runai submit "$DOWNLOADER_JOB" \
  --project "$PROJECT" \
  -g 0 \
  --image alpine:latest \
  --pvc "$PVC_NAME:$MOUNT_PATH" \
  --command -- sleep 3600

sleep 10
DOWNLOADER_POD=$(kubectl get pods -n runai-sacs-zu --no-headers -o custom-columns=":metadata.name" | grep "^${DOWNLOADER_JOB}-" | head -1)

mkdir -p "$LOCAL_RESULTS_DIR"
kubectl cp "runai-sacs-zu/$DOWNLOADER_POD:$PERSISTENT_ROOT/results" "$LOCAL_RESULTS_DIR"

runai delete job "$DOWNLOADER_JOB" --project "$PROJECT"

read -p "Delete main job $JOB_NAME? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  runai delete job "$JOB_NAME" --project "$PROJECT"
fi

echo "Results saved to $LOCAL_RESULTS_DIR"