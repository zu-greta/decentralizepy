#!/bin/bash
# submit_all_tables.sh
# This submits one job per table (or multiple repeats) using the runai submit command.

# Source the configuration variables
PROJECT="sacs-zu"
IMAGE="registry.rcp.epfl.ch/sacs-zu/faremark-custom:latest"  # your built image
PVC="sacs-scratch"
MOUNT="/mnt/nfs"
USER_UID=325874
USER_GID=11259
MEMORY="32Gi"

# Define experiments (you can add more)
CONFIGS=(
  "exp_fidelity.yaml"
  "exp_detection.yaml"
  "exp_free_rider.yaml"
  "exp_robustness.yaml"
  "exp_ablation.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
  # Create a unique output directory for each config
  OUTPUT_DIR="${MOUNT}/home/zu/results/${CONFIG%.yaml}_$(date +%Y%m%d_%H%M%S)"
  JOB_NAME="faremark-${CONFIG%.yaml}-$(date +%Y%m%d-%H%M%S)"

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
      mkdir -p $OUTPUT_DIR /mnt/nfs/home/zu/data

      # Clone the code (or use the already-mounted copy)
      cd /tmp
      if [ ! -d /tmp/faremark ]; then
        git clone https://github.com/zu-greta/faremark-reproduction.git /tmp/faremark
      else
        cd /tmp/faremark && git pull
      fi

      export PYTHONPATH=/tmp/faremark
      cd /tmp/faremark

      # Run the experiment with the given config
      python scripts/run_experiment.py --config configs/$CONFIG \
        --output_dir $OUTPUT_DIR \
        --data_root /mnt/nfs/home/zu/data \
        2>&1 | tee $OUTPUT_DIR/stdout.log

      # Keep pod alive for inspection (optional)
      sleep 600
    "
done