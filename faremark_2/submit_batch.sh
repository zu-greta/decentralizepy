#!/usr/bin/env bash
# submit_all.sh
# This script submits multiple jobs for different configs/repeats

CONFIG_INDICES=(0 1 2 3 4 5 6 7 8)   # indices corresponding to tables
REPEATS=0  # or 0..4 for multiple runs

for idx in "${CONFIG_INDICES[@]}"; do
  for rep in $(seq 0 $REPEATS); do
    # Create a unique output dir per run
    OUTPUT_DIR="/mnt/nfs/home/zu/results/table${idx}/rep${rep}"
    JOB_NAME="faremark-table${idx}-rep${rep}-$(date +%Y%m%d-%H%M%S)"
    
    # Submit with specific arguments
    runai submit "$JOB_NAME" \
      --project sacs-zu \
      -g 1 \
      --image registry.rcp.epfl.ch/sacs-zu/faremark-custom:latest \
      --pvc sacs-scratch:/mnt/nfs \
      --run-as-uid 325874 \
      --run-as-gid 11259 \
      --memory 32Gi \
      --command -- bash -c "
        set -euo pipefail
        mkdir -p $OUTPUT_DIR /mnt/nfs/home/zu/data
        cd /tmp
        git clone https://github.com/zu-greta/decentralizepy.git /tmp/decentralizepy 2>/dev/null || (cd /tmp/decentralizepy && git pull)
        export PYTHONPATH=/tmp/decentralizepy/faremark_paper
        cd /tmp/decentralizepy/faremark_paper
        python -u scripts/exp_table1.py --config_idx $idx --repeat $rep --device cuda --output_dir $OUTPUT_DIR --data_root /mnt/nfs/home/zu/data 2>&1 | tee $OUTPUT_DIR/stdout.log
        sleep 600
      "
  done
done