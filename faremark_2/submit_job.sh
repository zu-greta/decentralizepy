#!/bin/bash
# Submit a RunAI job for FareMark experiments

runai submit \
    -i registry.rcp.epfl.ch/sacs-zu/faremark-2:latest \
    -p sacs-zu \
    --name faremark-experiment \
    --gpu 1 \
    --node-pool default \
    --pvc sacs-scratch:/mnt/nfs \
    -e GIT_USER_NAME="zu-greta" \
    -e GIT_USER_EMAIL="gretarm.zu@gmail.com" \
    --command -- /bin/bash -c "cd /mnt/nfs/home/zu/decentralizepy/faremark_2 && python scripts/run_validation.py && sleep infinity"