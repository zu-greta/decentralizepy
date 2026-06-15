#!/usr/bin/env bash
# =============================================================================
# FareMark — RunAI Interactive Job (for debugging)
# =============================================================================
# Starts a Jupyter-accessible pod with 1 GPU for testing before mass submission.
# Max 12 hours (interactive limit per cluster docs).
#
# Usage:  bash submit_interactive.sh
# Then:   runai bash faremark-debug
# =============================================================================

PROJECT="${RUNAI_PROJECT:-your-lab-project}"
IMAGE="${DOCKER_IMAGE:-your-registry/faremark:latest}"
PVC="${RUNAI_PVC:-faremark-results}"
DATA_PVC="${RUNAI_DATA_PVC:-faremark-data}"

runai submit faremark-debug \
  --project "$PROJECT" \
  --image   "$IMAGE" \
  -g 1 \
  --pvc "${PVC}:/results" \
  --pvc "${DATA_PVC}:/data" \
  --interactive \
  -- sleep infinity

echo ""
echo "Interactive job submitted. Connect with:"
echo "  runai bash faremark-debug"
echo ""
echo "Then run a smoke test:"
echo "  cd /workspace && python run_experiments.py --exp smoke --device cuda --output_dir /results"
echo ""
echo "To delete when done:"
echo "  runai delete job faremark-debug --project $PROJECT"