#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Configuration
# -----------------------------
VENV_PATH="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/"
LOG_FILE="pipeline.log"

SCRIPTS=(
    /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/1_extract_swisstopo_dataset.py
    /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/2_transpose_swisstopo_dataset.py
    /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/3_add_dates.py
    /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/4_merge_zarr.py
    /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/5_analyse_demo.py
    /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/6_create_cogtiff.py
)

# -----------------------------
# Setup logging
# -----------------------------
# Redirect all output (stdout + stderr) to log file AND terminal
: > "$LOG_FILE"

echo "========================================"
echo "Pipeline started at: $(date)"
echo "========================================"

# -----------------------------
# Activate virtual environment
# -----------------------------
if [[ ! -d "$VENV_PATH" ]]; then
  echo "ERROR: Virtual environment not found at $VENV_PATH"
  exit 1
fi

source "$VENV_PATH/bin/activate"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo

# -----------------------------
# Run scripts sequentially
# -----------------------------
for SCRIPT in "${SCRIPTS[@]}"; do
  echo "----------------------------------------"
  echo "Running $SCRIPT"
  START_TIME=$(date +%s)

  python "$SCRIPT"

  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))

  echo "Finished $SCRIPT in ${ELAPSED}s"
  echo
done

echo "========================================"
echo "Pipeline completed successfully at: $(date)"
echo "========================================"
