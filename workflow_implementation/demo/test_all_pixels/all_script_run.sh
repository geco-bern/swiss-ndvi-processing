#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Configuration
# ============================================================
VENV_PATH="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv"
LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/pipeline.log"

# Read previous start date from file, or use default
if [[ -f /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/next_start_date.txt ]]; then
    START_DATE=$(cat /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/next_start_date.txt )
else
    START_DATE="${1:-2025-12-01}"
fi

END_DATE="${2:-$(date +%Y-%m-%d)}"

SCRIPTS=(
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/1_extract_swisstopo_dataset.py
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/2_transpose_swisstopo_dataset.py
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/3_add_dates.py
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr.py
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo.py
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py
)

# ============================================================
# Logging setup (terminal + file)
# ============================================================
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# Error handling
# ============================================================
CURRENT_SCRIPT="N/A"
trap 'echo "[ERROR] Script failed: $CURRENT_SCRIPT | Time: $(date)"' ERR

# ============================================================
# Helpers
# ============================================================
timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

format_seconds() {
  local s=$1
  printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# ============================================================
# Start pipeline
# ============================================================
echo "============================================================"
echo "Pipeline started at: $(timestamp)"
echo "Log file: $LOG_FILE"
echo "============================================================"
echo

# ============================================================
# Activate virtual environment
# ============================================================
if [[ ! -d "$VENV_PATH" ]]; then
  echo "[ERROR] Virtual environment not found: $VENV_PATH"
  exit 1
fi

source "$VENV_PATH/bin/activate"

echo "Virtual environment activated"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo

# remove all the data 

rm -rf /mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/

# ============================================================
# Run scripts sequentially
# ============================================================
PIPELINE_START=$(date +%s)

for SCRIPT in "${SCRIPTS[@]}"; do
  CURRENT_SCRIPT="$SCRIPT"

  if [[ ! -f "$SCRIPT" ]]; then
    echo "[ERROR] Script not found: $SCRIPT"
    exit 1
  fi

  echo "------------------------------------------------------------"
  echo "Running: $SCRIPT"
  echo "Start time: $(timestamp)"

  START_TIME=$(date +%s)

  case "$SCRIPT" in
    *"/1_extract_swisstopo_dataset.py" \
    | *"/3_add_dates.py" \
    | *"/4_merge_zarr.py" \
    | *"/5_analyse_demo.py")
      python "$SCRIPT" "$START_DATE" "$END_DATE"
      ;;
    *)
      python "$SCRIPT"
      ;;
  esac

  END_TIME=$(date +%s)


  ELAPSED=$((END_TIME - START_TIME))

  echo "Finished: $SCRIPT"
  echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
  echo
done

PIPELINE_END=$(date +%s)
PIPELINE_TIME=$((PIPELINE_END - PIPELINE_START))

# ============================================================
# Update start_date for next run (save to file)
# ============================================================
NEW_START_DATE=$(date -d "$END_DATE + 1 day" +%Y-%m-%d)
echo "$NEW_START_DATE" > /tmp/pipeline_next_start_date.txt

# ============================================================
# Finish pipeline
# ============================================================
echo "============================================================"
echo "Pipeline completed successfully"
echo "End time: $(timestamp)"
echo "Total runtime: $(format_seconds "$PIPELINE_TIME") (hh:mm:ss)"
echo "Next pipeline will use start_date: $NEW_START_DATE"
echo "Saved to: /tmp/pipeline_next_start_date.txt"
echo "============================================================"
