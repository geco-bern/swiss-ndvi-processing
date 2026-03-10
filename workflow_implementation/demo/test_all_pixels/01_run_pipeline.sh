#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Configuration
# ============================================================
VENV_PATH="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv"
LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/pipeline_FB.log"

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

# This function will check if there are new satellite images.
# No satellite images -> folder empty

is_directory_empty() {
  local dir="$1"
  # Check if directory exists and is empty (excluding . and ..)
  [ -d "$dir" ] && [ -z "$(find "$dir" -mindepth 1 -print -quit 2>/dev/null)" ]
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

rm -rf /mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr
rm -rf /mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/01_ndvi_dataset_spatial_.zarr

# ============================================================
# Run scripts sequentially
# ============================================================
PIPELINE_START=$(date +%s)

# ============================================================
# Define start and end
# ============================================================
# Read previous start date from file, or use default
START_DATE=$(python "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/00_get_last_date.py")
END_DATE="${2:-$(date +%Y-%m-%d)}"


# ============================================================
# Run script 1 and check results
# ============================================================
CURRENT_SCRIPT="${SCRIPTS[0]}"
echo "------------------------------------------------------------"
echo "Running: $CURRENT_SCRIPT"
echo "Start time: $(timestamp)"
echo

START_TIME=$(date +%s)
python "$CURRENT_SCRIPT" "$START_DATE" "$END_DATE"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "Finished: $CURRENT_SCRIPT"
echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
echo

# Check if data directory is empty (no satellite images found)
if is_directory_empty "$DATA_DIR"; then
  echo "------------------------------------------------------------"
  echo "NO SATELLITE IMAGES FOUND for $START_DATE/$END_DATE"
  echo "Data directory is empty: $DATA_DIR"
  echo "Skipping scripts 2-6 and updating date for next run"
  echo "------------------------------------------------------------"
else
  echo "Satellite images found. Continuing with full pipeline..."
  echo

  # ============================================================
  # Run scripts 2-6
  # ============================================================
  for i in {1..5}; do
    SCRIPT="${SCRIPTS[$i]}"
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
      *"/3_add_dates.py" | *"/4_merge_zarr.py" | *"/5_analyse_demo.py" | *"/6_create_cogtiff.py")
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
# Finish pipeline
# ============================================================
echo "============================================================"
echo "Pipeline completed successfully"
echo "End time: $(timestamp)"
echo "Total runtime: $(format_seconds "$PIPELINE_TIME") (hh:mm:ss)"
echo "============================================================"
