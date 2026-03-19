#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Configuration
# ============================================================
VENV_PATH="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv"
LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/pipeline_FB_$(date "+%Y-%m-%d_%Hh%Mm%S").log"

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

# ============================================================
# Run scripts sequentially
# ============================================================
PIPELINE_START=$(date +%s)

# ============================================================
# Define start and end
# ============================================================
# Read previous start date from file, or use default
# START_DATE="2025-11-30" # TODO: deactivate this
START_DATE=$(python "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/0_2_get_last_date.py")
echo $START_DATE
# END_DATE="${2:-$(date +%Y-%m-%d)}" # TODO activate this for
# END_DATE="2025-12-04" # TODO: deactivate this
# END_DATE="2025-12-06" # TODO: deactivate this
END_DATE="2025-12-12" # TODO: deactivate this
# END_DATE="2026-03-05" # TODO: deactivate this

echo "Running: with arguments: $START_DATE $END_DATE"
echo "------------------------------------------------------------"

# ============================================================
# run script 01 and check results
echo "------------------------------------------------------------"
echo "Running: ${SCRIPTS[0]}"; echo "Start time: $(timestamp)"
START_TIME=$(date +%s)
python -u "${SCRIPTS[0]}" "$START_DATE" "$END_DATE"
DOWNLOAD_FILE=$(awk 'END{print}' "$LOG_FILE") # Capture last print statement from python script
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Finished: ${SCRIPTS[0]}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
echo "Download script returned value: $DOWNLOAD_FILE"
echo

# ============================================================
# Run scripts 2-6
# ============================================================
# Check if data directory is empty (no new satellite images downloaded)
if [[ "${DOWNLOAD_FILE:-}" == "No data downloaded." ]] || [ ! -d "$DOWNLOAD_FILE" ] || [ -z "$(find "$DOWNLOAD_FILE" -mindepth 1 -print -quit)" ]; then
    echo "------------------------------------------------------------"
    echo "NO NEW SATELLITE IMAGE DOWNLOADS FOUND for $START_DATE/$END_DATE"
    echo "Data directory is empty: $DOWNLOAD_FILE"
    echo "Skipping execution of scripts 2-6"
    echo "------------------------------------------------------------"
else
  echo "New satellite image download found. Continuing with full pipeline..."
  echo

  # run script 02
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPTS[1]}"; echo "Start time: $(timestamp)"
  START_TIME=$(date +%s)
  python "${SCRIPTS[1]}" "$DOWNLOAD_FILE"
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPTS[1]}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"

  # run script 03
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPTS[2]}"; echo "Start time: $(timestamp)"
  START_TIME=$(date +%s)
  # python "${SCRIPTS[2]}" "$START_DATE" "$END_DATE" "$DOWNLOAD_FILE"
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPTS[2]}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"

  DOWNLOAD_FILE2="${DOWNLOAD_FILE/01_downloadedA/02-03_downloadedB}" # TODO: this could be replace by either combining scripts 1 to 3 or by returning the new name in script 2.

  # run script 04
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPTS[3]}"; echo "Start time: $(timestamp)"
  START_TIME=$(date +%s)
  # python "${SCRIPTS[3]}" "$START_DATE" "$END_DATE" "$DOWNLOAD_FILE2"
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPTS[3]}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"

  # run script 05
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPTS[4]}"; echo "Start time: $(timestamp)"
  START_TIME=$(date +%s)
  # python "${SCRIPTS[4]}" "$START_DATE" "$END_DATE"
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPTS[4]}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"

  # run script 06
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPTS[5]}"; echo "Start time: $(timestamp)"
  START_TIME=$(date +%s)
  # python "${SCRIPTS[5]}" "$START_DATE" "$END_DATE"
  # python "${SCRIPTS[5]}" "2025-08-22" "2025-09-30" # TODO remove this hardcoding
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPTS[5]}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
fi

PIPELINE_END=$(date +%s)
PIPELINE_TIME=$((PIPELINE_END - PIPELINE_START))

# ============================================================
# Clean up temporary output data
# ============================================================

# TODO: rm -rf /mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr
# TODO: rm -rf /mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/tmp_ndvi_02-03_downloadedB.zarr
# TODO: rm -rf /mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/01_ndvi_dataset_spatial_.zarr
# TODO: rm -rf /mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_01_downloadedA.zarr # not needed if we store into /var/tmp/

# ============================================================
# Finish pipeline
# ============================================================
echo "============================================================"
echo "Pipeline completed successfully"
echo "End time: $(timestamp)"
echo "Total runtime: $(format_seconds "$PIPELINE_TIME") (hh:mm:ss)"
echo "============================================================"
