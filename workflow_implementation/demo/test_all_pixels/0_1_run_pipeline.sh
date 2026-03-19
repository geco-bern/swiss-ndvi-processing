#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Script Configuration
# ============================================================
VENV_PATH="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv"
LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/pipeline_FB_$(date "+%Y-%m-%d_%Hh%Mm%S").log"

# Logging setup (terminal + file) -------------------------------
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Error handling ------------------------------------------------
CURRENT_SCRIPT="N/A"
trap 'echo "[ERROR] Script failed: $CURRENT_SCRIPT | Time: $(date)"' ERR

# Helpers -------------------------------------------------------
timestamp(){ date "+%Y-%m-%d %H:%M:%S"; }
format_seconds() { local s=$1; printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60)); }

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
# Pipeline Configuration
# ============================================================
# Define historical NDVI update mode --------------
# Script inputs: 
#   Define which historical NDVI file to use, and whether it will be udpated in-place or copied.
HISTO_INPUT="/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m_copy.zarr" 
HISTO_OUTPUT="/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_extended.zarr" # currently unused since we append INPUT

# Workaround to ensure we have the original untouched
HISTO_ORIG="/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m.zarr/" # Ensure we have the original untouched
if [ -d "$HISTO_INPUT" ]; then rm -r -- "$HISTO_INPUT"; fi
rsync -a $HISTO_ORIG $HISTO_INPUT # note the important trailing slash in source/

# Define start and end date for download script --------------
# End date:
END_DATE="${2:-$(date -d "yesterday" +%Y-%m-%d)}" # Yesterday
END_DATE="2025-12-12" # Or hardcode it alternatively
# Start date:
# Read previous start date from historical NDVI file
SCRIPT_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/0_2_get_last_date.py"
START_DATE=$(python $SCRIPT_FILE $HISTO_INPUT)

echo $START_DATE
echo $END_DATE

SCRIPTS=(
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/1_extract_swisstopo_dataset.py
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr.py
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo_efficient.py
  /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py
)

# ============================================================
# Start pipeline 
# ============================================================
echo "============================================================"
echo "Pipeline started at: $(timestamp)"
echo "Log file: $LOG_FILE"
echo "============================================================"
echo


# ============================================================
# Run scripts sequentially
# ============================================================
PIPELINE_START=$(date +%s)

echo "Running: with arguments: $START_DATE $END_DATE"
echo "------------------------------------------------------------"

# ============================================================
# run script 01 and check results
echo "------------------------------------------------------------"
echo "Running: ${SCRIPTS[0]}"; echo "Start time: $(timestamp)"
START_TIME=$(date +%s)
#python -u "${SCRIPTS[0]}" "$START_DATE" "$END_DATE"
#DOWNLOAD_FILE=$(awk 'END{print}' "$LOG_FILE") # Capture last print statement from python script
DOWNLOAD_FILE="/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12.zarr" # TODO: deactivate this
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

  # run script 04
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPTS[1]}"; echo "Start time: $(timestamp)"

  START_TIME=$(date +%s)
  python -u "${SCRIPTS[1]}" "$DOWNLOAD_FILE" "$HISTO_INPUT"
  NEW_NDVI=$(awk 'END{print}' "$LOG_FILE") # Capture last print statement from python script
  #NEW_NDVI="/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12_processed.zarr"
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPTS[1]}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
  echo "Merging script returned value: $NEW_NDVI"

  # run script 05
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPTS[2]}"; echo "Start time: $(timestamp)"
  START_TIME=$(date +%s)
  #python -u "${SCRIPTS[2]}" "$NEW_NDVI" "$HISTO_INPUT" --histo-output="$HISTO_OUTPUT"
  python -u "${SCRIPTS[2]}" "$NEW_NDVI" "$HISTO_INPUT" --histo-output="$HISTO_INPUT" # NOTE: this would overwrite.
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPTS[2]}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"

  # run script 06
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPTS[3]}"; echo "Start time: $(timestamp)"
  START_TIME=$(date +%s)
  # python "${SCRIPTS[3]}" "$START_DATE" "$END_DATE"
  # # python "${SCRIPTS[3]}" "2025-08-22" "2025-09-30" # TODO remove this hardcoding
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPTS[3]}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"

fi

PIPELINE_END=$(date +%s)
PIPELINE_TIME=$((PIPELINE_END - PIPELINE_START))

# ============================================================
# Clean up temporary output data
# ============================================================

#rm -rf $NEW_NDVI      # TODO: activate this
#rm -rf $DOWNLOAD_FILE # TODO: activate this

# ============================================================
# Finish pipeline
# ============================================================
echo "============================================================"
echo "Pipeline completed successfully"
echo "End time: $(timestamp)"
echo "Total runtime: $(format_seconds "$PIPELINE_TIME") (hh:mm:ss)"
echo "============================================================"
