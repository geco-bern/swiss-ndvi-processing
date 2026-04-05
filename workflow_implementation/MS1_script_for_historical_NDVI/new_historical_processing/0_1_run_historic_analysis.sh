#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Script Configuration
# ============================================================
VENV_PATH="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv"
LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/logs/historic_analysis_$(date "+%Y-%m-%d_%Hh%Mm%S").log"

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
# Configuration
# ============================================================
# Define start and end date for download script --------------
START_DATE="2017-04-01"
END_DATE="2025-11-30"

echo $START_DATE
echo $END_DATE

SCRIPT_0="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_historical_processing/0_create_lookup_table.py"
SCRIPT_1="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_historical_processing/1_download_satellite_images.py"
SCRIPT_2="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_historical_processing/2_historical_ndvi_test.py"

# ============================================================
# Log git repository status
# ============================================================
cd /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_historical_processing/
echo "Current commit:"
git rev-parse HEAD

echo
echo "Commit details:"
git log -10 --oneline --decorate --graph --all

echo
echo "Working tree status:"
git status

echo

# ============================================================
# Start historic processing 
# ============================================================
echo "============================================================"
echo "Historic processing started at: $(timestamp)"
echo "Log file: $LOG_FILE"
echo "============================================================"
echo


# ============================================================
# Run scripts sequentially
# ============================================================
PROCESSING_START=$(date +%s)

echo "Running: with arguments: $START_DATE $END_DATE"
echo "------------------------------------------------------------"


# # ============================================================
# # run preparation script to get median NDVI values for each pixel and DOY
# echo "------------------------------------------------------------"
# echo "Running: ${SCRIPT_0}"; echo "Start time: $(timestamp)"
# START_TIME=$(date +%s)
# python -u "${SCRIPT_0}"
# END_TIME=$(date +%s)
# ELAPSED=$((END_TIME - START_TIME))
# echo "Finished: ${SCRIPT_0}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
# echo
# # THE ABOVE GENERATED: /data_3/francesco/lookup_table_median_ndvi_v7.zarr


# ============================================================
# run download script and check results
echo "------------------------------------------------------------"
echo "Running: ${SCRIPT_1}"; echo "Start time: $(timestamp)"
START_TIME=$(date +%s)
## variant a: # SINGLE CORE RESULTING IN SINGLE FILE:
## variant a: #python -u "${SCRIPT_1}" "$START_DATE" "$END_DATE"
## variant a: #DOWNLOAD_FILE=$(awk 'END{print}' "$LOG_FILE") # Capture last print statement from python script
## variant a: #DOWNLOAD_FILE="/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12.zarr" # TODO: deactivate this

## variant b: MULTI CORE (9 parallel jobs) RESULTING IN MULTIPLE FILES
# START_YEAR=${START_DATE:0:4}
# END_YEAR=${END_DATE:0:4}
# export VENV_PATH SCRIPT_1 LOG_FILE
# seq "$START_YEAR" "$END_YEAR" | parallel -j9 \
#   '$VENV_PATH/bin/python -u $SCRIPT_1 {1}-01-01 {1}-12-31  > ${LOG_FILE}_{1}.log  2>&1'
# # wait is implicit when parallel finishes
# merge together and define DOWNLOAD_FILE
# manually run 1b_merge_satellite_image_downloads.py

DOWNLOAD_FILE="/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Finished: ${SCRIPT_1}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
echo "Download script returned value: $DOWNLOAD_FILE"
echo

# # ============================================================
# # run historical processing script
# echo "------------------------------------------------------------"
# echo "Running: ${SCRIPT_1}"; echo "Start time: $(timestamp)"

# START_TIME=$(date +%s)
# python -u "${SCRIPT_1}" "$DOWNLOAD_FILE" "$HISTO_INPUT"
# NEW_NDVI=$(awk 'END{print}' "$LOG_FILE") # Capture last print statement from python script
# #NEW_NDVI="/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12_processed.zarr"
# END_TIME=$(date +%s)
# ELAPSED=$((END_TIME - START_TIME))
# echo "Finished: ${SCRIPT_1}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
# echo "Merging script returned value: $NEW_NDVI"


# PROCESSING_END=$(date +%s)
# PROCESSING_TIME=$((PROCESSING_END - PROCESSING_START))

# # ============================================================
# # Clean up temporary output data
# # ============================================================

# #rm -rf $NEW_NDVI      # TODO: activate this
# #rm -rf $DOWNLOAD_FILE # TODO: activate this

# # ============================================================
# # Finish processing
# # ============================================================
# echo "============================================================"
# echo "Processing completed successfully"
# echo "End time: $(timestamp)"
# echo "Total runtime: $(format_seconds "$PROCESSING_TIME") (hh:mm:ss)"
# echo "============================================================"
