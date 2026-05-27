# NOTE: A good test is to run this pipeline with END_DATE="2025-12-12"
#       Use the variant where it updates the HISTO_INPUT.
#       Then modify the END_DATE="2025-12-28"
#       and comment out the creation the working copy from the backup.
#       Then rerun this script: now it should only download incrementally and append.
#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Script Configuration
# ============================================================
VENV_PATH="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv"
LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/pipeline_logs/pipeline_$(date "+%Y-%m-%d_%Hh%Mm%S").log"

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
#   Define which historical NDVI file to use, and whether it will be updated in-place or copied.
HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-focus-sites.zarr"           # NOTE: this is to run a subset
#HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-1kmX1km.zarr"              # NOTE: this is to run a subset
#HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-pixels10401-to-10500.zarr" # NOTE: this is to run a subset
#HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-focus-sites_1kmX1km.zarr"  # NOTE: this is to run a subset
#HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-10kmX10km.zarr"            # NOTE: this is to run a subset
#HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-100kmX100km.zarr"          # NOTE: this is to run a subset
#HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c.zarr"                             # TODO: this is to run the full data

HISTO_OUTPUT="/mnt/data2/UniBe-swiss-ndvi/output_data/tmp_out_ndvi_historic_extended_run-on-$(date +%Y%m%d-%H%M%S).zarr"

# (OPTIONAL): activate below if you want test overwriting and start with a fresh HISTO_INPUT. (OR RUN ONCE, IF YOUR HISTO_INPUT IS NOT YET EXISTING.)
# # Workaround creating the working copy from the backup:
# HISTO_BKP="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-focus-sites.zarr_bkp/"          # NOTE: requires trailing `/` NOTE: Ensure we have the original untouched
# HISTO_BKP="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-1kmX1km.zarr_bkp/"              # NOTE: requires trailing `/` NOTE: Ensure we have the original untouched
# HISTO_BKP="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-pixels10401-to-10500.zarr_bkp/" # NOTE: requires trailing `/` NOTE: Ensure we have the original untouched
# HISTO_BKP="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-focus-sites_1kmX1km.zarr_bkp/"  # NOTE: requires trailing `/` NOTE: Ensure we have the original untouched
# HISTO_BKP="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-10kmX10km.zarr_bkp/"            # NOTE: requires trailing `/` NOTE: Ensure we have the original untouched
# HISTO_BKP="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-100kmX100km.zarr_bkp/"          # NOTE: requires trailing `/` NOTE: Ensure we have the original untouched
# if [ -d "$HISTO_INPUT" ] && [ -d "$HISTO_BKP" ]; then rm -rf -- "$HISTO_INPUT"; else echo "Error backup file does not exist. Did not delete." >&2; fi  # CAREFUL THIS DELETES HISTO_INPUT (ensure you have a backup)
# rsync -rltDg --no-perms --chmod=ugo=rwX $HISTO_BKP $HISTO_INPUT # note the important trailing slash in source/

# Define start and end date for download script --------------
# End date:
## derive from date:
# END_DATE="${2:-$(date -d "yesterday" +%Y-%m-%d)}" # Yesterday # TODO: activate this for continuous application
## Or hardcode it alternatively:
END_DATE="2026-01-06"
# END_DATE="2026-01-15" # For a second test run after the first.
# Latter can e.g. be used to simulate two subsequent updates of the CI. In that case ensure that SCRIPT_2 does overwrite (ensure to have a backup).


# Start date:
# Read previous start date from historical NDVI file
SCRIPT_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/0_2_get_last_date.py"
START_DATE=$(python $SCRIPT_FILE $HISTO_INPUT)

echo $START_DATE
echo $END_DATE

SCRIPT_0=/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/1_extract_swisstopo_dataset.py
SCRIPT_1=/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/4_merge_zarr.py
SCRIPT_2=/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/5_analyse_demo_efficient.py
SCRIPT_3=/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/6_create_cogtiff.py

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
echo "Running: ${SCRIPT_0}"; echo "Start time: $(timestamp)"
START_TIME=$(date +%s)
#python -u "${SCRIPT_0}" "$START_DATE" "$END_DATE"
#DOWNLOAD_FILE=$(grep '1_... .py created file: ' "$LOG_FILE" | tail -n1 | sed 's/^1_... .py created file: //') # Capture output file path from python script
# instead of running the download each time provide the downloaded file.
#DOWNLOAD_FILE="/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-24_00h27_ndvi_01_downloaded_2025-11-30_2025-12-28.zarr"     # NOTE: if user wants to avoid running /1_extract_swisstopo_dataset, specify NEW_NDVI.
#DOWNLOAD_FILE="/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-04-21_08hXXXXXX_ndvi_01_downloaded_2026-01-01_2026-01-15.zarr" # NOTE: if user wants to avoid running /1_extract_swisstopo_dataset, specify NEW_NDVI.
DOWNLOAD_FILE="/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-05-25_22h36_ndvi_01_downloaded_2025-12-31_2026-01-06.zarr"      # NOTE: if user wants to avoid running /1_extract_swisstopo_dataset, specify NEW_NDVI.
# This can e.g. be used to simulate two subsequent updates of the CI:
#DOWNLOAD_FILE="/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12.zarr" # step 1 NOTE: run one step and then deactivate this # TODO: dates need to be updated
#DOWNLOAD_FILE="/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-19_18h40_ndvi_01_downloaded_2025-12-12_2025-12-28.zarr" # step 2 NOTE: run one step and then activate this  # TODO: dates need to be updated
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Finished: ${SCRIPT_0}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
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
  echo "Running: ${SCRIPT_1}"; echo "Start time: $(timestamp)"

  START_TIME=$(date +%s)
  python -u "${SCRIPT_1}" "$DOWNLOAD_FILE" "$HISTO_INPUT"
  NEW_NDVI=$(grep '4_merge_zarr.py created file: ' "$LOG_FILE" | tail -n1 | sed 's/^4_merge_zarr.py created file: //') # Capture output file path from python script
  # NEW_NDVI="/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12_processed.zarr" # NOTE: if user wants to avoid running 4_merge_zarr, specify NEW_NDVI. NOTE: this can only be skipped if HISTO_INPUT remains unchanged. DO NOT SKIP WHEN TESTING DIFFERENT EXAMPLE ZARRS!
  # NEW_NDVI="/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-19_18h40_ndvi_01_downloaded_2025-12-12_2025-12-28_processed.zarr" # NOTE: if user wants to avoid running 4_merge_zarr, specify NEW_NDVI. NOTE: this can only be skipped if HISTO_INPUT remains unchanged. DO NOT SKIP WHEN TESTING DIFFERENT EXAMPLE ZARRS!
  # NEW_NDVI="/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-24_00h27_ndvi_01_downloaded_2025-11-30_2025-12-28_processed.zarr" # NOTE: if user wants to avoid running 4_merge_zarr, specify NEW_NDVI. NOTE: this can only be skipped if HISTO_INPUT remains unchanged. DO NOT SKIP WHEN TESTING DIFFERENT EXAMPLE ZARRS!
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPT_1}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"
  echo "Merging script returned value: $NEW_NDVI"

  # run script 05
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPT_2}"; echo "Start time: $(timestamp)"
  START_TIME=$(date +%s)
  python -u "${SCRIPT_2}" "$NEW_NDVI" "$HISTO_INPUT" --histo-output="$HISTO_OUTPUT" # NOTE: this does not overwrite the existing HISTO_INPUT.
  #python -u "${SCRIPT_2}" "$NEW_NDVI" "$HISTO_INPUT" --histo-output="$HISTO_INPUT" # NOTE: this does overwrite the existing HISTO_INPUT. This can e.g. be used to simulate two subsequent updates of the CI.
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPT_2}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"

  # run script 06
  echo "------------------------------------------------------------"
  echo "Running: ${SCRIPT_3}"; echo "Start time: $(timestamp)"
  START_TIME=$(date +%s)
  # python "${SCRIPT_3}" "$START_DATE" "$END_DATE"
  # # python "${SCRIPT_3}" "2025-08-22" "2025-09-30" # TODO remove this hardcoding
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  echo "Finished: ${SCRIPT_3}"; echo "Duration: $(format_seconds "$ELAPSED") (hh:mm:ss)"

fi

PIPELINE_END=$(date +%s)
PIPELINE_TIME=$((PIPELINE_END - PIPELINE_START))

# ============================================================
# Clean up temporary output data
# ============================================================

#rm -rf $NEW_NDVI      # TODO: activate this for continuous application # NOTE: for the development and exploration this is deactivated on purpose
#rm -rf $DOWNLOAD_FILE # TODO: activate this for continuous application # NOTE: for the development and exploration this is deactivated on purpose

# ALSO DON'T FORGET TO MANUALLY clean up: HISTO_OUTPUT

# ============================================================
# Finish pipeline
# ============================================================
echo "============================================================"
echo "Pipeline completed successfully"
echo "End time: $(timestamp)"
echo "Total runtime: $(format_seconds "$PIPELINE_TIME") (hh:mm:ss)"
echo "============================================================"
