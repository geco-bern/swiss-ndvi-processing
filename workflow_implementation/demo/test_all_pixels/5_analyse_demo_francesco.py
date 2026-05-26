import datetime as dt
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import dask
import dask.array as da
import xarray as xr
import argparse
import os, shutil, sys
import time
from numcodecs import zarr3

INPUT_LOOKUPTABLE = "/mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr"

import warnings
warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

NO_COVERAGE =  2**15 - 1   # 32767  — pixels with no data for the given time step
INVALID     = -2**15       # -32768 — filtered out pixels, e.g. cloud shadows


# =============================================================================
# Core per-pixel function
# =============================================================================

def historical_ndvi(ndvi_arr_original, medians, mask_array_original,
                    is_observation_date, dates, starting_date):
    """
    Gap-fill and smooth NDVI for a single pixel over its full time series.

    Parameters
    ----------
    ndvi_arr_original     : 1-D int16 array  (scaled ×10000)
    medians               : 1-D int16 array  (scaled ×10000)
    mask_array_original   : 1-D int8  array
    is_observation_date   : 1-D bool  array
    dates                 : 1-D numpy datetime64 array  ← plain numpy, NOT dask
    starting_date         : numpy datetime64 scalar

    Mask values
    -----------
    0 – not an observation, not yet smoothed
    1 – not an observation, smoothed
    2 – observation, not yet smoothed
    3 – observation, smoothed
    4 – observation flagged as outlier
    """

    start_idx = np.searchsorted(dates, starting_date)
    obs_prior = np.nonzero(is_observation_date[:start_idx])[0]   # FIX: [0] to unpack tuple

    # Ensure mask_array is writable
    mask_array_original = np.array(mask_array_original, copy=True)

    if len(obs_prior) < 3:
        return ndvi_arr_original, mask_array_original

    crop_start = obs_prior[-3]          # start 3 observations before starting_date
    ndvi_arr            = ndvi_arr_original[crop_start:]
    medians             = medians[crop_start:]
    is_observation_date = is_observation_date[crop_start:]
    dates_cropped       = dates[crop_start:]
    mask_array          = mask_array_original[crop_start:]

    is_observation_date = is_observation_date.astype(bool)

    ndvi_not_analyzed      = ndvi_arr_original[:crop_start]
    mask_array_not_analyzed = mask_array_original[:crop_start]

    days_diff = (dates_cropped - dates_cropped[0]) / np.timedelta64(1, 'D')

    ndvi_arr = ndvi_arr / 10000.0
    medians  = medians  / 10000.0

    mask_valid_ndvi = (ndvi_arr > 0) & (ndvi_arr < 1)

    ndvi_valid    = ndvi_arr[mask_valid_ndvi]
    median_valid  = medians[mask_valid_ndvi]
    days_diff_2   = days_diff[mask_valid_ndvi]
    original_idx  = np.arange(len(ndvi_arr))[mask_valid_ndvi]

    obs_mask = (ndvi_arr > 0) & (ndvi_arr < 1) & is_observation_date

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------
    delta_threshold       = 0.1
    delta_delta_threshold = 0.1

    delta_ndvi        = ndvi_valid - median_valid
    delta_delta_left  = delta_ndvi[2:]
    delta_delta_right = delta_ndvi[:-2]

    outlier_mask = (
        (np.abs(delta_ndvi[1:-1]) > delta_threshold) &
        (np.abs(delta_delta_left) > delta_delta_threshold) &
        (np.abs(delta_delta_right) > delta_delta_threshold)
    )

    ndvi_valid   = ndvi_valid[1:-1][~outlier_mask]
    delta_ndvi   = delta_ndvi[1:-1][~outlier_mask]
    days_diff_2  = days_diff_2[1:-1][~outlier_mask]
    original_idx_2 = original_idx[1:-1][~outlier_mask]

    if len(delta_ndvi) <= 6:
        return ndvi_arr_original, mask_array_original

    # ------------------------------------------------------------------
    # LOESS smoothing on rolling 7-observation windows
    # ------------------------------------------------------------------
    delta_ndvi_to_interpolate = np.full(len(delta_ndvi) - 6, np.nan)
    idx = np.arange(len(delta_ndvi))

    for i in range(len(delta_ndvi) - 6):
        delta_window  = delta_ndvi[i:i + 7]
        ndvi_window   = ndvi_valid[i:i + 7]

        boundary_condition = np.any((ndvi_window < 0.05) | (ndvi_window > 0.95))
        extreme_negative   = np.sum(delta_window < -0.2) >= 5

        if boundary_condition or extreme_negative:
            delta_ndvi_to_interpolate[i] = delta_window[3]
        else:
            loess = sm.nonparametric.lowess(
                delta_window, idx, frac=1, it=3, return_sorted=False
            )
            delta_ndvi_to_interpolate[i] = loess[3]

    # Combine smoothed values with unsmoothed tail, then interpolate
    delta_ndvi_to_interpolate = np.concatenate([
        np.array([0]),
        loess[:-3],
        delta_ndvi[-3:],
        np.array([0])
    ])
    dates_to_interpolate = np.concatenate([
        np.array([0]),
        days_diff_2,
        np.array([days_diff[-1]])
    ])

    interpolated_values = np.interp(days_diff, dates_to_interpolate, delta_ndvi_to_interpolate)
    ndvi_smoothed = 10000.0 * (interpolated_values + medians)

    # ------------------------------------------------------------------
    # Update mask
    # ------------------------------------------------------------------
    mask_array[obs_mask] = 2
    before = np.arange(len(mask_array)) < original_idx_2[-4]

    outlier_idx       = original_idx[1:-1][outlier_mask]
    valid_outlier_idx = outlier_idx[is_observation_date[outlier_idx] == 1]

    mask_array[before & obs_mask]   = 3
    mask_array[before & ~obs_mask]  = 1
    mask_array[valid_outlier_idx]   = 4

    mask_array_final = np.concatenate([mask_array_not_analyzed, mask_array])
    final_ndvi_value = np.concatenate([ndvi_not_analyzed, ndvi_smoothed])

    return final_ndvi_value, mask_array_final


# =============================================================================
# Helpers
# =============================================================================

def show_ds_structure(ds):
    for c in list(ds.coords) + list(ds.data_vars):
        print(str(c).ljust(15) + ":   " + str(ds[c].encoding))


def get_existing_chunks(zarr_path):
    """Return (pixel_chunks, date_chunks) matching what is already on disk."""
    existing = xr.open_zarr(zarr_path)
    pc = existing.chunks.get('pixel', [40000])[0]
    dc = existing.chunks.get('date',  [365])[0]
    existing.close()
    return int(pc), int(dc)


def assert_append_compatible(ds_to_append, existing):
    """Raise an AssertionError with a clear message if structures differ."""
    assert sorted(ds_to_append.dims)      == sorted(existing.dims),      "Aborted append: dimensions differ"
    assert sorted(ds_to_append.coords)    == sorted(existing.coords),    "Aborted append: coordinates differ"
    assert sorted(ds_to_append.data_vars) == sorted(existing.data_vars), "Aborted append: data_vars differ"

    skip_coords = {'date', 'doy'}
    for c in [c for c in ds_to_append.coords if c not in skip_coords]:
        assert ds_to_append[c].dtype == existing[c].dtype, \
            f"dtype mismatch for coord '{c}': {ds_to_append[c].dtype} vs {existing[c].dtype}"
        assert ds_to_append[c].shape == existing[c].shape, \
            f"shape mismatch for coord '{c}': {ds_to_append[c].shape} vs {existing[c].shape}"
        assert (ds_to_append[c].values == existing[c].values).all(), \
            f"values mismatch for coord '{c}'"

    for c in ds_to_append.data_vars:
        assert ds_to_append[c].dtype == existing[c].dtype, \
            f"dtype mismatch for data_var '{c}': {ds_to_append[c].dtype} vs {existing[c].dtype}"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT_ZARR",       help="Path to newly downloaded NDVI zarr")
    parser.add_argument("HISTO_ZARR_INPUT", help="Path to historic NDVI zarr")
    parser.add_argument(
        "--histo-output", dest="HISTO_ZARR_OUTPUT", default=None,
        help="Path for updated historic zarr. Defaults to HISTO_ZARR_INPUT (append in-place)."
    )
    args = parser.parse_args()

    INPUT_ZARR        = args.INPUT_ZARR
    HISTO_ZARR_INPUT  = args.HISTO_ZARR_INPUT
    HISTO_ZARR_OUTPUT = args.HISTO_ZARR_OUTPUT or HISTO_ZARR_INPUT

    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # FIX 1: Configure Dask for robustness before creating the Client
    #   – longer TCP timeouts so slow workers are not dropped prematurely
    #   – aggressive memory spilling so workers don't go OOM
    # -------------------------------------------------------------------------
    dask.config.set({
        "distributed.comm.timeouts.connect": "120s",
        "distributed.comm.timeouts.tcp":     "600s",
        "distributed.worker.memory.target":  0.60,   # start spilling at 60 %
        "distributed.worker.memory.spill":   0.70,   # spill hard  at 70 %
        "distributed.worker.memory.pause":   0.85,   # pause tasks  at 85 %
        "distributed.worker.memory.terminate": 0.95, # restart worker at 95 %
    })

    # -------------------------------------------------------------------------
    # FIX 2: Fewer, larger workers → less inter-worker data transfer
    # -------------------------------------------------------------------------
    N_WORKERS            = 30
    N_THREADS_PER_WORKER = 1
    MEMORY_PER_WORKER    = "120GB"
    DASK_TEMP_DIR        = "/mnt/data2/UniBe-swiss-ndvi/tmp_data6/"

    # Chunk sizes — match existing zarr on disk to avoid write-time rechunking
    PIXEL_CHUNKS_EXISTING, DATE_CHUNKS_EXISTING = get_existing_chunks(HISTO_ZARR_INPUT)
    PIXEL_CHUNKS    = PIXEL_CHUNKS_EXISTING   # e.g. 40000
    DATE_CHUNKS_OUT = DATE_CHUNKS_EXISTING    # e.g. 365
    DATE_CHUNKS     = -1                      # load full time axis per pixel for apply_ufunc

    COMPRESSOR = zarr3.Blosc(cname="zstd", clevel=3, shuffle=2)

    client = Client(
        n_workers            = N_WORKERS,
        threads_per_worker   = N_THREADS_PER_WORKER,
        memory_limit         = MEMORY_PER_WORKER,
        local_directory      = DASK_TEMP_DIR,
        processes            = True,
        dashboard_address    = ":8343",
    )
    print(client, flush=True)
    print(client.dashboard_link, flush=True)

    # -------------------------------------------------------------------------
    # Load datasets (lazy)
    # -------------------------------------------------------------------------
    historic_ds = xr.open_zarr(HISTO_ZARR_INPUT, chunks={}).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    new_ds      = xr.open_zarr(INPUT_ZARR,        chunks={}).chunk({"pixel": PIXEL_CHUNKS, "date": -1})
    lookuptable = xr.open_zarr(INPUT_LOOKUPTABLE).chunk({"pixel": PIXEL_CHUNKS})

    # Minor fix: ensure correct dtypes in lookup table
    lookuptable = lookuptable.assign_coords(
        pixel=("pixel", lookuptable.pixel.values.astype(np.int32)),
        doy  =("date",  lookuptable.doy.values.astype(np.int32)),
    )

    print("Last dates in historic_ds:\n  " +
          "\n  ".join(np.datetime_as_string(historic_ds.date.isel(date=slice(-10, None)), unit="D")),
          flush=True)
    print("First dates in newly downloaded:\n  " +
          "\n  ".join(np.datetime_as_string(new_ds.date.isel(date=slice(0, 10)), unit="D")),
          flush=True)
    print("Current historic dataset:", flush=True);  print(historic_ds, flush=True)
    print("Newly downloaded dataset:", flush=True);  print(new_ds,      flush=True)

    # -------------------------------------------------------------------------
    # Join median NDVI from lookup table
    # -------------------------------------------------------------------------
    for ds in [new_ds, historic_ds]:
        doy_noLeap = xr.where(ds.doy == 366, 365, ds.doy)
        ds["median_ndvi"] = lookuptable["median_ndvi"].sel(doy=doy_noLeap, pixel=ds.pixel)

    # -------------------------------------------------------------------------
    # Build mask_array for new_ds
    # -------------------------------------------------------------------------
    mask_2or0 = (
        new_ds["obs_date"] &
        (new_ds["ndvi_obs"] < NO_COVERAGE) &
        (new_ds["ndvi_obs"] > INVALID)
    )
    new_ds["mask_array"] = xr.where(mask_2or0, np.int8(2), np.int8(0))

    # -------------------------------------------------------------------------
    # Rename / drop variables and concatenate
    # -------------------------------------------------------------------------
    new_ds = (
        new_ds
        .rename({"ndvi_obs": "ndvi_processed", "ndsi_obs": "ndsi_processed"})
        .drop_vars("ndsi_processed")
    )

    merged_ds = (
        xr.concat([historic_ds, new_ds], dim="date")
        .sortby("date")
        .chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    )

    # -------------------------------------------------------------------------
    # FIX 3: Convert dates to a plain numpy array before passing to apply_ufunc
    #   This prevents dates from being embedded in the Dask graph as a huge
    #   array, which was responsible for the 1.58 GiB graph size warning.
    # -------------------------------------------------------------------------
    dates_np   = merged_ds["date"].values          # tiny numpy array (~3200 × 8 bytes)
    start_date = historic_ds["date"].max().values  # numpy datetime64 scalar

    # -------------------------------------------------------------------------
    # FIX 4: Do NOT persist the large arrays.
    #   persist() forces the entire merged dataset (~1 TB) into worker RAM
    #   simultaneously. Let Dask stream chunks through workers instead.
    #   Only persist obs_dates (bool, tiny) since it is read by every pixel.
    # -------------------------------------------------------------------------
    ndvi_array   = merged_ds["ndvi_processed"]   # lazy
    median_array = merged_ds["median_ndvi"]      # lazy
    mask_array   = merged_ds["mask_array"]       # lazy
    obs_dates    = merged_ds["obs_date"].persist()  # bool 1-D, safe to persist

    output_dtypes = [ndvi_array.dtype, mask_array.dtype]

    # -------------------------------------------------------------------------
    # FIX 5: Set allow_rechunk=False to prevent hidden inter-worker reshuffling
    # -------------------------------------------------------------------------
    ndvi_processed, mask_processed = xr.apply_ufunc(
        historical_ndvi,
        ndvi_array,
        median_array,
        mask_array,
        obs_dates,
        input_core_dims  = [["date"], ["date"], ["date"], ["date"]],
        output_core_dims = [["date"], ["date"]],
        vectorize        = True,
        dask             = "parallelized",
        kwargs           = {
            "dates":         dates_np,    # plain numpy — not in the graph
            "starting_date": start_date,
        },
        output_dtypes        = output_dtypes,
        dask_gufunc_kwargs   = {"allow_rechunk": False},  # no hidden reshuffling
    )

    g = ndvi_processed.__dask_graph__()
    print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)

    # -------------------------------------------------------------------------
    # Prepare the slice to append (only dates after start_date)
    # -------------------------------------------------------------------------
    historic_ds_to_extend = historic_ds.drop_vars("median_ndvi")

    ndvi_processed_to_append = ndvi_processed.sel(date=slice(start_date + np.timedelta64(1, "D"), None))
    mask_processed_to_append = mask_processed.sel(date=slice(start_date + np.timedelta64(1, "D"), None))

    # -------------------------------------------------------------------------
    # FIX 6: Match chunk sizes to the existing zarr before writing
    #   Mismatched chunks force a write-time rechunk, adding huge overhead.
    # -------------------------------------------------------------------------
    ds_to_append = (
        xr.Dataset({
            "ndvi_processed": ndvi_processed_to_append,
            "mask_array":     mask_processed_to_append,
        })
        .chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS_OUT})
    )
    ds_to_append.attrs["pixel_definition"] = historic_ds.attrs["pixel_definition"]

    # -------------------------------------------------------------------------
    # Fallback: full rewrite to a separate file (used when append fails)
    # -------------------------------------------------------------------------
    def fallback_action_overwrite_zarr(outfile):
        print(f"Writing full dataset to {outfile}", flush=True)
        extended_historic_ds = (
            xr.concat([historic_ds_to_extend, ds_to_append], dim="date")
            .sortby("date")
            .chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS_OUT})
        )
        encoding = {v: {"compressors": COMPRESSOR} for v in extended_historic_ds.data_vars}
        for name in list(extended_historic_ds.coords) + list(extended_historic_ds.data_vars):
            extended_historic_ds[name].encoding.pop("chunks",     None)
            extended_historic_ds[name].encoding.pop("compressor", None)
            extended_historic_ds[name].encoding.pop("compressors", None)
        extended_historic_ds.to_zarr(outfile, mode="w", compute=True,
                                      encoding=encoding, zarr_format=3)

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------
    if len(ds_to_append["date"].values) == 0:
        warnings.warn("Did not modify historic NDVI — no new dates found.")
        raise ValueError("Did not modify historic NDVI — no new dates found.")

    if HISTO_ZARR_OUTPUT == HISTO_ZARR_INPUT:
        print(f"appending to file\n  {HISTO_ZARR_OUTPUT}", flush=True)
        try:
            print("Appending new dates to existing zarr store...", flush=True)

            existing = xr.open_zarr(HISTO_ZARR_OUTPUT)
            assert_append_compatible(ds_to_append, existing)
            existing.close()

            (ds_to_append
             .drop_vars(["x_idx", "y", "y_idx", "x"])
             .to_zarr(
                 HISTO_ZARR_OUTPUT,
                 mode       = "a-",
                 append_dim = "date",
                 compute    = True,
                 encoding   = {},
             ))

            # Post-write sanity check
            n_appended     = ds_to_append.sizes["date"]
            result         = xr.open_zarr(HISTO_ZARR_OUTPUT)
            boundary_dates = result.isel(date=slice(-n_appended - 1, -n_appended + 1)).date.values
            result.close()

            if (boundary_dates[1] - boundary_dates[0]) != np.timedelta64(1, "D"):
                raise ValueError(
                    f"Dates not exactly 1 day apart at boundary: {boundary_dates}"
                )

            print("Append successfully completed.", flush=True)

        except Exception as e:
            fallback_output = (HISTO_ZARR_INPUT + ".failedAppending_" +
                               dt.datetime.now().strftime("%Y%m%d%H%M%S"))
            print(f"Appending failed: {e}.\nWriting whole file to {fallback_output}", flush=True)
            fallback_action_overwrite_zarr(fallback_output)

    else:
        print(f"writing to new file\n  {HISTO_ZARR_INPUT}\n=> {HISTO_ZARR_OUTPUT}", flush=True)
        fallback_action_overwrite_zarr(HISTO_ZARR_OUTPUT)

    client.close()

    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} seconds", flush=True)
    print("Modified/Created file: ", flush=True)
    print(HISTO_ZARR_OUTPUT, flush=True)
    sys.exit(0)