TODO: test this here

from datetime import datetime
import os
import shutil
import sys

import dask
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from dask.distributed import Client
from numcodecs import zarr3

import warnings

warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3",
)


NO_COVERAGE = 2**15 - 1
INVALID = -2**15
COMPRESSOR = zarr3.Blosc(cname="zstd", clevel=3, shuffle=2)

INPUT_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"
INPUT_ZARR_LOOKUPTABLE = "/mnt/data2/UniBe-swiss-ndvi/input_data/lookup_table_median_ndvi_v7.zarr"
OUT_PATH = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_evaluation_sites.zarr"


def round_to_5_ending(n):
    return round((n - 5) / 10) * 10 + 5


RAW_EVALUATION_SITE_COORDS = {
    "Lowland broadleaf": (2694491, 1126023),
    "Highland broadleaf": (2692020, 1121443),
    "Lowland evergreen": (2761097, 1194613),
    "Highland evergreen": (2781537, 1182974),
    "Bitsch fire affected area": (2644035, 1133765),
    "Bitsch fire nearby non-affected area": (2644328, 1134342),
    "2018 Drought-affected area": (2690025, 1287413),
    "Vaia storm affected area": (2689564, 1154411),
}

EVALUATION_SITES = {
    "Lowland broadleaf": {"coords": (round_to_5_ending(2694491), round_to_5_ending(1126023)), "pixel": 90415334},
    "Highland broadleaf": {"coords": (round_to_5_ending(2692020), round_to_5_ending(1121443)), "pixel": 93677704},
    "Lowland evergreen": {"coords": (round_to_5_ending(2761097), round_to_5_ending(1194613)), "pixel": 46053259},
    "Highland evergreen": {"coords": (round_to_5_ending(2781537), round_to_5_ending(1182974)), "pixel": 54232662},
    "Bitsch fire affected area": {"coords": (round_to_5_ending(2644035), round_to_5_ending(1133765)), "pixel": 84856712},
    "Bitsch fire nearby non-affected area": {"coords": (round_to_5_ending(2644328), round_to_5_ending(1134342)), "pixel": 84468278},
    "2018 Drought-affected area": {"coords": (round_to_5_ending(2690025), round_to_5_ending(1287413)), "pixel": 427960},
    "Vaia storm affected area": {"coords": (round_to_5_ending(2689564), round_to_5_ending(1154411)), "pixel": 73583022},
}

SELECTED_SITE_NAMES = list(EVALUATION_SITES)
SELECTED_PIXEL_IDS = np.array([EVALUATION_SITES[name]["pixel"] for name in SELECTED_SITE_NAMES], dtype=np.int32)


def historical_ndvi_singleWindow(ndvi_arr, median_arr, is_observation_date, dates):
    original_idx = np.arange(len(ndvi_arr))
    days_diff = (dates - dates[0]) / np.timedelta64(1, "D")

    ndvi_arr = ndvi_arr / 10000
    median_arr = median_arr / 10000
    mask_valid_ndvi = (ndvi_arr > 0) & (ndvi_arr < 1)

    ndvi_valid = ndvi_arr[mask_valid_ndvi]
    median_valid = median_arr[mask_valid_ndvi]
    days_diff_1 = days_diff[mask_valid_ndvi]
    original_idx_1 = original_idx[mask_valid_ndvi]

    obs_mask = (ndvi_arr > 0) & (ndvi_arr < 1) & is_observation_date

    delta_threshold = 0.05
    delta_delta_threshold = 0.1

    delta_ndvi = ndvi_valid - median_valid

    delta_delta_left = delta_ndvi[:-2] - delta_ndvi[1:-1]
    delta_delta_right = delta_ndvi[2:] - delta_ndvi[1:-1]
    outlier_mask = (
        (abs(delta_ndvi[1:-1]) > delta_threshold)
        & (abs(delta_delta_left) > delta_delta_threshold)
        & (abs(delta_delta_right) > delta_delta_threshold)
    )

    delta_ndvi_2 = delta_ndvi[1:-1][~outlier_mask]
    days_diff_2 = days_diff_1[1:-1][~outlier_mask]
    original_idx_2 = original_idx_1[1:-1][~outlier_mask]
    outlier_idx = original_idx_1[1:-1][outlier_mask]

    if len(delta_ndvi_2) > 6:
        idx = np.arange(len(delta_ndvi_2))
        loess = sm.nonparametric.lowess(
            delta_ndvi_2,
            idx,
            frac=7 / len(delta_ndvi_2),
            it=3,
            return_sorted=False,
        )

        delta_ndvi_to_interpolate = np.concatenate(
            [
                np.array([0]),
                loess[:-4],
                delta_ndvi_2[-4:],
                np.array([0]),
            ]
        )
        dates_to_interpolate = np.concatenate(
            [
                np.array([0]),
                days_diff_2,
                np.array([days_diff[-1]]),
            ]
        )

        interpolated_values = np.interp(
            days_diff,
            dates_to_interpolate,
            delta_ndvi_to_interpolate,
        )

        ndvi_smoothed = 10000 * (interpolated_values + median_arr)

        mask_array = np.zeros(len(is_observation_date), dtype=object)
        mask_array[obs_mask] = 2

        before = np.arange(len(mask_array)) < original_idx_2[-4]
        mask_array[before & obs_mask] = 3
        mask_array[before & (~obs_mask)] = 1

        valid_outlier_idx = outlier_idx[is_observation_date[outlier_idx] == 1]
        mask_array[valid_outlier_idx] = 4

        return ndvi_smoothed, mask_array

    mask_array = np.zeros(len(is_observation_date), dtype=object)
    return 10000 * ndvi_arr, mask_array


def validate_selected_pixels(ds):
    subset = ds.sel(pixel=SELECTED_PIXEL_IDS)

    for site_name in SELECTED_SITE_NAMES:
        expected_x, expected_y = EVALUATION_SITES[site_name]["coords"]
        pixel_id = EVALUATION_SITES[site_name]["pixel"]
        actual_x = int(subset["x"].sel(pixel=pixel_id).values)
        actual_y = int(subset["y"].sel(pixel=pixel_id).values)

        if (actual_x, actual_y) != (expected_x, expected_y):
            raise ValueError(
                f"Pixel validation failed for {site_name}: expected {(expected_x, expected_y)}, "
                f"got {(actual_x, actual_y)} for pixel {pixel_id}."
            )

    return subset


if __name__ == "__main__":
    n_workers = min(len(SELECTED_PIXEL_IDS), os.cpu_count() or len(SELECTED_PIXEL_IDS))

    with Client(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit="4GB",
        processes=True,
        dashboard_address=":1236",
    ) as client:
        print(client, flush=True)
        print(client.dashboard_link, flush=True)
        print(dask.config.get("scheduler"), flush=True)

        print("Evaluation sites selected for processing:", flush=True)
        for site_name in SELECTED_SITE_NAMES:
            raw_x, raw_y = RAW_EVALUATION_SITE_COORDS[site_name]
            rounded_x, rounded_y = EVALUATION_SITES[site_name]["coords"]
            pixel_id = EVALUATION_SITES[site_name]["pixel"]
            print(
                f"  {site_name}: raw=({raw_x}, {raw_y}) rounded=({rounded_x}, {rounded_y}) pixel={pixel_id}",
                flush=True,
            )

        new_observations_ds = xr.open_dataset(
            INPUT_ZARR,
            chunks={},
            mask_and_scale=False,
            consolidated=True,
        ).drop_vars("ndsi")
        new_observations_ds = validate_selected_pixels(new_observations_ds)

        lookuptable = xr.open_zarr(INPUT_ZARR_LOOKUPTABLE, chunks={}, consolidated=True)

        print("Subset observation dataset:", flush=True)
        print(new_observations_ds, flush=True)

        observation_datetimes = pd.DatetimeIndex(new_observations_ds["datetime"].values)
        if not observation_datetimes.is_monotonic_increasing:
            new_observations_ds = new_observations_ds.sortby("datetime")
            observation_datetimes = pd.DatetimeIndex(new_observations_ds["datetime"].values)

        observation_dates = observation_datetimes.floor("D")
        first_obs_idx = np.flatnonzero(~observation_dates.duplicated(keep="first"))
        first_obs_dates = observation_dates[first_obs_idx]

        ndvi_daily_between_obs = (
            new_observations_ds
            .isel(datetime=first_obs_idx)
            .drop_vars("date", errors="ignore")
            .assign_coords(obs_day=("datetime", first_obs_dates.values))
            .swap_dims({"datetime": "obs_day"})
            .drop_vars("datetime")
            .rename({"obs_day": "date"})
        )

        ndvi_daily_between_obs["ndvi"] = xr.where(
            (ndvi_daily_between_obs["ndvi"] != NO_COVERAGE)
            & (ndvi_daily_between_obs["ndvi"] != INVALID),
            ndvi_daily_between_obs["ndvi"],
            np.int16(NO_COVERAGE),
        ).astype(np.int16)

        start_date = first_obs_dates.min()
        end_date = first_obs_dates.max()
        daily_dates = pd.date_range(
            start=pd.to_datetime(start_date).floor("D"),
            end=pd.to_datetime(end_date).floor("D"),
            freq="D",
        )

        new_ds = ndvi_daily_between_obs.reindex(
            date=daily_dates,
            fill_value=np.int16(NO_COVERAGE),
        )
        new_ds = new_ds.assign_coords(doy=("date", daily_dates.dayofyear.values.astype(np.int32)))
        new_ds["obs_date"] = new_ds.date.isin(observation_dates)

        doy_no_leap = xr.where(new_ds.doy == 366, 365, new_ds.doy)
        new_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
            doy=doy_no_leap,
            pixel=new_ds.pixel,
        )

        mask_2or0 = (
            new_ds["obs_date"]
            & (new_ds["ndvi"] < NO_COVERAGE)
            & (new_ds["ndvi"] > INVALID)
        )
        new_ds["mask_array"] = xr.where(mask_2or0, np.int8(2), np.int8(0))
        new_ds = new_ds.rename({"ndvi": "ndvi_processed"}).chunk({"pixel": len(SELECTED_PIXEL_IDS), "date": -1})

        dates_array_arg = new_ds["date"].values

        ndvi_out, mask_out = xr.apply_ufunc(
            historical_ndvi_singleWindow,
            new_ds["ndvi_processed"],
            new_ds["median_ndvi"],
            new_ds["obs_date"],
            input_core_dims=[["date"], ["date"], ["date"]],
            output_core_dims=[["date"], ["date"]],
            kwargs={"dates": dates_array_arg},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.dtype("int16"), np.dtype("int8")],
        )
        ndvi_out, mask_out = dask.compute(ndvi_out, mask_out)

        out_ds = xr.Dataset(
            {
                "ndvi_processed": ndvi_out.astype(np.int16),
                "mask_array": mask_out.astype(np.int8),
            },
            coords={name: new_ds.coords[name] for name in new_ds.coords},
        )
        out_ds.attrs = dict(new_ds.attrs)
        out_ds.attrs.pop("description_ndsi", None)
        out_ds.attrs["selected_sites"] = ", ".join(SELECTED_SITE_NAMES)
        out_ds.attrs["selected_pixels"] = ", ".join(str(pixel_id) for pixel_id in SELECTED_PIXEL_IDS.tolist())

        if os.path.exists(OUT_PATH):
            shutil.rmtree(OUT_PATH)

        encoding = {name: {"compressors": COMPRESSOR} for name in out_ds.data_vars}
        for name in list(out_ds.coords) + list(out_ds.data_vars):
            out_ds[name].encoding.pop("chunks", None)
            out_ds[name].encoding.pop("compressor", None)
            out_ds[name].encoding.pop("compressors", None)

        out_ds.to_zarr(OUT_PATH, mode="w", encoding=encoding, zarr_format=3)

    print(OUT_PATH, flush=True)
    sys.exit(0)