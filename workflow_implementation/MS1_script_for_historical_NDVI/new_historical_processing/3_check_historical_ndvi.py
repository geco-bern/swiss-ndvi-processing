# This plots time series of selected pixels, illustrating our processing

from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import xarray as xr
import os, sys
import shutil
import pandas as pd
from numcodecs import blosc, Blosc, zarr3
from zarr.codecs import BloscCodec
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)


NO_COVERAGE = 32767
NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step
INVALID     = -32768
INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows

N_WORKERS = 10

client = Client(
n_workers=N_WORKERS,
threads_per_worker=1,
memory_limit='50GB',
processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
dashboard_address=':2234')  
print(client.dashboard_link)

OBS_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"
PROC_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7-test6.zarr" # TODO: remove -test
INPUT_ZARR_LOOKUPTABLE = "/mnt/data2/UniBe-swiss-ndvi/input_data/lookup_table_median_ndvi_v7.zarr"

# =====================================================
#  Load data sets
# =====================================================
obs_ds  = xr.open_dataset(OBS_ZARR,  chunks={}, mask_and_scale= False)
proc_ds = xr.open_dataset(PROC_ZARR, chunks={}, mask_and_scale= False)

# --- load median values for each doy --------------------------
lookuptable  = xr.open_zarr(INPUT_ZARR_LOOKUPTABLE)


# --- add median NDVI from model ----------------------------------
# Append day-of-year (for merging of median expected NDVI from model)
doy_array = pd.to_datetime(obs_ds['datetime']).dayofyear
obs_ds = obs_ds.assign_coords(doy = ('datetime', doy_array))
doy_noLeap = xr.where(obs_ds.doy == 366, 365, obs_ds.doy) # remove leap year if encountered
obs_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
        doy=doy_noLeap,
        pixel=obs_ds.pixel) # this is to join by pixels and doy

doy_array = pd.to_datetime(proc_ds['date']).dayofyear
proc_ds = proc_ds.assign_coords(doy = ('date', doy_array))
doy_noLeap = xr.where(proc_ds.doy == 366, 365, proc_ds.doy) # remove leap year if encountered
proc_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
        doy=doy_noLeap,
        pixel=proc_ds.pixel) # this is to join by pixels and doy


# forest_mask = np.load(FOREST_MASK)
# height, width = forest_mask.shape
# forest_flat_indices = np.flatnonzero(forest_mask == 1)

# ds_temporal = zarr.open_group(TEMPORAL_DATASET_ZARR, mode="r")
# ds_spatial = zarr.open_group(SPATIAL_DATASET_ZARR, mode="r")
# params = ds_temporal["params_2"]
# params_lower = params["params_lower"]
# params_median = params["params_median"]
# params_upper = params["params_upper"]
# ndvi = ds_temporal["ndvi"]
# ndsi = ds_temporal["ndsi"]
# anomaly_values = ds_temporal["anomalies"]
# anomaly_scores = ds_temporal["anomaly_scores"]

# dates = pd.to_datetime([d.decode("utf-8") for d in ds_temporal["dates"][:]])
# dtindex = pd.DatetimeIndex(dates)
# doy = dtindex.dayofyear.to_numpy()
# is_leap = dtindex.is_leap_year.astype(int)
# t = torch.tensor((doy - 1) / (365 + is_leap), dtype=torch.float32)

# =====================================================
#  Plot figures
# =====================================================
START_DATE="2025-04-01"
END_DATE="2025-08-01"
# X_COORD, Y_COORD = "2720645", "1118245"
# X_COORD, Y_COORD = "2710385", "1116375"
# X_COORD, Y_COORD = "2710005", "1109995"
X_COORD, Y_COORD = "2644020", "1133790" # NOTE: Bitsch forest fire
    #     X_COORD="2710005" # TODO: check if this is indeed a forest pixel otherwise choose other test option
    #     Y_COORD="1109995" # TODO: check if this is indeed a forest pixel otherwise choose other test option
X_COORD, Y_COORD = proc_ds['x'].values[0], proc_ds['y'].values[0]
X_COORD, Y_COORD = proc_ds.isel(pixel=999)['x'].values, proc_ds['y'].isel(pixel=999).values


# Figure 00
# --- visual check of resulting data sets ----------------------------------
proc_ds_subset = proc_ds.isel(pixel=[0, 210, 350, 490]).drop(["y_idx","x_idx"])
obs_ds_subset  = obs_ds.isel(pixel=[0, 210, 350, 490]).drop(["y_idx","x_idx"])

smoothed_cmap = {
    # 0: ("no_obs_to_smooth", "black"),
    # 1: ("no_obs_smoothed",  "orange"),
    2: ("2: obs_to_smooth",    "orange"),
    3: ("3: obs_smoothed",     "black"),
    4: ("4: obs_smoothed_outlier", "red"),
}
obs_cmap     = {0: ("obs_raw",   "green")}
gapfill_cmap = {0: ("gapfilled", "black")}

# proc_ds_subset["median_ndvi"].plot.line(x='datetime',hue='pixel')
# plot all processed
# proc_ds_subset["ndvi_processed"].plot.scatter(x='date',hue='pixel',marker=".", edgecolors="none")
gr = proc_ds_subset["ndvi_processed"].plot.line(
    x='date',row='pixel', color = gapfill_cmap[0][1],
    figsize=(7.2*2, 7.2*2))

# plot processed observation
        # mask_array == 0: the data is not an observation and is yet to be smoothed
        # mask_array == 1: the data is not an observation and is smoothed
        # mask_array == 2: the data is an observation and is yet to be smoothed
        # mask_array == 3: the data is an observation and is smoothed
        # mask_array == 4: the data is an observation and is an outlier
indexer_proc = ((proc_ds_subset["mask_array"] == 2) |
                (proc_ds_subset["mask_array"] == 3) |
                (proc_ds_subset["mask_array"] == 4)).compute()
proc_ds_subset2 = proc_ds_subset.where(indexer_proc, drop=True)

# automatic facetting (doesn't work: https://github.com/pydata/xarray/issues/10176):
# proc_ds_subset2["ndvi_processed"].plot.scatter(x='date',col='pixel',marker="x")
# manual facetting (works):
colors = [smoothed_cmap[k][1] for k in sorted(smoothed_cmap)]
cmap = mcolors.ListedColormap(colors)
# boundaries = np.arange(-0.5, len(colors) + 0.5, 1.0)
# norm = mcolors.BoundaryNorm(boundaries, cmap.N)
for i in range(proc_ds_subset2.pixel.size):
    ax = gr.axs.flat[i]
    ax.set_prop_cycle(None)
    proc_ds_subset2.isel(pixel=i).plot.scatter(
        ax=ax, x='date', marker="x",
        y="ndvi_processed", hue="mask_array",
        cmap=cmap, add_colorbar=False) # norm=norm, 


# plot raw observation
indexer_obs = ((obs_ds_subset["ndvi"] < NO_COVERAGE) &
               (obs_ds_subset["ndvi"] > INVALID)).compute()
obs_ds_subset2 = obs_ds_subset.where(indexer_obs, drop=True)
# automatic facetting (doesn't work: https://github.com/pydata/xarray/issues/10176):
# obs_ds_subset2["ndvi"].plot.scatter(row='pixel',x='datetime',marker="o", hue=None, color="red", alpha=0.2)
# manual facetting (works):
for i in range(obs_ds_subset2.pixel.size):
    ax = gr.axs.flat[i]
    ax.set_prop_cycle(None)
    obs_ds_subset2.isel(pixel=i).plot.scatter(
        ax=ax, x='datetime',marker="o", hue=None, 
        color=obs_cmap[0][1],
        alpha=0.2, y = "ndvi")

# layouting/formatting
for i in range(obs_ds_subset2.pixel.size):
    ax = gr.axs.flat[i]
    ax.set_xlabel("") # remove x labels
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: val / 10000)) # fix y-axis tick labels

handles_for_legend = [ # legend entry for raw observations:
    Line2D([0], [0], marker='o', linestyle='', color=color, markerfacecolor=color, markersize=6, label=label)
    for _, (label, color) in obs_cmap.items()
] + [                  # legend entry for gapfilled line:
    Line2D([0], [0], marker='', linestyle='-', color=color, markerfacecolor=color, markersize=6, label=label)
    for _, (label, color) in gapfill_cmap.items()
] + [                  # legend entry for processed (smoothed) observations:
    Line2D([0], [0], marker='x', linestyle='', color=color, markerfacecolor=color, markersize=6, label=label)
    for _, (label, color) in smoothed_cmap.items()
]
for i in range(obs_ds_subset2.pixel.size):
    ax = gr.axs.flat[i]
    ax.legend(handles=handles_for_legend, title="", fontsize="small", loc="lower left") # add discrete legend

# save plot:    
plt.savefig('test4.png')
    
    
# Figure 0 (taken from workflow_implementation/demo/test_all_pixels/7_create_png_for_XY_demoFB.py)
# # c) subset historic data
# # 1) by date
# proc_ds_subset = proc_ds.sel(date = slice(pd.to_datetime(START_DATE), pd.to_datetime(END_DATE)))
# obs_ds_subset = obs_ds.sel(datetime = slice(pd.to_datetime(START_DATE), pd.to_datetime(END_DATE)))

# # 2) by coordinate
# #proc_ds_subset.x.compute() # from 2710005 to 2719945
# #proc_ds_subset.y.compute() # from 1109995 to 1100005
# indexer = ((proc_ds_subset.x==X_COORD) & (proc_ds_subset.y==Y_COORD))
# indexer = indexer.compute() # in order to use drop=True, we need to compute indexer so that dimension of result is known.
# proc_ds_subset2 = proc_ds_subset.where(indexer, drop=True)
# #proc_ds_subset2.compute()
# #proc_ds_subset2.sizes

# indexer = ((obs_ds_subset.x==X_COORD) & (obs_ds_subset.y==Y_COORD))
# indexer = indexer.compute() # in order to use drop=True, we need to compute indexer so that dimension of result is known.
# obs_ds_subset2 = obs_ds_subset.where(indexer, drop=True)



# print(proc_ds_subset2)
# print(proc_ds_subset2.compute())
# print(obs_ds_subset2.compute())

# # d) download raw data directly from swisstopo
# # if FLAG_DOWNLOAD:
# #     df_raw = download_timeseries_NDVI_singlePixel(
# #         x=X_COORD, 
# #         y=Y_COORD, 
# #         start_date = START_DATE, 
# #         end_date = END_DATE)

# # e) prepare plot
# # Get data
# dates      = proc_ds_subset2["date"].to_numpy()
# ndvi       = proc_ds_subset2["ndvi_processed"].load().to_numpy()[0]
# mask_array = proc_ds_subset2["mask_array"].load().to_numpy()[0]

# obs_dates  = obs_ds_subset2["datetime"].to_numpy()
# obs_ndvi   = obs_ds_subset2["ndvi"].load().to_numpy()


# print(ndvi)
# print(mask_array)

# # TODO: we did not include medians in historical data cube and need to append it each time: medians = ds_h["median_ndvi"].isel(date = slice(2800,3265)).load().to_numpy()

# # Filter based on mask_array
# no_obs_to_smooth = mask_array == 0
# no_obs_smoothed = mask_array == 1
# obs_to_smooth = mask_array == 2
# obs_smoothed = mask_array == 3
# outlier_smoothed = mask_array == 4

# # f) make plot
# plt.figure(figsize=(7.2, 4), dpi = 200)

# # plt.plot(dates[no_obs_to_smooth], ndvi[no_obs_to_smooth], marker="D", linestyle="None", markersize=2, color ="black",  label = "no obs to smooth") # TODO: what y-values do these have??? They have 32767.
# # plt.plot(dates[no_obs_smoothed],  ndvi[no_obs_smoothed],  marker="D", linestyle="None", markersize=2, color ="orange", label = "no obs smoothed")
# # plt.plot(dates[obs_to_smooth],    ndvi[obs_to_smooth],    marker="x", linestyle="None", markersize=4, color ="yellow", label = "obs to smooth")
# # plt.plot(dates[obs_smoothed],     ndvi[obs_smoothed],     marker="x", linestyle="None", markersize=4, color ="green",  label = "obs smoothed")
# # plt.plot(dates[outlier_smoothed], ndvi[outlier_smoothed], marker="x", linestyle="None", markersize=2, color ="red",    label = "outlier smoothed")


# plt.plot(obs_dates, obs_ndvi, marker="x", linestyle="None", markersize=2, color ="black",  label = "raw obs")

# # if FLAG_DOWNLOAD:
# #     # add crosses for raw downloaded observations
# #     plt.plot(
# #         df_raw.datetime, 
# #         df_raw.ndvi_scaled, 
# #         marker="x", alpha=.5, linestyle="None", markersize=5, color ="blue",
# #         label = "Raw download")
# #     # add vertical areas for days with observations
# #     obs_dates = [
# #         [obs_date.floor("D"), obs_date.ceil("D")]  for obs_date in df_raw.datetime]
# #     [plt.axvspan(_range[0], _range[1], color='grey', alpha=1.0) for _range in obs_dates]

# # TODO: ALSO ADD MEDIANS: plt.plot(dates, medians,color = "black", linestyle="-",label = "median_ndvi")
# # plt.ylim(0, 10000)
# plt.ylim(0, 33000) # TODO: reactivate cropping at 10000
# plt.xlabel("Date")
# plt.ylabel("NDVI")
# plt.title(f"NDVI Time Series of location: {(X_COORD, Y_COORD)}")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# # g) output figure

# plotpath = (
#     #"/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/report/fig/prova2/"+
#     # "TESTSUITE_"+
#     # f"{os.path.basename(HISTO_ZARR_INPUT)}_"+
#     # f"{START_DATE.replace("-","")}to{END_DATE.replace("-","")}_"+
#     # f"location_{X_COORD}x{Y_COORD}"+
#     PROC_ZARR+"-TESTSUITE_"+
#     f"{START_DATE.replace("-","")}to{END_DATE.replace("-","")}_"+
#     f"location_{X_COORD}x{Y_COORD}"+
#     ".png")
# plt.savefig(plotpath)
# plt.close()










# # Figure 1 (taken from https://github.com/SamanthaBiegel/s2-forest-browning-monitoring/blob/main/notebooks/plot_results.ipynb)

# indices = [10803621, 71181649, 96658205, 43074504]

# order = np.argsort(dates)
# dates_sorted = np.array(dates)[order]
# # t_sorted = t[order]

# fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True, dpi=600)
# axs = axs.flatten()

# # vmin = np.nanmin([np.nanmin(anomaly_scores[i]) for i in indices])
# # vmax = -1.5
# # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
# cmap = plt.cm.magma_r

# for i, idx in enumerate(indices):
#     lower = (
#         double_logistic_function(t_sorted, torch.as_tensor(params_lower[[idx]]).float())
#         .squeeze()
#         .numpy()
#     )
#     upper = (
#         double_logistic_function(t_sorted, torch.as_tensor(params_upper[[idx]]).float())
#         .squeeze()
#         .numpy()
#     )
#     ndvi_vals = np.asarray(ndvi[idx])[order]
#     ndsi_vals = np.asarray(ndsi[idx])[order]

#     is_valid = (
#         (ndvi_vals != -32768)
#         & (ndvi_vals != 32767)
#         & (ndsi_vals < 4300)
#         & ~(np.isnan(ndvi_vals))
#         & (ndvi_vals <= 10000)
#         & (ndvi_vals > -1000)
#     )
#     ndvi_valid = ndvi_vals[is_valid] / 10000.0
#     dates_valid = dates_sorted[is_valid]

#     iqr = np.abs(upper - lower)
#     low_thr = lower - 1.5 * iqr
#     high_thr = upper + 1.5 * iqr

#     ax = axs[i]
#     ax.plot(dates_sorted, high_thr, ls="--", lw=1, color="tab:green", alpha=0.8)
#     ax.plot(dates_sorted, low_thr, ls="--", lw=1, color="tab:red", alpha=0.8)
#     ax.plot(dates_sorted, lower, color="tab:red", lw=1.5)
#     ax.plot(dates_sorted, upper, color="tab:green", lw=1.5)
#     ax.fill_between(dates_sorted, lower, upper, color="tab:red", alpha=0.1)

#     ndvi_interp_low = np.interp(
#         dates_valid.astype("datetime64[D]").astype(float),
#         dates_sorted.astype("datetime64[D]").astype(float),
#         low_thr,
#     )
#     ndvi_interp_high = np.interp(
#         dates_valid.astype("datetime64[D]").astype(float),
#         dates_sorted.astype("datetime64[D]").astype(float),
#         high_thr,
#     )
#     is_anomaly = ndvi_valid < ndvi_interp_low

#     scores_valid = anomaly_scores[idx][order][is_valid]
#     scores_anom = scores_valid[is_anomaly]
#     colors = cmap(norm(scores_anom))

#     ax.scatter(
#         dates_valid[~is_anomaly],
#         ndvi_valid[~is_anomaly],
#         color="black",
#         s=10,
#         zorder=3,
#         label="Normal",
#     )
#     ax.scatter(
#         dates_valid[is_anomaly],
#         ndvi_valid[is_anomaly],
#         c=colors,
#         s=15,
#         zorder=4,
#         label="Anomaly",
#     )

#     ax.set_ylim(-0.1, 1.001)
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
#     ax.set_ylabel("NDVI")

# sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
# fig.colorbar(
#     sm, ax=axs, orientation="vertical", fraction=0.03, pad=0.02, label="Anomaly score"
# )

# plt.tight_layout(rect=[0, 0, 0.88, 1])
# plt.subplots_adjust(hspace=0.05)

# plt.savefig("../figs/figure_1.png", bbox_inches="tight", pad_inches=0)

# plt.show()

