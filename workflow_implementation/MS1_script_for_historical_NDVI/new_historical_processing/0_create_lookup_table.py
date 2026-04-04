
import numpy as np
import math
import zarr
import pandas as pd
import torch
import os
import time
from dask.distributed import Client, LocalCluster
import xarray as xr
import dask.array as da

if __name__ == '__main__':
    # NOTE: the data are in the workstation
    SRC_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
    ds0 = zarr.open_group(SRC_ZARR, mode="r")


    # SETUP PARALLELIZATION CLUSTER
    client = Client(
        n_workers=30,
        threads_per_worker=1,
        processes=True,  # Use separate processes (not threads, this appears to be much faster (even though using non-shared memory))
        dashboard_address=':2232'
    )  # start distributed scheduler locally.
    client.dashboard_link


    ds0 = zarr.open_group(SRC_ZARR, mode="r")

    # Lazy dask arrays from zarr
    ndvi_z = ds0["ndvi"]
    pl_z   = ds0["params_2"]["params_lower"]
    pu_z   = ds0["params_2"]["params_upper"]

    ndvi_da = da.from_zarr(ndvi_z)     # lazy
    pl_da   = da.from_zarr(pl_z)       # lazy
    pu_da   = da.from_zarr(pu_z)       # lazy

    param_labels = ['par0', 'par1', 'par2', 'par3', 'par4', 'par5'] # TODO: what are param names from Samanthas model?
    params_lower_xr = xr.DataArray(pl_da,   dims=("pixel", "param"), coords={               "pixel": np.arange(pl_da.shape[0]), "param": param_labels})
    params_upper_xr = xr.DataArray(pu_da,   dims=("pixel", "param"), coords={               "pixel": np.arange(pl_da.shape[0]), "param": param_labels})
    pl_sel   = params_lower_xr 
    pu_sel   = params_upper_xr


    def double_logistic(t, params):
        sos, mat_minus_sos, sen, eos_minus_sen, M, m = np.split(params, 6, axis=-1)
        mat_minus_sos = np.log1p(np.exp(mat_minus_sos))
        eos_minus_sen = np.log1p(np.exp(eos_minus_sen))
        t = t[None, :]  # shape (1, date)
        sigmoid_sos_mat = 1 / (1 + np.exp(2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-10)))
        sigmoid_sen_eos = 1 / (1 + np.exp(2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-10)))
        return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

    # calcualte the vallue for all the doys
    doy = np.arange(1,366)
    t_scaled = doy / 365.0  

    def build_median_ndvi_block(pl_block, pu_block):
        ndvi_lower = double_logistic(t_scaled, pl_block)
        ndvi_upper = double_logistic(t_scaled, pu_block)
        return ((ndvi_lower + ndvi_upper) / 2.0 * 10000).astype(np.int16)

    median_da = da.map_blocks(
        build_median_ndvi_block,
        pl_sel.data,
        pu_sel.data,
        dtype=np.int16,
        chunks=(pl_sel.chunks[0], (len(doy),))
    )

    median_ndvi_xr = xr.DataArray(
        median_da,
        dims=("pixel", "doy"),
        coords={"pixel": pl_sel["pixel"], "doy": doy},
        name="median_ndvi",
    )

    # change number types of dimensions
    median_ndvi_xr = median_ndvi_xr.assign_coords(
        pixel   = ('pixel', median_ndvi_xr.pixel.values.astype(np.int32)),
        doy     = ('doy', median_ndvi_xr.doy.values.astype(np.int32))
    )

    OUT = "/data_3/francesco/lookup_table_median_ndvi_v7.zarr"

    median_ndvi_xr.to_dataset().to_zarr(OUT, mode="w", consolidated=True)

