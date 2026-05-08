import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr
import dask.array as da
import pystac_client
from dask.distributed import Client
import zarr
#  nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tmp_extract_timeseries_for_report.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/logs/retirieve_images.log 2>&1 &


if __name__ == "__main__":


    INPUT_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7b.zarr" 

    INPUT_LOOKUPTABLE = "/mnt/data2/UniBe-swiss-ndvi/input_data/lookup_table_median_ndvi_v7.zarr"

    ds = xr.open_zarr(INPUT_ZARR, chunks={"date": -1, "pixel": 500})

    ds2 = xr.open_zarr(INPUT_LOOKUPTABLE, chunks={"date": -1, "pixel": 500})

    print(ds)
    print(ds2)

    
    dates = ds.date.values
    doys = ds.doy.values  

    START_DATE = str(dates[0])[:10]
    END_DATE = str(dates[-1])[:10]

    names = ["low_broad","high_broad","low_ever","high_ever","fire","non_fire","drought","storm"]

    names_better = ["Lowland broadleaf area","Highland broadleaf area","Lowland evergreen area","Highland evergreen area","Biscth fire affected area","Biscth fire non affected area","Drought affected area","Vaia storm affected area"]

    X = np.array([2694491, 2692020, 2761097, 2781537, 2644029, 2644328, 2690025, 2689564])
    Y = np.array([1126023, 1121443, 1194613, 1182975, 1134128, 1134342, 1287413, 1154411])

    z = zarr.open(INPUT_ZARR, mode="r")
    x_coords = z["x"][:] 
    y_coords = z["y"][:] 

    for i in range(8):
        name = names[i]
        x = X[i]
        y = Y[i]

        # Find closest X — pure numpy, instant
        closest_x = x_coords[np.abs(x_coords - x).argmin()]
        x_mask = x_coords == closest_x

        # Within X subset, find closest Y — pure numpy, instant
        y_sub = y_coords[x_mask]
        closest_y = y_sub[np.abs(y_sub - y).argmin()]

        # Get the single pixel index
        pixel_idx = np.where(x_mask & (y_coords == closest_y))[0][0]

        print(f"{name}: X={closest_x}, Y={closest_y}, pixel_idx={pixel_idx}")

        # Only touch xarray/Dask here for the actual timeseries
        ndvi = ds["ndvi_processed"].isel(pixel=pixel_idx).load().to_numpy() / 10000
        mask_array = ds["mask_array"].isel(pixel=pixel_idx).load().to_numpy()

        median = ds2["median_ndvi"].isel(pixel=pixel_idx).load().to_numpy() / 10000 

        valid = doys <= 365
        median_mapped = median[doys[valid] - 1] 




        no_obs_to_smooth = mask_array == 0
        no_obs_smoothed = mask_array == 1
        obs_to_smooth = mask_array == 2
        obs_smoothed = mask_array == 3
        outlier_smoothed = mask_array == 4
        valid_obs = obs_smoothed | outlier_smoothed

        # d) Make plot using your exact styling
        plt.figure(figsize=(7.2, 4), dpi=200)

        plt.plot(dates[valid_obs],     ndvi[valid_obs],     marker="x", linestyle="None", markersize=4, color="green",  label="obs smoothed")
        plt.plot(dates[valid], median_mapped, linestyle="-", linewidth=1.2, color="black", label="median NDVI values")
        plt.plot(dates[valid],     ndvi[valid],  linestyle="-", linewidth=1.2, color="green",  label="obs smoothed")



        plt.ylim(0, 1) 
        plt.xlabel("Date")
        plt.ylabel("NDVI")
        plt.title(f"NDVI Time Series of {names_better[i]}")
        plt.grid(True)
        plt.legend(fontsize='x-small', loc='upper left', ncol=2)
        plt.tight_layout()
        
        # e) Output figure using your specific naming convention
        output_dir = "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/report/fig/prova3/"
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        
        plotpath = (
            output_dir + name +
            ".png")
        
        plt.savefig(plotpath)
        print(f"Saved: {plotpath}")
        plt.close()