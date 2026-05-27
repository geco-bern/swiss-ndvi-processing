import matplotlib.pyplot as plt
# OUTPUT_ZARR = "foo.zarr" 
# test load this dataset:
# ds_test = xr.open_dataset(OUTPUT_ZARR)
# da = ds_test["ndvi"].isel(datetime = 0)

# test plot this dataset    
# xmin, xmax = 2650000, 2750000 # focus on Ticino
# ymin, ymax = 1070000, 1160000 # focus on Ticino
# pixels_subset_mask = (
#     (ds_test.x.values >= xmin) &
#     (ds_test.x.values <= xmax) &
#     (ds_test.y.values >= ymin) &
#     (ds_test.y.values <= ymax)
# )
# ds_test_subset = ds_test["ndvi"].isel(pixel=pixels_subset_mask.nonzero()[0])
# plot_da_map(ds_test_subset.isel(datetime = 0), png_fname = 'foo5.png')
# plot_da_map(ds_test_subset.isel(datetime = 0), reduction_factor = 1, png_fname = 'foo1.png')


def plot_da_map(da, *, reduction_factor = 5, png_fname = 'foo.png'): 
    # da is a 1D slice of a single timepoint (note: small enough to be plotted)

    # load (or compute) arrays (be careful with very large datasets)
    vals = da.data.compute() if hasattr(da.data, "compute") else da.data
    x_idx = da["x_idx"].data.compute() if hasattr(da["x_idx"].data, "compute") else da["x_idx"].data
    y_idx = da["y_idx"].data.compute() if hasattr(da["y_idx"].data, "compute") else da["y_idx"].data

    # handle fill / nodata
    INVALID = -2**15         # Filtered out pixels, e.g. cloud shadows
    NO_COVERAGE = 2**15 - 1  # Pixels with no data for the given time step
    vals = vals.astype(float)
    vals[vals == NO_COVERAGE] = np.nan
    vals[vals == INVALID] = np.nan

    # make grid indices zero-based
    x0 = int(x_idx.min()); y0 = int(y_idx.min())
    x_i = (x_idx - x0).astype(int)
    y_i = (y_idx - y0).astype(int)

    # grid shape
    nx = int(x_i.max() + 1)
    ny = int(y_i.max() + 1)

    grid = np.full((nx, ny), np.nan, dtype=float) # together with origin="upper" we
    grid[x_i, y_i] = vals / 10000                 # can use x as row and y as col index

    # optional: get real coords for extent if available
    if "x" in da.coords and "y" in da.coords:
        x_coords = da["x"].data.compute() if hasattr(da["x"].data, "compute") else da["x"].data
        y_coords = da["y"].data.compute() if hasattr(da["y"].data, "compute") else da["y"].data
        # unique sorted coordinates for grid cell size
        ux = np.unique(x_coords)
        uy = np.unique(y_coords)
        dx = ux[1] - ux[0] if ux.size>1 else 1
        dy = uy[1] - uy[0] if uy.size>1 else 1
        xmin = ux.min() - dx/2
        xmax = ux.max() + dx/2
        ymin = uy.min() - dy/2
        ymax = uy.max() + dy/2
        extent = [xmin, xmax, ymin, ymax]
    else:
        extent = None

    # reduce number of pixels to plot:
    grid_reduced = grid[::reduction_factor, ::reduction_factor] # takes only every x-th row

    # plot
    fig = plt.figure(figsize=(9,7))
    im = plt.imshow(grid_reduced, origin="upper", aspect='equal',
                    vmin=-1, vmax=1, cmap="RdYlGn",
                    extent=extent)
    plt.colorbar(im, label="NDVI")
    if "datetime" in da.coords:
        plt.title(f"NDVI {da['datetime'].dt.strftime("%Y-%m-%d %H:%M").values}")
    else:
        plt.title(f"NDVI {da['date'].dt.strftime("%Y-%m-%d").values}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()
    fig.savefig(png_fname)
