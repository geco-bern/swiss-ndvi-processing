# Gapfill NDVI

## How to Use the Package

To use the package, you have to download the required libraries using **Conda** or **Miniconda**.

### Setup with Conda
```bash
conda env create -f environment.yml --name ndvi
```

# Content

The demo is divided into two notebooks:  

1. **Smoothing options**  
2. **NDVI gapfilling**  

Both can be found here: `demo/notebook/`

---

## Pixel Selection

The notebooks can be used without any changes.  
We select **25 pixels** for each of the following biomes in this specific order:

| Biome                                | Coordinates (x, y)         | Pixel Range |
|--------------------------------------|----------------------------|-------------|
| Lowland broadleaf                     | 2694027.49, 1123239.84    | 0–24        |
| Highland broadleaf                    | 2694027.49, 1123239.84    | 25–49       |
| Lowland evergreen                     | 2613028.38, 1127777.24    | 50–74       |
| Highland evergreen                    | 2782037.00, 1183475.00    | 75–99       |
| Biscth fire affected area             | 2644008.20, 1133794.76    | 100–124     |
| Biscth fire nearby non-affected area  | 2644328.07, 1134342.81    | 125–149     |
| Drought-affected area                 | 2690025.48, 1287413.03    | 150–174     |
| Burlgim storm                         | —                         | 175–199     |

---

## Area of Interest

The areas of interest can be visualized directly in the notebook by inserting the coordinates.  

The data is stored here: `demo/pixel_biomes.zarr`  

To select any biome, choose the pixels accordingly in the given order.

---

## Smoothing Notebook

The **smoothing option notebook** presents three different smoothing methods:  

- **Savitzky–Golay filter**  
- **LOESS**  
- **Low-pass filter**  

We believe that the **Savitzky–Golay filter yields the best results**.

---

## NDVI Gapfilling Notebook

The **NDVI gapfilling notebook** provides a demo and explanation of the gapfilling process.  

It provides the full gapfilling using:  

- **L1**: linear interpolation  
- **L2**: smoothing with Savitzky–Golay  
- **Continuous integration setup**: L1, then smoothed with L2  
