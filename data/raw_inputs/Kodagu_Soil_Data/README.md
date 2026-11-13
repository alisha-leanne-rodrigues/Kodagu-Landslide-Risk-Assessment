# Kodagu soil data - ISRIC SoilGrids 2.0

Created (UTC): 2026-11-12T05:36:18
Rasters delivered: 12 of 12 requested (4 properties x 3 depths).
Failures:
none

## What this is
Real SoilGrids 2.0 predictions, **median (quantile 0.5)**, cropped to the Kodagu bounding box straight from the
official ISRIC WebDAV VRTs using GDAL `/vsicurl` (https://docs.isric.org/globaldata/soilgrids/webdav_from_Python.html).

| property | depths | stored values (mapped units) | divide by | conventional units |
|---|---|---|---|---|
| phh2o | 0-5, 5-15, 15-30 cm | pH x 10 | 10 | pH in water |
| clay | same | g/kg | 10 | % (g/100 g) |
| sand | same | g/kg | 10 | % (g/100 g) |
| soc | same | dg/kg | 10 | g/kg |

## Technical facts
- Native resolution **250 m**, native CRS **Interrupted Goode Homolosine** (`+proj=igh +datum=WGS84`). Nothing was
  resampled or reprojected: the crop is a pixel window on the source grid, so pixels are exactly ISRIC's.
- Values are integers in *mapped units* (see table). Convert yourself: `real = value / factor`. Nodata is kept as served by ISRIC (see `raster_summary.csv`).
- The rasters cover the district's bounding box (+1 pixel), not only the district; cells outside Kodagu hold real SoilGrids values. `raster_summary.csv` statistics use only cells touching the district polygon.
- Boundary used for the crop: OpenStreetMap administrative boundary via OSMnx geocode_to_gdf (Nominatim) (area 4104 km2, centroid (12.325 N, 75.801 E)). Stored in `kodagu_boundary.gpkg`.
- Filenames: `<property>_<depth>_Q0.5_IGH_250m.tif`.

## Using it with your 30 m EPSG:7755 layers
Sample the soil raster at your sample points (reproject the points to the soil raster CRS first), or warp it to your
grid with `rasterio.warp.reproject`/`gdalwarp -r near`. Note that 250 m pixels are ~8 x 8 of your 30 m pixels, so soil
will be constant over blocks of terrain pixels; treat it as a coarse regional covariate.

## Limitations
- SoilGrids is a machine-learning *model prediction*, not a soil survey. Uncertainty is large in steep, forested terrain such as the Western Ghats. Only the median is included; ISRIC also publishes Q0.05, Q0.95 and uncertainty layers.
- Predicted depth intervals are 0-5, 5-15 and 15-30 cm; landslide failure planes are often deeper (see 30-60+ cm layers on the same server if needed).

## Citation
See `CITATION.txt`. Poggio et al. (2021), SOIL 7, 217-240, doi:10.5194/soil-7-217-2021.
