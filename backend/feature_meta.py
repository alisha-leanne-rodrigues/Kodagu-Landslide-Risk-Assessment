"""Human-readable documentation of the 24 model features (source layer, unit, transformation)."""

DEM = "DEM_StudyArea_7755.tif (30 m, EPSG:7755)"
LC = "ESA WorldCover 2021 v200, 10 m, resampled/aligned to the 30 m grid (WorldCover_2021_StudyArea_7755_aligned.tif)"

FEATURE_META = {
    "elevation": dict(label="Elevation", unit="m", group="Terrain", source="DEM", layer=DEM,
                      transform="None (value of the DEM pixel)"),
    "slope": dict(label="Slope", unit="degrees", group="Terrain", source="DEM", layer="Slope_StudyArea_7755.tif (30 m)",
                  transform="None (supplied slope raster, degrees)"),
    "aspect_sin": dict(label="Aspect (sine)", unit="-1..1", group="Terrain", source="DEM", layer=DEM,
                       transform="sin(downslope direction), Horn 3x3 gradient; set to 0 where slope < 2 degrees"),
    "aspect_cos": dict(label="Aspect (cosine)", unit="-1..1", group="Terrain", source="DEM", layer=DEM,
                       transform="cos(downslope direction), Horn 3x3 gradient; set to 0 where slope < 2 degrees"),
    "curv_plan": dict(label="Plan curvature", unit="x100 1/m", group="Terrain", source="DEM", layer=DEM,
                      transform="Zevenbergen-Thorne plan curvature x100, clipped to +-5"),
    "curv_profile": dict(label="Profile curvature", unit="x100 1/m", group="Terrain", source="DEM", layer=DEM,
                         transform="Zevenbergen-Thorne profile curvature x100, clipped to +-5"),
    "tpi_150m": dict(label="Topographic position (150 m)", unit="m", group="Terrain", source="DEM", layer=DEM,
                     transform="elevation minus mean elevation of the 5x5 pixel window"),
    "tpi_900m": dict(label="Topographic position (900 m)", unit="m", group="Terrain", source="DEM", layer=DEM,
                     transform="elevation minus mean elevation of the 31x31 pixel window"),
    "relief_330m": dict(label="Local relief (330 m)", unit="m", group="Terrain", source="DEM", layer=DEM,
                        transform="max minus min elevation in the 11x11 pixel window"),
    "dist_river_m": dict(label="Distance to river", unit="m", group="Hydrology", source="HydroRIVERS v1.0",
                         layer="Distance_to_Rivers_StudyArea_7755.tif (30 m)", transform="None (Euclidean distance to nearest HydroRIVERS cell)"),
    "hand_m": dict(label="Height above nearest river", unit="m", group="Hydrology", source="DEM + HydroRIVERS",
                   layer=DEM + " + River_Raster_7755.tif", transform="elevation minus elevation of the nearest river cell, clipped >= 0"),
    "lc_10": dict(label="Land cover: tree cover", unit="0/1", group="Land cover", source="ESA WorldCover", layer=LC,
                  transform="One-hot of class 10"),
    "lc_20": dict(label="Land cover: shrubland", unit="0/1", group="Land cover", source="ESA WorldCover", layer=LC,
                  transform="One-hot of class 20"),
    "lc_30": dict(label="Land cover: grassland", unit="0/1", group="Land cover", source="ESA WorldCover", layer=LC,
                  transform="One-hot of class 30"),
    "lc_40": dict(label="Land cover: cropland", unit="0/1", group="Land cover", source="ESA WorldCover", layer=LC,
                  transform="One-hot of class 40"),
    "lc_50": dict(label="Land cover: built-up", unit="0/1", group="Land cover", source="ESA WorldCover", layer=LC,
                  transform="One-hot of class 50"),
    "lc_60": dict(label="Land cover: bare / sparse", unit="0/1", group="Land cover", source="ESA WorldCover", layer=LC,
                  transform="One-hot of class 60"),
    "lc_80": dict(label="Land cover: water", unit="0/1", group="Land cover", source="ESA WorldCover", layer=LC,
                  transform="One-hot of class 80"),
    "lc_90": dict(label="Land cover: wetland", unit="0/1", group="Land cover", source="ESA WorldCover", layer=LC,
                  transform="One-hot of class 90"),
    "soil_ph_5_15cm": dict(label="Soil pH (5-15 cm)", unit="pH", group="Soil", source="ISRIC SoilGrids 2.0",
                           layer="phh2o_5-15cm_Q0.5_IGH_250m.tif", transform="median prediction / 10; 250 m cell, nearest-valid fill <= 750 m"),
    "soil_clay_pct_5_15cm": dict(label="Soil clay (5-15 cm)", unit="%", group="Soil", source="ISRIC SoilGrids 2.0",
                                 layer="clay_5-15cm_Q0.5_IGH_250m.tif", transform="median prediction (g/kg) / 10; 250 m cell, nearest-valid fill <= 750 m"),
    "soil_sand_pct_5_15cm": dict(label="Soil sand (5-15 cm)", unit="%", group="Soil", source="ISRIC SoilGrids 2.0",
                                 layer="sand_5-15cm_Q0.5_IGH_250m.tif", transform="median prediction (g/kg) / 10; 250 m cell, nearest-valid fill <= 750 m"),
    "soil_soc_g_kg_5_15cm": dict(label="Soil organic carbon (5-15 cm)", unit="g/kg", group="Soil", source="ISRIC SoilGrids 2.0",
                                 layer="soc_5-15cm_Q0.5_IGH_250m.tif", transform="median prediction (dg/kg) / 10; 250 m cell, nearest-valid fill <= 750 m"),
    "rain_monsoon_mean_mm": dict(label="Mean monsoon rainfall", unit="mm", group="Rainfall", source="IMD gridded rainfall (RF25, 0.25 deg)",
                                 layer="RF25_ind2015..2025_rfp25.nc", transform="June-September total per year, mean of 2015-2025, bilinear interpolation to the pixel"),
}

BINARY_FEATURES = {f for f in FEATURE_META if f.startswith("lc_")}

LC_NAMES = {"lc_10": "tree cover", "lc_20": "shrubland", "lc_30": "grassland", "lc_40": "cropland",
            "lc_50": "built-up", "lc_60": "bare / sparse vegetation", "lc_80": "water", "lc_90": "wetland"}


def label(name: str) -> str:
    return FEATURE_META.get(name, {}).get("label", name)