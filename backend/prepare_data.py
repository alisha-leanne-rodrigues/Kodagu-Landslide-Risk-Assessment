#!/usr/bin/env python3
"""One-time preparation of the application data folder from the ORIGINAL project rasters.

What it does (all from real input files, nothing synthesised):
  1. Crops DEM, slope, distance-to-rivers, land cover to the Kodagu working window used when the
     training dataset was built (rows 3985-7422, cols 2140-5209 of the 30 m EPSG:7755 grid).
  2. Computes HAND (height above nearest river cell) exactly as in the dataset build.
  3. Reads the four SoilGrids 5-15 cm layers (pH, clay, sand, SOC), converts units, applies the
     same nearest-valid-cell fill (<= 3 cells = 750 m) used for the dataset.
  4. Rebuilds the mean monsoon (Jun-Sep) rainfall surface from the 2015-2025 IMD NetCDF files.
  5. Rasterises the OSM Kodagu boundary (from kodagu_boundary.gpkg) into a district mask and
     exports it as GeoJSON.
  6. Stores training-reference information (feature ranges, locations of training landslide
     pixels) for the spatial-support indicator, and copies the saved model bundle.

Usage:  python scripts/prepare_data.py --raw-dir /path/to/folder/with/unzipped/inputs
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from backend.geo import (GRID_X0, GRID_Y0, RES, rasterize_rings, read_gpkg_polygons,  # noqa: E402
                     polygon_area_m2)
from backend.layers import save_layer  # noqa: E402

# Working window of the training build (identical => identical HAND / filter edge behaviour)
WINDOW = (3985, 7422, 2140, 5209)  # r0, r1, c0, c1


def find(root, pattern, required=True):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits and required:
        raise FileNotFoundError("Could not find %s under %s" % (pattern, root))
    return hits


def read_tif(path, window=None):
    import tifffile
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        a = tifffile.memmap(path)
    except Exception:
        try:
            a = tifffile.imread(path)
        except Exception:
            a = np.array(Image.open(path))
    if window is not None:
        r0, r1, c0, c1 = window
        a = a[r0:r1, c0:c1]
    return np.array(a)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    r0, r1, c0, c1 = WINDOW
    manifest = {"created_utc": dt.datetime.utcnow().isoformat(timespec="seconds"), "window_rc": WINDOW,
                "grid": {"crs": "EPSG:7755", "res_m": RES, "origin_x": GRID_X0, "origin_y": GRID_Y0}, "sources": {}}

    # ---------------------------------------------------------------- terrain / hydrology / LC
    p = find(args.raw_dir, "DEM_StudyArea_7755.tif")[0]
    print("DEM", p)
    dem = read_tif(p, WINDOW).astype(np.float32)
    bad = (dem > 1e30) | (dem < -100)
    dem[bad] = np.nan
    manifest["sources"]["elevation"] = {"file": os.path.basename(p), "nodata": ">1e30", "units": "m"}

    p = find(args.raw_dir, "Slope_StudyArea_7755.tif")[0]
    slope = read_tif(p, WINDOW).astype(np.float32)
    slope[(slope < 0) | (slope > 1e30) | bad] = np.nan
    manifest["sources"]["slope"] = {"file": os.path.basename(p), "nodata": -9999, "units": "degrees"}

    p = find(args.raw_dir, "Distance_to_Rivers_StudyArea_7755.tif")[0]
    dist = read_tif(p, WINDOW).astype(np.float32)
    dist[(dist < 0) | (dist > 1e30) | bad] = np.nan
    manifest["sources"]["dist_river_m"] = {"file": os.path.basename(p), "units": "m",
                                            "origin": "HydroRIVERS v1.0 (Asia)"}

    p = find(args.raw_dir, "River_Raster_7755.tif")[0]
    river = read_tif(p, WINDOW) > 0
    print("river cells in window:", int(river.sum()))
    _, idx = ndi.distance_transform_edt(~river, return_indices=True)
    hand = dem - dem[idx[0], idx[1]]
    del idx
    hand[~np.isfinite(dem)] = np.nan
    hand = np.clip(hand, 0, None).astype(np.float32)
    manifest["sources"]["hand_m"] = {"derived_from": "DEM + River_Raster_7755.tif", "units": "m",
                                      "method": "elevation minus elevation of nearest river cell, clipped >= 0"}

    p = find(args.raw_dir, "WorldCover_2021_StudyArea_7755_aligned.tif")[0]
    lc = read_tif(p, WINDOW).astype(np.uint8)
    manifest["sources"]["landcover"] = {"file": os.path.basename(p), "product": "ESA WorldCover 10 m 2021 v200",
                                        "classes_used": [10, 20, 30, 40, 50, 60, 80, 90]}

    # ---------------------------------------------------------------- boundary / mask
    gpkg = find(args.raw_dir, "kodagu_boundary.gpkg")[0]
    polys = read_gpkg_polygons(gpkg)
    outer = [pl[0] for pl in polys]
    holes = [r for pl in polys for r in pl[1:]]
    mask = rasterize_rings(outer, r0, r1, c0, c1, holes)
    area = sum(polygon_area_m2(o) for o in outer) - sum(polygon_area_m2(h) for h in holes)
    print("boundary px", int(mask.sum()), "area km2", round(area / 1e6, 1))
    feat = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"name": "Kodagu district (Coorg)", "source": "OpenStreetMap administrative boundary, "
                       "relation 2020511 (via OSMnx/Nominatim, 2026-11-12)", "licence": "ODbL 1.0 (c) OpenStreetMap contributors",
                       "area_km2": round(area / 1e6, 1)},
        "geometry": {"type": "MultiPolygon" if len(polys) > 1 else "Polygon",
                     "coordinates": [[r.round(6).tolist() for r in pl] for pl in polys] if len(polys) > 1
                     else [r.round(6).tolist() for r in polys[0]]}}]}
    json.dump(feat, open(os.path.join(out, "kodagu_boundary.geojson"), "w"))
    manifest["boundary"] = {"file": "kodagu_boundary.geojson", "area_km2": round(area / 1e6, 1), "sha256_gpkg": sha256(gpkg)}

    # ---------------------------------------------------------------- soil (SoilGrids 5-15 cm)
    soil_defs = {"soil_ph_5_15cm": "phh2o_5-15cm", "soil_clay_pct_5_15cm": "clay_5-15cm",
                 "soil_sand_pct_5_15cm": "sand_5-15cm", "soil_soc_g_kg_5_15cm": "soc_5-15cm"}
    soil, soil_filled = {}, None
    for name, stem in soil_defs.items():
        p = find(args.raw_dir, "%s_Q0.5_IGH_250m.tif" % stem)[0]
        a = read_tif(p).astype(np.float64)
        nd = a == -32768
        a = np.where(nd, np.nan, a / 10.0)
        dd, ii = ndi.distance_transform_edt(nd, return_indices=True)
        filled = a.copy()
        use = nd & (dd <= 3.0)  # <= 3 cells (750 m)
        filled[use] = a[ii[0][use], ii[1][use]]
        soil[name] = filled.astype(np.float32)
        soil_filled = nd if soil_filled is None else (soil_filled | nd)
        manifest["sources"][name] = {"file": os.path.basename(p), "units": {"ph": "pH", "clay": "%", "sand": "%",
                                     "soc": "g/kg"}[name.split("_")[1]], "native_res_m": 250,
                                     "conversion": "raw / 10 (SoilGrids mapped units)"}
    np.savez_compressed(os.path.join(out, "soil_5_15cm.npz"), **soil, filled_mask=soil_filled,
                        igh_origin_x=8270250.0, igh_origin_y=1429000.0, cell_m=250.0)

    # ---------------------------------------------------------------- rainfall (mean monsoon total)
    from scipy.io import netcdf_file
    files = find(args.raw_dir, "RF25_ind*_rfp25.nc")
    years = sorted(int(os.path.basename(f)[8:12]) for f in files)
    print("rainfall years", years)
    mon = []
    lat = lon = None
    for y in years:
        f = netcdf_file(find(args.raw_dir, "RF25_ind%d_rfp25.nc" % y)[0], mmap=False)
        a = np.array(f.variables["RAINFALL"][:], dtype=np.float32)
        a[a < -900] = np.nan
        days = np.array(f.variables["TIME"][:]).astype(int)
        months = (np.datetime64("1900-12-31") + days.astype("timedelta64[D]")).astype("datetime64[M]").astype(int) % 12 + 1
        m = (months >= 6) & (months <= 9)
        mon.append(np.nansum(a[m], 0))
        lat = np.array(f.variables["LATITUDE"][:])
        lon = np.array(f.variables["LONGITUDE"][:])
        f.close()
    rain = np.mean(mon, 0).astype(np.float32)
    np.savez_compressed(os.path.join(out, "rain_monsoon_mean.npz"), grid=rain, lat=lat, lon=lon)
    manifest["sources"]["rain_monsoon_mean_mm"] = {"files": "IMD RF25 gridded daily rainfall %d-%d" % (years[0], years[-1]),
                                                     "units": "mm (June-September total, mean over years)",
                                                     "native_res_deg": 0.25, "interpolation": "bilinear"}

    # ---------------------------------------------------------------- save layers
    for name, arr in [("elevation", dem), ("slope", slope), ("dist_river_m", dist), ("hand_m", hand),
                      ("landcover", lc), ("district_mask", mask.astype(np.uint8))]:
        save_layer(os.path.join(out, "layer_%s.npz" % name), arr)
        print("saved layer", name, arr.dtype, arr.shape)

    # ---------------------------------------------------------------- training reference + model
    import pandas as pd
    csv = find(args.raw_dir, "kodagu_integrated_core.csv")[0]
    info = json.load(open(find(args.raw_dir, "random_forest_model_info.json")[0]))
    feats = info["features"]
    d = pd.read_csv(csv)
    ref = {"n_rows": int(len(d)), "n_positive": int(d.label.sum()),
           "feature_stats": {f: {"min": float(d[f].min()), "p1": float(d[f].quantile(.01)), "median": float(d[f].median()),
                                 "p99": float(d[f].quantile(.99)), "max": float(d[f].max())} for f in feats},
           "extent_rows": [int(d.row.min()), int(d.row.max())], "extent_cols": [int(d.col.min()), int(d.col.max())]}
    json.dump(ref, open(os.path.join(out, "training_reference.json"), "w"), indent=1)
    pos = d[d.label == 1]
    np.save(os.path.join(out, "training_positive_rc.npy"), pos[["row", "col"]].values.astype(np.int32))
    np.save(os.path.join(out, "training_all_rc.npy"), d[["row", "col"]].values.astype(np.int32))
    d[d.suggested_split == "validation"][feats + ["label"]].to_csv(os.path.join(out, "validation_features.csv"), index=False)
    d[d.suggested_split == "test"][["sample_id", "label", "row", "col"] + feats].to_csv(os.path.join(out, "test_features.csv"), index=False)

    shutil.copy(find(args.raw_dir, "kodagu_deployment_bundle.joblib")[0], os.path.join(out, "kodagu_deployment_bundle.joblib"))
    shutil.copy(find(args.raw_dir, "random_forest_model_info.json")[0], os.path.join(out, "random_forest_model_info.json"))
    for fn in ("data_dictionary.csv",):
        hits = find(args.raw_dir, fn, required=False)
        if hits:
            shutil.copy(hits[0], os.path.join(out, fn))
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("DONE ->", out)


if __name__ == "__main__":
    main()