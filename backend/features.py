"""Raster data store and feature extraction.

The 24 model inputs are re-created from the original layers with exactly the definitions that were
used when the training table was built:

  elevation, slope           DEM_StudyArea_7755 / Slope_StudyArea_7755 (30 m, EPSG:7755)
  aspect_sin/cos, curv_*     3x3 Horn gradient / Zevenbergen-Thorne curvature from the DEM
  tpi_150m, tpi_900m         DEM minus mean of a 5x5 / 31x31 window
  relief_330m                max-min of an 11x11 window
  dist_river_m, hand_m       HydroRIVERS distance raster; DEM minus DEM at nearest river cell
  lc_*                       one-hot of ESA WorldCover 2021 classes
  soil_*_5_15cm              SoilGrids 2.0 median, 250 m, nearest cell (+ nearest-valid fill)
  rain_monsoon_mean_mm       IMD 0.25 deg Jun-Sep mean total 2015-25, bilinear

Terrain derivatives use a neighbourhood, so windows are padded by PAD pixels and cropped afterwards;
results are identical to computing on the whole district at once.
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .geo import igh_forward_north_lobe2, rowcol_to_lonlat
from .layers import load_layer

PAD = 20
LC_CLASSES = (10, 20, 30, 40, 50, 60, 80, 90)
SOIL_NAMES = ("soil_ph_5_15cm", "soil_clay_pct_5_15cm", "soil_sand_pct_5_15cm", "soil_soc_g_kg_5_15cm")
WINDOW = (3985, 7422, 2140, 5209)  # must equal scripts/prepare_data.py


class DataStore:
    """Holds the Kodagu working-window rasters in memory (about 150 MB)."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.r0, self.r1, self.c0, self.c1 = WINDOW
        need = ["elevation", "slope", "dist_river_m", "hand_m", "landcover", "district_mask"]
        missing = [n for n in need if not os.path.exists(os.path.join(data_dir, "layer_%s.npz" % n))]
        for fn in ("soil_5_15cm.npz", "rain_monsoon_mean.npz"):
            if not os.path.exists(os.path.join(data_dir, fn)):
                missing.append(fn)
        if missing:
            raise FileNotFoundError("Missing data files in %s: %s. Run scripts/prepare_data.py" % (data_dir, missing))
        self.elevation = load_layer(os.path.join(data_dir, "layer_elevation.npz"))
        self.slope = load_layer(os.path.join(data_dir, "layer_slope.npz"))
        self.dist_river_m = load_layer(os.path.join(data_dir, "layer_dist_river_m.npz"))
        self.hand_m = load_layer(os.path.join(data_dir, "layer_hand_m.npz"))
        self.landcover = load_layer(os.path.join(data_dir, "layer_landcover.npz"))
        self.mask = load_layer(os.path.join(data_dir, "layer_district_mask.npz")).astype(bool)
        z = np.load(os.path.join(data_dir, "soil_5_15cm.npz"))
        self.soil = {n: z[n] for n in SOIL_NAMES}
        self.soil_filled_mask = z["filled_mask"]
        self.igh_x0, self.igh_y0, self.igh_cell = float(z["igh_origin_x"]), float(z["igh_origin_y"]), float(z["cell_m"])
        z = np.load(os.path.join(data_dir, "rain_monsoon_mean.npz"))
        self.rain_grid, self.rain_lat, self.rain_lon = z["grid"], z["lat"], z["lon"]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation.shape

    def in_window(self, rows, cols):
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        return (rows >= self.r0) & (rows < self.r1) & (cols >= self.c0) & (cols < self.c1)

    def district_mask_at(self, rows, cols):
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        ok = self.in_window(rows, cols)
        out = np.zeros(rows.shape, bool)
        out[ok] = self.mask[rows[ok] - self.r0, cols[ok] - self.c0]
        return out


# ---------------------------------------------------------------- terrain maths
def _nanfilter_mean(a: np.ndarray, size: int) -> np.ndarray:
    v = np.isfinite(a)
    a0 = np.where(v, a, 0).astype(np.float32)
    s = ndi.uniform_filter(a0, size, mode="nearest")
    n = ndi.uniform_filter(v.astype(np.float32), size, mode="nearest")
    return np.where(n > 0.02, s / np.maximum(n, 1e-6), np.nan).astype(np.float32)


def terrain_derivatives(dem: np.ndarray) -> Dict[str, np.ndarray]:
    """aspect_sin/cos, curvature, TPI and relief from a DEM array (NaN = no data)."""
    L = 30.0
    out = {}
    z = np.pad(dem, 1, mode="edge")
    z1, z2, z3 = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z4, z5, z6 = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
    z7, z8, z9 = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * L)
    dzdy = ((z1 + 2 * z2 + z3) - (z7 + 2 * z8 + z9)) / (8 * L)
    asp = np.arctan2(-dzdx, -dzdy)  # downslope direction, clockwise from north
    steep = np.hypot(dzdx, dzdy) > np.tan(np.radians(2))
    out["aspect_sin"] = np.where(steep, np.sin(asp), 0).astype(np.float32)
    out["aspect_cos"] = np.where(steep, np.cos(asp), 0).astype(np.float32)
    Dd = ((z4 + z6) / 2 - z5) / L ** 2
    Ee = ((z2 + z8) / 2 - z5) / L ** 2
    Ff = (-z1 + z3 + z7 - z9) / (4 * L ** 2)
    G = (-z4 + z6) / (2 * L)
    H = (z2 - z8) / (2 * L)
    g2 = G ** 2 + H ** 2
    ok = g2 > 1e-6
    g2s = np.where(ok, g2, 1)
    prof = np.where(ok, -2 * (Dd * G ** 2 + Ee * H ** 2 + Ff * G * H) / g2s, 0) * 100
    plan = np.where(ok, 2 * (Dd * H ** 2 + Ee * G ** 2 - Ff * G * H) / g2s, 0) * 100
    nodem = ~np.isfinite(dem)
    prof[nodem] = np.nan
    plan[nodem] = np.nan
    out["curv_profile"] = np.clip(prof, -5, 5).astype(np.float32)
    out["curv_plan"] = np.clip(plan, -5, 5).astype(np.float32)
    out["tpi_150m"] = (dem - _nanfilter_mean(dem, 5)).astype(np.float32)
    out["tpi_900m"] = (dem - _nanfilter_mean(dem, 31)).astype(np.float32)
    mx = ndi.maximum_filter(np.where(np.isfinite(dem), dem, -1e9), 11)
    mn = ndi.minimum_filter(np.where(np.isfinite(dem), dem, 1e9), 11)
    rel = (mx - mn).astype(np.float32)
    rel[nodem] = np.nan
    out["relief_330m"] = rel
    return out


def sample_soil(store: DataStore, lon: np.ndarray, lat: np.ndarray) -> Dict[str, np.ndarray]:
    lon = np.round(lon, 5)  # the training table stored lon/lat rounded to 5 decimals
    lat = np.round(lat, 5)
    x, y = igh_forward_north_lobe2(lon, lat)
    sc = np.floor((x - store.igh_x0) / store.igh_cell).astype(int)
    sr = np.floor((store.igh_y0 - y) / store.igh_cell).astype(int)
    out = {}
    for n in SOIL_NAMES:
        a = store.soil[n]
        ok = (sr >= 0) & (sr < a.shape[0]) & (sc >= 0) & (sc < a.shape[1])
        v = np.full(lon.shape, np.nan, np.float32)
        v[ok] = a[sr[ok], sc[ok]]
        out[n] = v
    return out


def sample_rain(store: DataStore, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    fi = (lat - store.rain_lat[0]) / 0.25
    fj = (lon - store.rain_lon[0]) / 0.25
    g = store.rain_grid
    gg = np.where(np.isfinite(g), g, 0)
    w = np.isfinite(g).astype(float)
    a = ndi.map_coordinates(gg, [fi.ravel(), fj.ravel()], order=1, mode="nearest")
    b = ndi.map_coordinates(w, [fi.ravel(), fj.ravel()], order=1, mode="nearest")
    return np.where(b > 0.5, a / np.maximum(b, 1e-6), np.nan).astype(np.float32).reshape(lon.shape)


# ---------------------------------------------------------------- extraction
def extract_window(store: DataStore, r0: int, r1: int, c0: int, c1: int) -> Dict[str, np.ndarray]:
    """All candidate model features for the global-grid window [r0:r1, c0:c1].
    Returns dict of 2D float32 arrays + 'inside_district' (bool)."""
    if r0 < store.r0 or c0 < store.c0 or r1 > store.r1 or c1 > store.c1 or r1 <= r0 or c1 <= c0:
        raise ValueError("window outside the prepared Kodagu raster area")
    pr0, pr1 = max(r0 - PAD, store.r0), min(r1 + PAD, store.r1)
    pc0, pc1 = max(c0 - PAD, store.c0), min(c1 + PAD, store.c1)
    sl = (slice(pr0 - store.r0, pr1 - store.r0), slice(pc0 - store.c0, pc1 - store.c0))
    dem = store.elevation[sl]
    der = terrain_derivatives(dem)
    cr = (slice(r0 - pr0, r1 - pr0), slice(c0 - pc0, c1 - pc0))  # crop back
    cs = (slice(r0 - store.r0, r1 - store.r0), slice(c0 - store.c0, c1 - store.c0))
    out = {
        "elevation": store.elevation[cs].astype(np.float32),
        "slope": store.slope[cs].astype(np.float32),
        "dist_river_m": store.dist_river_m[cs].astype(np.float32),
        "hand_m": store.hand_m[cs].astype(np.float32),
    }
    for k, v in der.items():
        out[k] = v[cr]
    lc = store.landcover[cs]
    for k in LC_CLASSES:
        out["lc_%d" % k] = (lc == k).astype(np.float32)
    rows = np.arange(r0, r1)[:, None] * np.ones((1, c1 - c0), int)
    cols = np.ones((r1 - r0, 1), int) * np.arange(c0, c1)[None, :]
    lon, lat = rowcol_to_lonlat(rows, cols)
    out.update(sample_soil(store, lon, lat))
    out["rain_monsoon_mean_mm"] = sample_rain(store, lon, lat)
    out["inside_district"] = store.mask[cs]
    out["landcover_code"] = lc.copy()
    out["lon"] = lon.astype(np.float64)
    out["lat"] = lat.astype(np.float64)
    return out


def valid_mask(feats: Dict[str, np.ndarray], feature_names) -> np.ndarray:
    ok = feats["inside_district"].copy()
    for f in feature_names:
        ok &= np.isfinite(feats[f])
    return ok


def extract_pixels(store: DataStore, rows, cols, feature_names) -> Tuple[np.ndarray, np.ndarray]:
    """Feature matrix (n, len(feature_names)) for arbitrary pixel indices + validity vector."""
    rows = np.asarray(rows, int)
    cols = np.asarray(cols, int)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    w = extract_window(store, r0, r1, c0, c1)
    X = np.column_stack([w[f][rows - r0, cols - c0] for f in feature_names]).astype(np.float64)
    ok = valid_mask(w, feature_names)[rows - r0, cols - c0]
    return X, ok