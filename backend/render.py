"""Renders pixel-level model results as lat/lon-aligned PNG overlays for Leaflet.

The model grid is EPSG:7755 (slightly rotated against north-up web maps), so each output pixel
(lon, lat) is mapped back into the 30 m grid and the value of the pixel containing it is used.
Nothing is smoothed or interpolated: every coloured output pixel shows the value of one model pixel."""
import base64
import io
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .geo import lonlat_to_rowcol, rowcol_to_lonlat

POS_RGBA = (201, 58, 38, 205)      # at/above threshold
NEG_RGBA = (83, 140, 101, 150)     # below threshold
RAMP = [(0.00, (36, 110, 80)), (0.25, (214, 196, 78)), (0.50, (224, 120, 47)), (0.80, (150, 28, 28))]


def window_lonlat_bounds(r0: int, r1: int, c0: int, c1: int) -> Tuple[float, float, float, float]:
    rr = np.concatenate([np.linspace(r0, r1, 40), np.full(40, r0), np.full(40, r1), np.linspace(r0, r1, 40)])
    cc = np.concatenate([np.full(40, c0), np.linspace(c0, c1, 40), np.linspace(c0, c1, 40), np.full(40, c1)])
    lon, lat = rowcol_to_lonlat(rr - 0.5, cc - 0.5)
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def _ramp(v: np.ndarray) -> np.ndarray:
    xs = np.array([p[0] for p in RAMP])
    out = np.zeros(v.shape + (4,), dtype=np.uint8)
    for k in range(3):
        out[..., k] = np.interp(v, xs, [p[1][k] for p in RAMP]).astype(np.uint8)
    out[..., 3] = 200
    return out


def render_overlay(scores: np.ndarray, r0: int, c0: int, step: int, mode: str, threshold: float,
                   max_px: int = 900) -> Dict:
    """scores[i, j] = model score of the pixel block whose top-left pixel is (r0 + i*step, c0 + j*step)
    (NaN = not assessed). mode: 'binary' or 'score'. Returns base64 PNG + Leaflet bounds."""
    nr, nc = scores.shape
    lon0, lat0, lon1, lat1 = window_lonlat_bounds(r0, r0 + nr * step, c0, c0 + nc * step)
    mid = np.radians((lat0 + lat1) / 2)
    wkm, hkm = (lon1 - lon0) * 111.32 * np.cos(mid), (lat1 - lat0) * 110.57
    sc = max_px / max(wkm, hkm)
    w, h = max(2, int(wkm * sc)), max(2, int(hkm * sc))
    lon = lon0 + (np.arange(w) + 0.5) / w * (lon1 - lon0)
    lat = lat1 - (np.arange(h) + 0.5) / h * (lat1 - lat0)
    LON, LAT = np.meshgrid(lon, lat)
    row, col = lonlat_to_rowcol(LON, LAT)
    i = np.floor((row - r0) / step).astype(int)
    j = np.floor((col - c0) / step).astype(int)
    ok = (i >= 0) & (i < nr) & (j >= 0) & (j < nc)
    v = np.full(LON.shape, np.nan)
    v[ok] = scores[i[ok], j[ok]]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    fin = np.isfinite(v)
    if mode == "binary":
        pos = fin & (v >= threshold)
        neg = fin & (v < threshold)
        rgba[pos] = POS_RGBA
        rgba[neg] = NEG_RGBA
    else:
        rgba[fin] = _ramp(v[fin])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "bounds": [[lat0, lon0], [lat1, lon1]], "width_px": w, "height_px": h, "mode": mode}