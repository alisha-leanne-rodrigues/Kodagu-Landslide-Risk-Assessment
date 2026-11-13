"""Coordinate maths (no GIS libraries needed).

* EPSG:7755 (WGS 84 / India NSF LCC) forward + inverse, the CRS of the training grid.
* Interrupted Goode Homolosine (SoilGrids native CRS), sinusoidal zone, central meridian 30E.
* Pixel <-> coordinate helpers for the project's 30 m grid.
* Polygon rasterisation (Pillow) and a WKB reader for the boundary GeoPackage.
Both projections were verified against raster extents / the training CSV (see tests).
"""
import sqlite3
import struct
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- EPSG:7755
_A = 6378137.0
_F = 1 / 298.257223563
_E = np.sqrt(2 * _F - _F * _F)
_PHI1, _PHI2 = np.radians(12.472955), np.radians(35.172806)
_PHI0, _LAM0 = np.radians(24.0), np.radians(80.0)
_FE = _FN = 4000000.0


def _m(p):
    return np.cos(p) / np.sqrt(1 - _E ** 2 * np.sin(p) ** 2)


def _t(p):
    return np.tan(np.pi / 4 - p / 2) / ((1 - _E * np.sin(p)) / (1 + _E * np.sin(p))) ** (_E / 2)


_N = (np.log(_m(_PHI1)) - np.log(_m(_PHI2))) / (np.log(_t(_PHI1)) - np.log(_t(_PHI2)))
_FC = _m(_PHI1) / (_N * _t(_PHI1) ** _N)
_R0 = _A * _FC * _t(_PHI0) ** _N

# Pixel grid of DEM_StudyArea_7755.tif (top-left corner of pixel (0,0)), 30 m
GRID_X0 = 3430068.978841963
GRID_Y0 = 2908526.76881167
RES = 30.0


def lcc_forward(lon, lat):
    lon = np.radians(np.asarray(lon, float))
    lat = np.radians(np.asarray(lat, float))
    r = _A * _FC * _t(lat) ** _N
    th = _N * (lon - _LAM0)
    return _FE + r * np.sin(th), _FN + _R0 - r * np.cos(th)


def lcc_inverse(x, y):
    xp = np.asarray(x, float) - _FE
    yp = _R0 - (np.asarray(y, float) - _FN)
    r = np.sign(_N) * np.hypot(xp, yp)
    tp = (r / (_A * _FC)) ** (1 / _N)
    th = np.arctan2(xp, yp)
    lon = th / _N + _LAM0
    phi = np.pi / 2 - 2 * np.arctan(tp)
    for _ in range(8):
        phi = np.pi / 2 - 2 * np.arctan(tp * ((1 - _E * np.sin(phi)) / (1 + _E * np.sin(phi))) ** (_E / 2))
    return np.degrees(lon), np.degrees(phi)


def lonlat_to_rowcol(lon, lat):
    """Float (row, col) on the 30 m grid (pixel-edge origin). floor() gives the pixel index."""
    x, y = lcc_forward(lon, lat)
    return (GRID_Y0 - y) / RES, (x - GRID_X0) / RES


def rowcol_to_lonlat(row, col):
    """Lon/lat of pixel CENTRES for integer row/col."""
    x = GRID_X0 + (np.asarray(col, float) + 0.5) * RES
    y = GRID_Y0 - (np.asarray(row, float) + 0.5) * RES
    return lcc_inverse(x, y)


# ------------------------------------------- Interrupted Goode Homolosine
_R_IGH = 6378137.0


def igh_forward_north_lobe2(lon, lat):
    """IGH sinusoidal zone (|lat| < 40.7), lobe with central meridian 30E (covers India)."""
    lam = np.radians(np.asarray(lon, float))
    phi = np.radians(np.asarray(lat, float))
    l0 = np.radians(30.0)
    return _R_IGH * (lam - l0) * np.cos(phi) + _R_IGH * l0, _R_IGH * phi


# ------------------------------------------------------------ rasterising
def rasterize_rings(rings_lonlat: Sequence, r0: int, r1: int, c0: int, c1: int,
                    holes_lonlat: Sequence = ()) -> np.ndarray:
    """Boolean mask (r1-r0, c1-c0) of grid pixels whose CENTRE lies inside the polygon rings.
    Rings are (N,2) lon/lat arrays; holes are cut out."""
    h, w = r1 - r0, c1 - c0
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for ring, fill in [(r, 1) for r in rings_lonlat] + [(r, 0) for r in holes_lonlat]:
        ring = np.asarray(ring, float)
        rr, cc = lonlat_to_rowcol(ring[:, 0], ring[:, 1])
        # PIL treats integer coordinates as pixel centres; ours are edge-origin floats -> shift by -0.5
        d.polygon(list(zip((cc - c0 - 0.5).tolist(), (rr - r0 - 0.5).tolist())), fill=fill)
    return np.array(img, dtype=bool)


def polygon_area_m2(ring_lonlat) -> float:
    ring = np.asarray(ring_lonlat, float)
    x, y = lcc_forward(ring[:, 0], ring[:, 1])
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def ring_is_valid(ring) -> bool:
    try:
        ring = np.asarray(ring, float)
    except (TypeError, ValueError):
        return False
    return (ring.ndim == 2 and ring.shape[1] == 2 and len(ring) >= 3 and bool(np.isfinite(ring).all())
            and polygon_area_m2(ring) > 0)


# --------------------------------------------------------- GeoPackage WKB
def read_gpkg_polygons(path: str, table: str = None):
    """Return list of polygons; each polygon = list of rings (first = outer), lon/lat arrays."""
    con = sqlite3.connect(path)
    tn = table or con.execute("select table_name from gpkg_contents where data_type='features'").fetchone()[0]
    gc = con.execute("select column_name from gpkg_geometry_columns where table_name=?", (tn,)).fetchone()[0]
    blob = con.execute(f'select "{gc}" from "{tn}"').fetchone()[0]
    con.close()
    flags = blob[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    wkb = blob[8 + env:]
    bo = "<" if wkb[0] == 1 else ">"

    def read_poly(pos):
        nr = struct.unpack(bo + "I", wkb[pos:pos + 4])[0]
        pos += 4
        rings = []
        for _ in range(nr):
            n = struct.unpack(bo + "I", wkb[pos:pos + 4])[0]
            pos += 4
            rings.append(np.frombuffer(wkb[pos:pos + 16 * n], dtype=bo + "f8").reshape(-1, 2).copy())
            pos += 16 * n
        return rings, pos

    t = struct.unpack(bo + "I", wkb[1:5])[0]
    if t == 3:
        rings, _ = read_poly(5)
        return [rings]
    if t == 6:
        n = struct.unpack(bo + "I", wkb[5:9])[0]
        pos, polys = 9, []
        for _ in range(n):
            pos += 5
            rings, pos = read_poly(pos)
            polys.append(rings)
        return polys
    raise ValueError("unsupported geometry type %d" % t)