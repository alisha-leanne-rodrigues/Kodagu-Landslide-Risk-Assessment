"""Lossless compact storage for raster layers (byte-shuffle + zlib inside .npz).

Smooth float32 rasters compress several times better after byte-shuffling, and the values
are bit-identical after loading (no precision change)."""
import numpy as np


def save_layer(path: str, arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr)
    if a.dtype.itemsize == 1:
        planes = a.reshape(1, -1).view(np.uint8)
    else:
        planes = a.view(np.uint8).reshape(-1, a.dtype.itemsize).T.copy()
    np.savez_compressed(path, planes=planes, shape=np.array(a.shape), dtype=np.array(str(a.dtype)))


def load_layer(path: str) -> np.ndarray:
    z = np.load(path)
    dtype = np.dtype(str(z["dtype"]))
    shape = tuple(int(s) for s in z["shape"])
    planes = z["planes"]
    if dtype.itemsize == 1:
        return planes.reshape(shape).view(dtype).copy()
    flat = planes.T.copy().view(dtype).reshape(shape)
    return flat