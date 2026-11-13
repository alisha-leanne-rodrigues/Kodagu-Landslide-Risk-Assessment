"""Spatial / covariate support indicator.

The model was trained only on labels from north / north-west Kodagu. For every pixel we report how well
the TRAINING DATA covers it:

  * geographic distance (km) to the nearest training landslide pixel
  * number of continuous features outside the range seen in the training table

Levels (transparent heuristics, not statistical confidence):
  well_supported : <= 3 km from a training landslide pixel and no feature outside the training range
                   (3 km = the radius within which training negatives were sampled)
  partial        : <= 10 km and at most one feature outside the training range
  extrapolation  : otherwise
"""
import json
import os
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree

from .feature_meta import BINARY_FEATURES

LEVELS = ("well_supported", "partial", "extrapolation")
WELL_KM = 3.0
PARTIAL_KM = 10.0
PIXEL_KM = 0.03


class SupportModel:
    def __init__(self, data_dir: str, feature_names: List[str]):
        self.feature_names = feature_names
        pos = np.load(os.path.join(data_dir, "training_positive_rc.npy")).astype(float)
        self.tree = cKDTree(pos)
        ref = json.load(open(os.path.join(data_dir, "training_reference.json")))
        self.ref = ref
        self.cont_idx = [i for i, f in enumerate(feature_names) if f not in BINARY_FEATURES]
        self.lo = np.array([ref["feature_stats"][feature_names[i]]["min"] for i in self.cont_idx])
        self.hi = np.array([ref["feature_stats"][feature_names[i]]["max"] for i in self.cont_idx])
        self.p1 = {f: ref["feature_stats"][f]["p1"] for f in feature_names}
        self.p99 = {f: ref["feature_stats"][f]["p99"] for f in feature_names}
        self.median = {f: ref["feature_stats"][f]["median"] for f in feature_names}
        self.n_train, self.n_train_pos = ref["n_rows"], ref["n_positive"]

    def distance_km(self, rows, cols) -> np.ndarray:
        pts = np.column_stack([np.asarray(rows, float), np.asarray(cols, float)])
        d, _ = self.tree.query(pts)
        return d * PIXEL_KM

    def out_of_range(self, X: np.ndarray) -> np.ndarray:
        """Boolean (n, n_cont) - True where the feature lies outside the training [min, max]."""
        Xc = X[:, self.cont_idx]
        return (Xc < self.lo) | (Xc > self.hi)

    def assess(self, rows, cols, X: np.ndarray) -> Dict[str, np.ndarray]:
        dist = self.distance_km(rows, cols)
        oor = self.out_of_range(X)
        n_out = oor.sum(axis=1)
        level = np.full(len(dist), 2, dtype=np.uint8)
        level[(dist <= PARTIAL_KM) & (n_out <= 1)] = 1
        level[(dist <= WELL_KM) & (n_out == 0)] = 0
        return {"distance_km": dist, "n_out_of_range": n_out, "level": level, "oor": oor}

    def out_of_range_names(self, oor_row: np.ndarray) -> List[str]:
        return [self.feature_names[self.cont_idx[i]] for i in np.where(oor_row)[0]]

    def describe(self) -> Dict:
        return {"well_supported_km": WELL_KM, "partial_km": PARTIAL_KM,
                "training_rows": self.n_train, "training_positive_pixels": self.n_train_pos,
                "training_extent_rows": self.ref["extent_rows"], "training_extent_cols": self.ref["extent_cols"],
                "definition": "Distance to the nearest training landslide pixel and number of continuous features outside "
                              "the training range. Heuristic indicator of evidential support, not a confidence interval."}