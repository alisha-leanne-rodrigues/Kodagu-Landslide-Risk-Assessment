"""Loads the saved Random Forest bundle and runs inference. The model is NEVER refitted here."""
import json
import os
import warnings
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

EXPECTED_THRESHOLD = 0.25


class ModelService:
    def __init__(self, data_dir: str):
        path = os.path.join(data_dir, "kodagu_deployment_bundle.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError("Model bundle not found: %s" % path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = joblib.load(path)
        self.model = bundle["model"]
        self.feature_names: List[str] = list(bundle["feature_names"])
        self.threshold: float = float(bundle["threshold"])
        self.bundle_meta = {k: (v if isinstance(v, (str, int, float)) else None)
                            for k, v in bundle.items() if k not in ("model", "feature_names")}
        info_path = os.path.join(data_dir, "random_forest_model_info.json")
        self.info = json.load(open(info_path)) if os.path.exists(info_path) else {}
        self._validate()

    def _validate(self):
        m = self.model
        if len(self.feature_names) != 24 or m.n_features_in_ != 24:
            raise RuntimeError("Expected 24 model features, bundle has %d" % len(self.feature_names))
        if list(getattr(m, "feature_names_in_", self.feature_names)) != self.feature_names:
            raise RuntimeError("Feature order in bundle does not match the model's own feature_names_in_")
        if self.info.get("features") and self.info["features"] != self.feature_names:
            raise RuntimeError("random_forest_model_info.json features differ from the bundle")
        if abs(self.threshold - EXPECTED_THRESHOLD) > 1e-12:
            raise RuntimeError("Threshold in bundle is %s, expected 0.25 - refusing to run" % self.threshold)

    def predict_scores(self, X: np.ndarray, chunk: int = 40000) -> np.ndarray:
        """Class-1 score (mean tree probability) for X with columns in bundle order."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X must have shape (n, %d)" % len(self.feature_names))
        if not np.isfinite(X).all():
            raise ValueError("Feature matrix contains NaN/inf - invalid pixels must be removed first")
        out = np.empty(len(X), dtype=np.float64)
        for i in range(0, len(X), chunk):
            df = pd.DataFrame(X[i:i + chunk], columns=self.feature_names)
            out[i:i + chunk] = self.model.predict_proba(df)[:, 1]
        return out

    def classify(self, scores: np.ndarray) -> np.ndarray:
        return (np.asarray(scores) >= self.threshold).astype(int)

    def describe(self) -> Dict:
        return {
            "algorithm": "Random Forest (scikit-learn RandomForestClassifier)",
            "n_estimators": int(self.model.n_estimators),
            "min_samples_leaf": int(self.model.min_samples_leaf),
            "max_depth": None if self.model.max_depth is None else int(self.model.max_depth),
            "classification_threshold": self.threshold,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "shap_version_in_bundle": self.bundle_meta.get("shap_version"),
            "score_meaning": "Mean tree probability for the landslide class. It is a model score, NOT a calibrated "
                             "real-world probability of landslide occurrence.",
        }