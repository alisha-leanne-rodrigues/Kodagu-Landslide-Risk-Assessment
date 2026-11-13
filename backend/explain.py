"""SHAP explanations for the deployed Random Forest.

Primary engine : shap.TreeExplainer (the SHAP library, tree-path-dependent TreeSHAP).
Fallback engine: a pure-Python port of the same exact TreeSHAP algorithm (Lundberg et al. 2018,
                 Algorithm 2). It is verified against brute-force Shapley values in the test-suite,
                 but it is slow (seconds per pixel), so it is only used when `shap` cannot be imported
                 and it is limited to a handful of pixels.

SHAP values are in units of the model score (mean tree probability of the landslide class):
    score(x) = base_value + sum_j phi_j(x)
They describe the model's behaviour, not causal effects in the field.
"""
import sys
from typing import Tuple

import numpy as np

sys.setrecursionlimit(10000)
FALLBACK_MAX_ROWS = 12


# ----------------------------------------------------------------------------- internal TreeSHAP
def _tree_arrays(est):
    t = est.tree_
    val = t.value[:, 0, :]
    prob1 = val[:, 1] / val.sum(axis=1)
    return (t.children_left, t.children_right, t.feature, t.threshold, prob1, t.weighted_n_node_samples)


def _expected_value(arrs) -> float:
    cl, cr, _, _, v, w = arrs
    n = len(cl)
    ev = np.zeros(n)
    for i in range(n - 1, -1, -1):  # children always have larger ids than parents in sklearn trees
        if cl[i] < 0:
            ev[i] = v[i]
        else:
            ev[i] = (ev[cl[i]] * w[cl[i]] + ev[cr[i]] * w[cr[i]]) / w[i]
    return float(ev[0])


def _extend(path, d, zf, of, fi):
    path[d] = [fi, zf, of, 1.0 if d == 0 else 0.0]
    for i in range(d - 1, -1, -1):
        path[i + 1][3] += of * path[i][3] * (i + 1) / (d + 1)
        path[i][3] = zf * path[i][3] * (d - i) / (d + 1)


def _unwind(path, d, idx):
    of, zf = path[idx][2], path[idx][1]
    nxt = path[d][3]
    for i in range(d - 1, -1, -1):
        if of != 0:
            tmp = path[i][3]
            path[i][3] = nxt * (d + 1) / ((i + 1) * of)
            nxt = tmp - path[i][3] * zf * (d - i) / (d + 1)
        else:
            path[i][3] = (path[i][3] * (d + 1)) / (zf * (d - i))
    for i in range(idx, d):
        path[i][0], path[i][1], path[i][2] = path[i + 1][0], path[i + 1][1], path[i + 1][2]


def _unwound_sum(path, d, idx):
    of, zf = path[idx][2], path[idx][1]
    nxt = path[d][3]
    total = 0.0
    for i in range(d - 1, -1, -1):
        if of != 0:
            tmp = nxt * (d + 1) / ((i + 1) * of)
            total += tmp
            nxt = path[i][3] - tmp * zf * ((d - i) / (d + 1))
        elif zf != 0:
            total += (path[i][3] / zf) / ((d - i) / (d + 1))
    return total


def _tree_shap_one(arrs, x, phi):
    cl, cr, feat, thr, val, w = arrs

    def recurse(node, d, parent_path, pzf, pof, pfi):
        path = [list(p) for p in parent_path[:d]]   # entries 0..d-1 inherited from the parent
        path.append([0, 0, 0, 0])                   # slot d for this node
        _extend(path, d, pzf, pof, pfi)
        if cl[node] < 0:
            for i in range(1, d + 1):
                wgt = _unwound_sum(path, d, i)
                el = path[i]
                phi[int(el[0])] += wgt * (el[2] - el[1]) * val[node]
            return
        s = feat[node]
        hot = cl[node] if x[s] <= thr[node] else cr[node]
        cold = cr[node] if hot == cl[node] else cl[node]
        hzf = w[hot] / w[node]
        czf = w[cold] / w[node]
        izf = iof = 1.0
        k = 0
        while k <= d:
            if int(path[k][0]) == s:
                break
            k += 1
        dd = d
        if k != d + 1:
            izf, iof = path[k][1], path[k][2]
            _unwind(path, d, k)
            dd = d - 1
        recurse(hot, dd + 1, path, hzf * izf, iof, s)
        recurse(cold, dd + 1, path, czf * izf, 0.0, s)

    recurse(0, 0, [], 1.0, 1.0, -1)


def internal_tree_shap(model, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Exact TreeSHAP for a sklearn RandomForestClassifier, class-1 probability output."""
    X = np.asarray(X, float)
    phi = np.zeros(X.shape, float)
    base = 0.0
    n_est = len(model.estimators_)
    for est in model.estimators_:
        arrs = _tree_arrays(est)
        base += _expected_value(arrs)
        for r in range(len(X)):
            p = np.zeros(X.shape[1])
            _tree_shap_one(arrs, X[r], p)
            phi[r] += p
    return phi / n_est, base / n_est


# ----------------------------------------------------------------------------- engine wrapper
class ShapExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)
        self.engine = "internal-treeshap (slow fallback)"
        self.error = None
        self._ex = None
        try:
            import shap  # type: ignore
            self._ex = shap.TreeExplainer(model)
            self.engine = "shap.TreeExplainer (shap %s)" % getattr(shap, "__version__", "?")
        except Exception as e:  # noqa
            self.error = "%s: %s" % (type(e).__name__, str(e)[:160])
        self._fallback_base = None

    @property
    def is_library(self) -> bool:
        return self._ex is not None

    @property
    def max_rows(self) -> int:
        return 10 ** 9 if self.is_library else FALLBACK_MAX_ROWS

    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (phi[n, n_features], base_value) for the landslide-class score."""
        X = np.asarray(X, float)
        if len(X) > self.max_rows:
            raise ValueError("SHAP fallback engine can explain at most %d pixels at a time; install `shap`" % self.max_rows)
        if self._ex is not None:
            import pandas as pd
            sv = self._ex.shap_values(pd.DataFrame(X, columns=self.feature_names), check_additivity=False)
            if isinstance(sv, list):          # older SHAP: list per class
                phi = np.asarray(sv[1])
            else:
                sv = np.asarray(sv)
                phi = sv[:, :, 1] if sv.ndim == 3 else sv   # newer SHAP: (n, features, classes)
            ev = np.asarray(self._ex.expected_value).ravel()
            base = float(ev[1] if ev.size > 1 else ev[0])
            return phi, base
        phi, base = internal_tree_shap(self.model, X)
        return phi, base