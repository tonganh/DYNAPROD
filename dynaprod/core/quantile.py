from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import lightgbm as lgb

from ..config import LEVELS


class QuantileGBM:

    def __init__(self, levels=LEVELS, n_estimators=120, num_leaves=15, learning_rate=0.08,
                 min_child_samples=20, max_depth=-1, clip=None, seed=0):
        self.levels = np.asarray(levels, float)
        self.params = dict(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate,
                           min_child_samples=min_child_samples, max_depth=max_depth, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.9, verbose=-1, random_state=seed, n_jobs=1)
        self.clip = clip
        self.models = []
        self.const = None

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if len(y) < 30:
            self.const = np.quantile(y, self.levels) if len(y) else np.zeros(len(self.levels))
            return self
        self.models = []
        for a in self.levels:
            m = lgb.LGBMRegressor(objective="quantile", alpha=float(a), **self.params)
            m.fit(X, y, sample_weight=sample_weight)
            self.models.append(m)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        if self.const is not None:
            Q = np.tile(self.const, (len(X), 1))
        else:
            Q = np.stack([m.predict(X) for m in self.models], axis=1)
        Q = np.sort(Q, axis=1)
        if self.clip is not None:
            Q = np.clip(Q, *self.clip)
        return Q


def interp_quantile(Q: np.ndarray, levels: np.ndarray, v: np.ndarray, lo=None, hi=None) -> np.ndarray:
    L = len(levels)
    d_lo = (Q[:, 1] - Q[:, 0]) / (levels[1] - levels[0]) * levels[0]
    d_hi = (Q[:, -1] - Q[:, -2]) / (levels[-1] - levels[-2]) * (1 - levels[-1])
    q0 = Q[:, 0] - d_lo
    q1 = Q[:, -1] + d_hi
    if lo is not None:
        q0 = np.maximum(q0, lo)
    if hi is not None:
        q1 = np.minimum(q1, hi)
    lv = np.concatenate([[0.0], levels, [1.0]])
    QQ = np.concatenate([q0[:, None], Q, q1[:, None]], axis=1)
    v = np.asarray(v, float)
    idx = np.clip(np.searchsorted(lv, v, side="right") - 1, 0, L)
    lo_l, hi_l = lv[idx], lv[idx + 1]
    w = (v - lo_l) / np.maximum(hi_l - lo_l, 1e-12)
    n_idx = np.broadcast_to(np.arange(Q.shape[0]), v.shape)
    out = QQ[n_idx, idx] * (1 - w) + QQ[n_idx, idx + 1] * w
    if lo is not None or hi is not None:
        out = np.clip(out, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
    return out


def pit(Q: np.ndarray, levels: np.ndarray, y: np.ndarray, rng, lo=None, hi=None, grid=201) -> np.ndarray:
    vs = np.linspace(0, 1, grid)
    V = np.tile(vs, (len(y), 1)).T
    inv = interp_quantile(Q, levels, V, lo, hi)
    below = (inv < y[None, :]).mean(0)
    at = (np.abs(inv - y[None, :]) < 1e-9).mean(0)
    return np.clip(below + rng.random(len(y)) * at, 0, 1)


class Recalibrator:

    def __init__(self):
        self.pits = None

    def fit(self, pits: np.ndarray):
        self.pits = np.sort(np.asarray(pits, float))
        return self

    def __call__(self, v: np.ndarray) -> np.ndarray:
        if self.pits is None or len(self.pits) < 20:
            return v
        return np.quantile(self.pits, np.clip(v, 0, 1))
