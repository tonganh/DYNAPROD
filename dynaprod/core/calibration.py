from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .strata import Stratification, family_size

from ..config import ALPHA_C, ALPHA_T, ALPHA_Y, GAMMA, GRID, MIN_STRATUM, TAIL


def F_left(samples: np.ndarray, g) -> np.ndarray:
    return (samples < g).mean(axis=-1)


def coverage_score(samples: np.ndarray, y: float, grid: np.ndarray, u: float) -> float:
    pts = np.concatenate([[0.0], grid, [np.inf]])
    i = int(np.searchsorted(pts, y, side="right") - 1)
    i = min(max(i, 0), len(pts) - 2)
    g_lo, g_hi = pts[i], pts[i + 1]
    F_lo = 0.0 if g_lo == 0.0 else float(F_left(samples, g_lo))
    F_hi = 1.0 if np.isinf(g_hi) else float(F_left(samples, g_hi))
    return F_lo + u * (F_hi - F_lo)


def coverage_bound_from_threshold(samples: np.ndarray, grid: np.ndarray, thr: float) -> float:
    Fg = np.array([F_left(samples, g) for g in grid])
    ok = np.where(Fg < thr)[0]
    return float(grid[ok.max()]) if len(ok) else 0.0


def set_tail_scales(beta_C: float, beta_T: float):
    TAIL["C"], TAIL["T"] = float(max(beta_C, 1e-3)), float(max(beta_T, 1e-3))


def _tail_cdf(samples: np.ndarray, c: float, beta: float, left: bool) -> float:
    s = np.asarray(samples, float)
    M = len(s)
    smax = float(s.max())
    if c <= smax or (left and c <= smax):
        cnt = float((s < c).sum()) if left else float((s <= c).sum())
        return cnt / (M + 1)
    z = math.log(max(c, 1e-300)) - math.log(max(smax, 1e-300))
    return M / (M + 1) + (1.0 - math.exp(-z / beta)) / (M + 1)


def upper_score(samples: np.ndarray, c: float, u: float, beta: Optional[float] = None) -> float:
    beta = TAIL["C"] if beta is None else beta
    lo = _tail_cdf(samples, c, beta, left=True)
    hi = _tail_cdf(samples, c, beta, left=False)
    return lo + u * (hi - lo)


def upper_bound_from_threshold(samples: np.ndarray, thr: float, beta: Optional[float] = None) -> float:
    beta = TAIL["C"] if beta is None else beta
    if thr >= 1.0:
        return float("inf")
    s = np.sort(np.asarray(samples, float))
    M = len(s)
    i = int(math.floor(thr * (M + 1)))
    if i < M:
        return float(s[max(i, 0)])
    p = thr * (M + 1) - M
    return float(s[-1] * math.exp(-beta * math.log(max(1.0 - p, 1e-300))))


def k_lower(alpha: float, n: int) -> int:
    return int(math.floor(alpha * (n + 1)))


def j_upper(alpha: float, n: int) -> int:
    return int(math.ceil((1 - alpha) * (n + 1)))


@dataclass
class Bounds:
    ell: np.ndarray
    UC: np.ndarray
    UT: np.ndarray
    risk: np.ndarray
    ell_raw: Optional[np.ndarray] = None


class Calibrator:

    def __init__(self, grid=GRID, alpha_Y=ALPHA_Y, alpha_C=ALPHA_C, alpha_T=ALPHA_T, min_stratum=MIN_STRATUM,
                 weights=None, raw=False, strata: Optional[Stratification] = None):
        self.grid = np.asarray(grid, float)
        self.aY, self.aC, self.aT = alpha_Y, alpha_C, alpha_T
        self.min_stratum = min_stratum
        self.raw = raw
        a = alpha_Y + alpha_C + alpha_T
        self.w = np.array(weights if weights is not None else [alpha_Y / a, alpha_C / a, alpha_T / a])
        self.strata = strata if strata is not None else family_size(min_stratum)
        self.scores: Dict[tuple, Dict[str, np.ndarray]] = {}

    def _records_by_stratum(self, entries: List[dict], make):
        self.strata.fit(entries)
        by = defaultdict(list)
        for e in entries:
            key = self.strata.key(e)
            rec = make(e)
            if key is not None:
                by[key].append(rec)
        return by

    def stratum(self, family, size, group=None):
        return self.strata.key(dict(family=family, size=size, group=group if group is not None else "*"))

    def n(self, family, size, group=None) -> int:
        k = self.stratum(family, size, group)
        return 0 if k is None or k not in self.scores else len(self.scores[k]["Y"])

    def coverage_scores(self, family, size, group=None) -> np.ndarray:
        k = self.stratum(family, size, group)
        return self.scores[k]["Y"] if k is not None and k in self.scores else np.zeros(0)

    def _keys(self, family, size, K, groups):
        if groups is None:
            k = self.stratum(family, size)
            return [k] * K
        return [self.stratum(family, size, g) for g in groups]

    def fit(self, entries: List[dict], rng: np.random.Generator):
        U = {id(e): rng.random(3) for e in entries}

        def make(e):
            u = U[id(e)]
            sY = coverage_score(e["Y"], e["y"], self.grid, u[0])
            sC = upper_score(e["C"], e["c"], u[1], TAIL["C"])
            sT = upper_score(e["T"], e["t"], u[2], TAIL["T"])
            r = min(sY / self.w[0], (1 - sC) / self.w[1], (1 - sT) / self.w[2])
            return (sY, sC, sT, r)

        by = self._records_by_stratum(entries, make)
        self.scores = {k: {n: np.sort(np.array([v[i] for v in vals])) for i, n in enumerate(["Y", "C", "T", "J"])}
                       for k, vals in by.items()}
        return self

    def bounds(self, family, size, Y: np.ndarray, C: np.ndarray, T: np.ndarray,
               alpha_Y=None, alpha_C=None, alpha_T=None, groups=None) -> Bounds:
        aY = self.aY if alpha_Y is None else alpha_Y
        aC = self.aC if alpha_C is None else alpha_C
        aT = self.aT if alpha_T is None else alpha_T
        K = Y.shape[0]
        ell, UC, UT = np.zeros(K), np.full(K, np.inf), np.full(K, np.inf)
        risk = np.ones((K, len(self.grid)))
        if self.raw:
            for i in range(K):
                ell[i] = coverage_bound_from_threshold(Y[i], self.grid, aY)
                UC[i] = upper_bound_from_threshold(C[i], 1 - aC, TAIL["C"])
                UT[i] = upper_bound_from_threshold(T[i], 1 - aT, TAIL["T"])
            return Bounds(ell, UC, UT, risk, ell.copy())
        keys = self._keys(family, size, K, groups)
        for i in range(K):
            key = keys[i]
            if key is None or key not in self.scores:
                continue
            S = self.scores[key]
            n = len(S["Y"])
            k = k_lower(aY, n) if aY > 0 else 0
            jc, jt = j_upper(aC, n), j_upper(aT, n)
            if aY > 0 and k > n:
                ell[i] = float(self.grid[-1])
            elif k >= 1:
                ell[i] = coverage_bound_from_threshold(Y[i], self.grid, S["Y"][k - 1])
            if 1 <= jc <= n:
                UC[i] = upper_bound_from_threshold(C[i], S["C"][jc - 1], TAIL["C"])
            if 1 <= jt <= n:
                UT[i] = upper_bound_from_threshold(T[i], S["T"][jt - 1], TAIL["T"])
            Fg = np.array([F_left(Y[i], g) for g in self.grid])
            risk[i] = (1 + np.searchsorted(S["Y"], Fg, side="right")) / (n + 1)
        return Bounds(ell, UC, UT, risk, ell.copy())

    def joint_bounds(self, family, size, Y, C, T, alpha: float, groups=None) -> Bounds:
        K = Y.shape[0]
        ell, UC, UT = np.zeros(K), np.full(K, np.inf), np.full(K, np.inf)
        keys = self._keys(family, size, K, groups)
        for i in range(K):
            if keys[i] is None or keys[i] not in self.scores:
                continue
            S = self.scores[keys[i]]["J"]
            k = k_lower(alpha, len(S))
            if 1 <= k <= len(S):
                tau = S[k - 1]
                ell[i] = coverage_bound_from_threshold(Y[i], self.grid, tau * self.w[0])
                UC[i] = upper_bound_from_threshold(C[i], 1 - tau * self.w[1], TAIL["C"])
                UT[i] = upper_bound_from_threshold(T[i], 1 - tau * self.w[2], TAIL["T"])
        return Bounds(ell, UC, UT, np.ones((K, len(self.grid))))


def coherent(ell: np.ndarray, closure: np.ndarray) -> np.ndarray:
    out = ell.copy()
    for i in range(len(ell)):
        js = np.where(closure[i])[0]
        if len(js):
            out[i] = min(out[i], ell[js].min())
    return out


def count_incoherent(ell: np.ndarray, closure: np.ndarray) -> int:
    I, J = np.where(closure)
    return int(np.sum(ell[I] > ell[J] + 1e-12))


class MenuCalibrator:

    def __init__(self, grid=GRID, alpha=ALPHA_Y):
        self.grid = np.asarray(grid, float)
        self.alpha = alpha
        self.L = None

    def fit(self, entries: List[dict], rng):
        Ls = []
        for e in entries:
            u = rng.random()
            s = [coverage_score(e["Yplus"][k], e["y"][k], self.grid, u) for k in range(len(e["y"]))]
            Ls.append(min(s))
        self.L = np.sort(np.array(Ls))
        return self

    def bounds(self, Yplus: np.ndarray) -> np.ndarray:
        n = len(self.L)
        k = k_lower(self.alpha, n)
        if k < 1:
            return np.zeros(Yplus.shape[0])
        r = self.L[k - 1]
        return np.array([coverage_bound_from_threshold(Yplus[i], self.grid, r) for i in range(Yplus.shape[0])])


class OnlineLevel:

    def __init__(self, alpha=ALPHA_Y, gamma=GAMMA):
        self.alpha, self.gamma = alpha, gamma
        self.a = alpha
        self.history = []

    def level(self) -> float:
        return self.a

    def update(self, err: int):
        self.a = self.a + self.gamma * (self.alpha - err)
        self.history.append((self.a, err))

    @staticmethod
    def bound(samples: np.ndarray, scores_sorted: np.ndarray, a_t: float, grid=GRID) -> float:
        n = len(scores_sorted)
        if a_t <= 0:
            return 0.0
        k = int(math.floor(a_t * (n + 1)))
        if k > n:
            return float(grid[-1])
        if k < 1:
            return 0.0
        return coverage_bound_from_threshold(samples, np.asarray(grid), scores_sorted[k - 1])


def floor_grid(v: float, grid) -> float:
    ok = [g for g in grid if g <= v + 1e-12]
    return float(max(ok)) if ok else 0.0


class CQRCalibrator(Calibrator):

    LEVELS = np.round(np.arange(0.01, 1.0, 0.01), 2)

    def fit(self, entries, rng):
        super().fit(entries, rng)
        by = self._records_by_stratum(entries, lambda e: e)
        self.cqr = {}
        for key, es in by.items():
            Ys = np.stack([np.asarray(e["Y"], float) for e in es])
            y = np.array([e["y"] for e in es], float)
            self.cqr[key] = dict(Ys=Ys, y=y, Cs=np.stack([np.asarray(e["C"], float) for e in es]),
                                 c=np.array([e["c"] for e in es], float),
                                 Ts=np.stack([np.asarray(e["T"], float) for e in es]),
                                 t=np.array([e["t"] for e in es], float),
                                 EY=np.sort(np.quantile(Ys, self.LEVELS, axis=1) - y[None, :], axis=1))
        return self

    @staticmethod
    def _upper(samples, truth, n_alpha, q_alpha):
        scores = np.sort(truth / np.maximum(np.quantile(samples, 1 - n_alpha, axis=1), 1e-12))
        j = j_upper(n_alpha, len(scores))
        return q_alpha * scores[j - 1] if 1 <= j <= len(scores) else np.inf

    def _lower(self, S, Yi, alpha):
        if alpha <= 0:
            return 0.0
        n = len(S["y"])
        scores = np.sort(np.quantile(S["Ys"], alpha, axis=1) - S["y"])
        j = max(1, j_upper(alpha, n))
        if j > n:
            return 0.0
        return floor_grid(float(np.quantile(Yi, alpha)) - scores[j - 1], self.grid)

    def _risk(self, S, Yi):
        n = len(S["y"])
        js = np.array([j_upper(a, n) for a in self.LEVELS])
        q = np.quantile(Yi, self.LEVELS)
        ok = js <= n
        ell = np.zeros(len(self.LEVELS))
        ell[ok] = q[ok] - S["EY"][np.flatnonzero(ok), js[ok] - 1]
        ell = np.minimum.accumulate(ell[::-1])[::-1]
        risk = np.ones(len(self.grid))
        for gi, g in enumerate(self.grid):
            reach = np.flatnonzero(ok & (ell >= g - 1e-12))
            if len(reach):
                risk[gi] = self.LEVELS[reach[0]]
        return risk

    def bounds(self, family, size, Y, C, T, alpha_Y=None, alpha_C=None, alpha_T=None, groups=None):
        aY = self.aY if alpha_Y is None else alpha_Y
        aC = self.aC if alpha_C is None else alpha_C
        aT = self.aT if alpha_T is None else alpha_T
        K = Y.shape[0]
        ell, UC, UT = np.zeros(K), np.full(K, np.inf), np.full(K, np.inf)
        risk = np.ones((K, len(self.grid)))
        keys = self._keys(family, size, K, groups)
        for i in range(K):
            if keys[i] is None or keys[i] not in self.cqr:
                continue
            S = self.cqr[keys[i]]
            ell[i] = self._lower(S, Y[i], aY)
            UC[i] = self._upper(S["Cs"], S["c"], aC, float(np.quantile(C[i], 1 - aC)))
            UT[i] = self._upper(S["Ts"], S["t"], aT, float(np.quantile(T[i], 1 - aT)))
            risk[i] = self._risk(S, Y[i])
        return Bounds(ell, UC, UT, risk, ell.copy())
