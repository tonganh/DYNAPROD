from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def draw_error(M: int, K: int, delta: float = 0.05) -> float:
    return math.sqrt(math.log(6.0 * K / delta) / (2.0 * M))


def draws_needed(eps: float, K: int, delta: float = 0.05) -> int:
    return int(math.ceil(math.log(6.0 * K / delta) / (2.0 * eps * eps)))


def coupling_bound(etas: Iterable[float]) -> float:
    prod = 1.0
    for e in etas:
        prod *= 1.0 - min(max(float(e), 0.0), 1.0)
    return 1.0 - prod


def model_epsilon(etas: Iterable[float], M: int, K: int, delta: float = 0.05) -> float:
    return coupling_bound(etas) + draw_error(M, K, delta)


def min_calibration(alpha: float) -> int:
    return max(0, int(math.ceil(1.0 / alpha - 1e-12)) - 1)


def conformal_k(alpha: float, n: int) -> int:
    return int(math.floor(alpha * (n + 1)))


def lemma1_miss_bound(alpha: float, n: int) -> float:
    return conformal_k(alpha, n) / (n + 1)


def sharpness_floor(alpha: float, n: int, eps: float = 0.0) -> float:
    return alpha - 1.0 / (n + 1) - 2.0 * eps


def thm5_miss_bound(alpha: float, eps: float) -> float:
    return alpha + 2.0 * eps


def sidak_level(alpha: float, K: int) -> float:
    return 1.0 - (1.0 - alpha) ** (1.0 / max(K, 1))


def union_bound(alpha_Y: float, alpha_C: float, alpha_T: float) -> float:
    return alpha_Y + alpha_C + alpha_T


def slack(Y, ell, UC, C, UT, T) -> np.ndarray:
    return np.stack([np.asarray(Y) - np.asarray(ell), np.asarray(UC) - np.asarray(C),
                     np.asarray(UT) - np.asarray(T)], 1)


def regret_bound(weights: Sequence[float], delta_best: Sequence[float]) -> float:
    return float(np.dot(np.asarray(weights, float), np.asarray(delta_best, float)))


def online_miss_upper(alpha: float, gamma: float, T: int, d: int = 0) -> float:
    return alpha + (alpha + gamma * (d + 1)) / (gamma * T) + d / T


def online_miss_lower(alpha: float, gamma: float, T: int, A: float, d: int = 0) -> float:
    return alpha - (A - alpha) / (gamma * T) - d / T


def hoeffding_halfwidth(n_units: int, delta: float = 0.05) -> float:
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n_units))
