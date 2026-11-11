from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def dominates(a, b) -> bool:
    ge = a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2]
    st = a[0] > b[0] or a[1] < b[1] or a[2] < b[2]
    return ge and st


def efficient(points: np.ndarray) -> np.ndarray:
    K = len(points)
    keep = []
    for i in range(K):
        if not any(dominates(points[j], points[i]) for j in range(K) if j != i):
            keep.append(i)
    return np.array(keep, int)


def option_set(ell, UC, UT, m: int = 8) -> np.ndarray:
    P = np.stack([ell, UC, UT], 1)
    idx = efficient(P)
    if len(idx) > m:
        order = sorted(idx, key=lambda i: (-ell[i], UC[i], UT[i]))
        idx = np.array(order[:m], int)
    return idx


def pareto_recall(true_pts: np.ndarray, chosen: Sequence[int]) -> float:
    eff = set(efficient(true_pts).tolist())
    if not eff:
        return 1.0
    return len(eff & set(int(c) for c in chosen)) / len(eff)


def hypervolume(pts: np.ndarray, ref_cost: float, ref_time: float) -> float:
    if len(pts) == 0:
        return 0.0
    P = np.array(pts, float)
    P = P[(P[:, 1] <= ref_cost) & (P[:, 2] <= ref_time)]
    if len(P) == 0:
        return 0.0
    ys = np.unique(np.concatenate([[0.0], P[:, 0]]))
    cs = np.unique(np.concatenate([P[:, 1], [ref_cost]]))
    ts = np.unique(np.concatenate([P[:, 2], [ref_time]]))
    vol = 0.0
    for a in range(len(ys) - 1):
        for b in range(len(cs) - 1):
            for c in range(len(ts) - 1):
                y, cc, tt = ys[a + 1], cs[b], ts[c]
                if np.any((P[:, 0] >= y) & (P[:, 1] <= cc) & (P[:, 2] <= tt)):
                    vol += (ys[a + 1] - ys[a]) * (cs[b + 1] - cs[b]) * (ts[c + 1] - ts[c])
    return vol


def buyer_regret(true_Y, true_C, true_T, ell, UC, UT, options, n_buyers=50, rng=None, scales=None) -> float:
    rng = rng or np.random.default_rng(0)
    if len(options) == 0:
        return np.nan
    if scales is None:
        scales = (max(np.max(true_C), 1e-12), max(np.max(true_T), 1e-12))
    cbar, tbar = scales
    W = rng.dirichlet([1, 1, 1], n_buyers)
    U_true = W[:, [0]] * true_Y[None] - W[:, [1]] * true_C[None] / cbar - W[:, [2]] * true_T[None] / tbar
    bUC = np.where(np.isfinite(UC), UC, 10 * cbar)
    bUT = np.where(np.isfinite(UT), UT, 10 * tbar)
    U_b = W[:, [0]] * ell[None] - W[:, [1]] * bUC[None] / cbar - W[:, [2]] * bUT[None] / tbar
    opts = np.asarray(options, int)
    pick = opts[np.argmax(U_b[:, opts], axis=1)]
    best = U_true.max(1)
    return float(np.mean(best - U_true[np.arange(n_buyers), pick]) * 100)


def requirement_choice(ell, UC, UT, options, target, budget, deadline) -> Optional[int]:
    ok = [i for i in options if ell[i] >= target - 1e-12 and UC[i] <= budget and UT[i] <= deadline]
    if not ok:
        return None
    return sorted(ok, key=lambda i: (-ell[i], UC[i]))[0]


def requirement_met(choice, true_Y, true_C, true_T, target, budget, deadline) -> bool:
    if choice is None:
        return False
    return bool(true_Y[choice] >= target - 1e-12 and true_C[choice] <= budget and true_T[choice] <= deadline)
