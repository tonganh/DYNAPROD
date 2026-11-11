from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def anyof_frechet(a: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, float)
    if a.size == 0:
        return 0.0, 0.0
    return float(a.max()), float(min(1.0, a.sum()))


def allof_frechet(a: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, float)
    if a.size == 0:
        return 0.0, 0.0
    return float(max(0.0, a.sum() - a.size + 1)), float(a.min())


def atleast_interval(lo: np.ndarray, hi: np.ndarray, k: int) -> Tuple[float, float]:
    n = len(lo)
    l = (np.sum(lo) - k + 1) / max(1, n - k + 1)
    h = np.sum(hi) / k
    return float(np.clip(l, 0, 1)), float(np.clip(h, 0, 1))


def anyof_with_overlaps(a: np.ndarray, overlaps: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, float)
    n = a.size
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(a[0]), float(a[0])
    O = np.array(overlaps, float)
    np.fill_diagonal(O, 0.0)
    W = np.triu(O.max() + 1.0 - O, 1) * (np.triu(np.ones_like(O), 1) > 0)
    mst = minimum_spanning_tree(W).toarray()
    tree = mst > 0
    upper = a.sum() - O[np.triu(tree | tree.T, 1)].sum()
    S1 = a.sum()
    S2 = np.triu(O, 1).sum()
    lower = 0.0
    if S1 > 0:
        j = 1 + int(np.floor(2 * S2 / S1))
        lower = 2 * S1 / (j + 1) - 2 * S2 / (j * (j + 1))
    f_lo, f_hi = anyof_frechet(a)
    return float(np.clip(max(lower, f_lo), 0, 1)), float(np.clip(min(upper, f_hi), 0, 1))


def thm2_eligible(node) -> bool:
    if node is None:
        return False
    if node[0] == "ev":
        return True
    return node[0] in ("any", "all") and all(thm2_eligible(c) for c in node[1])


def circuit_interval(node, leaf_share: Dict[tuple, float],
                     overlap: Optional[Callable[[tuple, tuple], float]] = None) -> Tuple[float, float]:
    if node is None:
        return 0.0, 0.0
    k = node[0]
    if k == "ev":
        s = leaf_share.get(node, 0.0)
        return s, s
    if k == "any" and overlap is not None and all(c[0] == "ev" for c in node[1]):
        ch = list(node[1])
        O = np.array([[overlap(a, b) for b in ch] for a in ch], float)
        return anyof_with_overlaps(np.array([leaf_share.get(c, 0.0) for c in ch]), O)
    if k in ("any", "all", "atl"):
        ch = node[1] if k != "atl" else node[2]
        iv = np.array([circuit_interval(c, leaf_share, overlap) for c in ch])
        lo, hi = iv[:, 0], iv[:, 1]
        if k == "any":
            return float(lo.max()), float(min(1.0, hi.sum()))
        if k == "all":
            return float(max(0.0, lo.sum() - len(lo) + 1)), float(hi.min())
        return atleast_interval(lo, hi, node[1])
    if k == "fill":
        lo, _ = circuit_interval(node[2], leaf_share, overlap)
        return lo, 1.0
    if k == "pool":
        lo, hi = circuit_interval(node[2], leaf_share, overlap)
        return lo, 1.0 if hi > 0 else 0.0
    raise ValueError(k)


def independent_anyof(p: np.ndarray) -> float:
    return float(1.0 - np.prod(1.0 - np.asarray(p, float)))


def independent_allof(p: np.ndarray) -> float:
    return float(np.prod(np.asarray(p, float)))


def poisson_binomial_tail(p: np.ndarray, k: int) -> float:
    dp = np.zeros(len(p) + 1)
    dp[0] = 1.0
    for pi in p:
        dp[1:] = dp[1:] * (1 - pi) + dp[:-1] * pi
        dp[0] *= (1 - pi)
    return float(dp[k:].sum())


def independent_circuit(node, prob: Dict[tuple, np.ndarray]) -> np.ndarray:
    k = node[0]
    if k == "ev":
        return prob[node]
    if k == "any":
        q = np.ones_like(next(iter(prob.values())), dtype=float)
        for c in node[1]:
            q = q * (1 - independent_circuit(c, prob))
        return 1 - q
    if k == "all":
        q = np.ones_like(next(iter(prob.values())), dtype=float)
        for c in node[1]:
            q = q * independent_circuit(c, prob)
        return q
    if k == "atl":
        ps = np.stack([independent_circuit(c, prob) for c in node[2]], axis=0)
        n, U = ps.shape
        dp = np.zeros((n + 1, U))
        dp[0] = 1
        for i in range(n):
            dp[1:] = dp[1:] * (1 - ps[i]) + dp[:-1] * ps[i]
            dp[0] *= (1 - ps[i])
        return dp[node[1]:].sum(0)
    if k == "fill":
        base = independent_circuit(node[2], prob)
        acc = prob.get(("acc", node[1]), 0.5)
        return 1 - (1 - base) * (1 - acc)
    if k == "pool":
        base = independent_circuit(node[2], prob)
        cmap, n_coarse = prob[("pool", node[1])]
        out = np.ones(n_coarse)
        valid = cmap >= 0
        np.multiply.at(out, cmap[valid], 1 - base[valid])
        return 1 - out
    raise ValueError(k)
