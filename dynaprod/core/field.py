from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class FieldSampler:

    def __init__(self, coords: np.ndarray, lam: float, grid=None, spacing=1.0):
        self.coords = np.asarray(coords, float)
        self.n = len(self.coords)
        self.lam = lam
        self.grid = grid
        if lam <= 0:
            self.mode = "iid"
        elif grid is not None:
            self.mode = "fft"
            H, W = grid
            P, Q = 2 * H, 2 * W
            dy = np.minimum(np.arange(P), P - np.arange(P))[:, None] * spacing
            dx = np.minimum(np.arange(Q), Q - np.arange(Q))[None, :] * spacing
            k = np.exp(-np.sqrt(dx ** 2 + dy ** 2) / lam)
            eig = np.real(np.fft.fft2(k))
            self.sqrt_eig = np.sqrt(np.clip(eig, 0, None))
            self.H, self.W, self.P, self.Q = H, W, P, Q
        else:
            self.mode = "chol"
            D = cdist(self.coords, self.coords)
            K = np.exp(-D / lam) + 1e-6 * np.eye(self.n)
            try:
                self.L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(K)
                self.L = V * np.sqrt(np.clip(w, 0, None))

    def sample(self, M: int, rng: np.random.Generator) -> np.ndarray:
        if self.mode == "iid":
            return rng.standard_normal((M, self.n))
        if self.mode == "fft":
            w = rng.standard_normal((M, self.P, self.Q))
            f = np.real(np.fft.ifft2(self.sqrt_eig[None] * np.fft.fft2(w)))
            return f[:, : self.H, : self.W].reshape(M, -1)
        return rng.standard_normal((M, self.n)) @ self.L.T


def copula_normals(D: np.ndarray, lam: float, M: int, rng: np.random.Generator) -> np.ndarray:
    n = D.shape[0]
    if lam <= 0 or n == 1:
        return rng.standard_normal((M, n))
    R = np.exp(-D / lam)
    R = 0.5 * (R + R.T) + 1e-6 * np.eye(n)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(R)
        L = V * np.sqrt(np.clip(w, 0, None))
    return rng.standard_normal((M, n)) @ L.T


def place_evidence(field: np.ndarray, n: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(field, axis=1), axis=1)
    return ranks < n[:, None]


def place_evidence_blocks(field: np.ndarray, blocks: np.ndarray, counts: np.ndarray) -> np.ndarray:
    M, S = field.shape
    counts = np.asarray(counts)
    per_draw = counts.ndim == 2
    B = np.zeros((M, S), dtype=bool)
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        k = counts[:, b] if per_draw else np.full(M, int(counts[b]))
        if not np.any(k > 0):
            continue
        sub = field[:, idx]
        r = np.argsort(np.argsort(sub, axis=1), axis=1)
        B[:, idx] = r < k[:, None]
    return B
