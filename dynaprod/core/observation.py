from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ObservationPolicy:
    enabled: bool = False
    error_model: bool = True
    items: Optional[frozenset] = None

    def observe(self, domain, ctx, x):
        if not self.enabled or (self.items is not None and x not in self.items):
            return None
        return domain.observe(ctx, x)


class OverviewErrorModel:
    def __init__(self, n_levels: int = 17, n_prior_bins: int = 10, smoothing: float = 0.5):
        self.L = int(n_levels)
        self.n_a = int(n_prior_bins)
        self.smoothing = smoothing
        self.levels = np.linspace(0.0, 1.0, self.L)
        self.lik = np.full((self.L, self.L), 1.0 / self.L)
        self.prior = np.full((self.n_a, self.L), 1.0 / self.L)
        self.n_blocks = 0
        self.mean_abs_error = float("nan")
        self.inversion_rate = float("nan")

    def _lvl(self, share: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(np.asarray(share, float) * (self.L - 1)), 0, self.L - 1).astype(int)

    def _abin(self, a: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(a, float) * self.n_a).astype(int), 0, self.n_a - 1)

    def fit(self, domain, records: List[dict]) -> "OverviewErrorModel":
        by_q: Dict[str, Tuple[object, set]] = {}
        for r in records:
            q = r["request"]
            ent = by_q.setdefault(q.qid, (r["ctx"], set()))
            ent[1].update(r["comp"].read_set)
        T, O, A = [], [], []
        for ctx, xs in by_q.values():
            xs = sorted(xs)
            if not xs:
                continue
            ev = domain.truth_evidence(ctx, xs)
            for x in xs:
                obs = domain.observe(ctx, x)
                if obs is None:
                    continue
                blocks, counts = obs
                b = ev[x][0]
                if len(b) != len(blocks) or len(b) == 0:
                    continue
                size = np.bincount(blocks, minlength=len(counts)).astype(float)
                true = np.bincount(blocks, weights=b.astype(float), minlength=len(counts))
                ok = size > 0
                T.append(true[ok] / size[ok])
                O.append(np.asarray(counts, float)[ok] / size[ok])
                A.append(np.full(int(ok.sum()), b.mean()))
        if not T:
            return self
        t, o, a = np.concatenate(T), np.concatenate(O), np.concatenate(A)
        ti, oi, ai = self._lvl(t), self._lvl(o), self._abin(a)
        J = np.full((self.L, self.L), self.smoothing / self.L)
        np.add.at(J, (ti, oi), 1.0)
        self.lik = J / J.sum(1, keepdims=True)
        P = np.full((self.n_a, self.L), self.smoothing / self.L)
        np.add.at(P, (ai, ti), 1.0)
        marg = np.bincount(ti, minlength=self.L) + self.smoothing / self.L
        marg = marg / marg.sum()
        w = P.sum(1, keepdims=True)
        P = (P + 5.0 * marg[None, :]) / (w + 5.0)
        self.prior = P / P.sum(1, keepdims=True)
        self.n_blocks = len(t)
        self.mean_abs_error = float(np.mean(np.abs(o - t)))
        self.inversion_rate = float(np.mean(np.abs(o - t) > 0.5))
        return self

    def posterior(self, obs_share: np.ndarray, a: np.ndarray) -> np.ndarray:
        oi = self._lvl(obs_share)
        lik = self.lik[:, oi].T
        pri = self.prior[self._abin(a)]
        post = lik[None, :, :] * pri[:, None, :]
        post /= post.sum(-1, keepdims=True)
        return post

    def sample_counts(self, blocks: np.ndarray, counts: np.ndarray, a: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
        size = np.bincount(blocks, minlength=len(counts)).astype(float)
        share = np.where(size > 0, np.asarray(counts, float) / np.maximum(size, 1.0), 0.0)
        post = self.posterior(share, a)
        cdf = np.cumsum(post, -1)
        u = rng.random(post.shape[:2])[..., None]
        lvl = np.minimum((cdf < u).sum(-1), self.L - 1)
        return np.rint(self.levels[lvl] * size[None, :]).astype(int)

    def summary(self) -> dict:
        return dict(n_blocks=self.n_blocks, mean_abs_error=self.mean_abs_error,
                    inversion_rate=self.inversion_rate)
