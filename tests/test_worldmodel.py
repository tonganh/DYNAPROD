from types import SimpleNamespace

import numpy as np

from dynaprod.core.field import place_evidence_blocks
from dynaprod.core.observation import OverviewErrorModel


def test_place_evidence_blocks_per_draw():
    rng = np.random.default_rng(4)
    blocks = np.repeat(np.arange(4), 4)
    f = rng.standard_normal((3, 16))
    counts = np.array([[0, 1, 2, 3], [4, 4, 4, 4], [1, 0, 0, 0]])
    B = place_evidence_blocks(f, blocks, counts)
    got = np.stack([np.bincount(blocks, weights=B[k].astype(float), minlength=4) for k in range(3)])
    assert np.array_equal(got, counts)


class _StubObsDomain:

    def __init__(self, p=0.075, n_items=400, seed=0):
        rng = np.random.default_rng(seed)
        self.blocks = np.repeat(np.arange(4), 16)
        self.B, self.obs = {}, {}
        for x in range(n_items):
            share = rng.choice([0.0, 0.25, 1.0], size=4, p=[0.3, 0.2, 0.5])
            b = np.concatenate([np.arange(16) < int(16 * s) for s in share])
            self.B[x] = b
            true = np.bincount(self.blocks, weights=b.astype(float), minlength=4).astype(int)
            flip = rng.random(4) < p
            self.obs[x] = (self.blocks, np.where(flip, 16 - true, true))

    def truth_evidence(self, ctx, xs):
        return {x: self.B[x][None, :] for x in xs}

    def observe(self, ctx, x):
        return self.obs[x]


def test_overview_error_model_learns_inversions():
    dom = _StubObsDomain()
    recs = [dict(request=SimpleNamespace(qid="q"), ctx=None, comp=SimpleNamespace(read_set=list(dom.B)))]
    em = OverviewErrorModel().fit(dom, recs)
    assert abs(em.inversion_rate - 0.075) < 0.03
    assert abs(em.lik[0, -1] - 0.075) < 0.04 and em.lik[-1, -1] > 0.85
    lo = em.posterior(np.array([1.0]), np.array([0.3]))[0, 0]
    hi = em.posterior(np.array([1.0]), np.array([0.9]))[0, 0]
    assert lo[0] > hi[0]
    counts = em.sample_counts(dom.blocks, np.array([16, 0, 4, 16]), np.full(500, 0.6), np.random.default_rng(0))
    assert counts.shape == (500, 4) and counts.min() >= 0 and counts.max() <= 16
    assert np.mean(counts[:, 0] == 16) > 0.7
