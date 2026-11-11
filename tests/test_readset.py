import itertools

import numpy as np

from dynaprod.core.lineage import AllOf, AnyOf, Ev
from dynaprod.core.readset import greedy_readset


def test_readset_certificate(toy_context):
    rng = np.random.default_rng(6)
    M, U, n = 40, 30, 8
    ev = {x: rng.random((M, U)) < rng.uniform(0.1, 0.6) for x in range(n)}
    ctx = toy_context(U, n)
    cost = {x: float(rng.uniform(1, 3)) for x in range(n)}
    root = AnyOf([Ev(x) for x in range(n)])
    res = greedy_readset(root, ev, list(range(n)), cost, ctx, None, M, budget=5.0)
    assert res["cost"] <= 5.0 + 1e-9
    best = 0.0
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            if sum(cost[x] for x in S) <= 5.0:
                v = np.mean(np.any(np.stack([ev[x] for x in S]), 0)) if S else 0.0
                best = max(best, v)
    assert res["f"] / best >= res["certificate"] - 1e-9
    assert res["f"] >= (1 - 1 / np.e) * best * 0.5


def test_readset_allof_maximises_coverage(toy_context):
    rng = np.random.default_rng(7)
    M, U, n = 30, 20, 8
    ev = {x: rng.random((M, U)) < rng.uniform(0.3, 0.8) for x in range(n)}
    ctx = toy_context(U, n)
    cost = {x: float(rng.uniform(1, 3)) for x in range(n)}
    groups = [list(range(0, 4)), list(range(4, 8))]
    root = AllOf([AnyOf([Ev(x) for x in g]) for g in groups])
    res = greedy_readset(root, ev, list(range(n)), cost, ctx, None, M, budget=6.0)
    assert res["objective"] == "allof" and np.isnan(res["certificate"])
    assert res["cost"] <= 6.0 + 1e-9 and abs(res["f"] - res["f_obj"]) < 1e-9 and res["f"] > 0
