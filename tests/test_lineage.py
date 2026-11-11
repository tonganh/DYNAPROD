import numpy as np

from dynaprod.core.lineage import AllOf, AnyOf, AtLeast, EvalContext, Ev, at_unit, eval_formula


def test_lineage_eval_matches_formula():
    rng = np.random.default_rng(1)
    M, U = 50, 6
    ev = {x: rng.random((M, U)) < 0.4 for x in range(4)}
    um = {x: {"own": np.arange(U)} for x in range(4)}
    root = AnyOf([AllOf([Ev(0), Ev(1)]), AtLeast(2, [Ev(1), Ev(2), Ev(3)])])
    out = EvalContext(ev, um, U, M).eval(root)
    for m in range(5):
        for u in range(U):
            f = at_unit(root, u, um, {})
            assign = {("B", x, u): bool(ev[x][m, u]) for x in range(4)}
            assert eval_formula(f, assign) == out[m, u]


def test_incremental_lineage_update():
    ev = {1: np.ones((2, 4), bool), 2: np.zeros((2, 4), bool), 3: np.zeros((2, 4), bool)}
    um = {x: {"own": np.arange(4)} for x in ev}
    a, b = AnyOf([Ev(1), Ev(2)]), AllOf([Ev(2), Ev(3)])
    e = EvalContext(ev, um, 4, 2)
    assert e.coverage(a).mean() == 1.0 and e.coverage(b).mean() == 0.0
    n = e.update_evidence({3: np.ones((2, 4), bool)})
    assert n == 1 and a in e.cache and b not in e.cache
    e.update_evidence({2: np.ones((2, 4), bool)})
    assert e.coverage(b).mean() == 1.0
