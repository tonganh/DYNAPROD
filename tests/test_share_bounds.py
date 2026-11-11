import numpy as np

from dynaprod.core.lineage import AllOf, AnyOf, AtLeast, Ev, Fill
from dynaprod.core.share_bounds import anyof_with_overlaps, circuit_interval, independent_anyof, thm2_eligible


def test_theorem2_intervals_contain_truth():
    rng = np.random.default_rng(2)
    U = 200
    for _ in range(200):
        n = int(rng.integers(2, 6))
        B = rng.random((n, U)) < rng.random((n, 1))
        a = B.mean(1)
        y_any = B.any(0).mean()
        lo, hi = circuit_interval(AnyOf([Ev(i) for i in range(n)]), {Ev(i): a[i] for i in range(n)})
        assert lo - 1e-12 <= y_any <= hi + 1e-12
        y_all = B.all(0).mean()
        lo, hi = circuit_interval(AllOf([Ev(i) for i in range(n)]), {Ev(i): a[i] for i in range(n)})
        assert lo - 1e-12 <= y_all <= hi + 1e-12
        k = int(rng.integers(1, n + 1))
        y_k = (B.sum(0) >= k).mean()
        lo, hi = circuit_interval(AtLeast(k, [Ev(i) for i in range(n)]), {Ev(i): a[i] for i in range(n)})
        assert lo - 1e-12 <= y_k <= hi + 1e-12
        O = (B[:, None, :] & B[None, :, :]).mean(2)
        lo, hi = anyof_with_overlaps(a, O)
        assert lo - 1e-9 <= y_any <= hi + 1e-9
        if n >= 4:
            g1, g2 = [Ev(i) for i in range(2)], [Ev(i) for i in range(2, n)]
            y_nested = (B[:2].any(0) & B[2:].any(0)).mean()
            lo, hi = circuit_interval(AllOf([AnyOf(g1), AnyOf(g2)]), {Ev(i): a[i] for i in range(n)},
                                      lambda u, v: O[u[1], v[1]])
            assert lo - 1e-9 <= y_nested <= hi + 1e-9


def test_theorem3_independence_overstates_anyof_under_association():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(20000, 1))
    X = (0.8 * z + 0.6 * rng.normal(size=(20000, 3))) > 0.3
    p = X.mean(0)
    assert independent_anyof(p) >= X.any(1).mean() - 0.01


def test_thm2_eligible():
    assert thm2_eligible(AnyOf([Ev(1), Ev(2)]))
    assert thm2_eligible(AllOf([Ev(1), Ev(2)]))
    assert not thm2_eligible(Fill("spatial", AnyOf([Ev(1), Ev(2)])))
    assert not thm2_eligible(AtLeast(2, [Ev(1), Ev(2), Ev(3)]))
    assert thm2_eligible(AllOf([AnyOf([Ev(1), Ev(2)]), AnyOf([Ev(3), Ev(4)])]))
    assert not thm2_eligible(AllOf([Fill("spatial", AnyOf([Ev(1), Ev(2)])), Ev(3)]))
