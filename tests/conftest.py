import numpy as np
import pytest

from dynaprod.core.types import Request, RequestContext


@pytest.fixture
def make_entries():
    def make(spec, rng, M=50):
        out = []
        for fam, size, group, n in spec:
            for _ in range(n):
                out.append(dict(family=fam, size=size, group=group, Y=rng.beta(5, 2, M),
                                C=rng.lognormal(0, 0.2, M), T=rng.lognormal(0, 0.2, M),
                                y=float(rng.beta(5, 2)), c=1.0, t=1.0))
        return out
    return make


@pytest.fixture
def toy_context():
    def make(U, n):
        req = Request("q", "d", "f", "c", 0, "", {})
        return RequestContext(req, None, U, np.zeros((U, 1)), {x: np.arange(U) for x in range(n)},
                              {x: {"own": np.arange(U)} for x in range(n)})
    return make
