import numpy as np

from dynaprod.core.containment import _brute, implies, to_phi, to_psi


def V(i):
    return ("var", ("B", i, 0))


def rand_phi(rng, n=7):
    terms = []
    for _ in range(rng.integers(1, 3)):
        vs = list(rng.permutation(n))
        groups = []
        for _ in range(rng.integers(1, 3)):
            sz = int(rng.integers(1, 3))
            S, vs = vs[:sz], vs[sz:]
            if not S:
                break
            k = int(rng.integers(1, len(S) + 1))
            ch = [V(int(i)) for i in S]
            groups.append(ch[0] if len(ch) == 1 else ("atl", k, tuple(ch)) if 1 < k < len(ch) else
                          (("any", tuple(ch)) if k == 1 else ("all", tuple(ch))))
        terms.append(groups[0] if len(groups) == 1 else ("all", tuple(groups)))
    return terms[0] if len(terms) == 1 else ("any", tuple(terms))


def rand_psi(rng, n=7):
    clauses = []
    for _ in range(rng.integers(1, 3)):
        vs = list(rng.permutation(n))
        S = [V(int(i)) for i in vs[: rng.integers(0, 4)]]
        A = [V(int(i)) for i in vs[4: 4 + rng.integers(0, 3)]]
        parts = []
        if len(S) >= 2:
            k = int(rng.integers(1, len(S) + 1))
            parts.append(("atl", k, tuple(S)) if 1 < k < len(S) else (("any", tuple(S)) if k == 1 else ("all", tuple(S))))
        elif len(S) == 1:
            A = A + S
        parts += A
        if not parts:
            parts = [V(0)]
        clauses.append(parts[0] if len(parts) == 1 else ("any", tuple(parts)))
    return clauses[0] if len(clauses) == 1 else ("all", tuple(clauses))


def test_theorem1_fragment_matches_brute_force():
    rng = np.random.default_rng(0)
    n_checked = 0
    for _ in range(600):
        phi, psi = rand_phi(rng), rand_psi(rng)
        if to_phi(phi) is None or to_psi(psi) is None:
            continue
        assert implies(phi, psi) == _brute(phi, psi), (phi, psi)
        n_checked += 1
    assert n_checked > 300
