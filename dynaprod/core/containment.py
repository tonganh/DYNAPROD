from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lineage import at_unit, formula_vars, leaves

MAX_TERMS = 4096
MAX_BRUTE_VARS = 14


def to_phi(f) -> Optional[List[List[Tuple[frozenset, int]]]]:
    if f is None:
        return []
    k = f[0]
    if k == "var":
        return [[(frozenset([f[1]]), 1)]]
    if k == "atl":
        sets = _disjoint_anyof_sets(f[2])
        if sets is None:
            return None
        if all(len(S) == 1 for S in sets):
            return [[(frozenset().union(*sets), f[1])]]
        if _n_choose(len(sets), f[1]) > MAX_TERMS:
            return None
        return [[(sets[i], 1) for i in I] for I in itertools.combinations(range(len(sets)), f[1])]
    if k == "any":
        out = []
        for c in f[1]:
            t = to_phi(c)
            if t is None:
                return None
            out.extend(t)
            if len(out) > MAX_TERMS:
                return None
        return out
    if k == "all":
        parts = []
        for c in f[1]:
            t = to_phi(c)
            if t is None:
                return None
            parts.append(t)
        n = 1
        for p in parts:
            n *= max(1, len(p))
            if n > MAX_TERMS:
                return None
        out = []
        for combo in itertools.product(*parts):
            groups = [g for term in combo for g in term]
            seen = set()
            ok = True
            for S, _ in groups:
                if seen & S:
                    ok = False
                    break
                seen |= S
            if not ok:
                return None
            out.append(groups)
        return out
    return None


def to_psi(f) -> Optional[List[Tuple[frozenset, int, frozenset]]]:
    if f is None:
        return None
    k = f[0]
    if k == "var":
        return [(frozenset(), 1, frozenset([f[1]]))]
    if k == "atl":
        sets = _disjoint_anyof_sets(f[2])
        if sets is None:
            return None
        if all(len(S) == 1 for S in sets):
            return [(frozenset().union(*sets), f[1], frozenset())]
        D, kk = len(sets), f[1]
        if _n_choose(D, D - kk + 1) > MAX_TERMS:
            return None
        return [(frozenset(), 1, frozenset().union(*(sets[i] for i in J)))
                for J in itertools.combinations(range(D), D - kk + 1)]
    if k == "all":
        out = []
        for c in f[1]:
            t = to_psi(c)
            if t is None:
                return None
            out.extend(t)
        return out
    if k == "any":
        S, kk, A = frozenset(), 1, set()
        for c in f[1]:
            t = to_psi(c)
            if t is None or len(t) != 1:
                return None
            cS, ck, cA = t[0]
            if cS:
                if S:
                    return None
                S, kk = cS, ck
            A |= cA
        return [(S, kk, frozenset(A))]
    return None


def _disjoint_anyof_sets(children) -> Optional[List[frozenset]]:
    sets = []
    for c in children:
        if c[0] == "var":
            sets.append(frozenset([c[1]]))
        elif c[0] == "any" and all(g[0] == "var" for g in c[1]):
            sets.append(frozenset(g[1] for g in c[1]))
        else:
            return None
    seen = set()
    for S in sets:
        if seen & S:
            return None
        seen |= S
    return sets


def _n_choose(n: int, k: int) -> int:
    from math import comb
    return comb(n, k) if 0 <= k <= n else 0


def _check_fragment(phi_terms, psi_clauses) -> bool:
    for term in phi_terms:
        for Sp, kp, Ap in psi_clauses:
            if any(len(S - Ap) < k for S, k in term):
                continue
            need = sum(max(0, k - len(S - (Sp | Ap))) for S, k in term)
            if need >= kp and Sp:
                continue
            return False
    return True


def _brute(phi, psi) -> Optional[bool]:
    vs = sorted(formula_vars(phi) | formula_vars(psi), key=repr)
    if len(vs) > MAX_BRUTE_VARS:
        return None
    from .lineage import eval_formula
    for bits in itertools.product([False, True], repeat=len(vs)):
        a = dict(zip(vs, bits))
        if eval_formula(phi, a) and not eval_formula(psi, a):
            return False
    return True


def implies(phi, psi) -> Optional[bool]:
    if phi is None:
        return True
    if psi is None:
        return False
    p, q = to_phi(phi), to_psi(psi)
    if p is not None and q is not None:
        return _check_fragment(p, q)
    return _brute(phi, psi)


def _signature(u, leaf_list, unitmap):
    sig = []
    cells = {}
    for (_, x, role) in leaf_list:
        um = unitmap[x].get(role)
        c = -1 if um is None else int(um[u])
        if c >= 0:
            key = (x, c)
            cells.setdefault(key, len(cells))
            sig.append(cells[key])
        else:
            sig.append(-1)
    return tuple(sig)


def _relabel(f, names):
    if f is None:
        return None
    k = f[0]
    if k == "var":
        v = f[1]
        if v[0] == "B":
            if v not in names:
                names[v] = len(names)
            return ("var", ("B", names[v]))
        if v not in names:
            names[v] = len(names)
        return ("var", ("A", v[1], names[v]))
    if k in ("any", "all"):
        return (k, tuple(_relabel(c, names) for c in f[1]))
    if k == "atl":
        return (k, f[1], tuple(_relabel(c, names) for c in f[2]))
    return f


_IMPLIES_CACHE: Dict[tuple, Optional[bool]] = {}


def contained(root_a, root_b, space_a: str, space_b: str, n_units_space: int,
              unitmap, pool_inv) -> bool:
    if space_a != space_b:
        return False
    leaf_list = sorted(leaves(root_a) | leaves(root_b), key=repr)
    seen: Dict[tuple, bool] = {}
    for u in range(n_units_space):
        if space_a == "fine":
            sig = _signature(u, leaf_list, unitmap)
            if sig in seen:
                if not seen[sig]:
                    return False
                continue
        fa = at_unit(root_a, u, unitmap, pool_inv)
        fb = at_unit(root_b, u, unitmap, pool_inv)
        names = {}
        key = (_relabel(fa, names), _relabel(fb, names))
        if key in _IMPLIES_CACHE:
            r = _IMPLIES_CACHE[key]
        else:
            r = implies(fa, fb) is True
            if len(_IMPLIES_CACHE) > 200000:
                _IMPLIES_CACHE.clear()
            _IMPLIES_CACHE[key] = r
        if space_a == "fine":
            seen[sig] = r
        if not r:
            return False
    return True


def certified_pairs(roots: Sequence, spaces: Sequence[str], n_units: Dict[str, int],
                    unitmap, pool_inv) -> set:
    pairs = set()
    K = len(roots)
    for i in range(K):
        for j in range(K):
            if i == j or roots[i] is None or roots[j] is None:
                continue
            if contained(roots[i], roots[j], spaces[i], spaces[j], n_units[spaces[i]], unitmap, pool_inv):
                pairs.add((i, j))
    return pairs


def transitive_closure(pairs: set, K: int) -> np.ndarray:
    R = np.zeros((K, K), dtype=bool)
    for i, j in pairs:
        R[i, j] = True
    for k in range(K):
        R |= R[:, [k]] & R[[k], :]
    np.fill_diagonal(R, False)
    return R
