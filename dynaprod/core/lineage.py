from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np


def Ev(x: int, role: str = "own"):
    return ("ev", int(x), role)


def _canon(children: Iterable) -> tuple:
    return tuple(sorted(set(children), key=repr))


def AnyOf(children):
    flat = []
    for c in children:
        if c is None:
            continue
        if c[0] == "any":
            flat.extend(c[1])
        else:
            flat.append(c)
    flat = _canon(flat)
    if not flat:
        return None
    if len(flat) == 1:
        return flat[0]
    return ("any", flat)


def AllOf(children):
    flat = []
    for c in children:
        if c is None:
            return None
        if c[0] == "all":
            flat.extend(c[1])
        else:
            flat.append(c)
    flat = _canon(flat)
    if not flat:
        return None
    if len(flat) == 1:
        return flat[0]
    return ("all", flat)


def AtLeast(k: int, children):
    ch = _canon([c for c in children if c is not None])
    if k <= 0:
        raise ValueError("Th_k needs k >= 1 (no constant-true output)")
    if len(ch) < k:
        return None
    if k == 1:
        return AnyOf(ch)
    if k == len(ch):
        return AllOf(ch)
    return ("atl", int(k), ch)


def Fill(method: str, child):
    return ("fill", method, child)


def Pool(mapname: str, child):
    if child is None:
        return None
    return ("pool", mapname, child)


def leaves(node, acc=None) -> set:
    acc = set() if acc is None else acc
    if node is None:
        return acc
    k = node[0]
    if k == "ev":
        acc.add(node)
    elif k in ("any", "all"):
        for c in node[1]:
            leaves(c, acc)
    elif k == "atl":
        for c in node[2]:
            leaves(c, acc)
    elif k in ("fill", "pool"):
        leaves(node[2], acc)
    return acc


def items_of(node) -> set:
    return {l[1] for l in leaves(node)}


def size(node) -> int:
    if node is None:
        return 0
    k = node[0]
    if k == "ev":
        return 1
    if k in ("any", "all"):
        return 1 + sum(size(c) for c in node[1])
    if k == "atl":
        return 1 + sum(size(c) for c in node[2])
    return 1 + size(node[2])


def unit_space(node, base: str = "fine") -> str:
    if node is None:
        return base
    if node[0] == "pool":
        return node[1]
    if node[0] in ("any", "all"):
        return unit_space(node[1][0], base)
    if node[0] == "atl":
        return unit_space(node[2][0], base)
    if node[0] == "fill":
        return unit_space(node[2], base)
    return base


class EvalContext:

    def __init__(self, evidence: Dict[int, np.ndarray], unitmap: Dict[int, Dict[str, np.ndarray]],
                 n_units: int, M: int, pool_maps: Optional[dict] = None,
                 fill_fn: Optional[Callable] = None):
        self.evidence = evidence
        self.unitmap = unitmap
        self.n_units = n_units
        self.M = M
        self.pool_maps = pool_maps or {}
        self.fill_fn = fill_fn
        self.cache: Dict[tuple, np.ndarray] = {}
        self.n_evaluated = 0

    def _n(self, space: str) -> int:
        return self.n_units if space == "fine" else self.pool_maps[space][1]

    def leaf(self, x: int, role: str) -> np.ndarray:
        B = self.evidence.get(x)
        um = self.unitmap[x].get(role)
        if B is None or um is None:
            return np.zeros((self.M, self.n_units), dtype=bool)
        padded = np.concatenate([B, np.zeros((B.shape[0], 1), dtype=bool)], axis=1)
        idx = np.where(um >= 0, um, padded.shape[1] - 1)
        return padded[:, idx]

    def eval(self, node) -> np.ndarray:
        if node is None:
            return np.zeros((self.M, self.n_units), dtype=bool)
        hit = self.cache.get(node)
        if hit is not None:
            return hit
        self.n_evaluated += 1
        k = node[0]
        if k == "ev":
            out = self.leaf(node[1], node[2])
            return out
        if k == "any":
            out = None
            for c in node[1]:
                v = self.eval(c)
                out = v.copy() if out is None else np.logical_or(out, v, out=out)
        elif k == "all":
            out = None
            for c in node[1]:
                v = self.eval(c)
                out = v.copy() if out is None else np.logical_and(out, v, out=out)
        elif k == "atl":
            cnt = None
            for c in node[2]:
                v = self.eval(c).astype(np.int16)
                cnt = v if cnt is None else cnt + v
            out = cnt >= node[1]
        elif k == "fill":
            child = self.eval(node[2])
            space = unit_space(node[2])
            acc = self.fill_fn(node[1], child, space) if self.fill_fn else np.zeros_like(child)
            out = child | acc
        elif k == "pool":
            child = self.eval(node[2])
            cmap, n_coarse = self.pool_maps[node[1]]
            out = np.zeros((child.shape[0], n_coarse), dtype=bool)
            valid = cmap >= 0
            np.logical_or.at(out.T, cmap[valid], child[:, valid].T)
        else:
            raise ValueError(f"unknown node {k}")
        self.cache[node] = out
        return out

    def coverage(self, node) -> np.ndarray:
        return self.eval(node).mean(axis=1)

    def update_evidence(self, updates: Dict[int, np.ndarray]) -> int:
        changed = set(int(x) for x in updates)
        self.evidence.update({int(x): v for x, v in updates.items()})
        stale = [n for n in self.cache if items_of(n) & changed]
        for n in stale:
            del self.cache[n]
        return len(stale)


def at_unit(node, u: int, unitmap: Dict[int, Dict[str, np.ndarray]], pool_inv: dict, space: str = "fine"):
    k = node[0]
    if k == "ev":
        um = unitmap[node[1]].get(node[2])
        if um is None or um[u] < 0:
            return None
        return ("var", ("B", node[1], int(um[u])))
    if k == "any":
        return AnyOfF([at_unit(c, u, unitmap, pool_inv, space) for c in node[1]])
    if k == "all":
        return AllOfF([at_unit(c, u, unitmap, pool_inv, space) for c in node[1]])
    if k == "atl":
        ch = [at_unit(c, u, unitmap, pool_inv, space) for c in node[2]]
        ch = [c for c in ch if c is not None]
        if len(ch) < node[1]:
            return None
        if node[1] == 1:
            return AnyOfF(ch)
        return ("atl", node[1], tuple(ch))
    if k == "fill":
        return AnyOfF([at_unit(node[2], u, unitmap, pool_inv, space), ("var", ("A", node[1], unit_space(node[2]), u))])
    if k == "pool":
        fines = pool_inv[node[1]][u]
        return AnyOfF([at_unit(node[2], f, unitmap, pool_inv, "fine") for f in fines])
    raise ValueError(k)


def AnyOfF(ch):
    ch = [c for c in ch if c is not None]
    flat = []
    for c in ch:
        flat.extend(c[1] if c[0] == "any" else [c])
    flat = list(dict.fromkeys(flat))
    if not flat:
        return None
    return flat[0] if len(flat) == 1 else ("any", tuple(flat))


def AllOfF(ch):
    if any(c is None for c in ch):
        return None
    flat = []
    for c in ch:
        flat.extend(c[1] if c[0] == "all" else [c])
    flat = list(dict.fromkeys(flat))
    if not flat:
        return None
    return flat[0] if len(flat) == 1 else ("all", tuple(flat))


def eval_formula(f, assign: dict) -> bool:
    if f is None:
        return False
    k = f[0]
    if k == "var":
        return bool(assign.get(f[1], False))
    if k == "any":
        return any(eval_formula(c, assign) for c in f[1])
    if k == "all":
        return all(eval_formula(c, assign) for c in f[1])
    if k == "atl":
        return sum(eval_formula(c, assign) for c in f[2]) >= f[1]
    raise ValueError(k)


def formula_vars(f, acc=None) -> set:
    acc = set() if acc is None else acc
    if f is None:
        return acc
    k = f[0]
    if k == "var":
        acc.add(f[1])
    elif k in ("any", "all"):
        for c in f[1]:
            formula_vars(c, acc)
    elif k == "atl":
        for c in f[2]:
            formula_vars(c, acc)
    return acc
