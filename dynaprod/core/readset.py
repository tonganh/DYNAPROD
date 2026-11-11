from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence

import numpy as np

from .lineage import EvalContext, items_of


def _ctx(evidence, S, ctx, fill_fn, M):
    return EvalContext({x: evidence[x] for x in S}, ctx.unitmap, ctx.n_units, M, ctx.pool_maps, fill_fn)


def coverage_with(root, evidence: Dict[int, np.ndarray], S: Sequence[int], ctx, fill_fn, M: int) -> np.ndarray:
    return _ctx(evidence, S, ctx, fill_fn, M).coverage(root)


def core_gate(root):
    node = root
    while node is not None and node[0] in ("fill", "pool"):
        node = node[2]
    return node


def objective(root):
    core = core_gate(root)
    if core is not None and core[0] == "atl":
        k, ch = core[1], core[2]

        def f(e):
            cnt = sum(e.eval(c).astype(np.int16) for c in ch)
            return (np.minimum(cnt, k) / k).mean(1)
        return "atleast", f
    if core is not None and core[0] == "all":
        return "allof", lambda e: e.coverage(root)
    return "anyof", lambda e: e.coverage(root)


def greedy_readset(root, evidence: Dict[int, np.ndarray], items: Sequence[int], cost: Dict[int, float], ctx,
                   fill_fn, M: int, budget: float = np.inf, target: Optional[float] = None,
                   quantile: float = 0.1) -> dict:
    kind, fobj = objective(root)
    if kind == "allof":
        return _plain_greedy(root, fobj, evidence, items, cost, ctx, fill_fn, M, budget, target, quantile)
    S: List[int] = []
    spent = 0.0
    cur_obj = 0.0
    cur_Y = np.zeros(M)
    heap = [(-np.inf, int(x), -1) for x in items]
    heapq.heapify(heap)
    ub = {int(x): np.inf for x in items}
    rnd = 0
    while heap:
        neg, x, r = heapq.heappop(heap)
        if spent + cost[x] > budget + 1e-12:
            continue
        if r != rnd:
            e = _ctx(evidence, S + [x], ctx, fill_fn, M)
            gain = float(fobj(e).mean()) - cur_obj
            ub[x] = max(gain, 0.0)
            heapq.heappush(heap, (-(ub[x] / max(cost[x], 1e-12)), x, rnd))
            continue
        if ub[x] <= 1e-15:
            break
        S.append(x)
        spent += cost[x]
        cur_obj += ub[x]
        ub.pop(x)
        rnd += 1
        cur_Y = coverage_with(root, evidence, S, ctx, fill_fn, M)
        if target is not None and np.quantile(cur_Y, quantile) >= target:
            break
    if S:
        e = _ctx(evidence, S, ctx, fill_fn, M)
        cur_obj = float(fobj(e).mean())
    B = budget if np.isfinite(budget) else spent
    deltas = sorted(((g, cost[x]) for x, g in ub.items() if np.isfinite(g)), key=lambda d: -d[0] / max(d[1], 1e-12))
    unknown = [x for x, g in ub.items() if not np.isfinite(g)]
    extra, cap = 0.0, B
    if unknown:
        for x in unknown:
            g = max(0.0, float(fobj(_ctx(evidence, S + [x], ctx, fill_fn, M)).mean()) - cur_obj)
            deltas.append((g, cost[x]))
        deltas.sort(key=lambda d: -d[0] / max(d[1], 1e-12))
    for d, c in deltas:
        if cap <= 0:
            break
        z = min(1.0, cap / max(c, 1e-12))
        extra += z * d
        cap -= z * c
    upper = cur_obj + extra
    return dict(S=S, f=float(cur_Y.mean()), draws=cur_Y, cost=spent, objective=kind, f_obj=cur_obj,
                certificate=cur_obj / upper if upper > 0 else 1.0)


def _plain_greedy(root, fobj, evidence, items, cost, ctx, fill_fn, M, budget, target, quantile) -> dict:
    S, spent = [], 0.0
    left = [int(x) for x in items]
    for child in core_gate(root)[1]:
        mine = [x for x in left if x in items_of(child)]
        if any(x in items_of(child) for x in S) or not mine:
            continue
        ratio = {x: float(_ctx(evidence, S + [x], ctx, fill_fn, M).eval(child).mean()) / max(cost[x], 1e-12)
                 for x in mine if spent + cost[x] <= budget + 1e-12}
        if ratio:
            x = max(ratio, key=ratio.get)
            S.append(x)
            left.remove(x)
            spent += cost[x]
    cur_obj = float(fobj(_ctx(evidence, S, ctx, fill_fn, M)).mean()) if S else 0.0
    cur_Y = coverage_with(root, evidence, S, ctx, fill_fn, M) if S else np.zeros(M)
    while left and not (target is not None and np.quantile(cur_Y, quantile) >= target):
        best, best_ratio, best_obj = None, 0.0, cur_obj
        for x in left:
            if spent + cost[x] > budget + 1e-12:
                continue
            val = float(fobj(_ctx(evidence, S + [x], ctx, fill_fn, M)).mean())
            ratio = (val - cur_obj) / max(cost[x], 1e-12)
            if ratio > best_ratio + 1e-15:
                best, best_ratio, best_obj = x, ratio, val
        if best is None:
            break
        S.append(best)
        left.remove(best)
        spent += cost[best]
        cur_obj = best_obj
        cur_Y = coverage_with(root, evidence, S, ctx, fill_fn, M)
        if target is not None and np.quantile(cur_Y, quantile) >= target:
            break
    return dict(S=S, f=float(cur_Y.mean()), draws=cur_Y, cost=spent, objective="allof", f_obj=cur_obj,
                certificate=float("nan"))


def metadata_order_readset(root, evidence, items, cost, e_of, ctx, fill_fn, M, budget) -> dict:
    S, spent = [], 0.0
    for x in sorted(items, key=lambda x: -e_of[x]):
        if spent + cost[x] > budget + 1e-12:
            continue
        S.append(x)
        spent += cost[x]
    cur = coverage_with(root, evidence, S, ctx, fill_fn, M) if S else np.zeros(M)
    return dict(S=S, f=float(cur.mean()), draws=cur, cost=spent)
