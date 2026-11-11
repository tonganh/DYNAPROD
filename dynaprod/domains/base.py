from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.compute import execute_compute
from ..core.run_logs import RunLogStore
from ..core.lineage import EvalContext
from ..core.types import Compiled, OpSpec, Outcome, Request, RequestContext
from ..utils import rng_for


class ValidationError(Exception):
    pass


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Domain:

    name = "base"
    families: List[str] = []
    psi_keys: List[str] = []
    e_cols: List[str] = ["e"]
    fill_reference: Dict[str, tuple] = {}
    field_scales = np.array([1.0])
    time_scales = np.array([1.0])
    has_observations = False

    def __init__(self, seed: int = 0, quick: bool = False):
        self.seed = seed
        self.quick = quick
        self._ctx_cache: Dict[str, RequestContext] = {}
        self.run_logs: Optional[RunLogStore] = None

    def use_run_logs(self, path: str) -> "Domain":
        self.run_logs = RunLogStore(path)
        return self

    def requests(self) -> List[Request]:
        raise NotImplementedError

    def context(self, request: Request) -> RequestContext:
        if request.qid not in self._ctx_cache:
            self._ctx_cache[request.qid] = self._build_context(request)
        return self._ctx_cache[request.qid]

    def _build_context(self, request: Request) -> RequestContext:
        raise NotImplementedError

    def item_features(self, ctx: RequestContext, xs) -> pd.DataFrame:
        raise NotImplementedError

    def item_distance(self, ctx: RequestContext, xs) -> np.ndarray:
        raise NotImplementedError

    def order_key(self, ctx: RequestContext, xs) -> np.ndarray:
        return ctx.items.loc[list(xs), "time"].to_numpy(float)

    def true_evidence(self, ctx: RequestContext, x: int) -> np.ndarray:
        raise NotImplementedError

    def observe(self, ctx: RequestContext, x: int):
        return None

    def resolve_scope(self, request: Request, req: Optional[dict] = None) -> dict:
        ctx = self.context(request)
        scope = (req or request.req).get("scope")
        return dict(ctx=ctx, scope=scope, n_units=int(ctx.n_units), n_items=int(len(ctx.items)),
                    matches_request=(scope == request.req.get("scope")) if scope is not None else True)

    def source_group(self, comp: Compiled) -> str:
        return "*"

    def item_read_op(self, ctx: RequestContext, comp: Compiled, x: int) -> Optional[OpSpec]:
        reads = [o for o in comp.ops if o.op == "read"]
        if not reads:
            return None
        o = reads[0]
        n_in = max(1.0, float(o.psi.get("n_inputs", len(comp.read_set) or 1)))
        rec = float(np.mean(o.records)) if not callable(o.records) else 0.0
        b = self.item_mb(ctx, x) * 1e6 if hasattr(self, "item_mb") else o.bytes / n_in
        return OpSpec("read", dict(o.psi, n_inputs=1.0), float(b), rec / n_in)

    def size_measure(self, request: Request, ctx: RequestContext) -> float:
        return float(ctx.n_units)

    def templates(self, family: str) -> Dict[str, dict]:
        raise NotImplementedError

    def unseen(self, family: str) -> Dict[str, dict]:
        return {}

    def vocabulary(self) -> dict:
        raise NotImplementedError

    def compile(self, spec: dict, request: Request, ctx: RequestContext, name: str = "w") -> Compiled:
        raise NotImplementedError

    def candidate_pool(self, request: Request, ctx: RequestContext, include_unseen: bool) -> List[Compiled]:
        out = []
        for nm, sp in self.templates(request.family).items():
            try:
                out.append(self.compile(sp, request, ctx, name=nm))
            except ValidationError:
                pass
        if include_unseen:
            for nm, sp in self.unseen(request.family).items():
                try:
                    c = self.compile(sp, request, ctx, name=nm)
                    c.seen = False
                    out.append(c)
                except ValidationError:
                    pass
        return out

    def catalogue_text(self, request: Request, ctx: RequestContext) -> str:
        return ""

    def describe_request(self, req: dict) -> str:
        raise NotImplementedError

    def parse_rule(self, text: str) -> dict:
        raise NotImplementedError

    def truth_evidence(self, ctx: RequestContext, xs) -> Dict[int, np.ndarray]:
        cache = ctx.extra.setdefault("truth", {})
        out = {}
        for x in xs:
            if x not in cache:
                cache[x] = self.true_evidence(ctx, x).astype(bool)
            out[x] = cache[x][None, :]
        return out

    def true_fill_for(self, ctx: RequestContext, comp: Compiled):
        accepted = self.run_logs.accepted_fill(ctx.request.qid, comp) if self.run_logs is not None else {}
        return self.true_fill_fn(ctx, accepted)

    def true_fill_fn(self, ctx: RequestContext, accepted: Optional[Dict[str, list]] = None):
        U = {}

        def fn(method, child, space):
            n = child.shape[1]
            if accepted and method in accepted:
                acc = np.zeros(n, bool)
                acc[np.asarray(accepted[method], int)] = True
                return acc[None, :] & ~child
            if space not in U:
                U[space] = rng_for(self.seed, ctx.request.qid, "fill", space).random(n)
            A = ctx.adjacency[space]
            frac = np.asarray(A @ child.T.astype(float)).T
            a, b = self.fill_reference[method]
            p = sigmoid(a + b * frac)
            return (U[space][None, :] < p) & ~child

        return fn

    def execute(self, ctx: RequestContext, comp: Compiled, rng: np.random.Generator) -> Outcome:
        ev = self.truth_evidence(ctx, comp.read_set)
        ectx = EvalContext(ev, ctx.unitmap, ctx.n_units, 1, ctx.pool_maps, self.true_fill_for(ctx, comp))
        usable = ectx.eval(comp.root)
        yplus = float(usable.mean()) if comp.root is not None else 0.0
        records = [float(np.mean(o.records(ectx))) if callable(o.records) else float(o.records) for o in comp.ops]
        run = self.run_logs.run(ctx.request.qid, comp, records) if self.run_logs is not None else None
        if run is None:
            if self.run_logs is not None and comp.template != "composed" and comp.spec.get("items") is None:
                raise ValueError(f"{ctx.request.qid}: workflow {comp.template!r} {comp.spec} is not in the "
                                 f"execution logs {self.run_logs.path}")
            run = execute_compute(comp, records, rng)
        fill_logs = []
        for node, val in ectx.cache.items():
            if node[0] == "fill":
                child = ectx.cache.get(node[2], None)
                if child is None:
                    child = ectx.eval(node[2])
                from ..core.lineage import unit_space
                space = unit_space(node[2])
                frac = np.asarray(ctx.adjacency[space] @ child[0].astype(float))
                gap = ~child[0]
                fill_logs.append((node[1], frac[gap], val[0][gap]))
        Y = (1 - run.E) * yplus
        return Outcome(yplus, Y, run.C, run.T, run.E, log=dict(run=run, fill=fill_logs))
