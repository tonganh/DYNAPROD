from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.readset import greedy_readset
from ..core.types import Compiled, Request
from ..domains.base import ValidationError
from ..utils import rng_for


def canon(spec: dict) -> str:
    s = {k: v for k, v in spec.items() if v is not None and k != "items"}
    if "sources" in s:
        s["sources"] = sorted(s["sources"])
    return json.dumps(s, sort_keys=True)


class ScopeResolver:
    def __init__(self, domain):
        self.domain = domain

    def __call__(self, q: Request, req: Optional[dict] = None) -> dict:
        out = self.domain.resolve_scope(q, req)
        out["summary"] = {k: v for k, v in out.items() if k not in ("ctx",)}
        return out


class CatalogueSummary:
    def __init__(self, domain):
        self.domain = domain

    def __call__(self, q: Request, ctx) -> str:
        return self.domain.catalogue_text(q, ctx)


class Composer:

    def __init__(self, domain):
        self.domain = domain
        self.seen_specs = {canon(sp): nm for fam in domain.families for nm, sp in domain.templates(fam).items()}
        self.errors: List[str] = []

    def vocabulary(self) -> dict:
        return self.domain.vocabulary()

    def compile(self, spec: dict, q: Request, ctx, name: str) -> Compiled:
        c = self.domain.compile(spec, q, ctx, name=name)
        key = canon(spec)
        if key in self.seen_specs and spec.get("items") is None:
            c.template, c.seen = self.seen_specs[key], True
        else:
            c.template, c.seen = "composed", False
        return c

    def validate(self, spec: dict, q: Request, ctx, name: str) -> Tuple[Optional[Compiled], Optional[str]]:
        try:
            return self.compile(spec, q, ctx, name), None
        except ValidationError as e:
            self.errors.append(str(e))
            return None, str(e)


class WorldModelTool:
    def __init__(self, wm, M: int = 64, seed: int = 0):
        self.wm, self.M, self.seed = wm, M, seed
        self.calls = 0

    def rollout(self, q: Request, ctx, comps: Sequence[Compiled], observe=False):
        self.calls += 1
        return self.wm.rollout(q, ctx, list(comps), M=self.M, seed=self.seed, observe=observe)

    def __call__(self, q: Request, ctx, comps: Sequence[Compiled]) -> List[dict]:
        if not comps:
            return []
        R = self.rollout(q, ctx, comps)
        return [dict(coverage_median=float(np.median(R.Y[k])), coverage_p10=float(np.quantile(R.Y[k], 0.1)),
                     cost_usd_median=float(np.median(R.C[k])), time_s_median=float(np.median(R.T[k])),
                     failure_prob=float(R.E[k].mean())) for k in range(len(comps))]


class ReadSetSelector:
    def __init__(self, wm, M: int = 64, seed: int = 0, cost: str = "eq2"):
        self.wm, self.M, self.seed, self.cost = wm, M, seed, cost

    def costs(self, ctx, comp: Compiled, xs) -> Dict[int, float]:
        if self.cost == "mb":
            return {int(x): float(self.wm.domain.item_mb(ctx, x)) for x in xs}
        return self.wm.read_costs(ctx, comp, xs)

    def __call__(self, q: Request, ctx, comp: Compiled, target: Optional[float] = None,
                 budget: float = np.inf) -> dict:
        xs = sorted(set(comp.read_set))
        draws = self.wm.world_draws(q, ctx, self.M, self.seed)
        ev = {x: draws[x] for x in xs}
        fill = self.wm.fill_fn(ctx, rng_for(self.seed, q.qid, "shared"), self.M, key=(self.seed, q.qid))
        return greedy_readset(comp.root, ev, xs, self.costs(ctx, comp, xs), ctx, fill, self.M,
                              budget=budget, target=target)


@dataclass
class AgentTools:
    resolver: ScopeResolver
    catalogue: CatalogueSummary
    composer: Composer
    world_model: WorldModelTool
    readset: ReadSetSelector

    @classmethod
    def default(cls, domain, wm, M_look: Optional[int] = None, seed: int = 0, readset_cost: str = "eq2") -> "AgentTools":
        M = M_look or wm.cfg.M
        return cls(ScopeResolver(domain), CatalogueSummary(domain), Composer(domain),
                   WorldModelTool(wm, M, seed), ReadSetSelector(wm, M, seed, readset_cost))

    def describe(self) -> List[dict]:
        return [
            dict(name="resolve_scope", description="map the request's named scope to units and items"),
            dict(name="catalogue_summary", description="metadata of the items over the request's units"),
            dict(name="compose", description="validate a workflow spec against the vocabulary and compile "
                                             "its lineage", grammar=self.composer.vocabulary().get("spec_fields")),
            dict(name="world_model", description="predicted coverage, cost, time and failure of compiled "
                                                 "workflows on shared draws"),
            dict(name="select_read_set", description="cheapest read set reaching the target on the draws"),
        ]
