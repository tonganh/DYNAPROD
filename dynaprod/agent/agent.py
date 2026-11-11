from __future__ import annotations

import itertools
import json
import time
from typing import Dict, List, Optional

import numpy as np

from ..core.types import Compiled, Request
from ..domains.base import ValidationError
from ..llm.openrouter import LLMError, OpenRouter, available, parse_json
from .tools import AgentTools, canon

SYSTEM_INTERPRET = """You are the requirement parser of a data seller. Turn the buyer's request into a JSON requirement.
Return ONLY a JSON object with keys:
  "family": one of {families},
  "scope": object (domain-specific: see example),
  "target": usable-coverage fraction in [0,1],
  "budget": number in USD or null,
  "deadline": seconds or null,
  "floors": object of quality floors (e.g. max_resolution_m, min_clear_obs, min_valid_hours) or {{}}.
Unspecified fields get the family's defaults. Example of the schema for this domain:
{example}"""

SYSTEM_COMPOSE = """You are the data agent of a data seller. A buyer's requirement and the archive's catalogue
summary over the request's units are given. Compose up to {m} candidate workflows (products) that trade quality
against cost in different ways: other sources, other combination rules, fusion, gap filling, a coarser
resolution or a cheaper model. The target usable coverage is {target}.

A workflow is a JSON spec in this grammar (only these fields; the composer rejects anything else):
{grammar}

Reply with ONLY a JSON object {{"candidates": [spec, ...]}} (at most {m} specs)."""

SYSTEM_REVISE = """You are the data agent of a data seller. The world model predicted the outcomes of your
candidate workflows before anything was read, and none is predicted to reach the target usable coverage
{target}. Add candidates aimed at the shortfall (more sources, fusion, gap filling, radar, a coarser resolution,
a cascade), in the same grammar:
{grammar}

Reply with ONLY a JSON object {{"candidates": [spec, ...]}} with at most {m} new specs."""


class DataAgent:
    def __init__(self, domain, wm, mode: str = "llm", parse: str = "llm", model: Optional[str] = None,
                 max_cands: int = 8, rounds: int = 2, M_look: Optional[int] = None, seed: int = 0, use_readset: bool = True,
                 tools: Optional[AgentTools] = None, readset_cost: str = "eq2"):
        self.domain = domain
        self.wm = wm
        self.mode = mode
        self.parse = parse
        self.max_cands = max_cands
        self.rounds = rounds
        self.M_look = M_look or wm.cfg.M
        self.seed = seed
        self.use_readset = use_readset
        self.llm = None
        if (mode == "llm" or parse == "llm"):
            if not available():
                raise LLMError("the LLM agent needs OPENROUTER_API_KEY in .env")
            self.llm = OpenRouter(model=model)
        self.calls = 0
        self.seconds = 0.0
        self.trace: List[dict] = []
        self.tools = tools or AgentTools.default(domain, wm, self.M_look, seed, readset_cost)
        self._seen_specs = self.tools.composer.seen_specs

    def interpret(self, q: Request) -> dict:
        if self.parse == "gold":
            return dict(q.req)
        if self.parse == "rule" or self.llm is None:
            req = self.domain.parse_rule(q.text)
        else:
            ex = json.dumps({k: v for k, v in q.req.items() if k != "budget"} | {"budget": 1.0}, default=str)
            msgs = [{"role": "system", "content": SYSTEM_INTERPRET.format(families=self.domain.families, example=ex)},
                    {"role": "user", "content": q.text}]
            t0 = time.time()
            try:
                req = self.llm.json(msgs, max_tokens=1500)
            except ValueError:
                req = None
            if not isinstance(req, dict):
                req = self.domain.parse_rule(q.text)
            self.calls += 1
            self.seconds += time.time() - t0
        return self._fill_defaults(req, q)

    def _fill_defaults(self, req: dict, q: Request) -> dict:
        req = dict(req or {})
        if req.get("family") not in self.domain.families:
            req["family"] = self.domain.parse_rule(q.text).get("family", self.domain.families[0])
        req.setdefault("floors", {})
        try:
            req["target"] = float(req.get("target", 0.8))
            if req["target"] > 1:
                req["target"] /= 100.0
        except (TypeError, ValueError):
            req["target"] = 0.8
        for k in ("budget", "deadline"):
            try:
                req[k] = float(req[k]) if req.get(k) is not None else np.inf
            except (TypeError, ValueError):
                req[k] = np.inf
        return req

    def parse_accuracy(self, req: dict, q: Request) -> float:
        g = q.req
        checks = [req.get("family") == g["family"], abs(req.get("target", -1) - g["target"]) < 1e-6,
                  (g.get("budget") is None) or abs(req.get("budget", np.inf) - g["budget"]) <= 0.02 * g["budget"] + 1e-9,
                  abs(req.get("deadline", np.inf) - g["deadline"]) < 1.0]
        for k, v in g.get("floors", {}).items():
            checks.append(req.get("floors", {}).get(k) == v)
        return float(np.mean(checks))

    def _compile(self, spec, q, ctx, name):
        return self.tools.composer.compile(spec, q, ctx, name)

    def _predict(self, q, ctx, comps):
        return self.tools.world_model(q, ctx, comps)

    def _readset(self, q, ctx, comp, target, budget=np.inf):
        return self.tools.readset(q, ctx, comp, target=target, budget=budget)

    def compose(self, q: Request, req: dict, ctx) -> List[Compiled]:
        if self.mode == "templates":
            return self._templates(q, ctx)
        if self.mode == "grid":
            return self._grid(q, ctx, req)
        if self.mode == "search":
            return self._search(q, ctx, req)
        return self._llm_compose(q, ctx, req)

    def _templates(self, q, ctx):
        out = []
        for nm, sp in self.domain.templates(q.family).items():
            try:
                out.append(self._compile(sp, q, ctx, nm))
            except ValidationError:
                pass
        return out[: self.max_cands]

    def _all_specs(self, family):
        specs = list(self.domain.templates(family).values()) + list(self.domain.unseen(family).values())
        keys = sorted({k for s in specs for k in s})
        values = {k: sorted({json.dumps(s.get(k)) for s in specs}) for k in keys}
        out, seen = [], set()
        for combo in itertools.product(*[values[k] for k in keys]):
            sp = {k: json.loads(v) for k, v in zip(keys, combo) if json.loads(v) is not None}
            c = canon(sp)
            if c not in seen:
                seen.add(c)
                out.append(sp)
        return out

    def _grid(self, q, ctx, req):
        out = self._templates(q, ctx)
        names = {canon(c.spec) for c in out}
        for nm, sp in self.domain.templates(q.family).items():
            for k, v in sp.items():
                for v2 in {json.dumps(s.get(k)) for s in self.domain.templates(q.family).values()}:
                    s2 = dict(sp, **{k: json.loads(v2)})
                    if canon(s2) in names:
                        continue
                    try:
                        out.append(self._compile(s2, q, ctx, f"grid{len(out)}"))
                        names.add(canon(s2))
                    except ValidationError:
                        pass
        return self._rank(q, ctx, out, req)

    def _search(self, q, ctx, req):
        comps = []
        for i, sp in enumerate(self._all_specs(q.family)):
            try:
                comps.append(self._compile(sp, q, ctx, f"s{i}"))
            except ValidationError:
                pass
        return self._rank(q, ctx, comps, req)

    def _rank(self, q, ctx, comps, req):
        if len(comps) <= self.max_cands:
            return comps
        preds = self._predict(q, ctx, comps)
        P = np.array([[p["coverage_median"], p["cost_usd_median"], p["time_s_median"]] for p in preds])
        from ..core.options import efficient
        eff = list(efficient(P))
        rest = [i for i in range(len(comps)) if i not in eff]
        eff.sort(key=lambda i: (-int(P[i, 0] >= req["target"]), P[i, 1]))
        rest.sort(key=lambda i: -P[i, 0])
        keep = (eff + rest)[: self.max_cands]
        return [comps[i] for i in keep]

    def _llm_call(self, system, user):
        t0 = time.time()
        try:
            out = self.llm.json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                max_tokens=2500)
        except ValueError:
            out = {}
        self.calls += 1
        self.seconds += time.time() - t0
        specs = out.get("candidates", []) if isinstance(out, dict) else []
        return [sp for sp in specs if isinstance(sp, dict)]

    def _compile_specs(self, specs, q, ctx, prefix, keys):
        comps = []
        for i, sp in enumerate(specs):
            if canon(sp) in keys or (sp.get("items") is not None and json.dumps(sp, sort_keys=True) in keys):
                continue
            try:
                comps.append(self._compile(sp, q, ctx, f"{prefix}{i}"))
                keys.add(canon(sp) if sp.get("items") is None else json.dumps(sp, sort_keys=True))
            except ValidationError:
                pass
        return comps

    def _llm_compose(self, q, ctx, req):
        voc = self.domain.vocabulary()
        grammar = json.dumps(voc["spec_fields"], indent=1)
        examples = json.dumps(list(self.domain.templates(q.family).values())[:2])
        scope = self.tools.resolver(q, req)["summary"]
        user = (f"Request: {q.text}\nParsed requirement: {json.dumps(req, default=str)}\n"
                f"Scope: {json.dumps(scope, default=str)}\nCatalogue: {self.tools.catalogue(q, ctx)}")
        sys_c = SYSTEM_COMPOSE.format(grammar=grammar + "\nExamples: " + examples, m=self.max_cands,
                                      target=req["target"])
        keys = set()
        comps = self._compile_specs(self._llm_call(sys_c, user), q, ctx, "agent", keys)[: self.max_cands]
        for rnd in range(self.rounds):
            if not comps:
                break
            preds = self._predict(q, ctx, comps)
            P = np.array([[p["coverage_median"], -p["cost_usd_median"], -p["time_s_median"]] for p in preds])
            beaten = [i for i in range(len(comps))
                      if any(np.all(P[j] > P[i]) for j in range(len(comps)) if j != i)]
            comps = [c for i, c in enumerate(comps) if i not in beaten]
            preds = [p for i, p in enumerate(preds) if i not in beaten]
            reach = [i for i, p in enumerate(preds) if p["coverage_median"] >= req["target"]]
            if not reach:
                table = [dict(spec={k: v for k, v in c.spec.items() if k != "items"}, **p) for c, p in zip(comps, preds)]
                sys_r = SYSTEM_REVISE.format(grammar=grammar, m=max(1, self.max_cands - len(comps)),
                                             target=req["target"])
                new = self._llm_call(sys_r, user + "\nPredictions: " + json.dumps(table, default=float))
                comps += self._compile_specs(new, q, ctx, f"rev{rnd}_", keys)
            if reach and self.use_readset:
                for i in reach:
                    c = comps[i]
                    if len(c.read_set) < 2 or c.spec.get("items") is not None:
                        continue
                    res = self._readset(q, ctx, c, req["target"])
                    if 0 < len(res["S"]) < len(c.read_set):
                        comps += self._compile_specs([dict(c.spec, items=[int(x) for x in res["S"]])], q, ctx,
                                                     f"rs{rnd}_", keys)
        self.trace.append(dict(qid=q.qid, n_specs=len(comps), calls=self.calls))
        if not comps:
            comps = self._templates(q, ctx)
        return self._rank(q, ctx, comps, req)

    def run(self, q: Request, ctx=None) -> tuple:
        req = self.interpret(q)
        if ctx is None:
            ctx = self.tools.resolver(q, req)["ctx"]
        comps = self.compose(q, req, ctx)
        for i, c in enumerate(comps):
            c.name = f"{c.template}#{i}" if c.template != "composed" else f"composed#{i}"
        return req, comps
