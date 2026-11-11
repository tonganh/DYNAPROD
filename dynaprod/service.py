from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .core.audit import AuditRecord, AuditStream, audit
from .core.calibration import GRID, Calibrator, F_left, MenuCalibrator, OnlineLevel, coherent
from .core.containment import certified_pairs, transitive_closure
from .core.observation import ObservationPolicy
from .core.options import option_set
from .core.types import Compiled, Outcome, Request, Rollouts
from .config import DAY, GAMMA
from .utils import rng_for


@dataclass
class PlanningSnapshot:
    period: int
    wm: object
    calibrator: Calibrator
    observation: ObservationPolicy = field(default_factory=ObservationPolicy)
    agent_kw: Dict = field(default_factory=dict)
    presented: Optional[Calibrator] = None
    menu: Optional[MenuCalibrator] = None
    M: int = 200
    m: int = 8
    seed: int = 0
    frozen_at: float = 0.0


@dataclass
class OptionQuote:
    index: int
    name: str
    spec: Dict
    coverage: float
    cost: float
    time: float
    failure_prob: float
    ell: float
    UC: float
    UT: float
    shortfall_risk: Dict[float, float]
    F_hat: Dict[float, float]
    ell_menu: Optional[float] = None


@dataclass
class Quote:
    qid: str
    request: Request
    requirement: Dict
    options: List[OptionQuote]
    candidates: List[Compiled]
    rollouts: Rollouts
    level: float
    period: int
    created_at: float
    expires_at: float
    closure: np.ndarray
    accepted: Optional[int] = None
    delivered: bool = False

    def valid(self, now: float) -> bool:
        return now <= self.expires_at

    def option(self, index: int) -> OptionQuote:
        for o in self.options:
            if o.index == index:
                return o
        raise KeyError(f"candidate {index} is not an option of quote {self.qid}")


class QuoteExpired(Exception):
    pass


class DynaProd:
    def __init__(self, domain, snapshot: PlanningSnapshot, gamma: float = GAMMA, ttl: float = DAY, agent=None):
        self.domain = domain
        self.snapshot = snapshot
        self.ttl = ttl
        self.online = OnlineLevel(snapshot.calibrator.aY, gamma)
        self.agent = agent or self._make_agent()
        self.stream = AuditStream()
        self.quotes: Dict[str, Quote] = {}
        self.open: set = set()
        self.max_open = 0

    def _make_agent(self):
        from .agent.agent import DataAgent
        from .llm.openrouter import available
        kw = dict(mode="llm", parse="llm") if available() else dict(mode="templates", parse="rule")
        kw.update(self.snapshot.agent_kw)
        return DataAgent(self.domain, self.snapshot.wm, seed=self.snapshot.seed, **kw)

    def closure(self, ctx, cands: Sequence[Compiled]) -> np.ndarray:
        n_units = {"fine": ctx.n_units, **{k: v[1] for k, v in ctx.pool_maps.items()}}
        pairs = certified_pairs([c.root for c in cands], [c.space for c in cands], n_units, ctx.unitmap,
                                ctx.pool_inv())
        return transitive_closure(pairs, len(cands))

    def estimate(self, q: Request, now: float = 0.0, candidates: Optional[List[Compiled]] = None) -> Quote:
        snap, dom = self.snapshot, self.domain
        if candidates is None:
            req, cands = self.agent.run(q)
        else:
            req, cands = dict(q.req), list(candidates)
        ctx = dom.context(q)
        cands = [c for c in cands if c.root is not None]
        if not cands:
            raise ValueError(f"no valid candidate for request {q.qid}")
        R = snap.wm.rollout(q, ctx, cands, M=snap.M, seed=snap.seed, observe=snap.observation)
        groups = [dom.source_group(c) for c in cands]
        a_t = self.online.level()
        b = snap.calibrator.bounds(q.family, q.size_class, R.Y, R.C, R.T, alpha_Y=a_t, groups=groups)
        Rc = self.closure(ctx, cands)
        ell = coherent(b.ell, Rc)
        opts = [int(i) for i in option_set(ell, b.UC, b.UT, snap.m)]
        UC, UT, risk = b.UC, b.UT, b.risk
        if snap.presented is not None:
            b2 = snap.presented.bounds(q.family, q.size_class, R.Y, R.C, R.T, alpha_Y=a_t, groups=groups)
            ell, UC, UT, risk = coherent(b2.ell, Rc), b2.UC, b2.UT, b2.risk
        menu = snap.menu.bounds(R.Yplus) if snap.menu is not None else None
        grid = snap.calibrator.grid
        options = []
        for k in opts:
            options.append(OptionQuote(
                index=k, name=cands[k].name, spec={kk: v for kk, v in cands[k].spec.items() if kk != "items"},
                coverage=float(R.Y[k].mean()), cost=float(np.median(R.C[k])), time=float(np.median(R.T[k])),
                failure_prob=float(R.E[k].mean()), ell=float(ell[k]), UC=float(UC[k]), UT=float(UT[k]),
                shortfall_risk={float(g): float(r) for g, r in zip(grid, risk[k])},
                F_hat={float(g): float(F_left(R.Y[k], g)) for g in grid},
                ell_menu=float(menu[k]) if menu is not None else None))
        quote = Quote(q.qid, q, req, options, cands, R, a_t, snap.period, now, now + self.ttl, Rc)
        self.quotes[q.qid] = quote
        return quote

    def accept(self, quote: Quote, index: int, now: float) -> OptionQuote:
        if not quote.valid(now):
            raise QuoteExpired(f"quote {quote.qid} expired at {quote.expires_at:.0f}s")
        opt = quote.option(index)
        quote.accepted = index
        self.open.add(quote.qid)
        self.max_open = max(self.max_open, len(self.open))
        return opt

    def deliver(self, quote: Quote, outcome: Outcome) -> int:
        if quote.accepted is None or quote.delivered:
            raise ValueError("deliver an accepted, undelivered quote")
        err = int(outcome.Y < quote.option(quote.accepted).ell - 1e-12)
        self.online.update(err)
        quote.delivered = True
        self.open.discard(quote.qid)
        return err

    def build(self, quote: Quote, rng: Optional[np.random.Generator] = None) -> Outcome:
        ctx = self.domain.context(quote.request)
        rng = rng or rng_for(self.snapshot.seed, quote.qid, "build")
        return self.domain.execute(ctx, quote.candidates[quote.accepted], rng)

    def audit(self, q: Request, stage: str = "candidate", menu: bool = False,
              candidates: Optional[List[Compiled]] = None) -> AuditRecord:
        quote = self.estimate(q, candidates=candidates)
        ctx = self.domain.context(q)
        rng = rng_for(self.snapshot.seed, self.snapshot.period, q.qid, "audit", stage)
        pool = list(range(len(quote.candidates))) if stage == "candidate" else [o.index for o in quote.options]
        k = int(pool[int(rng.integers(len(pool)))])
        out = self.domain.execute(ctx, quote.candidates[k], rng)
        R = quote.rollouts
        rec = AuditRecord(q, q.family, q.size_class, self.domain.source_group(quote.candidates[k]), k,
                          R.Y[k], R.C[k], R.T[k], out.Y, out.C, out.T, stage)
        if menu:
            rec.Yplus_all = R.Yplus
            rec.certificates = audit(self.domain, ctx, quote.candidates).certificates
        self.stream.add(rec)
        return rec

    def next_period(self, now: float = 0.0, keep_scores_if_empty: bool = True) -> PlanningSnapshot:
        old = self.snapshot
        cal_old = old.calibrator
        entries = self.stream.entries("candidate")
        rng = rng_for(old.seed, old.period + 1, "calib-u")
        cal = cal_old
        if entries or not keep_scores_if_empty:
            cal = Calibrator(cal_old.grid, cal_old.aY, cal_old.aC, cal_old.aT, min_stratum=cal_old.min_stratum,
                             strata=copy.deepcopy(cal_old.strata)).fit(entries, rng)
        presented = old.presented
        opt_entries = self.stream.entries("option")
        if opt_entries:
            presented = Calibrator(cal.grid, cal.aY, cal.aC, cal.aT, min_stratum=cal.min_stratum,
                                   strata=copy.deepcopy(cal_old.strata)).fit(opt_entries, rng)
        menu = old.menu
        if self.stream.menu_entries():
            menu = MenuCalibrator(cal.grid, cal.aY).fit(self.stream.menu_entries(), rng)
        self.snapshot = PlanningSnapshot(old.period + 1, old.wm, cal, old.observation, dict(old.agent_kw), presented,
                                         menu, old.M, old.m, old.seed, now)
        self.online = OnlineLevel(cal.aY, self.online.gamma)
        self.stream = AuditStream()
        return self.snapshot

    def rescale_rates(self, compute: Optional[float] = None, tokens: Optional[float] = None,
                      transfer: Optional[float] = None):
        self.snapshot.wm.set_rates(compute, tokens, transfer)
