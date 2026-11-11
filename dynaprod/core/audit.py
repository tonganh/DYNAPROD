from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .lineage import EvalContext


@dataclass
class AuditResult:
    certificates: np.ndarray
    items_read: List[int]
    mb_read: float
    union_mb: float
    full: bool


def audit(domain, ctx, cands: Sequence, ratio: float = 1.0, rng: Optional[np.random.Generator] = None) -> AuditResult:
    xs = sorted(set().union(*[set(c.read_set) for c in cands])) if cands else []
    mb = {x: (domain.item_mb(ctx, x) if hasattr(domain, "item_mb") else 1.0) for x in xs}
    union_mb = float(sum(mb.values()))
    read = list(xs)
    if ratio < 1.0:
        rng = rng if rng is not None else np.random.default_rng(0)
        budget, read, spent = ratio * union_mb, [], 0.0
        for x in rng.permutation(xs):
            if spent + mb[x] <= budget + 1e-12:
                read.append(int(x))
                spent += mb[x]
    ev = domain.truth_evidence(ctx, read)
    cert = np.array([float(EvalContext(ev, ctx.unitmap, ctx.n_units, 1, ctx.pool_maps, domain.true_fill_for(ctx, c))
                           .eval(c.root).mean()) if c.root is not None else 0.0 for c in cands])
    return AuditResult(cert, [int(x) for x in read], float(sum(mb[x] for x in read)), union_mb, ratio >= 1.0)


def audit_certificates(domain, ctx, cands, ratio: float = 1.0, rng=None) -> np.ndarray:
    return audit(domain, ctx, cands, ratio, rng).certificates


@dataclass
class AuditRecord:
    request: object
    family: str
    size: str
    group: str
    k: int
    Y: np.ndarray
    C: np.ndarray
    T: np.ndarray
    y: float
    c: float
    t: float
    stage: str = "candidate"
    Yplus_all: Optional[np.ndarray] = None
    certificates: Optional[np.ndarray] = None

    def entry(self) -> dict:
        return dict(family=self.family, size=self.size, group=self.group, Y=self.Y, C=self.C, T=self.T,
                    y=self.y, c=self.c, t=self.t)


@dataclass
class AuditStream:
    records: List[AuditRecord] = field(default_factory=list)
    exclude_from_refit: set = field(default_factory=set)

    def add(self, rec: AuditRecord):
        self.records.append(rec)
        self.exclude_from_refit.add(rec.request.qid)

    def entries(self, stage: str = "candidate") -> List[dict]:
        return [r.entry() for r in self.records if r.stage == stage]

    def menu_entries(self) -> List[dict]:
        return [dict(Yplus=r.Yplus_all, y=r.certificates) for r in self.records
                if r.Yplus_all is not None and r.certificates is not None]

    def clear(self):
        self.records.clear()
