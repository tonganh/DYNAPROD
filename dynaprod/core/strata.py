from __future__ import annotations


from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ANY = "*"


@dataclass
class Stratification:
    fields: Tuple[str, ...] = ("family", "size")
    coarsen: Tuple[str, ...] = ("size",)
    min_n: int = 30
    valid: Dict[tuple, int] = field(default_factory=dict)

    def __post_init__(self):
        if "family" not in self.fields:
            raise ValueError("strata are always split by family")
        if "family" in self.coarsen:
            raise ValueError("strata never pool across families")
        if any(c not in self.fields for c in self.coarsen):
            raise ValueError("coarsen must name fields of the stratification")

    def levels(self) -> List[Tuple[str, ...]]:
        out, merged = [], []
        out.append(tuple(f for f in self.fields))
        for c in self.coarsen:
            merged.append(c)
            out.append(tuple(f for f in self.fields if f not in merged))
        return out

    def _key(self, rec: dict, kept: Sequence[str]) -> tuple:
        return tuple(str(rec.get(f, ANY)) if f in kept else ANY for f in self.fields)

    def fit(self, entries: Iterable[dict]) -> "Stratification":
        entries = list(entries)
        levels = self.levels()[::-1]
        self.valid = {}

        def split(recs: List[dict], depth: int):
            key = self._key(recs[0], levels[depth])
            if depth + 1 < len(levels):
                kids: Dict[tuple, List[dict]] = {}
                for r in recs:
                    kids.setdefault(self._key(r, levels[depth + 1]), []).append(r)
                if len(kids) > 1 and all(len(v) >= self.min_n for v in kids.values()):
                    for v in kids.values():
                        split(v, depth + 1)
                    return
                if len(kids) == 1:
                    split(next(iter(kids.values())), depth + 1)
                    return
            self.valid[key] = len(recs)

        fams: Dict[tuple, List[dict]] = {}
        for e in entries:
            fams.setdefault(self._key(e, levels[0]), []).append(e)
        for recs in fams.values():
            if len(recs) >= self.min_n:
                split(recs, 0)
        return self

    def key(self, rec: dict) -> Optional[tuple]:
        for kept in self.levels():
            k = self._key(rec, kept)
            if k in self.valid:
                return k
        return None

    def describe(self) -> str:
        rows = [f"{'/'.join(k)}: n={n}" for k, n in sorted(self.valid.items())]
        return f"strata over {self.fields} (min {self.min_n}): " + ("; ".join(rows) if rows else "none")


def family_size(min_n: int = 30) -> Stratification:
    return Stratification(("family", "size"), ("size",), min_n)


def family_group(min_n: int = 30) -> Stratification:
    return Stratification(("family", "group", "size"), ("group", "size"), min_n)
