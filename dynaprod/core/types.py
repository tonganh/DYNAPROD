from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Request:
    qid: str
    domain: str
    family: str
    cluster: str
    t: float
    text: str
    req: Dict[str, Any]
    split: str = "train"
    size_class: str = "small"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpSpec:
    op: str
    psi: Dict[str, float]
    bytes: float
    records: Any
    transfer_bytes: float = 0.0


@dataclass
class Compiled:
    name: str
    spec: Dict[str, Any]
    template: str
    seen: bool
    root: Any
    space: str
    read_set: List[int]
    ops: List[OpSpec]
    features: Dict[str, float] = field(default_factory=dict)
    output_units: int = 0


@dataclass
class RequestContext:
    request: Request
    items: pd.DataFrame
    n_units: int
    cell_coords: np.ndarray
    scope: Dict[int, np.ndarray]
    unitmap: Dict[int, Dict[str, np.ndarray]]
    pool_maps: Dict[str, tuple] = field(default_factory=dict)
    adjacency: Dict[str, Any] = field(default_factory=dict)
    group: Dict[int, Any] = field(default_factory=dict)
    item_area: Dict[int, float] = field(default_factory=dict)
    unit_weight_space: Dict[str, int] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def n_space(self, space: str) -> int:
        return self.n_units if space == "fine" else self.pool_maps[space][1]

    def pool_inv(self) -> Dict[str, List[np.ndarray]]:
        out = {}
        for name, (cmap, n) in self.pool_maps.items():
            inv = [[] for _ in range(n)]
            for f, c in enumerate(cmap):
                if c >= 0:
                    inv[c].append(f)
            out[name] = [np.array(v, int) for v in inv]
        return out


@dataclass
class Outcome:
    Yplus: float
    Y: float
    C: float
    T: float
    E: int
    log: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rollouts:
    Yplus: np.ndarray
    Y: np.ndarray
    C: np.ndarray
    T: np.ndarray
    E: np.ndarray
    seconds: float = 0.0
    bytes_read: float = 0.0
