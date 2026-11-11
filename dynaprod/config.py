from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WORKER = dict(vcpu=4, mem_gb=16.0, io_mb_s=180.0, usd_per_cpu_s=0.0504 / 3600,
              usd_per_gb_in=0.01, usd_per_gb_out=0.09, transient_rate=0.03)

OMEGA = {
    "select": 2e-3, "read": 0.0, "clean": 4e-8, "composite": 3e-8, "fuse": 2e-8, "fill": 1.2e-7,
    "pool": 5e-9, "export": 1e-8, "join": 4e-7, "extract": 0.0, "vote": 2e-6, "coarsen": 1e-7,
    "resample": 1.5e-8, "index": 2e-8,
}

LM = {"cheap": dict(usd=0.15e-6, tps=6000.0), "accurate": dict(usd=3.0e-6, tps=1500.0)}

GRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
ALPHA_Y, ALPHA_C, ALPHA_T = 0.10, 0.05, 0.05
MIN_STRATUM = 30
GAMMA = 0.02
TAIL = {"C": 0.1, "T": 0.1}

LEVELS = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
ITEM_GBM = dict(max_depth=2, num_leaves=4, n_estimators=300, learning_rate=0.05, min_child_samples=20)

DAY = 24 * 3600.0


@dataclass
class WMConfig:
    M: int = 200
    ablate: frozenset = frozenset()
    independent: bool = False
    use_h0_for_all: bool = False
    seed: int = 0
    obs_error_model: bool = True
