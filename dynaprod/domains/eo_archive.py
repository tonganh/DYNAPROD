from __future__ import annotations

import math
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..core.field import FieldSampler
from ..utils import CACHE, rng_for

SOURCES = ["S2", "L8", "L9", "S1"]
REVISIT = {"S2": 5, "L8": 16, "L9": 16, "S1": 6}
HAZE = {"S2": 0.02, "L8": 0.03, "L9": 0.03, "S1": 0.03}
LARGE_SIDE_KM = 100


class MaskArchive:

    def __init__(self, root: str):
        self.root = root
        table = pd.read_csv(os.path.join(root, "aois.csv"))
        table = pd.concat([table[table.side_km < LARGE_SIDE_KM], table[table.side_km >= LARGE_SIDE_KM]])
        self.aois = [dict(id=str(r.id), region=r.region, side=float(r.side_km)) for r in table.itertuples()]
        self.n_large_aoi = int((table.side_km >= LARGE_SIDE_KM).sum())
        self.n_aoi = len(self.aois) - self.n_large_aoi
        raw = [np.load(os.path.join(root, "scenes", f"{A['id']}.npz"), allow_pickle=False) for A in self.aois]
        self.day0 = min(_days(d["day"]).min() for d in raw if len(d["day"]))
        self._scenes = {a: self._scenes_of(a, d) for a, d in enumerate(raw)}
        sizes = {len(np.unpackbits(s["mask"])) for sc in self._scenes.values() for s in sc[:1]}
        self.G = int(round(math.sqrt(min(sizes)))) if sizes else 32
        self.horizon = 1 + max(s["day"] for sc in self._scenes.values() for s in sc)
        self._catalogue = self._catalogue_by_region()
        path = os.path.join(root, "activations.csv")
        self.activations = None
        if os.path.exists(path):
            acts = pd.read_csv(path)
            acts["day"] = _days(acts["day"].to_numpy()) - self.day0
            self.activations = acts

    def _scenes_of(self, a: int, d) -> list:
        aoi = self.aois[a]["id"]
        clear = np.asarray(d["clear"], bool).reshape(len(d["day"]), -1)
        G = int(round(math.sqrt(clear.shape[1])))
        if G * G != clear.shape[1]:
            raise ValueError(f"{aoi}: clear masks must hold G*G cells, got {clear.shape[1]}")
        unknown = set(map(str, d["source"])) - set(SOURCES)
        if unknown:
            raise ValueError(f"{aoi}: unknown sources {sorted(unknown)}, expected {SOURCES}")
        days = _days(d["day"]) - self.day0
        cf = np.nan_to_num(np.asarray(d["cf"], float), nan=0.0)
        return [dict(source=str(s), day=int(t), cf=float(c), footprint=float(f), mask=np.packbits(m))
                for s, t, c, f, m in zip(d["source"], days, cf, d["footprint"], clear)]

    def scenes(self, a: int) -> list:
        return self._scenes[a]

    def _catalogue_by_region(self) -> dict:
        by_region = {}
        for a, A in enumerate(self.aois):
            by_region.setdefault(A["region"], []).extend((s["day"], s["cf"]) for s in self._scenes[a] if s["source"] != "S1")
        return {r: tuple(np.array(v, float) for v in zip(*obs)) for r, obs in by_region.items() if obs}

    def climatology(self, region, day: float, before: Optional[float] = None) -> float:
        days, cf = self._catalogue.get(region, (np.zeros(0), np.zeros(0)))
        if before is not None:
            days, cf = days[days < before], cf[days < before]
        if len(days) == 0:
            return 0.5
        gap = np.abs(days % 365 - day % 365)
        near = np.minimum(gap, 365 - gap) <= 30
        return float(np.clip(cf[near].mean() if near.any() else cf.mean(), 0.03, 0.97))


class SyntheticArchive:

    activations = None

    def __init__(self, seed: int, n_aoi: int, G: int, large: bool, horizon: int = 500):
        self.seed, self.n_aoi, self.G, self.horizon = seed, n_aoi, G, horizon
        rng = rng_for(seed, "eo-archive")
        n_reg = max(4, n_aoi // 6)
        self.regions = [dict(base=rng.uniform(0.25, 0.65), amp=rng.uniform(0.05, 0.3), phase=rng.uniform(0, 365),
                             xy=rng.uniform(0, 1000, 2), L_km=rng.uniform(4, 14)) for _ in range(n_reg)]
        sides = np.array([5, 8, 10, 15, 20, 30, 40, 50])
        self.aois = []
        for a in range(n_aoi):
            side = float(rng.choice(sides, p=[.16, .14, .16, .14, .14, .1, .08, .08]))
            self.aois.append(dict(id=f"A{a:03d}", region=a % n_reg, side=side, off=rng.integers(0, G + 1, 2),
                                  s2_off=int(rng.integers(0, 5)), l8_off=int(rng.integers(0, 16)),
                                  s1_off=int(rng.integers(0, 6))))
        self.n_large_aoi = max(8, n_aoi // 4) if large else 0
        for j in range(self.n_large_aoi):
            lr = rng_for(seed, "eo-large-aoi", j)
            self.aois.append(dict(id=f"X{j:03d}", region=int(lr.integers(n_reg)),
                                  side=float(lr.choice([100, 150, 200, 300, 400, 500])),
                                  off=lr.integers(0, G + 1, 2), s2_off=int(lr.integers(0, 5)),
                                  l8_off=int(lr.integers(0, 16)), s1_off=int(lr.integers(0, 6))))
        self._scenes: Dict[int, list] = {}

    def climatology(self, region: int, day: float, before: Optional[float] = None) -> float:
        R = self.regions[region]
        return float(np.clip(R["base"] + R["amp"] * np.cos(2 * np.pi * (day - R["phase"]) / 365.0), 0.03, 0.97))

    def scenes(self, a: int) -> list:
        if a in self._scenes:
            return self._scenes[a]
        name = f"aoi{a}.pkl" if a < self.n_aoi else f"large{a - self.n_aoi}.pkl"
        path = os.path.join(CACHE, f"eo_s{self.seed}_G{self.G}_n{self.n_aoi}", name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                self._scenes[a] = pickle.load(f)
            return self._scenes[a]
        sc = self._simulate(a)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(sc, f)
        self._scenes[a] = sc
        return sc

    def _simulate(self, a: int) -> list:
        A = self.aois[a]
        R = self.regions[A["region"]]
        G = self.G
        rng = rng_for(self.seed, "aoi", a)
        sampler = FieldSampler(np.zeros((1, 2)), R["L_km"], grid=(2 * G, 2 * G), spacing=A["side"] / G)
        acq = {}
        for d in range(self.horizon):
            for s in SOURCES:
                off = {"S2": A["s2_off"], "L8": A["l8_off"], "L9": (A["l8_off"] + 8) % 16, "S1": A["s1_off"]}[s]
                if (d - off) % REVISIT[s] == 0:
                    acq.setdefault(d, []).append(s)
        zeta = sampler.sample(1, rng)[0]
        eta = 0.0
        out = []
        oy, ox = A["off"]
        rho_day = 0.55
        for d in range(self.horizon):
            zeta = rho_day * zeta + math.sqrt(1 - rho_day ** 2) * sampler.sample(1, rng)[0]
            eta = 0.6 * eta + math.sqrt(1 - 0.36) * rng.normal() * 0.9
            if d not in acq:
                continue
            f = float(norm.cdf(norm.ppf(self.climatology(A["region"], d)) + eta))
            thr = np.quantile(zeta, 1 - f)
            cloud = (zeta > thr).reshape(2 * G, 2 * G)[oy:oy + G, ox:ox + G].reshape(-1)
            for s in acq[d]:
                foot = np.ones(G * G, bool)
                if rng.random() < 0.12:
                    ang = rng.uniform(0, 2 * np.pi)
                    yy, xx = np.mgrid[0:G, 0:G]
                    proj = (np.cos(ang) * xx + np.sin(ang) * yy).reshape(-1)
                    foot = proj <= np.quantile(proj, rng.uniform(0.3, 0.9))
                haze = rng.random(G * G) < HAZE[s]
                if s == "S1":
                    clear, cf = foot & ~haze, 0.0
                else:
                    clear = foot & ~cloud & ~haze
                    cf = float(np.clip(f + rng.normal(0, 0.03), 0, 1))
                out.append(dict(source=s, day=d, cf=cf, footprint=float(foot.mean()), mask=np.packbits(clear)))
        return out


def _days(values) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return values.astype(int)
    return (pd.to_datetime(values).values.astype("datetime64[D]").astype(int)).astype(int)


def load_archive(root: Optional[str], seed: int, n_aoi: int, G: int, large: bool):
    if root and os.path.exists(os.path.join(root, "aois.csv")):
        return MaskArchive(root)
    return SyntheticArchive(seed, n_aoi, G, large)
