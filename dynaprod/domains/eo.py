from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
from scipy import sparse

from ..core.lineage import AllOf, AnyOf, AtLeast, Ev, Fill, Pool
from ..core.types import Compiled, OpSpec, Request, RequestContext
from ..utils import rng_for
from .base import Domain, ValidationError
from .eo_archive import SOURCES, load_archive
from ..core.compute import WORKER


NATIVE_RES = {"S2": 10, "L8": 30, "L9": 30, "S1": 10}
BANDS = {"S2": 4, "L8": 4, "L9": 4, "S1": 2}
FAMILIES = ["rapid", "monitoring", "composite"]

TEMPLATES = {
    "rapid": {
        "T1_latest_clear_s2": dict(sources=["S2"], rule="any", resolution_m=10),
        "T2_latest_clear_landsat": dict(sources=["L"], rule="any", resolution_m=30),
        "T3_radar_flood_s1": dict(sources=["S1"], rule="any", resolution_m=10),
        "T4_single_best_s2": dict(sources=["S2"], rule="any", best_only=True, resolution_m=10),
    },
    "monitoring": {
        "T5_timeseries_s2": dict(sources=["S2"], rule="count", resolution_m=10),
        "T6_timeseries_landsat": dict(sources=["L"], rule="count", resolution_m=30),
        "T7_timeseries_s2_gapfill": dict(sources=["S2"], rule="count", fill="temporal", resolution_m=10),
        "T8_resample_timeseries_s2": dict(sources=["S2"], rule="count", resolution_m=20),
    },
    "composite": {
        "T9_median_s2": dict(sources=["S2"], rule="median", resolution_m=10),
        "T10_best_pixel_s2": dict(sources=["S2"], rule="any", resolution_m=10),
        "T11_best_pixel_landsat": dict(sources=["L"], rule="any", resolution_m=30),
        "T12_max_ndvi_s2": dict(sources=["S2"], rule="any", index="ndvi", resolution_m=10),
    },
}
UNSEEN = {
    "rapid": {
        "U1_fuse_latest_clear_s2_l": dict(sources=["S2", "L"], rule="any", resolution_m=10),
        "U6_latest_clear_s2_radar_fill": dict(sources=["S2"], rule="any", fill="radar", resolution_m=10),
    },
    "monitoring": {
        "U2_fuse_timeseries_gapfill_s2_l": dict(sources=["S2", "L"], rule="count", fill="temporal", resolution_m=10),
        "UL_timeseries_landsat_gapfill": dict(sources=["L"], rule="count", fill="temporal", resolution_m=30),
    },
    "composite": {
        "U3_fuse_best_pixel_s2_l": dict(sources=["S2", "L"], rule="any", resolution_m=10),
        "U4_best_pixel_s2_radar_fill": dict(sources=["S2"], rule="any", fill="radar", resolution_m=10),
    },
}
GROUPS = {"S2": ["S2"], "L": ["L8", "L9"], "L8": ["L8"], "L9": ["L9"], "S1": ["S1"]}
GROUP_OF = {"S2": "S2", "L8": "L", "L9": "L", "S1": "S1"}


class EarthObservation(Domain):
    name = "eo"
    families = FAMILIES
    psi_keys = ["n_inputs", "n_sources", "remote_store", "resolution", "radar", "cached", "method_temporal", "method_radar",
                "rule_median", "area_km2"]
    e_cols = ["e", "cf", "footprint"]
    fill_reference = {"spatial": (-2.2, 3.4), "temporal": (-2.0, 3.0), "radar": (-1.2, 2.6)}
    fill_fallback = {"radar": "temporal", "spatial": "temporal"}
    field_scales = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    time_scales = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    has_observations = True

    def __init__(self, seed=0, quick=False, n_aoi=None, per_family=None, G=None, month_days=30,
                 overview_factor=4, overview_noise=0.075, reduced=False, large=0, archive=None,
                 range_read_s=0.6):
        super().__init__(seed, quick)
        self.reduced = reduced
        self.large = int(large)
        if reduced and per_family is None:
            per_family = 40 if quick else 100
        if reduced:
            self.name = "eo_reduced"
        self.per_family = per_family or (60 if quick else 200)
        self.month_days = month_days
        self.ov_f = overview_factor
        self.ov_noise = overview_noise
        self.range_read_s = range_read_s
        self.archive = load_archive(archive, seed, n_aoi or (60 if quick else 117), G or (16 if quick else 32),
                                    bool(self.large))
        self.G, self.horizon = self.archive.G, self.archive.horizon
        self.aois, self.n_aoi, self.n_large_aoi = self.archive.aois, self.archive.n_aoi, self.archive.n_large_aoi
        self._aoi_by_id = {A["id"]: a for a, A in enumerate(self.aois)}

    def climatology(self, region, day: float, before=None) -> float:
        return self.archive.climatology(region, day, before)

    def scenes(self, a: int) -> list:
        return self.archive.scenes(a)

    def requests(self) -> List[Request]:
        if hasattr(self, "_requests"):
            return self._requests
        rng = rng_for(self.seed, "eo-requests")
        reqs = []
        small = [i for i in range(self.n_aoi) if self.aois[i]["side"] <= 20]
        replayed = self._replayed_activations()
        for fam in FAMILIES:
            if fam == "rapid" and replayed:
                reqs += [self._make_request(fam, i, a, rng, "std", day=day, hazard=hz)
                         for i, (a, day, hz) in enumerate(replayed)]
                continue
            for i in range(self.per_family):
                a = small[int(rng.integers(len(small)))] if fam == "rapid" else int(rng.integers(self.n_aoi))
                reqs.append(self._make_request(fam, i, a, rng, "std"))
        if self.large:
            lrng = rng_for(self.seed, "eo-large-requests")
            for fam in ("monitoring", "composite"):
                for i in range(self.large):
                    a = self.n_aoi + int(lrng.integers(self.n_large_aoi))
                    reqs.append(self._make_request(fam, f"xl{i}", a, lrng, "xl"))
        self._requests = reqs
        return reqs

    def _replayed_activations(self) -> list:
        acts = self.archive.activations
        if acts is None:
            return []
        return [(self._aoi_by_id[str(r.aoi)], float(r.day), str(r.hazard)) for r in acts.itertuples()
                if str(r.aoi) in self._aoi_by_id][: self.per_family]

    def _make_request(self, fam, i, a, rng, size_class, day=None, hazard=None):
        A = self.aois[a]
        t = float(rng.integers(min(140, self.horizon // 3), self.horizon - 20)) if day is None else day
        target = float(rng.choice([0.6, 0.7, 0.8, 0.9]))
        floors = {}
        if fam == "rapid":
            N = int(rng.integers(3, 15))
            win = [int(t - N // 2), int(t + (N - N // 2))]
            deadline = 3600.0
            floors["max_resolution_m"] = int(rng.choice([10, 20]))
            hazard = hazard or str(rng.choice(["flood", "burned-area"]))
        elif fam == "monitoring":
            L = 1 if self.reduced else int(rng.integers(3, 13))
            win = [int(t - L * self.month_days), int(t)]
            deadline = 86400.0
            floors["min_clear_obs"] = int(rng.choice([1, 2, 3], p=[.4, .4, .2]))
            floors["max_resolution_m"] = 20
            target = float(rng.choice([0.5, 0.6, 0.7, 0.8]))
            hazard = None
        else:
            L = 1 if self.reduced else int(rng.integers(1, 4))
            win = [int(t - L * self.month_days), int(t)]
            deadline = 86400.0
            floors["max_resolution_m"] = int(rng.choice([20, 60], p=[.7, .3]))
            hazard = None
        req = dict(family=fam, scope=dict(aoi=A["id"], side_km=A["side"], window=win),
                   target=target, budget=None, deadline=deadline, floors=floors, hazard=hazard)
        return Request(qid=f"{self.name}-{fam}-{i}", domain=self.name, family=fam, cluster=f"R{A['region']}", t=t,
                       text="", req=req, size_class=size_class, extra=dict(aoi=a))

    def describe_request(self, req: dict, rng=None) -> str:
        rng = rng or np.random.default_rng(0)
        s = req["scope"]
        fam = req["family"]
        budget = f" Our budget is about ${req['budget']:.2f}." if req.get("budget") else ""
        if fam == "rapid":
            n = s["window"][1] - s["window"][0]
            txt = (f"Urgent: we need a {req['hazard']} map of the {s['side_km']:.0f} km x {s['side_km']:.0f} km area "
                   f"{s['aoi']} from the latest clear optical or radar view between day {s['window'][0]} and day "
                   f"{s['window'][1]} ({n} days). At least {req['target']*100:.0f}% of the area must be usable, at "
                   f"{req['floors']['max_resolution_m']} m or finer, delivered within one hour.{budget}")
        elif fam == "monitoring":
            months = (s["window"][1] - s["window"][0]) // self.month_days
            txt = (f"Please set up a monitoring time series over our agricultural region {s['aoi']} "
                   f"({s['side_km']:.0f} km on a side) for the {months} months ending on day {s['window'][1]}. "
                   f"Each cell needs at least {req['floors']['min_clear_obs']} clear observation(s) per month, and "
                   f"we want that for {req['target']*100:.0f}% of cell-months. Delivery within a day is fine.{budget}")
        else:
            txt = (f"We would like a seasonal land-cover composite of area {s['aoi']} ({s['side_km']:.0f} km side) "
                   f"for the season from day {s['window'][0]} to day {s['window'][1]}, with at least "
                   f"{req['target']*100:.0f}% usable coverage at {req['floors']['max_resolution_m']} m resolution or "
                   f"better. Within a day.{budget}")
        return txt

    def parse_rule(self, text: str) -> dict:
        import re
        t = text.lower()
        fam = "rapid" if ("flood" in t or "burned" in t or "urgent" in t) else \
            ("monitoring" if "time series" in t or "monitoring" in t else "composite")
        aoi = re.search(r"\b(a\d{3})\b", t)
        side = re.search(r"\(?(\d+) km", t)
        days = [int(x) for x in re.findall(r"day (\d+)", t)]
        pct = re.search(r"(\d+)%", t)
        budget = re.search(r"\$([\d.]+)", t)
        res = re.search(r"(\d+) m\b", t)
        req = dict(family=fam, scope=dict(aoi=aoi.group(1).upper() if aoi else None,
                                          side_km=float(side.group(1)) if side else None),
                   target=float(pct.group(1)) / 100 if pct else 0.8,
                   budget=float(budget.group(1).rstrip(".")) if budget else None,
                   deadline=3600.0 if fam == "rapid" else 86400.0, floors={})
        if fam == "monitoring":
            months = re.search(r"(\d+) months", t)
            k = re.search(r"at least (\d+) clear", t)
            end = days[-1] if days else None
            m = int(months.group(1)) if months else 6
            req["scope"]["window"] = [end - m * self.month_days, end] if end is not None else None
            req["floors"]["min_clear_obs"] = int(k.group(1)) if k else 1
            req["floors"]["max_resolution_m"] = 20
        else:
            req["scope"]["window"] = days[:2] if len(days) >= 2 else None
            req["floors"]["max_resolution_m"] = int(res.group(1)) if res else 20
        if fam == "rapid":
            req["hazard"] = "flood" if "flood" in t else "burned-area"
        return req

    def _aoi_index(self, aoi_id: str) -> int:
        return self._aoi_by_id[aoi_id]

    def _build_context(self, q: Request) -> RequestContext:
        G = self.G
        a = q.extra["aoi"]
        A = self.aois[a]
        win = q.req["scope"]["window"]
        fam = q.family
        rows = []
        sc = self.scenes(a)
        for j, s in enumerate(sc):
            if win[0] <= s["day"] <= win[1]:
                rows.append(dict(item=a * 10000 + j, source=s["source"], time=float(s["day"]),
                                 cf=s["cf"], footprint=s["footprint"], local=j))
        items = pd.DataFrame(rows).set_index("item") if rows else pd.DataFrame(
            columns=["source", "time", "cf", "footprint", "local"])
        exists = items["time"] <= q.t
        items["exists"] = exists
        items["e"] = np.where(items["source"] == "S1", 0.97 * items["footprint"],
                              (1 - items["cf"]) * items["footprint"])
        clim = np.array([self.climatology(A["region"], d, before=q.t) for d in items["time"]]) if len(items) else np.zeros(0)
        items["clim"] = clim
        cell_km = A["side"] / G
        yy, xx = np.mgrid[0:G, 0:G]
        coords = np.stack([yy.reshape(-1), xx.reshape(-1)], 1) * cell_km
        n_cells = G * G
        if fam == "monitoring":
            L = max(1, (win[1] - win[0]) // self.month_days)
            n_units = n_cells * L
        else:
            L = 1
            n_units = n_cells
        scope, unitmap, group = {}, {}, {}
        for x, r in items.iterrows():
            scope[x] = np.arange(n_cells)
            if fam == "monitoring":
                m = min(L - 1, int((r["time"] - win[0]) // self.month_days))
                um = np.full(n_units, -1)
                um[m * n_cells:(m + 1) * n_cells] = np.arange(n_cells)
            else:
                um = np.arange(n_cells)
            unitmap[x] = {"own": um}
            group[x] = "sar" if r["source"] == "S1" else "opt"
        cb = (yy // 2) * (G // 2) + (xx // 2)
        cb = cb.reshape(-1)
        nb = (G // 2) ** 2
        cmap = np.concatenate([cb + m * nb for m in range(L)])
        pool_maps = {"c2": (cmap, nb * L)}
        adjacency = {"fine": self._adjacency(G, L, temporal=(fam == "monitoring")),
                     "c2": self._adjacency(G // 2, L, temporal=(fam == "monitoring"))}
        ctx = RequestContext(q, items, n_units, coords, scope, unitmap, pool_maps, adjacency, group)
        ctx.extra.update(grid=(G, G), spacing=cell_km, months=L, area_km2=A["side"] ** 2, aoi=a)
        return ctx

    @staticmethod
    def _adjacency(G, L, temporal):
        n = G * G
        rows, cols = [], []
        if temporal:
            for m in range(L):
                for mm in (m - 1, m + 1):
                    if 0 <= mm < L:
                        rows.append(np.arange(n) + m * n)
                        cols.append(np.arange(n) + mm * n)
        else:
            idx = np.arange(n).reshape(G, G)
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                src = idx[max(0, -dy):G - max(0, dy), max(0, -dx):G - max(0, dx)]
                dst = idx[max(0, dy):G + min(0, dy), max(0, dx):G + min(0, dx)]
                for m in range(L):
                    rows.append(src.reshape(-1) + m * n)
                    cols.append(dst.reshape(-1) + m * n)
        if not rows:
            return sparse.csr_matrix((n * L, n * L))
        r, c = np.concatenate(rows), np.concatenate(cols)
        A = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n * L, n * L))
        deg = np.asarray(A.sum(1)).ravel()
        deg[deg == 0] = 1
        return sparse.diags(1 / deg) @ A

    def item_features(self, ctx, xs) -> pd.DataFrame:
        I = ctx.items.loc[list(xs)]
        q = ctx.request
        A = self.aois[ctx.extra["aoi"]]
        F = pd.DataFrame(index=I.index)
        ex = I["exists"].to_numpy(bool)
        F["e"] = np.where(ex, I["e"], np.nan)
        F["cf"] = np.where(ex, I["cf"], np.nan)
        F["footprint"] = np.where(ex, I["footprint"], np.nan)
        for s in SOURCES:
            F[f"src_{s}"] = (I["source"] == s).astype(float)
        F["doy_sin"] = np.sin(2 * np.pi * I["time"] / 365)
        F["doy_cos"] = np.cos(2 * np.pi * I["time"] / 365)
        F["clim"] = I["clim"]
        F["side_km"] = A["side"]
        F["lead_days"] = I["time"] - q.t
        F["exists"] = ex
        return F

    def item_distance(self, ctx, xs) -> np.ndarray:
        I = ctx.items.loc[list(xs)]
        t = I["time"].to_numpy(float)
        radar = (I["source"] == "S1").to_numpy()
        D = np.abs(t[:, None] - t[None, :]) + 1e4 * (radar[:, None] != radar[None, :])
        return D

    def true_evidence(self, ctx, x) -> np.ndarray:
        a = ctx.extra["aoi"]
        j = int(ctx.items.loc[x, "local"])
        return np.unpackbits(self.scenes(a)[j]["mask"])[: self.G * self.G].astype(bool)

    def observe(self, ctx, x):
        r = ctx.items.loc[x]
        if r["source"] == "S1" or r["time"] > ctx.request.t - 1:
            return None
        cache = ctx.extra.setdefault("_obs", {})
        if x not in cache:
            G, f = self.G, self.ov_f
            B = self.true_evidence(ctx, x)
            rng = rng_for(self.seed, "ov", ctx.extra["aoi"], int(r["local"]))
            yy, xx = np.mgrid[0:G, 0:G]
            blocks = ((yy // f) * (G // f) + (xx // f)).reshape(-1)
            nb = (G // f) ** 2
            true = np.bincount(blocks, weights=B.astype(float), minlength=nb).astype(int)
            size = np.bincount(blocks, minlength=nb)
            flip = rng.random(nb) < self.ov_noise
            counts = np.where(flip, size - true, true)
            cache[x] = (blocks, counts)
        return cache[x]

    def templates(self, family):
        return TEMPLATES[family]

    def unseen(self, family):
        return UNSEEN[family]

    def vocabulary(self) -> dict:
        return {
            "spec_fields": {
                "sources": "non-empty subset of ['S2','L','S1'] (L = Landsat 8/9; S1 = Sentinel-1 radar, sees "
                           "through clouds); two optical groups are fused",
                "rule": "'any' (latest-clear / best-pixel: a cell is usable if any selected scene is clear), "
                        "'median' (median composite: at least two clear scenes), "
                        "'count' (time series: at least k clear scenes per cell and month; monitoring only)",
                "k": "integer >= 1, only for rule='count'",
                "fill": "null, 'temporal' (fill from adjacent months; monitoring), 'spatial' (fill cloud gaps from "
                        "clear neighbours; rapid/composite) or 'radar' (fill cells the optical product leaves empty "
                        "where a Sentinel-1 scene covers them; rapid/composite)",
                "resolution_m": "10, 20, 30 or 60 (60 aggregates to 2x coarser product cells)",
                "latest_only": "true to read only the single latest scene",
                "best_only": "true to read only the scene with the lowest catalogue cloud fraction",
                "index": "null or 'ndvi' (compute an index; best pixel by maximum NDVI)",
            },
            "operators": ["select", "read", "clean", "resample", "index", "composite", "fuse", "fill", "pool", "export"],
        }

    def compile(self, spec, q, ctx, name="w") -> Compiled:
        fam = q.family
        groups = spec.get("sources") or []
        if not groups or any(g not in GROUPS for g in groups):
            raise ValidationError("sources must be a non-empty subset of ['S2', 'L', 'S1'] (L = Landsat 8/9)")
        srcs = sorted({s_ for g in groups for s_ in GROUPS[g]})
        n_groups = len({GROUP_OF[s_] for s_ in srcs})
        rule = spec.get("rule", "any")
        if rule not in ("any", "median", "count", "best", "latest"):
            raise ValidationError("rule must be any|median|count")
        rule = "any" if rule in ("best", "latest") else rule
        if rule == "count" and fam != "monitoring":
            raise ValidationError("rule 'count' is only for monitoring time series")
        if fam == "monitoring" and rule == "median":
            raise ValidationError("monitoring needs rule 'count' or 'any'")
        fill = spec.get("fill")
        if fill not in (None, "spatial", "temporal", "radar"):
            raise ValidationError("fill must be null, 'spatial', 'temporal' or 'radar'")
        if fill == "temporal" and fam != "monitoring":
            raise ValidationError("temporal fill only for monitoring")
        if fill in ("spatial", "radar") and fam == "monitoring":
            raise ValidationError("spatial or radar fill not available for time series")
        if fill == "radar" and "S1" in srcs:
            raise ValidationError("radar fill takes optical sources; it adds Sentinel-1 itself")
        res = int(spec.get("resolution_m", 20))
        if res not in (10, 20, 30, 60):
            raise ValidationError("resolution_m must be 10, 20, 30 or 60")
        I = ctx.items
        sel = I[I["source"].isin(srcs)]
        if "items" in spec and spec["items"] is not None:
            sel = sel.loc[[x for x in spec["items"] if x in sel.index]]
        if spec.get("latest_only"):
            sel = sel[sel["time"] <= q.t] if (sel["time"] <= q.t).any() else sel
            sel = sel.sort_values("time").iloc[-1:]
        if spec.get("best_only"):
            ex = sel[sel["exists"]] if sel["exists"].any() else sel
            sel = ex.sort_values("cf").iloc[:1]
        xs = [int(x) for x in sel.index]
        if not xs:
            raise ValidationError("no scene of the selected sources in the window")
        k = int(spec.get("k", q.req["floors"].get("min_clear_obs", 1))) if rule == "count" else 1
        if rule == "median":
            root = AtLeast(2, [Ev(x) for x in xs]) if len(xs) >= 2 else None
            if len(xs) < 2:
                raise ValidationError("median composite needs at least two scenes")
        elif rule == "count":
            root = AtLeast(k, [Ev(x) for x in xs]) if len(xs) >= k else None
        else:
            root = AnyOf([Ev(x) for x in xs])
        radar_xs = []
        if fill == "radar":
            rad = I[I["source"] == "S1"]
            radar_xs = [int(x) for x in rad.index]
            if root is not None and radar_xs:
                s1 = AnyOf([Ev(x) for x in radar_xs])
                root = AnyOf([root, AllOf([s1, Fill("radar", root)])])
        elif fill:
            root = Fill(fill, root) if root is not None else None
        space = "fine"
        if res >= 60:
            root = Pool("c2", root)
            space = "c2"
        xs_all = xs + radar_xs
        area = ctx.extra["area_km2"]
        px = area * 1e6 / (res ** 2)
        months = ctx.extra["months"]
        ops = []
        shards = float(max(1, math.ceil(area / 2500.0)))
        read_srcs = sorted(set(srcs) | ({"S1"} if radar_xs else set()))
        n_src = n_groups
        sel_all = I.loc[xs_all]
        for s_ in read_srcs:
            ns = int((sel_all["source"] == s_).sum())
            if ns == 0:
                continue
            nat = NATIVE_RES[s_]
            b = ns * area * 1e6 / nat ** 2 * BANDS[s_] * 2
            common = dict(n_sources=n_src, remote_store=float(s_ == "S1"), resolution=float(nat),
                          radar=float(s_ == "S1"), area_km2=area, shards=shards)
            ops.append(OpSpec("select", dict(common, n_inputs=ns), 0.0, ns))
            ops.append(OpSpec("read", dict(common, n_inputs=ns), b, px * ns))
            ops.append(OpSpec("clean", dict(common, n_inputs=ns), 0.0, px * ns))
        n = len(xs)
        common = dict(n_sources=n_src, resolution=float(res), area_km2=area, radar=float(bool(radar_xs) or "S1" in srcs),
                      shards=shards)
        if res > min(NATIVE_RES[s_] for s_ in srcs):
            ops.append(OpSpec("resample", dict(common, n_inputs=n), 0.0, px * n))
        if spec.get("index"):
            ops.append(OpSpec("index", dict(common, n_inputs=n), 0.0, px * n))
        ops.append(OpSpec("composite", dict(common, n_inputs=n, rule_median=float(rule != "any")), 0.0, px * max(n, 1)))
        if n_groups > 1:
            ops.append(OpSpec("fuse", dict(common, n_inputs=n_groups), 0.0, px * n_groups))
        if fill:
            node = root[2] if root is not None and root[0] == "pool" else root
            if node is not None and node[0] == "fill":
                child = node[2]
            elif node is not None and fill == "radar":
                child = next(c for c in node[1] if c[0] == "all")[1]
                child = next(c[2] for c in child if c[0] == "fill")
            else:
                child = None

            def fill_records(ectx, child=child, px=px):
                if child is None:
                    return np.full(ectx.M, px)
                return px * (1.0 - ectx.eval(child).mean(axis=1)) + 1.0

            ops.append(OpSpec("fill", dict(common, n_inputs=n, method_temporal=float(fill == "temporal"),
                                           method_radar=float(fill == "radar"), _nominal=0.4 * px), 0.0, fill_records))
        if space == "c2":
            ops.append(OpSpec("pool", dict(common, n_inputs=1), 0.0, px))
        ops.append(OpSpec("export", dict(common, n_inputs=1), 0.0, px * months, transfer_bytes=px * months * 4 * 2))
        e = I.loc[xs, "e"].to_numpy(float) if xs else np.zeros(1)
        feats = dict(n_items=n, n_sources=n_src, has_s1=float("S1" in srcs), rule_median=float(rule == "median"),
                     rule_count=float(rule == "count"), k=k, fill=float(fill is not None), resolution=res,
                     coarse=float(space == "c2"), area_km2=area, window_days=q.req["scope"]["window"][1] -
                     q.req["scope"]["window"][0], months=months, e_mean=float(np.mean(e)) if n else 0.0,
                     e_max=float(np.max(e)) if n else 0.0, e_min=float(np.min(e)) if n else 0.0,
                     fam_rapid=float(fam == "rapid"), fam_monitoring=float(fam == "monitoring"),
                     frac_future=float((I.loc[xs, "exists"] == False).mean()) if n else 0.0)
        feats.update(n_groups=n_groups, radar_fill=float(fill == "radar"), index=float(bool(spec.get("index"))))
        return Compiled(name, dict(spec), name, True, root, space, xs_all, ops, feats,
                        output_units=ctx.n_space(space))

    def source_group(self, comp) -> str:
        groups = comp.spec.get("sources") or []
        g = sorted({GROUP_OF[s_] for gr in groups for s_ in GROUPS.get(gr, [])})
        if comp.spec.get("fill") == "radar" and "S1" not in g:
            g.append("S1")
        return "+".join(sorted(g)) or "*"

    def item_read_op(self, ctx, comp, x):
        s_ = ctx.items.loc[x, "source"]
        area = ctx.extra["area_km2"]
        nat = NATIVE_RES[s_]
        reads = [o for o in comp.ops if o.op == "read" and o.psi.get("resolution") == float(nat)
                 and o.psi.get("radar") == float(s_ == "S1")]
        if not reads:
            return super().item_read_op(ctx, comp, x)
        o = reads[0]
        px = float(o.records) / max(1.0, float(o.psi.get("n_inputs", 1.0)))
        return OpSpec("read", dict(o.psi, n_inputs=1.0), self.item_mb(ctx, x) * 1e6, px)

    def resolve_scope(self, request, req=None):
        out = super().resolve_scope(request, req)
        ctx = out["ctx"]
        out.update(area_km2=float(ctx.extra["area_km2"]), months=int(ctx.extra["months"]),
                   window=list(request.req["scope"]["window"]))
        return out

    def size_measure(self, q, ctx) -> float:
        return float(ctx.extra["area_km2"])

    def item_mb(self, ctx, x) -> float:
        s = ctx.items.loc[x, "source"]
        return ctx.extra["area_km2"] * 1e6 / NATIVE_RES[s] ** 2 * BANDS[s] * 2 / 1e6

    def overview_mb(self, ctx) -> float:
        return (self.G // self.ov_f) ** 2 * 4 / 1e6 + 0.5

    def overview_seconds(self, ctx, n_overviews: int) -> float:
        return n_overviews * (self.range_read_s + self.overview_mb(ctx) / WORKER["io_mb_s"])

    def catalogue_text(self, q, ctx) -> str:
        I = ctx.items
        lines = [f"Area {q.req['scope']['aoi']}: {ctx.extra['area_km2']:.0f} km2, request day {q.t:.0f}, "
                 f"{len(I)} scenes in the window."]
        for s in SOURCES:
            sub = I[I["source"] == s]
            if len(sub):
                fut = int((~sub["exists"]).sum())
                ex = sub[sub["exists"]]
                cf = f"mean tile cloud {ex['cf'].mean():.2f}" if len(ex) and s != "S1" else "radar"
                lines.append(f"  {s}: {len(sub)} scenes ({fut} not yet acquired), {cf}, "
                             f"climatological cloudiness {sub['clim'].mean():.2f}")
        return "\n".join(lines)
