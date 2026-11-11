from __future__ import annotations

import glob
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from ..core.lineage import AllOf, AnyOf, Ev, Fill, Pool
from ..core.types import Compiled, OpSpec, Request, RequestContext
from ..utils import DATA, rng_for
from .base import Domain, ValidationError

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
STATIONS = {
    "Aotizhongxin": (39.982, 116.397), "Changping": (40.217, 116.230), "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417), "Guanyuan": (39.929, 116.339), "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628), "Nongzhanguan": (39.937, 116.461), "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407), "Wanliu": (39.987, 116.287), "Wanshouxigong": (39.878, 116.352),
}
ST = list(STATIONS)

TEMPLATES = {
    "own": dict(fallback=False, fill=None, coarsen_days=1),
    "fallback": dict(fallback=True, fill=None, coarsen_days=1),
    "own_fill": dict(fallback=False, fill="interp", coarsen_days=1),
    "own_2day": dict(fallback=False, fill=None, coarsen_days=2),
    "own_weekly": dict(fallback=False, fill=None, coarsen_days=7),
    "fallback_2day": dict(fallback=True, fill=None, coarsen_days=2),
}
UNSEEN = {
    "fallback_fill": dict(fallback=True, fill="interp", coarsen_days=1),
    "own_fill_2day": dict(fallback=False, fill="interp", coarsen_days=2),
}


def _km(a, b):
    (la1, lo1), (la2, lo2) = STATIONS[a], STATIONS[b]
    return float(np.hypot((la1 - la2) * 111.0, (lo1 - lo2) * 85.0))


class SensorPanels(Domain):
    name = "sensors"
    families = ["panel"]
    psi_keys = ["n_inputs", "n_stations", "n_pollutants", "hours_req", "fallback", "resolution", "cached",
                "method_temporal"]
    e_cols = ["e", "e7", "e_hours"]
    fill_reference = {"interp": (-2.5, 4.0)}
    field_scales = np.array([0.5, 1, 2, 4, 8, 16, 32])
    time_scales = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])

    def __init__(self, seed=0, quick=False, n_requests=None, path=None):
        super().__init__(seed, quick)
        self.n_requests = n_requests or (120 if quick else 300)
        self.path = path or os.path.join(DATA, "raw", "beijing", "PRSA_Data_20130301-20170228")
        self._load()
        self.D = {(a, b): _km(a, b) for a in ST for b in ST}
        self.nearest = {s: min((t for t in ST if t != s), key=lambda t: self.D[(s, t)]) for s in ST}

    def _load(self):
        cache = os.path.join(DATA, "beijing_valid_hours.npy")
        if os.path.exists(cache):
            self.valid = np.load(cache)
        else:
            files = sorted(glob.glob(os.path.join(self.path, "PRSA_Data_*.csv")))
            if not files:
                raise FileNotFoundError(f"Beijing air-quality CSVs not found under {self.path}; see README, Data")
            V = None
            for f in files:
                st = re.search(r"PRSA_Data_(.+?)_2013", os.path.basename(f)).group(1)
                df = pd.read_csv(f)
                df["date"] = pd.to_datetime(df[["year", "month", "day"]])
                day = (df["date"] - pd.Timestamp("2013-03-01")).dt.days.to_numpy()
                nd = day.max() + 1
                if V is None:
                    V = np.zeros((len(ST), len(POLLUTANTS), nd), np.int16)
                for j, p in enumerate(POLLUTANTS):
                    ok = df[p].notna().to_numpy().astype(np.int16)
                    np.add.at(V[ST.index(st), j], day, ok)
            self.valid = V
            np.save(cache, V)
        self.n_days = self.valid.shape[2]

    def requests(self) -> List[Request]:
        if hasattr(self, "_requests"):
            return self._requests
        rng = rng_for(self.seed, "sensor-requests")
        block = 90
        reqs = []
        for i in range(self.n_requests):
            L = int(rng.integers(30, 121))
            start = int(rng.integers(31, self.n_days - L))
            n_st = int(rng.integers(2, 7))
            n_p = int(rng.integers(2, 7))
            stations = sorted(rng.choice(ST, n_st, replace=False).tolist())
            chosen = set(rng.choice(POLLUTANTS, n_p, replace=False).tolist())
            pols = [p for p in POLLUTANTS if p in chosen]
            H = int(rng.choice([18, 21, 24]))
            target = float(rng.choice([0.6, 0.7, 0.8, 0.9]))
            req = dict(family="panel", scope=dict(stations=stations, pollutants=pols, window=[start, start + L]),
                       target=target, budget=None, deadline=3600.0, floors=dict(min_valid_hours=H))
            reqs.append(Request(f"sens-{i}", "sensors", "panel", f"B{start // block}", float(start), "", req,
                                size_class="small" if n_st * L <= 300 else "large"))
        self._requests = reqs
        return reqs

    def describe_request(self, req, rng=None) -> str:
        s = req["scope"]
        b = f" Budget about ${req['budget']:.4f}." if req.get("budget") else ""
        return (f"We need a daily panel for stations {', '.join(s['stations'])} with pollutants "
                f"{', '.join(s['pollutants'])} from day {s['window'][0]} to day {s['window'][1]} "
                f"({s['window'][1] - s['window'][0]} days). A station-day counts if every pollutant has at least "
                f"{req['floors']['min_valid_hours']} valid hourly readings; we need {req['target']*100:.0f}% of "
                f"station-days usable, within an hour.{b}")

    def parse_rule(self, text):
        t = text
        sts = [s for s in ST if s in t]
        pols = [p for p in POLLUTANTS if re.search(r"(?<![A-Za-z0-9.])" + re.escape(p) + r"(?![0-9])", t)]
        days = [int(x) for x in re.findall(r"day (\d+)", t)]
        H = re.search(r"at least (\d+) valid", t)
        pct = re.search(r"(\d+)%", t)
        bud = re.search(r"\$([\d.]+)", t)
        return dict(family="panel", scope=dict(stations=sts, pollutants=pols, window=days[:2] if len(days) >= 2 else None),
                    target=float(pct.group(1)) / 100 if pct else 0.8,
                    budget=float(bud.group(1).rstrip(".")) if bud else None, deadline=3600.0,
                    floors=dict(min_valid_hours=int(H.group(1)) if H else 24))

    def _build_context(self, q):
        s = q.req["scope"]
        stations, pols = s["stations"], s["pollutants"]
        d0, d1 = s["window"]
        L = d1 - d0
        H = q.req["floors"]["min_valid_hours"]
        n_units = len(stations) * L
        files = []
        for st in stations:
            for p in pols:
                files.append((st, p))
                nb = self.nearest[st]
                if (nb, p) not in files:
                    files.append((nb, p))
        files = sorted(set(files), key=lambda f: (ST.index(f[0]), POLLUTANTS.index(f[1])))
        rows, scope, unitmap, group = [], {}, {}, {}
        for (st, p) in files:
            x = ST.index(st) * 10 + POLLUTANTS.index(p)
            v = self.valid[ST.index(st), POLLUTANTS.index(p)]
            past = v[max(0, d0 - 30):d0]
            rows.append(dict(item=x, station=st, pollutant=p, time=float(d0),
                             e=float((past >= H).mean()) if len(past) else np.nan,
                             e7=float((v[max(0, d0 - 7):d0] >= H).mean()) if d0 > 0 else np.nan,
                             e_hours=float(past.mean() / 24) if len(past) else np.nan, exists=True))
            scope[x] = np.arange(L)
            own = np.full(n_units, -1)
            near = np.full(n_units, -1)
            for i, st2 in enumerate(stations):
                if st2 == st:
                    own[i * L:(i + 1) * L] = np.arange(L)
                if self.nearest[st2] == st:
                    near[i * L:(i + 1) * L] = np.arange(L)
            unitmap[x] = {"own": own, "near": near}
            group[x] = "all"
        items = pd.DataFrame(rows).set_index("item")
        pool_maps = {}
        adjacency = {}
        for k in (1, 2, 7):
            nb = int(np.ceil(L / k))
            cmap = np.concatenate([i * nb + np.arange(L) // k for i in range(len(stations))])
            name = "fine" if k == 1 else f"d{k}"
            if k > 1:
                pool_maps[name] = (cmap, nb * len(stations))
            n = nb * len(stations)
            r, c = [], []
            for i in range(len(stations)):
                for d in range(nb):
                    for dd in (d - 1, d + 1):
                        if 0 <= dd < nb:
                            r.append(i * nb + d)
                            c.append(i * nb + dd)
            A = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
            deg = np.asarray(A.sum(1)).ravel()
            deg[deg == 0] = 1
            adjacency[name] = sparse.diags(1 / deg) @ A
        coords = np.arange(L, dtype=float)[:, None]
        ctx = RequestContext(q, items, n_units, coords, scope, unitmap, pool_maps, adjacency, group)
        ctx.extra.update(L=L, H=H, stations=stations, pollutants=pols)
        return ctx

    def item_features(self, ctx, xs):
        I = ctx.items.loc[list(xs)]
        F = pd.DataFrame(index=I.index)
        F["e"], F["e7"], F["e_hours"] = I["e"], I["e7"], I["e_hours"]
        for s in ST:
            F[f"st_{s}"] = (I["station"] == s).astype(float)
        for p in POLLUTANTS:
            F[f"p_{p}"] = (I["pollutant"] == p).astype(float)
        F["season_sin"] = np.sin(2 * np.pi * (I["time"] + 59) / 365.25)
        F["season_cos"] = np.cos(2 * np.pi * (I["time"] + 59) / 365.25)
        F["L"] = ctx.extra["L"]
        F["H"] = ctx.extra["H"]
        F["exists"] = True
        return F

    def item_distance(self, ctx, xs):
        I = ctx.items.loc[list(xs)]
        st, p = I["station"].tolist(), I["pollutant"].tolist()
        n = len(xs)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = self.D[(st[i], st[j])] / 10.0 + 1.0 * (p[i] != p[j])
        return D

    def order_key(self, ctx, xs):
        I = ctx.items.loc[list(xs)]
        return np.array([ST.index(s) * 10 + POLLUTANTS.index(p) for s, p in zip(I["station"], I["pollutant"])], float)

    def true_evidence(self, ctx, x):
        st, p = ST[x // 10], POLLUTANTS[x % 10]
        d0, d1 = ctx.request.req["scope"]["window"]
        return self.valid[ST.index(st), POLLUTANTS.index(p), d0:d1] >= ctx.extra["H"]

    def templates(self, family):
        return TEMPLATES

    def unseen(self, family):
        return UNSEEN

    def vocabulary(self):
        return {"spec_fields": {
            "fallback": "true to fall back on the nearest station's sensor when a station's own file has a gap",
            "fill": "null or 'interp' (fill a missing station-day from adjacent days)",
            "coarsen_days": "1 (daily), 2 or 7: aggregate the panel to coarser periods",
        }, "operators": ["select", "read", "clean", "join", "fill", "coarsen", "export"]}

    def compile(self, spec, q, ctx, name="w"):
        fb = bool(spec.get("fallback", False))
        fill = spec.get("fill")
        k = int(spec.get("coarsen_days", 1))
        if fill not in (None, "interp"):
            raise ValidationError("fill must be null or 'interp'")
        if k not in (1, 2, 7):
            raise ValidationError("coarsen_days must be 1, 2 or 7")
        pols = ctx.extra["pollutants"]
        I = ctx.items
        allowed = set(spec["items"]) if spec.get("items") is not None else None
        per_p = []
        used = set()
        for p in pols:
            leaves_ = []
            for x, r in I[I["pollutant"] == p].iterrows():
                if allowed is not None and x not in allowed:
                    continue
                if (ctx.unitmap[x]["own"] >= 0).any():
                    leaves_.append(Ev(x, "own"))
                    used.add(int(x))
                if fb and (ctx.unitmap[x]["near"] >= 0).any():
                    leaves_.append(Ev(x, "near"))
                    used.add(int(x))
            if not leaves_:
                raise ValidationError(f"no item covers pollutant {p}")
            per_p.append(AnyOf(leaves_))
        root = AllOf(per_p)
        if fill:
            root = Fill(fill, root) if root is not None else None
        space = "fine"
        if k > 1:
            root = Pool(f"d{k}", root)
            space = f"d{k}"
        xs = sorted(used)
        if not xs:
            raise ValidationError("no items selected")
        L, H = ctx.extra["L"], ctx.extra["H"]
        nst = len(ctx.extra["stations"])
        common = dict(n_stations=nst, n_pollutants=len(pols), hours_req=H, fallback=float(fb), resolution=float(k))
        rows = 24 * L * len(xs)
        ops = [OpSpec("select", dict(common, n_inputs=len(xs), remote_store=1.0), 0.0, len(xs)),
               OpSpec("read", dict(common, n_inputs=len(xs), remote_store=1.0), rows * 64.0, rows),
               OpSpec("clean", dict(common, n_inputs=len(xs)), 0.0, rows),
               OpSpec("join", dict(common, n_inputs=len(pols)), 0.0, ctx.n_units * len(pols) * 24)]
        if fill:
            child = root[2] if root[0] == "fill" else root[2][2]

            def fill_records(ectx, child=child, n=ctx.n_units):
                return n * 24 * (1.0 - ectx.eval(child).mean(axis=1)) + 1.0

            ops.append(OpSpec("fill", dict(common, n_inputs=len(pols), method_temporal=1.0,
                                           _nominal=0.2 * ctx.n_units * 24), 0.0, fill_records))
        if k > 1:
            ops.append(OpSpec("coarsen", dict(common, n_inputs=1), 0.0, ctx.n_units * 24))
        n_out = ctx.n_space(space)
        ops.append(OpSpec("export", dict(common, n_inputs=1), 0.0, n_out, transfer_bytes=n_out * len(pols) * 8.0))
        e = I.loc[xs, "e"].to_numpy(float)
        feats = dict(n_items=len(xs), n_stations=nst, n_pollutants=len(pols), H=H, L=L, fallback=float(fb),
                     fill=float(fill is not None), coarsen=k, e_mean=float(np.nanmean(e)), e_min=float(np.nanmin(e)),
                     e_max=float(np.nanmax(e)))
        return Compiled(name, dict(spec), name, True, root, space, xs, ops, feats, output_units=n_out)

    def item_mb(self, ctx, x):
        return 24 * ctx.extra["L"] * 64 / 1e6

    def catalogue_text(self, q, ctx):
        I = ctx.items
        lines = [f"{len(I)} sensor files (station x pollutant) for the window; nearest stations: " +
                 ", ".join(f"{s}->{self.nearest[s]}" for s in ctx.extra["stations"])]
        for st, sub in I.groupby("station"):
            lines.append(f"  {st}: past-30-day availability " +
                         ", ".join(f"{p} {e:.2f}" for p, e in zip(sub["pollutant"], sub["e"])))
        return "\n".join(lines)
