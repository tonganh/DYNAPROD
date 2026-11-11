from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .compute import WORKER, LM, physical, physical_time
from .field import FieldSampler, copula_normals, place_evidence, place_evidence_blocks
from .lineage import EvalContext
from .observation import ObservationPolicy, OverviewErrorModel
from .quantile import LEVELS, QuantileGBM, Recalibrator, interp_quantile, pit
from .run_logs import spec_key
from .types import Compiled, Request, RequestContext, Rollouts
from ..config import ITEM_GBM, WMConfig
from ..utils import rng_for

OPS = ["select", "read", "clean", "composite", "fuse", "fill", "pool", "export", "join", "extract",
       "vote", "coarsen", "resample", "index"]


class WorldModel:
    def __init__(self, domain, cfg: Optional[WMConfig] = None):
        self.domain = domain
        self.cfg = cfg or WMConfig()
        self._draws: Dict[tuple, dict] = {}
        self.lam_t = 1.0
        self.lam_s = 1.0
        self.lam_f = 1.0
        self._samplers: Dict[tuple, FieldSampler] = {}
        self.obs_model: Optional[OverviewErrorModel] = None
        self.rates = dict(compute=1.0, tokens=1.0, transfer=1.0)

    def fit(self, records: List[dict]):
        t0 = time.time()
        self._fit_items(records)
        self._fit_dependence(records)
        self._fit_acceptance(records)
        self._fit_operators(records)
        if getattr(self.domain, "has_observations", False):
            self.obs_model = OverviewErrorModel().fit(self.domain, records)
        self.fit_seconds = time.time() - t0
        return self

    def _item_rows(self, records):
        seen = set()
        feats, ys, groups, keys = [], [], [], []
        by_q = defaultdict(set)
        for r in records:
            by_q[r["request"].qid].update(r["comp"].read_set)
        ctxs = {r["request"].qid: r["ctx"] for r in records}
        for qid, xs in by_q.items():
            ctx = ctxs[qid]
            xs = sorted(xs)
            if not xs:
                continue
            ev = self.domain.truth_evidence(ctx, xs)
            F = self.domain.item_features(ctx, xs)
            for x in xs:
                if len(ctx.scope[x]) == 0:
                    continue
                feats.append(F.loc[x])
                ys.append(float(ev[x].mean()))
                groups.append(ctx.request.cluster)
                keys.append((qid, x))
        F = pd.DataFrame(feats).reset_index(drop=True)
        return F, np.array(ys), np.array(groups), keys

    def _fit_items(self, records):
        F, y, g, keys = self._item_rows(records)
        self.item_cols = [c for c in F.columns if c != "exists"]
        self.item_cols0 = [c for c in self.item_cols if c not in self.domain.e_cols]
        X, X0 = F[self.item_cols].to_numpy(float), F[self.item_cols0].to_numpy(float)
        rng = np.random.default_rng(self.cfg.seed)
        pits, pits0 = np.zeros(len(y)), np.zeros(len(y))
        n_splits = min(5, len(np.unique(g)))
        if n_splits >= 2:
            for tr, te in GroupKFold(n_splits).split(X, y, g):
                m = QuantileGBM(clip=(0, 1), **ITEM_GBM).fit(X[tr], y[tr])
                pits[te] = pit(m.predict(X[te]), LEVELS, y[te], rng, 0, 1)
                m0 = QuantileGBM(clip=(0, 1), **ITEM_GBM).fit(X0[tr], y[tr])
                pits0[te] = pit(m0.predict(X0[te]), LEVELS, y[te], rng, 0, 1)
        self.h = QuantileGBM(clip=(0, 1), **ITEM_GBM).fit(X, y)
        self.h0 = QuantileGBM(clip=(0, 1), **ITEM_GBM).fit(X0, y)
        self.recal = Recalibrator().fit(pits) if n_splits >= 2 else Recalibrator()
        self.recal0 = Recalibrator().fit(pits0) if n_splits >= 2 else Recalibrator()
        self.item_train = dict(F=F, y=y, pits=pits, keys=keys)

    def item_quantiles(self, F: pd.DataFrame):
        exists = F["exists"].to_numpy(bool) if "exists" in F else np.ones(len(F), bool)
        if self.cfg.use_h0_for_all:
            exists[:] = False
        Q = np.zeros((len(F), len(LEVELS)))
        if exists.any():
            Q[exists] = self.h.predict(F.loc[exists, self.item_cols].to_numpy(float))
        if (~exists).any():
            Q[~exists] = self.h0.predict(F.loc[~exists, self.item_cols0].to_numpy(float))
        return Q, exists

    def _fit_dependence(self, records):
        dom = self.domain
        rng = np.random.default_rng(self.cfg.seed + 1)
        F, y, pits = self.item_train["F"], self.item_train["y"], self.item_train["pits"]
        z_of = {k: norm.ppf(np.clip(p, 0.01, 0.99)) for k, p in zip(self.item_train["keys"], pits)}
        ctxs = {}
        xs_of = defaultdict(set)
        for r in records:
            ctxs[r["request"].qid] = r["ctx"]
            xs_of[r["request"].qid].update(r["comp"].read_set)
        ds, prods = [], []
        pers = []
        mask_samples = []
        for qid, xs in xs_of.items():
            ctx = ctxs[qid]
            xs = [x for x in sorted(xs) if (qid, x) in z_of]
            if len(xs) < 2:
                continue
            D = dom.item_distance(ctx, xs)
            z = np.array([z_of[(qid, x)] for x in xs])
            iu = np.triu_indices(len(xs), 1)
            if len(iu[0]) > 400:
                sel = rng.choice(len(iu[0]), 400, replace=False)
                iu = (iu[0][sel], iu[1][sel])
            ds.append(D[iu])
            prods.append(z[iu[0]] * z[iu[1]])
            ev = dom.truth_evidence(ctx, xs)
            groups = defaultdict(list)
            for x in xs:
                groups[ctx.group.get(x, 0)].append(x)
            for gx in groups.values():
                if len(gx) < 2:
                    continue
                key = dom.order_key(ctx, gx)
                order = [gx[i] for i in np.argsort(key, kind="stable")]
                Dg = dom.item_distance(ctx, order)
                for i in range(1, len(order)):
                    x0, x1 = order[i - 1], order[i]
                    common, i0, i1 = np.intersect1d(ctx.scope[x0], ctx.scope[x1], return_indices=True)
                    if len(common) < 2:
                        continue
                    b0, b1 = ev[x0][0][i0], ev[x1][0][i1]
                    pers.append((Dg[i - 1, i], (b0 & b1).mean(), b0.mean(), b1.mean()))
            if len(mask_samples) < 150:
                for x in xs[:: max(1, len(xs) // 3)]:
                    b = ev[x][0]
                    if 0.05 < b.mean() < 0.95 and len(b) >= 4:
                        mask_samples.append((qid, x, b))
        if ds:
            d = np.concatenate(ds)
            p = np.concatenate(prods)
            best, bl = np.inf, dom.time_scales[0]
            for lam in dom.time_scales:
                err = np.mean((p - np.exp(-d / lam)) ** 2)
                if err < best:
                    best, bl = err, lam
            self.lam_t = float(bl)
        if pers:
            P = np.array(pers)
            d, o, a0, a1 = P.T
            lo, hi = a0 * a1, np.minimum(a0, a1)
            best, bl = np.inf, dom.time_scales[0]
            for lam in dom.time_scales:
                pk = np.exp(-d / lam)
                err = np.mean((o - (pk * hi + (1 - pk) * lo)) ** 2)
                if err < best:
                    best, bl = err, lam
            self.lam_f = float(bl)
        if mask_samples:
            best, bl = np.inf, dom.field_scales[0]
            for lam in dom.field_scales:
                err = []
                for qid, x, b in mask_samples:
                    ctx = ctxs[qid]
                    sc = ctx.scope[x]
                    nb = self._neighbor_pairs(ctx, sc)
                    if nb is None:
                        continue
                    obs = (b[nb[0]] == b[nb[1]]).mean()
                    f = self._sampler(ctx, lam).sample(4, rng)[:, sc]
                    B = place_evidence(f, np.full(4, int(b.sum())))
                    sim = (B[:, nb[0]] == B[:, nb[1]]).mean()
                    err.append((obs - sim) ** 2)
                if err and np.mean(err) < best:
                    best, bl = np.mean(err), lam
            self.lam_s = float(bl)

    def _neighbor_pairs(self, ctx, sc):
        key = ("nb", len(sc), int(sc[0]) if len(sc) else -1, int(sc[-1]) if len(sc) else -1)
        cache = ctx.extra.setdefault("_nb", {})
        if key in cache:
            return cache[key]
        C = ctx.cell_coords[sc]
        if len(C) < 4:
            cache[key] = None
            return None
        from scipy.spatial import cKDTree
        tree = cKDTree(C)
        dd, _ = tree.query(C, k=2)
        h = np.median(dd[:, 1]) * 1.01
        pairs = np.array(sorted(tree.query_pairs(h)))
        cache[key] = None if len(pairs) == 0 else (pairs[:, 0], pairs[:, 1])
        return cache[key]

    def _sampler(self, ctx: RequestContext, lam: float) -> FieldSampler:
        key = (ctx.request.qid, round(lam, 6))
        s = self._samplers.get(key)
        if s is None:
            grid = ctx.extra.get("grid")
            spacing = ctx.extra.get("spacing", 1.0)
            s = FieldSampler(ctx.cell_coords, lam, grid=grid, spacing=spacing)
            if len(self._samplers) > 64:
                self._samplers.clear()
            self._samplers[key] = s
        return s

    def _fit_acceptance(self, records):
        rows = defaultdict(lambda: ([], []))
        for r in records:
            for method, frac, acc in r["outcome"].log.get("fill", []):
                rows[method][0].append(frac)
                rows[method][1].append(acc)
        self.acc = {}
        for m, (fr, ac) in rows.items():
            f = np.concatenate(fr)
            a = np.concatenate(ac).astype(int)
            if len(np.unique(a)) < 2:
                self.acc[m] = (np.log(max(a.mean(), 1e-3) / max(1 - a.mean(), 1e-3)), 0.0)
                continue
            if len(f) > 50000:
                idx = np.random.default_rng(0).choice(len(f), 50000, replace=False)
                f, a = f[idx], a[idx]
            lr = LogisticRegression(C=10.0).fit(f[:, None], a)
            self.acc[m] = (float(lr.intercept_[0]), max(0.0, float(lr.coef_[0, 0])))

    def _op_features(self, op: str, b: float, n: float, psi: Dict[str, float]) -> List[float]:
        sh = max(1.0, psi.get("shards", 1.0))
        return [OPS.index(op), np.log1p(b / sh), np.log1p(n / sh)] + [float(psi.get(k, 0.0)) for k in self.domain.psi_keys]

    def _fit_operators(self, records):
        Xr, yc, yt, ym = [], [], [], []
        speeds, heads, trans = [], [], []
        for r in records:
            run = r["outcome"].log["run"]
            speeds.append(run.speed)
            heads.append(run.headroom)
            trans.append(run.transient)
            for o in run.ops:
                ph = physical(o.op, o.bytes, o.records, o.psi)
                Xr.append(self._op_features(o.op, o.bytes, o.records, o.psi))
                yc.append(np.log(o.cpu_s * run.speed / ph))
                yt.append(np.log(o.time_s / physical_time(o.op, ph / run.speed)))
                ym.append(np.log(o.mem_gb))
        rc, rt = [], []
        for r in records:
            run = r["outcome"].log["run"]
            if not run.ops:
                continue
            ph = sum(physical(o.op, o.bytes, o.records, o.psi) for o in run.ops)
            pt = sum(physical_time(o.op, physical(o.op, o.bytes, o.records, o.psi)) for o in run.ops)
            cs, ts = sum(o.cpu_s for o in run.ops), sum(o.time_s for o in run.ops)
            if ph > 0 and cs > 0 and pt > 0 and ts > 0:
                rc.append(np.log(cs / ph))
                rt.append(np.log(ts / pt))
        self.tail_scales = (float(np.std(rc)) if len(rc) > 1 else 0.1, float(np.std(rt)) if len(rt) > 1 else 0.1)
        Xr = np.array(Xr)
        self.seen_ops = {OPS[int(i)] for i in Xr[:, 0]}
        self.cost_q = QuantileGBM().fit(Xr, np.array(yc))
        self.time_q = QuantileGBM().fit(Xr, np.array(yt))
        self.mem_q = QuantileGBM(n_estimators=150).fit(Xr, np.array(ym))
        ls = np.log(speeds)
        self.speed_mu, self.speed_sd = float(ls.mean()), float(ls.std() + 1e-6)
        self.headrooms = np.array(heads)
        self.transient_rate = float(np.mean(trans))

    def sample_shares(self, ctx, xs, M, rng, independent=False):
        F = self.domain.item_features(ctx, xs)
        Q, exists = self.item_quantiles(F)
        if independent or "copula" in self.cfg.ablate:
            V = rng.random((M, len(xs)))
        else:
            D = self.domain.item_distance(ctx, xs)
            V = norm.cdf(copula_normals(D, self.lam_t, M, rng))
        V = np.where(exists[None, :], self.recal(V), self.recal0(V))
        A = interp_quantile(Q, LEVELS, V, 0.0, 1.0)
        return A, Q

    def policy(self, observe) -> ObservationPolicy:
        if isinstance(observe, ObservationPolicy):
            return observe
        return ObservationPolicy(enabled=bool(observe), error_model=self.cfg.obs_error_model)

    def sample_evidence(self, ctx, xs, M, rng, observe=False, independent=False, known=None):
        pol = self.policy(observe)
        A, _ = self.sample_shares(ctx, xs, M, rng, independent)
        col = {x: i for i, x in enumerate(xs)}
        lam_s = 0.0 if (independent or "field" in self.cfg.ablate) else self.lam_s
        sampler = self._sampler(ctx, lam_s)
        groups = defaultdict(list)
        for x in xs:
            groups[ctx.group.get(x, 0)].append(x)
        evidence = {}
        persist = not (independent or "persistence" in self.cfg.ablate)
        for gx in groups.values():
            key = self.domain.order_key(ctx, gx)
            order = [gx[i] for i in np.argsort(key, kind="stable")]
            Dg = self.domain.item_distance(ctx, order) if persist and len(order) > 1 else None
            prev = None
            for i, x in enumerate(order):
                sc = ctx.scope[x]
                S = len(sc)
                if S == 0:
                    evidence[x] = np.zeros((M, 0), bool)
                    continue
                full = sampler.sample(M, rng)
                if prev is not None and persist:
                    keep = rng.random(M) < np.exp(-Dg[i - 1, i] / max(self.lam_f, 1e-9))
                    full = np.where(keep[:, None], prev, full)
                prev = full
                fld = full[:, sc]
                obs = pol.observe(self.domain, ctx, x)
                if known is not None and x in known:
                    pos, val = known[x]
                    rest = np.setdiff1d(np.arange(S), pos)
                    B = np.zeros((M, S), bool)
                    B[:, pos] = val[None, :]
                    if len(rest):
                        n_rest = np.rint(A[:, col[x]] * len(rest)).astype(int)
                        B[:, rest] = place_evidence(fld[:, rest], n_rest)
                    evidence[x] = B
                elif obs is not None:
                    blocks, counts = obs
                    if pol.error_model and self.obs_model is not None:
                        counts = self.obs_model.sample_counts(blocks, counts, A[:, col[x]], rng)
                    evidence[x] = place_evidence_blocks(fld, blocks, counts)
                else:
                    n = np.rint(A[:, col[x]] * S).astype(int)
                    evidence[x] = place_evidence(fld, n)
        return evidence, A

    def fill_fn(self, ctx, rng, M, key=None):
        U = {}

        def fn(method, child, space):
            n = child.shape[1]
            if space not in U:
                U[space] = (rng_for(*key, "fill", space) if key is not None else rng).random((M, n))
            frac = np.asarray(ctx.adjacency[space] @ child.T.astype(float)).T
            if method in self.acc:
                a, b = self.acc[method]
            else:
                fb = getattr(self.domain, "fill_fallback", {}).get(method)
                a, b = self.acc.get(fb, next(iter(self.acc.values()), (0.0, 0.0)))
            p = 1.0 / (1.0 + np.exp(-(a + b * frac)))
            return (U[space] < p) & ~child

        return fn

    def rollout(self, request: Request, ctx: RequestContext, cands: Sequence[Compiled], M: Optional[int] = None,
                seed: int = 0, observe: bool = False, known=None) -> Rollouts:
        t0 = time.perf_counter()
        M = M or self.cfg.M
        independent = self.cfg.independent
        separate = independent or "shared" in self.cfg.ablate
        K = len(cands)
        Yp = np.zeros((K, M))
        if separate:
            ectxs = []
            for k, c in enumerate(cands):
                rng = rng_for(seed, request.qid, k, "sep")
                ev, _ = self.sample_evidence(ctx, sorted(set(c.read_set)), M, rng, observe, independent, known)
                ectx = EvalContext(ev, ctx.unitmap, ctx.n_units, M, ctx.pool_maps, self.fill_fn(ctx, rng, M))
                Yp[k] = ectx.coverage(c.root) if c.root is not None else 0.0
                ectxs.append(ectx)
        else:
            pol = self.policy(observe)
            if pol.enabled:
                pol = replace(pol, items=frozenset(int(x) for c in cands for x in c.read_set))
            ev = self.world_draws(request, ctx, M, seed, pol, known)
            rng = rng_for(seed, request.qid, "shared")
            ectx = EvalContext(ev, ctx.unitmap, ctx.n_units, M, ctx.pool_maps,
                               self.fill_fn(ctx, rng, M, key=(seed, request.qid)))
            for k, c in enumerate(cands):
                Yp[k] = ectx.coverage(c.root) if c.root is not None else 0.0
            ectxs = [ectx] * K
        rng_c = rng_for(seed, request.qid, "compute")
        run_rng = lambda c: rng_for(seed, request.qid, spec_key(c.spec), "run")
        C, T, E = self.compute_rollout(cands, ectxs, M, rng_c, shared_theta=not separate, run_rng=run_rng)
        Y = (1 - E) * Yp
        out = Rollouts(Yp, Y, C, T, E)
        out.seconds = time.perf_counter() - t0
        out.n_nodes = sum(e.n_evaluated for e in set(ectxs)) if ectxs else 0
        return out

    def world_draws(self, request, ctx, M, seed, observe=False, known=None) -> dict:
        pol = self.policy(observe)
        key = (request.qid, seed, M, pol.enabled, pol.error_model, pol.items)
        if known is None and key in self._draws:
            return self._draws[key]
        xs = sorted(int(x) for x in ctx.items.index)
        rng = rng_for(seed, request.qid, "shared")
        ev = self.sample_evidence(ctx, xs, M, rng, observe, known=known)[0] if xs else {}
        if known is None:
            if len(self._draws) >= 4:
                self._draws.pop(next(iter(self._draws)))
            self._draws[key] = ev
        return ev

    def compute_rollout(self, cands, ectxs, M, rng, shared_theta=True, run_rng=None):
        K = len(cands)
        C, T, E = np.zeros((K, M)), np.zeros((K, M)), np.zeros((K, M))
        rows, owners = [], []
        recs = []
        for k, c in enumerate(cands):
            for i, o in enumerate(c.ops):
                n = o.records(ectxs[k]) if callable(o.records) else np.full(M, float(o.records))
                n = np.broadcast_to(np.asarray(n, float), (M,))
                recs.append(n)
                rows.append(self._op_features(o.op, o.bytes, float(np.mean(n)), o.psi))
                owners.append((k, i))
        if not rows:
            return C, T, E + 1
        X = np.array(rows)
        Qc, Qt, Qm = self.cost_q.predict(X), self.time_q.predict(X), self.mem_q.predict(X)
        if "residual" in self.cfg.ablate:
            Qc = np.zeros_like(Qc)
            Qt = np.zeros_like(Qt)
        else:
            unseen = np.array([OPS[int(i)] not in self.seen_ops for i in X[:, 0]])
            if unseen.any():
                Qc, Qt = Qc.copy(), Qt.copy()
                Qc[unseen] = 0.0
                Qt[unseen] = 0.0

        def theta():
            speed = np.exp(rng.normal(self.speed_mu, self.speed_sd, M))
            head = rng.choice(self.headrooms, M) if len(self.headrooms) else np.ones(M)
            trans = rng.random(M) < self.transient_rate
            return speed, head, trans

        th = theta()
        r = 0
        for k, c in enumerate(cands):
            nops = len(c.ops)
            if not shared_theta:
                th = theta()
            speed, head, trans = th
            run = run_rng(c) if run_rng is not None else rng
            nu = run.random(M)
            sl = slice(r, r + nops)
            Rc = interp_quantile(Qc[sl], LEVELS, np.tile(nu[:, None], (1, nops)))
            Rt = interp_quantile(Qt[sl], LEVELS, np.tile(nu[:, None], (1, nops)))
            Rm = interp_quantile(Qm[sl], LEVELS, np.tile(nu[:, None], (1, nops)))
            usd = np.zeros((M, nops))
            tm = np.zeros((M, nops))
            for i, o in enumerate(c.ops):
                n = recs[r + i]
                ph = np.array([physical(o.op, o.bytes, ni, o.psi) for ni in n]) if callable(o.records) else \
                    np.full(M, physical(o.op, o.bytes, n[0], o.psi))
                cpu = ph * np.exp(Rc[:, i]) / speed
                tm[:, i] = physical_time(o.op, ph / speed) * np.exp(Rt[:, i])
                usd[:, i] = n * LM[o.psi.get("model_name", "cheap")]["usd"] * self.rates["tokens"] \
                    if o.op == "extract" else cpu * WORKER["usd_per_cpu_s"] * self.rates["compute"]
            memfail = np.exp(Rm) > WORKER["mem_gb"] * head[:, None]
            fail_op = np.where(trans, run.integers(0, nops, M), nops)
            mf = np.where(memfail.any(1), memfail.argmax(1), nops)
            first = np.minimum(fail_op, mf)
            if len(c.read_set) == 0:
                first[:] = 0
            failed = first < nops
            upto = np.arange(nops)[None, :] <= first[:, None]
            cu = (usd * upto).sum(1)
            tt = (tm * upto).sum(1)
            transfer = (sum(o.transfer_bytes for o in c.ops) / 1e9 * WORKER["usd_per_gb_out"] +
                        sum(o.bytes for o in c.ops if o.op == "read") / 1e9 * WORKER["usd_per_gb_in"]) * \
                self.rates["transfer"]
            C[k] = np.where(failed, 0.5 * cu, cu + transfer)
            T[k] = tt
            E[k] = failed
            r += nops
        return C, T, E

    def read_cost(self, ctx, comp: Compiled, x: int, q: float = 0.5) -> float:
        op = self.domain.item_read_op(ctx, comp, x)
        if op is None:
            return 0.0
        ph = physical(op.op, op.bytes, float(op.records), op.psi)
        res = 0.0
        if op.op in self.seen_ops and "residual" not in self.cfg.ablate:
            Q = self.cost_q.predict(np.array([self._op_features(op.op, op.bytes, float(op.records), op.psi)]))
            res = float(interp_quantile(Q, LEVELS, np.array([[q]]))[0, 0])
        cpu = ph * np.exp(res) / np.exp(self.speed_mu)
        usd = op.records * LM[op.psi.get("model_name", "cheap")]["usd"] * self.rates["tokens"] if op.op == "extract" \
            else cpu * WORKER["usd_per_cpu_s"] * self.rates["compute"]
        return float(usd + op.bytes / 1e9 * WORKER["usd_per_gb_in"] * self.rates["transfer"])

    def read_costs(self, ctx, comp: Compiled, xs=None) -> Dict[int, float]:
        xs = sorted(set(comp.read_set)) if xs is None else xs
        return {int(x): self.read_cost(ctx, comp, int(x)) for x in xs}

    def set_rates(self, compute: Optional[float] = None, tokens: Optional[float] = None,
                  transfer: Optional[float] = None):
        for k, v in (("compute", compute), ("tokens", tokens), ("transfer", transfer)):
            if v is not None:
                self.rates[k] = float(v)
