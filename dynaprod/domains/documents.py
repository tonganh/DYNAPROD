from __future__ import annotations

import os
import pickle
import re
from collections import Counter, defaultdict
from typing import List

import numpy as np
import pandas as pd
from scipy import sparse

from ..core.lineage import AnyOf, AtLeast, Ev
from ..core.types import Compiled, OpSpec, Request, RequestContext
from ..utils import DATA, rng_for, stable_seed
from .base import Domain, ValidationError

ATTRS = {
    "year": re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b"),
    "version": re.compile(r"\b\d+\.\d+(\.\d+)?\b"),
    "number": re.compile(r"\b\d{2,}\b"),
    "contact": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\(\d{3}\)\s?\d{3}-\d{4}"),
}
STOP = set("""The This That These Those There Then They Their What When Where Which While Who Why How And But
For Not You Your Our All Any Are Can Could Did Does Don Each From Have Has Had Her His Its Just Like May Might
Most Much Must Now One Only Other Some Such Than Also Here Into More Many Very Well Will Would With Without
About After Again Before Being Both Even Every First From Get Got Into Last Let Lines Subject Organization
Article Writes Wrote Distribution Reply Keywords Summary Path Newsgroups Message Date Sender References
Thanks Thank Please Yes Well Sure Maybe Actually Anyway However Therefore Also Since Because Still Though
Mon Tue Wed Thu Fri Sat Sun Jan Feb Mar Apr Jun Jul Aug Sep Oct Nov Dec University Inc NNTP Posting Host
Re In On At To Of If It Is As An Be By Do Go He Me My No Or So Up Us We""".split())

TEMPLATES = {
    "cheap_all": dict(model="cheap", docs="all", rule="any"),
    "accurate_all": dict(model="accurate", docs="all", rule="any"),
    "cheap_top3": dict(model="cheap", docs="top3", rule="any"),
    "accurate_top3": dict(model="accurate", docs="top3", rule="any"),
    "cheap_vote2": dict(model="cheap", docs="all", rule="vote2"),
    "cascade_all": dict(model="cascade", docs="all", rule="any"),
}
UNSEEN = {
    "cascade_top3": dict(model="cascade", docs="top3", rule="any"),
    "accurate_vote2": dict(model="accurate", docs="all", rule="vote2"),
}
P_MODEL = {"cheap": (0.88, 0.14, 0.35, 0.92), "accurate": (0.97, 0.05, 0.7, 0.99)}


def _p_success(model, tokens):
    a, b, lo, hi = P_MODEL[model]
    return float(np.clip(a - b * np.log10(max(tokens, 50) / 200.0), lo, hi))


class Documents(Domain):
    name = "documents"
    families = ["extract"]
    psi_keys = ["n_inputs", "model_accurate", "mean_len", "n_entities", "cascade", "vote", "top_k"]
    e_cols = ["e", "value_rate", "attr_rate"]
    field_scales = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    time_scales = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])

    def __init__(self, seed=0, quick=False, n_requests=None, max_entities=80):
        super().__init__(seed, quick)
        self.n_requests = n_requests or (120 if quick else 300)
        self.max_entities = max_entities
        self._load()

    def _load(self):
        cache = os.path.join(DATA, "20ng_index_v2.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                self.__dict__.update(pickle.load(f))
            return
        from sklearn.datasets import fetch_20newsgroups
        d = fetch_20newsgroups(subset="all", data_home=os.path.join(DATA, "raw", "20ng"))
        docs, topics = d.data, np.array(d.target)
        names = list(d.target_names)
        tok_len = np.array([max(1, len(t.split())) for t in docs])
        ent_re = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z]{2,})\b")
        doc_ents, df = [], Counter()
        for t in docs:
            ents = {m for m in ent_re.findall(t) if m not in STOP}
            doc_ents.append(ents)
            df.update(ents)
        vocab = [e for e, c in df.items() if 5 <= c <= 300]
        vocab = sorted(vocab, key=lambda e: -df[e])[:5803]
        eid = {e: i for i, e in enumerate(vocab)}
        mentions = {}
        value_rate = np.zeros(len(docs))
        attr_rate = np.zeros((len(ATTRS), len(docs)))
        for di, t in enumerate(docs):
            lines = t.split("\n")
            vr = 0
            for ln in lines:
                hits = [bool(r.search(ln)) for r in ATTRS.values()]
                vr += any(hits)
                attr_rate[:, di] += hits
            value_rate[di] = vr / max(1, len(lines))
            attr_rate[:, di] /= max(1, len(lines))
            ents = [e for e in doc_ents[di] if e in eid]
            if not ents:
                continue
            line_attrs = [[a for a, r in ATTRS.items() if r.search(ln)] for ln in lines]
            for li, ln in enumerate(lines):
                present = [e for e in ents if e in ln]
                if not present:
                    continue
                attrs = set()
                for lj in range(max(0, li - 2), min(len(lines), li + 3)):
                    attrs.update(line_attrs[lj])
                for e in present:
                    rec = mentions.setdefault((di, eid[e]), [0, set()])
                    rec[0] += 1
                    rec[1].update(attrs)
        state = dict(topics=topics, topic_names=names, tok_len=tok_len, vocab=vocab, value_rate=value_rate,
                     attr_rate=attr_rate,
                     mentions={k: (v[0], frozenset(v[1])) for k, v in mentions.items()},
                     doc_bytes=np.array([len(t.encode()) for t in docs]))
        with open(cache, "wb") as f:
            pickle.dump(state, f)
        self.__dict__.update(state)

    def _entity_docs(self):
        if not hasattr(self, "_ed"):
            ed = defaultdict(list)
            for (di, e), (cnt, attrs) in self.mentions.items():
                ed[e].append((di, cnt, attrs))
            self._ed = ed
        return self._ed

    def requests(self) -> List[Request]:
        if hasattr(self, "_requests"):
            return self._requests
        rng = rng_for(self.seed, "doc-requests")
        ed = self._entity_docs()
        reqs = []
        topic_df = defaultdict(Counter)
        for e, lst in ed.items():
            for di, _, _ in lst:
                topic_df[int(self.topics[di])][e] += 1
        for i in range(self.n_requests):
            topic = int(rng.integers(len(self.topic_names)))
            attr = str(rng.choice(list(ATTRS)))
            cand = [e for e, c in topic_df[topic].most_common(200) if c >= 3]
            n = int(rng.integers(20, self.max_entities + 1))
            ents = sorted(rng.choice(cand, min(n, len(cand)), replace=False).tolist())
            target = float(rng.choice([0.5, 0.6, 0.7, 0.8]))
            req = dict(family="extract", scope=dict(topic=self.topic_names[topic], entities=[self.vocab[e] for e in ents],
                                                    attribute=attr), target=target, budget=None, deadline=1800.0,
                       floors={})
            reqs.append(Request(f"doc-{i}", "documents", "extract", f"T{topic}", 0.0, "", req,
                                size_class="small" if len(ents) <= 50 else "large",
                                extra=dict(topic=topic, ent_ids=ents)))
        self._requests = reqs
        return reqs

    def describe_request(self, req, rng=None):
        s = req["scope"]
        b = f" Budget about ${req['budget']:.3f}." if req.get("budget") else ""
        return (f"From the '{s['topic']}' discussions, extract the {s['attribute']} mentioned for each of these "
                f"{len(s['entities'])} entities: {', '.join(s['entities'])}. We need a value for at least "
                f"{req['target']*100:.0f}% of them, within 30 minutes.{b}")

    def parse_rule(self, text):
        m = re.search(r"From the '([^']+)'", text)
        a = re.search(r"extract the (\w+)", text)
        ents = re.search(r"entities: (.+?)\. We need", text)
        pct = re.search(r"(\d+)%", text)
        bud = re.search(r"\$([\d.]+)", text)
        return dict(family="extract", scope=dict(topic=m.group(1) if m else None,
                                                 entities=[e.strip() for e in ents.group(1).split(",")] if ents else [],
                                                 attribute=a.group(1) if a else "number"),
                    target=float(pct.group(1)) / 100 if pct else 0.7,
                    budget=float(bud.group(1).rstrip(".")) if bud else None, deadline=1800.0, floors={})

    def _build_context(self, q):
        topic = q.extra["topic"]
        ents = q.extra["ent_ids"]
        attr = q.req["scope"]["attribute"]
        ai = list(ATTRS).index(attr)
        col = {e: j for j, e in enumerate(ents)}
        ed = self._entity_docs()
        per_doc = defaultdict(dict)
        for e in ents:
            for di, cnt, attrs in ed[e]:
                if self.topics[di] == topic:
                    per_doc[di][e] = (cnt, attr in attrs)
        top = defaultdict(set)
        for e in ents:
            lst = sorted(((per_doc[d][e][0], -d) for d in per_doc if e in per_doc[d]), reverse=True)[:3]
            for _, nd in lst:
                top[-nd].add(e)
        rows, scope, unitmap, group, truth = [], {}, {}, {}, {}
        U = len(ents)
        for di, emap in sorted(per_doc.items()):
            sc = np.array(sorted(col[e] for e in emap))
            for mi, model in enumerate(("cheap", "accurate")):
                x = di * 2 + mi
                rows.append(dict(item=x, doc=di, model=model, time=float(di), tokens=float(self.tok_len[di]),
                                 hits=len(emap), value_rate=float(self.value_rate[di]),
                                 attr_rate=float(self.attr_rate[ai, di]),
                                 e=float(min(1.0, 5 * self.attr_rate[ai, di])) * (0.7 if model == "cheap" else 0.9),
                                 exists=True))
                scope[x] = sc
                um = np.full(U, -1)
                um[sc] = np.arange(len(sc))
                tp = np.full(U, -1)
                for k, j in enumerate(sc):
                    if ents[j] in top[di]:
                        tp[j] = k
                unitmap[x] = {"own": um, "top": tp}
                group[x] = "all"
        items = pd.DataFrame(rows).set_index("item") if rows else pd.DataFrame(
            columns=["doc", "model", "time", "tokens", "hits", "value_rate", "attr_rate", "e", "exists"])
        docs = sorted(per_doc)
        Minc = np.zeros((U, max(1, len(docs))))
        for k, d in enumerate(docs):
            for e in per_doc[d]:
                Minc[col[e], k] = 1.0
        Minc = Minc / np.maximum(1, Minc.sum(1, keepdims=True)) ** 0.5
        try:
            u, s, _ = np.linalg.svd(Minc, full_matrices=False)
            coords = u[:, :3] * s[:3]
        except np.linalg.LinAlgError:
            coords = np.zeros((U, 3))
        coords = coords / (np.abs(coords).max() + 1e-9)
        ctx = RequestContext(q, items, U, coords, scope, unitmap, {}, {"fine": sparse.csr_matrix((U, U))}, group)
        ctx.extra.update(per_doc=per_doc, ents=ents, attr=attr)
        return ctx

    def item_features(self, ctx, xs):
        I = ctx.items.loc[list(xs)]
        F = pd.DataFrame(index=I.index)
        F["e"] = I["e"]
        F["value_rate"] = I["value_rate"]
        F["attr_rate"] = I["attr_rate"]
        F["log_tokens"] = np.log(I["tokens"])
        F["hits"] = I["hits"]
        F["accurate"] = (I["model"] == "accurate").astype(float)
        for a in ATTRS:
            F[f"attr_{a}"] = float(ctx.extra["attr"] == a)
        F["n_entities"] = ctx.n_units
        F["exists"] = True
        return F

    def item_distance(self, ctx, xs):
        I = ctx.items.loc[list(xs)]
        d = I["doc"].to_numpy()
        m = (I["model"] == "accurate").to_numpy()
        lt = np.log(I["tokens"].to_numpy(float))
        D = 1.0 * (d[:, None] != d[None, :]) + 0.3 * np.abs(lt[:, None] - lt[None, :]) + 0.1 * (m[:, None] != m[None, :])
        return D

    def order_key(self, ctx, xs):
        I = ctx.items.loc[list(xs)]
        docs = np.sort(ctx.items["doc"].unique())
        rank = np.searchsorted(docs, I["doc"].to_numpy())
        return (rank * 2 + (I["model"] == "accurate").to_numpy()).astype(float) / 2.0

    def true_evidence(self, ctx, x):
        di, model = x // 2, ("cheap", "accurate")[x % 2]
        ents = ctx.extra["ents"]
        sc = ctx.scope[x]
        p = _p_success(model, self.tok_len[di])
        out = np.zeros(len(sc), bool)
        for k, j in enumerate(sc):
            cnt, stated = ctx.extra["per_doc"][di][ents[j]]
            u = (stable_seed(self.seed, "extract", di, ents[j], ctx.extra["attr"]) % 100000) / 100000.0
            out[k] = stated and u < p
        return out

    def templates(self, family):
        return TEMPLATES

    def unseen(self, family):
        return UNSEEN

    def vocabulary(self):
        return {"spec_fields": {
            "model": "'cheap', 'accurate' or 'cascade' (cheap first; accurate only on documents of entities the cheap model left unresolved)",
            "docs": "'all' documents mentioning an entity, or 'top3' (the three documents that mention it most)",
            "rule": "'any' (one document returning a value suffices) or 'vote2' (two documents must return it)",
        }, "operators": ["select", "read", "extract", "vote", "export"]}

    def compile(self, spec, q, ctx, name="w"):
        model = spec.get("model", "cheap")
        docs = spec.get("docs", "all")
        rule = spec.get("rule", "any")
        if model not in ("cheap", "accurate", "cascade"):
            raise ValidationError("model must be cheap, accurate or cascade")
        if docs not in ("all", "top3"):
            raise ValidationError("docs must be all or top3")
        if rule not in ("any", "vote2"):
            raise ValidationError("rule must be any or vote2")
        I = ctx.items
        role = "own" if docs == "all" else "top"
        allowed = set(spec["items"]) if spec.get("items") is not None else None
        sel = []
        for x, r in I.iterrows():
            if model != "cascade" and r["model"] != model:
                continue
            if (ctx.unitmap[x][role] >= 0).any() and (allowed is None or x in allowed):
                sel.append(int(x))
        cheap = [x for x in sel if x % 2 == 0]
        acc = [x for x in sel if x % 2 == 1]
        if rule == "any":
            root = AnyOf([Ev(x, role) for x in sel])
        else:
            if model == "cascade":
                per_doc = defaultdict(list)
                for x in sel:
                    per_doc[x // 2].append(Ev(x, role))
                root = AtLeast(2, [AnyOf(v) for v in per_doc.values()]) if len(per_doc) >= 2 else None
            else:
                root = AtLeast(2, [Ev(x, role) for x in sel]) if len(sel) >= 2 else None
        if root is None:
            raise ValidationError("no candidate documents")
        tok = {x: float(I.loc[x, "tokens"] + 60 * len(ctx.scope[x])) for x in sel}
        nbytes = float(sum(self.doc_bytes[x // 2] for x in set(sel)))
        n_docs = len({x // 2 for x in sel})
        common = dict(n_entities=ctx.n_units, cascade=float(model == "cascade"), vote=float(rule == "vote2"),
                      top_k=float(docs == "top3"),
                      mean_len=float(np.mean([I.loc[x, "tokens"] for x in sel])))
        ops = [OpSpec("select", dict(common, n_inputs=n_docs), 0.0, n_docs),
               OpSpec("read", dict(common, n_inputs=n_docs), nbytes, n_docs)]
        if model in ("cheap", "accurate"):
            ops.append(OpSpec("extract", dict(common, n_inputs=len(sel), model_name=model,
                                              model_accurate=float(model == "accurate")), 0.0,
                              sum(tok.values())))
        else:
            ops.append(OpSpec("extract", dict(common, n_inputs=len(cheap), model_name="cheap", model_accurate=0.0),
                              0.0, sum(tok[x] for x in cheap)))
            cheap_root = AnyOf([Ev(x, role) for x in cheap])
            S = np.zeros((len(acc), ctx.n_units))
            for i, x in enumerate(acc):
                S[i, ctx.unitmap[x][role] >= 0] = 1
            tk = np.array([tok[x] for x in acc])

            def acc_tokens(ectx, cheap_root=cheap_root, S=S, tk=tk):
                unres = ~ectx.eval(cheap_root) if cheap_root is not None else np.ones((ectx.M, S.shape[1]), bool)
                need = (unres.astype(float) @ S.T) > 0
                return need.astype(float) @ tk + 1.0

            ops.append(OpSpec("extract", dict(common, n_inputs=len(acc), model_name="accurate", model_accurate=1.0,
                                              _nominal=0.4 * float(tk.sum())), 0.0, acc_tokens))
        if rule == "vote2":
            ops.append(OpSpec("vote", dict(common, n_inputs=len(sel)), 0.0, len(sel) * ctx.n_units))
        ops.append(OpSpec("export", dict(common, n_inputs=1), 0.0, ctx.n_units, transfer_bytes=ctx.n_units * 64.0))
        e = I.loc[sel, "e"].to_numpy(float)
        feats = dict(n_items=len(sel), n_docs=n_docs, accurate=float(model == "accurate"), cascade=float(model == "cascade"),
                     top3=float(docs == "top3"), vote2=float(rule == "vote2"), n_entities=ctx.n_units,
                     tokens=float(sum(tok.values())), e_mean=float(e.mean()), e_max=float(e.max()),
                     **{f"attr_{a}": float(ctx.extra["attr"] == a) for a in ATTRS})
        return Compiled(name, dict(spec), name, True, root, "fine", sorted(sel), ops, feats, output_units=ctx.n_units)

    def item_mb(self, ctx, x):
        return self.doc_bytes[x // 2] / 1e6

    def catalogue_text(self, q, ctx):
        I = ctx.items
        n_docs = I["doc"].nunique() if len(I) else 0
        return (f"{n_docs} documents of topic '{q.req['scope']['topic']}' mention at least one requested entity "
                f"(median length {I['tokens'].median():.0f} tokens, mean entity hits per document "
                f"{I['hits'].mean():.1f}). Extractors: cheap (fast, weaker on long documents) and accurate (20x price).")
