import json

import numpy as np
import pandas as pd
import pytest

from dynaprod.core.run_logs import spec_key
from dynaprod.domains.eo import EarthObservation
from dynaprod.domains.eo_archive import MaskArchive
from dynaprod.utils import rng_for


def test_eo_reads_the_scene_dataset(tmp_path):
    sim = EarthObservation(seed=0, quick=True).archive
    (tmp_path / "scenes").mkdir()
    rows = []
    start = np.datetime64("2024-01-01")
    for a in range(6):
        A, sc = sim.aois[a], sim.scenes(a)
        rows.append(dict(id=A["id"], region=f"zone{A['region']}", side_km=A["side"]))
        clear = np.stack([np.unpackbits(s["mask"])[: sim.G ** 2].astype(bool) for s in sc])
        np.savez(tmp_path / "scenes" / f"{A['id']}.npz", source=np.array([s["source"] for s in sc]),
                 day=np.array([str(start + s["day"]) for s in sc]),
                 cf=np.array([np.nan if s["source"] == "S1" else s["cf"] for s in sc]),
                 footprint=np.array([s["footprint"] for s in sc]), clear=clear)
    pd.DataFrame(rows).to_csv(tmp_path / "aois.csv", index=False)
    pd.DataFrame([dict(aoi=rows[0]["id"], day="2024-07-19", hazard="flood")]).to_csv(tmp_path / "activations.csv", index=False)
    dom = EarthObservation(seed=0, quick=True, archive=str(tmp_path), per_family=5)
    day0 = min(s["day"] for a in range(6) for s in sim.scenes(a))
    assert isinstance(dom.archive, MaskArchive) and dom.G == sim.G and dom.n_aoi == 6
    assert np.array_equal(dom.scenes(2)[0]["mask"], sim.scenes(2)[0]["mask"])
    assert dom.scenes(2)[0]["day"] == sim.scenes(2)[0]["day"] - day0
    rapid = [q for q in dom.requests() if q.family == "rapid"]
    assert len(rapid) == 1 and rapid[0].t == 200 - day0 and rapid[0].req["hazard"] == "flood"
    ctx = dom.context(rapid[0])
    assert len(ctx.items) > 0 and ctx.items["clim"].between(0.03, 0.97).all()
    region = rows[0]["region"]
    assert dom.archive.climatology(region, 200, before=-1) == 0.5


def test_execution_uses_logged_runs(tmp_path):
    dom = EarthObservation(seed=0, quick=True)
    q = dom.requests()[0]
    ctx = dom.context(q)
    logged, unlogged = dom.candidate_pool(q, ctx, include_unseen=False)[:2]
    ops = [dict(op=o.op, cpu_s=1.0 + i, time_s=2.0, mem_gb=1.0, failed=False) for i, o in enumerate(logged.ops)]
    rec = dict(request=q.qid, workflow=logged.spec, failed=False, cost_usd=0.42, time_s=9.0, speed=1.1, ops=ops)
    (tmp_path / "runs.jsonl").write_text(json.dumps(rec, default=str) + "\n")
    dom.use_run_logs(str(tmp_path / "runs.jsonl"))
    out = dom.execute(ctx, logged, rng_for(0, "x"))
    assert (out.C, out.T, out.E) == (0.42, 9.0, 0) and out.log["run"].speed == 1.1
    assert [o.cpu_s for o in out.log["run"].ops] == [o["cpu_s"] for o in ops]
    assert spec_key(unlogged.spec) != spec_key(logged.spec)
    with pytest.raises(ValueError, match="not in the execution logs"):
        dom.execute(ctx, unlogged, rng_for(0, "x"))
    unlogged.template = "composed"
    other = dom.execute(ctx, unlogged, rng_for(0, "x"))
    assert other.C != 0.42 and dom.run_logs.hits == 1 and dom.run_logs.misses == 2
