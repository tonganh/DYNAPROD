# Settings

`dynaprod` takes its settings from `dynaprod/config.py`, which is plain Python,
so a deployment overrides them in code rather than through a config file:

```python
from dynaprod.config import LM, WORKER, WMConfig
from dynaprod.core import Calibrator, WorldModel, family_size, set_tail_scales
from dynaprod.service import DynaProd, PlanningSnapshot

WORKER["usd_per_cpu_s"] = 0.0504 / 3600                    # worker price per CPU second
LM["cheap"] = dict(usd=0.15e-6, tps=6000.0)                 # extraction model: USD per token, tokens/s
wm = WorldModel(domain, WMConfig(M=200, seed=0)).fit(records)
set_tail_scales(*wm.tail_scales)                           # cost and time tails beyond the draws
cal = Calibrator(alpha_Y=0.10, alpha_C=0.05, alpha_T=0.05,  # levels for coverage, cost, time
                 strata=family_size(min_n=30)).fit(entries, rng)
snap = PlanningSnapshot(period=0, wm=wm, calibrator=cal, M=200, m=8,
                        agent_kw=dict(max_cands=8, rounds=2))
svc = DynaProd(domain, snap, gamma=0.02, ttl=24 * 3600)     # online step, quote lifetime (s)
```

The language model of the data agent comes from the environment (`.env`):

| variable | what it names |
| --- | --- |
| `OPENROUTER_API_KEY` | the key; without it the agent parses by rule and composes from templates |
| `OPENROUTER_MODEL` | the agent's model |
| `OPENROUTER_MODEL_2` | a second model family |
| `OPENROUTER_BASE_URL` | the endpoint (default `https://openrouter.ai/api/v1`) |
