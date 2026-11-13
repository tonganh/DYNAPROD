# DynaProd

Code for *Towards Dynamic Data Pricing: On-Demand Agentic Data Products with
Probabilistic World Models*.

DynaProd estimates the **coverage, cost and running time** of an on-demand data
product before any workflow builds it. A data agent reads the buyer's request
and composes candidate workflows. A probabilistic world model compiles each
candidate into a monotone outcome lineage and rolls all candidates out on
shared draws. Conformal calibration turns the rollouts into bounds that stay
coherent across related products. The undominated candidates become the
options the buyer chooses from or the seller prices.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

LLM calls are optional and go through [OpenRouter](https://openrouter.ai). To
enable them, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`,
`OPENROUTER_MODEL` and `OPENROUTER_MODEL_2`. Without a key, the agent falls
back to a rule parser and composes from templates.

## Data

| Domain | Source |
|---|---|
| Earth observation | [Sentinel-2 L2A](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED), [Landsat 8](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2) / [Landsat 9](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2) C2L2 and [Sentinel-1](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) scenes (Google Earth Engine) with their clear masks in `data/eo/`, plus [Copernicus EMS](https://mapping.emergency.copernicus.eu/) activations |
| Sensor panels | [UCI Beijing Multi-Site Air-Quality](https://archive.ics.uci.edu/dataset/501), in `data/raw/beijing/` |
| Documents | [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset), fetched into `data/raw/20ng/` |

## Use

```python
from dynaprod import DynaProd

svc = DynaProd(domain, snapshot)                 # snapshot: fitted world model and calibrator
quote = svc.estimate(request, now=0)             # options with coverage, cost and time bounds
opt = svc.accept(quote, quote.options[0].index, now=3600)
svc.deliver(quote, svc.build(quote))             # the online level moves on the delivery
svc.audit(other_request)                         # audits feed the next calibration period
svc.next_period()
```

How to build the snapshot and set the worker rates, levels and strata is in
[configs/README.md](configs/README.md).

## Layout

```
dynaprod/
  agent/           data agent and its tools
  core/            lineage, world model, fields, quantile heads, cost model,
                   containment, share bounds, read sets, calibration, strata,
                   audits, options
  domains/         Earth observation, sensor panels, documents
  llm/             cached OpenRouter client
  config.py        worker rates, levels, world-model settings
  service.py       snapshots, quotes, delivery, audits
configs/           how to override the settings
tests/             unit tests
```

## License

MIT, see [LICENSE](LICENSE).
