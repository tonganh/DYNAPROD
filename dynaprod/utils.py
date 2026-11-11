import hashlib
import json
import os
import time

import numpy as np


def stable_seed(*args) -> int:
    h = hashlib.sha256(repr(args).encode()).hexdigest()
    return int(h[:8], 16)


def rng_for(*args) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*args))


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache")
OUTPUTS = os.environ.get("DYNAPROD_OUT", os.path.join(ROOT, "outputs"))
