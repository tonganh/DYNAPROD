from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from .compute import WORKER, OpLog, RunLog, op_usd
from .types import Compiled


def spec_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, default=str)


class RunLogStore:
    def __init__(self, path: str):
        self.path = path
        self.runs: Dict[Tuple[str, str], dict] = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    self.runs[(str(r["request"]), spec_key(r["workflow"]))] = r
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self.runs)

    def record(self, qid: str, comp: Compiled) -> Optional[dict]:
        return self.runs.get((qid, spec_key(comp.spec)))

    def accepted_fill(self, qid: str, comp: Compiled) -> Dict[str, list]:
        r = self.record(qid, comp)
        return dict(r.get("fill", {})) if r is not None else {}

    def run(self, qid: str, comp: Compiled, records: List[float]) -> Optional[RunLog]:
        r = self.record(qid, comp)
        if r is None:
            self.misses += 1
            return None
        self.hits += 1
        ops = r.get("ops", [])
        if [o["op"] for o in ops] != [o.op for o in comp.ops[:len(ops)]]:
            raise ValueError(f"{qid}: logged operators {[o['op'] for o in ops]} do not match the compiled "
                             f"workflow {[o.op for o in comp.ops]}")
        failed = bool(r["failed"])
        log = RunLog(float(r.get("speed", 1.0)), float(r.get("headroom", 1.0)),
                     bool(r.get("transient", failed and not any(o.get("mem_gb", 0) > WORKER["mem_gb"] for o in ops))))
        for o, spec, n in zip(ops, comp.ops, records):
            log.ops.append(OpLog(o["op"], dict(spec.psi), spec.bytes, float(o.get("records", n)), float(o["cpu_s"]),
                                 float(o["time_s"]), float(o.get("mem_gb", 0.0)), bool(o.get("failed", False))))
        log.C = float(r["cost_usd"]) if "cost_usd" in r else self._cost(comp, log, failed)
        log.T = float(r["time_s"]) if "time_s" in r else sum(o.time_s for o in log.ops)
        log.E = int(failed)
        return log

    @staticmethod
    def _cost(comp: Compiled, log: RunLog, failed: bool) -> float:
        usd = sum(op_usd(spec, o.cpu_s, o.records) for spec, o in zip(comp.ops, log.ops))
        if failed:
            return 0.5 * usd
        transfer = sum(o.transfer_bytes for o in comp.ops)
        read_bytes = sum(o.bytes for o in comp.ops if o.op == "read")
        return usd + transfer / 1e9 * WORKER["usd_per_gb_out"] + read_bytes / 1e9 * WORKER["usd_per_gb_in"]
