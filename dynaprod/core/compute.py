from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import LM, OMEGA, WORKER
from .types import Compiled, OpSpec


def physical(op: str, b: float, n: float, psi: Dict[str, float]) -> float:
    if op == "extract":
        return n / LM[psi.get("model_name", "cheap")]["tps"]
    return b / (WORKER["io_mb_s"] * 1e6) + n * OMEGA.get(op, 1e-7) + 1e-3


def physical_time(op: str, phys: float):
    if op == "extract":
        return phys
    return phys / (WORKER["vcpu"] * 0.7)


def _reference_residual(op: str, psi: Dict[str, float]) -> float:
    fan = psi.get("n_inputs", 1.0)
    r = 0.25 * np.log1p(fan) - 0.15 * psi.get("cached", 0.0)
    r += 0.35 * psi.get("remote_store", 0.0) + 0.2 * np.sin(psi.get("resolution", 20.0) / 15.0)
    r += 0.12 * np.log1p(psi.get("n_sources", 1.0)) + 0.1 * psi.get("radar", 0.0)
    if op == "fill":
        r += 0.4 * psi.get("method_temporal", 0.0) + 0.3 * psi.get("method_radar", 0.0)
    if op == "extract":
        r += 0.15 * np.log1p(psi.get("mean_len", 500) / 500.0)
    return float(r)


def _reference_memory_gb(op: str, b: float, n: float, psi: Dict[str, float]) -> float:
    sh = max(1.0, psi.get("shards", 1.0))
    b, n = b / sh, n / sh
    base = 0.3 + 2.2e-9 * n * (1.0 + 0.5 * psi.get("n_inputs", 1.0) ** 0.5) + b / 1e9 * 0.8
    if op in ("composite", "fuse"):
        base *= 1.6
    return float(base)


@dataclass
class OpLog:
    op: str
    psi: Dict[str, float]
    bytes: float
    records: float
    cpu_s: float
    time_s: float
    mem_gb: float
    failed: bool


@dataclass
class RunLog:
    speed: float
    headroom: float
    transient: bool
    ops: List[OpLog] = field(default_factory=list)
    C: float = 0.0
    T: float = 0.0
    E: int = 0


def op_usd(op: OpSpec, cpu_s: float, records: float) -> float:
    if op.op == "extract":
        return records * LM[op.psi.get("model_name", "cheap")]["usd"]
    return cpu_s * WORKER["usd_per_cpu_s"]


def execute_compute(compiled: Compiled, records: List[float], rng: np.random.Generator) -> RunLog:
    speed = float(np.exp(rng.normal(0.0, 0.25)))
    headroom = float(rng.uniform(0.8, 1.0))
    transient = bool(rng.random() < WORKER["transient_rate"])
    fail_at = int(rng.integers(len(compiled.ops))) if transient else -1
    z_run = rng.normal()
    log = RunLog(speed, headroom, transient)
    usd, T, failed = 0.0, 0.0, False
    if len(compiled.read_set) == 0:
        failed = True
    for i, (op, n) in enumerate(zip(compiled.ops, records)):
        if failed:
            break
        phys = physical(op.op, op.bytes, n, op.psi)
        res = _reference_residual(op.op, op.psi) + 0.12 * z_run + 0.08 * rng.normal()
        cpu = phys * np.exp(res) / speed
        t = cpu / (WORKER["vcpu"] * 0.7) + 0.05 * op.psi.get("n_inputs", 1.0) * np.exp(0.1 * rng.normal())
        if op.op == "extract":
            t = cpu
        mem = _reference_memory_gb(op.op, op.bytes, n, op.psi) * np.exp(0.1 * rng.normal())
        f = (mem > WORKER["mem_gb"] * headroom) or (i == fail_at)
        usd += op_usd(op, cpu, n)
        T += t
        log.ops.append(OpLog(op.op, dict(op.psi), op.bytes, float(n), float(cpu), float(t), float(mem), bool(f)))
        if f:
            failed = True
    transfer = sum(o.transfer_bytes for o in compiled.ops)
    read_bytes = sum(o.bytes for o in compiled.ops if o.op == "read")
    if failed:
        usd = 0.5 * usd
    else:
        usd += transfer / 1e9 * WORKER["usd_per_gb_out"] + read_bytes / 1e9 * WORKER["usd_per_gb_in"]
    log.C, log.T, log.E = float(usd), float(T), int(failed)
    return log
