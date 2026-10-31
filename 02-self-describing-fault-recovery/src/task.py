"""Self-describing task structure and the deterministic 3-stage demo job.

A ``TaskDesc`` carries everything a worker needs to (re)start execution: task/job id,
next stage id, the full stage plan, a reference to its input (original partition or a
checkpoint stored at the coordinator), attempt counter and an input checksum.

Stages are pure functions on numpy arrays (key, value); the per-record padding field
is *not* materialised, its length is kept as metadata so that intermediate-result
sizes can be varied (combo C) without using gigabytes of RAM.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

N_KEYS = 64
BASE_RECORD_BYTES = 16  # int64 key + float64 value


@dataclass(frozen=True)
class StageSpec:
    name: str
    cost_us_per_record: float   # simulated compute cost per input record


@dataclass(frozen=True)
class Combo:
    name: str
    records_per_task: int
    filter_cost_us: float
    transform_cost_us: float
    aggregate_cost_us: float
    pad_bytes: int              # logical padding for the simulator; not allocated
    desc: str = ""
    wire_payload: int = 0       # real bytes/record carried in multiprocess checkpoints

    def plan(self) -> Tuple[StageSpec, ...]:
        return (StageSpec("filter", self.filter_cost_us), StageSpec("transform", self.transform_cost_us),
                StageSpec("aggregate", self.aggregate_cost_us))


COMBOS: Dict[str, Combo] = {
    "A": Combo("A", 20000, 5.0, 10.0, 2.0, 16, "计算轻、状态小"),
    "B": Combo("B", 20000, 5.0, 120.0, 2.0, 16, "计算重、状态小"),
    "C": Combo("C", 20000, 5.0, 10.0, 2.0, 4096, "计算轻、状态大"),
    "D": Combo("D", 20000, 5.0, 120.0, 2.0, 1024, "计算重、状态较大"),
}
TUNE_COMBO = Combo("T", 16000, 5.0, 60.0, 2.0, 1024, "调参配置（不参与正式实验）")


@dataclass(frozen=True)
class TaskDesc:
    task_id: str
    job_id: str
    stage_id: int                       # next stage to run (0-based)
    stage_plan: Tuple[StageSpec, ...]
    input_ref: str                      # "orig:<task_id>" or "ckpt:<task_id>:<stage>"
    attempt: int = 0
    saved_stages: Tuple[int, ...] = ()
    checksum: str = ""
    needed_refs: Tuple[str, ...] = ()  # intermediates required to (re)start; usually (input_ref,)

    def next_attempt(self, stage_id: int, input_ref: str) -> "TaskDesc":
        return replace(self, stage_id=stage_id, input_ref=input_ref, attempt=self.attempt + 1)


@dataclass
class Data:
    key: np.ndarray
    val: np.ndarray
    pad: int
    extra: Optional[np.ndarray] = None  # real payload pickled with multiprocess checkpoints

    @property
    def nbytes(self) -> int:
        """Logical size used by the single-process simulator (includes unallocated pad)."""
        return int(len(self.key) * (BASE_RECORD_BYTES + self.pad))


def make_partition(task_idx: int, combo: Combo, seed: int) -> Data:
    rng = np.random.default_rng(seed * 100003 + task_idx)
    n = combo.records_per_task
    extra = np.zeros((n, combo.wire_payload), dtype=np.uint8) if combo.wire_payload else None
    return Data(rng.integers(0, N_KEYS, n), rng.normal(0.0, 1.0, n), 0, extra)


def digest_data(d: Data) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(d.key).tobytes()); h.update(np.round(d.val, 9).tobytes())
    return h.hexdigest()[:16]


def run_stage(i: int, d: Data, combo: Combo) -> Data:
    if i == 0:   # filter: keep records with value > -0.1 (about 54 %)
        m = d.val > -0.1
        extra = None if d.extra is None else d.extra[m]
        return Data(d.key[m], d.val[m], combo.pad_bytes, extra)
    if i == 1:   # transform: deterministic numeric map
        v = d.val
        for _ in range(3):
            v = np.tanh(v * 1.5) + 0.1 * v * v
        return Data(d.key, v, combo.pad_bytes, d.extra)
    # aggregate: per-key sum and count; the bulk payload is not part of the final vector
    s = np.bincount(d.key, weights=d.val, minlength=N_KEYS)
    c = np.bincount(d.key, minlength=N_KEYS).astype(np.float64)
    return Data(np.arange(N_KEYS), np.concatenate([s, c]), 0, None)


def stage_cost_s(i: int, n_in: int, combo: Combo) -> float:
    return n_in * combo.plan()[i].cost_us_per_record * 1e-6


def global_digest(partials: List[Data]) -> str:
    tot = np.zeros(2 * N_KEYS)
    for p in partials:
        tot += p.val
    return hashlib.sha256(np.round(tot, 6).tobytes()).hexdigest()[:16]
