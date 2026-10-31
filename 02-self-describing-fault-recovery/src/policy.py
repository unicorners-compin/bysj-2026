"""Checkpoint policy shared by the event-driven simulator and the multiprocess runtime.

Selective saving keeps a stage output when the compute spent since the previous
checkpoint is at least ``lam`` times the cost of persisting that output:

    C_k >= lam * P_k,   P_k = nbytes / bandwidth + fixed save cost.
"""
from __future__ import annotations


def should_save(strategy: str, lam: float, acc_compute: float, nbytes: int, bw: float, s0: float) -> bool:
    if strategy == "recompute":
        return False
    if strategy == "per_stage":
        return True
    if strategy == "selective":
        return acc_compute >= lam * (nbytes / bw + s0)
    raise ValueError(f"unknown strategy {strategy}")
