"""Latency-aware early-exit for a multi-stage pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class StageResult:
    stage: int
    confidence: float
    latency_ms: float
    output: str


StageFn = Callable[[], Tuple[float, str, float]]  # confidence, output, cost_ms


def run_with_budget(
    stages: List[StageFn],
    budget_ms: float,
    confidence_threshold: float = 0.85,
) -> StageResult:
    """Run stages sequentially; exit early on confidence or remaining budget."""
    elapsed = 0.0
    last: Optional[StageResult] = None
    for i, stage in enumerate(stages):
        # Peek cost by calling; stages are pure/demo and cheap.
        conf, out, cost = stage()
        if last is not None and elapsed + cost > budget_ms:
            return last
        elapsed += cost
        last = StageResult(stage=i + 1, confidence=conf, latency_ms=elapsed, output=out)
        if conf >= confidence_threshold or elapsed >= budget_ms:
            return last
    assert last is not None
    return last


def demo_stages(seed: int = 0):
    """Deterministic toy stages for demos/tests."""
    seq = [
        (0.55, "coarse", 8.0),
        (0.72, "mid", 12.0),
        (0.91, "fine", 20.0),
    ]
    out = []
    for conf, label, cost in seq:
        def make(c=conf, lab=label, cst=cost):
            return lambda: (c, lab, cst)
        out.append(make())
    return out


def random_stages(rng, n_stages: int = 4):
    """Toy multi-stage pipeline: confidence rises, cost grows (simulated)."""
    import random
    assert isinstance(rng, random.Random)
    base_conf = rng.uniform(0.35, 0.55)
    stages = []
    for i in range(n_stages):
        conf = min(0.99, base_conf + i * rng.uniform(0.12, 0.22) + rng.uniform(-0.03, 0.03))
        cost = rng.uniform(5.0, 10.0) + i * rng.uniform(6.0, 12.0)
        label = f"stage{i+1}"
        def make(c=conf, lab=label, cst=cost):
            return lambda: (c, lab, cst)
        stages.append(make())
    return stages
