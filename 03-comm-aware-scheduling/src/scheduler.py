"""Communication-load-aware task placement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Node:
    name: str
    cpu_load: float  # 0..1
    comm_load: float  # 0..1 outbound/inbound aggregate


@dataclass
class Placement:
    task: str
    node: str
    score: float


def score_node(node: Node, alpha: float = 0.4, beta: float = 0.6) -> float:
    """Lower is better. Emphasize communication load by default."""
    return alpha * node.cpu_load + beta * node.comm_load


def place_tasks(tasks: List[str], nodes: List[Node]) -> List[Placement]:
    """Greedy: assign each task to the current best-scoring node, then bump load."""
    loads: Dict[str, Tuple[float, float]] = {
        n.name: (n.cpu_load, n.comm_load) for n in nodes
    }
    names = {n.name: n for n in nodes}
    out: List[Placement] = []
    for task in tasks:
        best_name = None
        best_score = float("inf")
        for name, (cpu, comm) in loads.items():
            s = score_node(Node(name, cpu, comm))
            if s < best_score:
                best_score = s
                best_name = name
        assert best_name is not None
        out.append(Placement(task=task, node=best_name, score=best_score))
        cpu, comm = loads[best_name]
        loads[best_name] = (min(1.0, cpu + 0.15), min(1.0, comm + 0.2))
    return out
