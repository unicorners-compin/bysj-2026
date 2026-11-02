"""In-process simulation of a K-node serial inference pipeline.

Compute delays come from measured single-sample forward times (figs/compute_timing.csv)
multiplied by node speed factors. Link delay is computed in this process as

    t_tx = S*8/R + max(0, d + N(0, sigma^2))

No packet is sent and no network interface is used. Every random draw uses a numpy
Generator seeded from the experiment seed so that all strategies face the same delays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

FEATURE_BYTES = 32 * 7 * 7 * 4  # 6272 B, float32 32x7x7


@dataclass(frozen=True)
class Link:
    bw_mbps: float
    delay_ms: float
    jitter_ms: float

    def serialization_ms(self, nbytes: int = FEATURE_BYTES) -> float:
        return nbytes * 8 / (self.bw_mbps * 1e6) * 1e3


NETWORKS: Dict[str, Link] = {
    "good": Link(100.0, 5.0, 1.0),
    "medium": Link(20.0, 20.0, 4.0),
    "poor": Link(5.0, 50.0, 10.0),
}
NET_CN = {"good": "良好", "medium": "一般", "poor": "较差"}

# Stable regime: one bandwidth and one propagation delay for the whole run.
STABLE_LINK = Link(40.0, 8.0, 0.0)
# Prior used before any dynamic sample is observed (the provisioned good state).
DYNAMIC_PRIOR = Link(80.0, 6.0, 1.5)


@dataclass(frozen=True)
class NodeCfg:
    speed1: float = 4.0     # node 1 = weak end device (x4 slower than this PC)
    speed2: float = 1.0     # node 2 = edge server
    speed_relay: float = 1.0
    relay_fwd_ms: float = 0.05
    t_dec_ms: float = 0.01  # decision overhead (upper bound of measured cost)


@dataclass
class Trace:
    """Pre-drawn per-sample delays for one (seed, network, K) combination."""
    t1: np.ndarray        # simulated compute on node 1 (stage1 + exit1 head)
    t_link: np.ndarray    # total link (+relay) delay if the sample continues
    t2: np.ndarray        # simulated compute on node 2
    nominal_link: float   # configured mean of t_link (initial estimate)
    nominal_std: float


def dynamic_link_sequence(n: int, rng: np.random.Generator) -> List[Link]:
    """Time-varying link: slow bandwidth wander, congestion bursts, and jitter.

    Good state is 80 Mbit/s, 6 ms, jitter 1.5 ms. A burst (3.5 Mbit/s, 35 ms,
    jitter 6 ms) starts with probability 0.02 per sample and ends with
    probability 0.08. Bandwidth is scaled by ``1 + 0.15*sin(2π i / 500)``.
    """
    bad = False
    seq: List[Link] = []
    for i in range(n):
        if not bad and rng.random() < 0.02:
            bad = True
        elif bad and rng.random() < 0.08:
            bad = False
        wander = 1.0 + 0.15 * np.sin(2.0 * np.pi * i / 500.0)
        if bad:
            seq.append(Link(float(3.5 * wander), 35.0, 6.0))
        else:
            seq.append(Link(float(80.0 * wander), 6.0, 1.5))
    return seq


def make_trace(n: int, link: Link, timing: Dict[str, float], node: NodeCfg,
               rng: np.random.Generator, k_nodes: int = 2, link_seq: List[Link] | None = None) -> Trace:
    t1 = np.full(n, node.speed1 * (timing["stage1"] + timing["exit1_head"]))
    t2 = np.full(n, node.speed2 * timing["stage2_with_final_exit"])
    if link_seq is not None:  # time-varying network: one Link per sample
        ser = np.array([l.serialization_ms() for l in link_seq])
        d = np.array([l.delay_ms for l in link_seq])
        j = np.array([l.jitter_ms for l in link_seq])
        t_link = ser + np.maximum(0.0, d + rng.normal(0, 1, n) * j)
        return Trace(t1, t_link, t2, link.serialization_ms() + link.delay_ms, link.jitter_ms)
    if k_nodes == 2:
        t_link = link.serialization_ms() + np.maximum(0.0, link.delay_ms + rng.normal(0, link.jitter_ms, n))
        return Trace(t1, t_link, t2, link.serialization_ms() + link.delay_ms, link.jitter_ms)
    # K=3: node1 -> relay -> node2, two hops each with half the propagation delay
    half = Link(link.bw_mbps, link.delay_ms / 2, link.jitter_ms / np.sqrt(2))
    hop = lambda: half.serialization_ms() + np.maximum(0.0, half.delay_ms + rng.normal(0, half.jitter_ms, n))
    relay = node.speed_relay * node.relay_fwd_ms
    t_link = hop() + relay + hop()
    return Trace(t1, t_link, t2, 2 * half.serialization_ms() + link.delay_ms + relay, link.jitter_ms)
