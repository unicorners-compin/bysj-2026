"""Stable link is flat; dynamic link varies. Held-out seed 0 keeps the qualitative gap."""
import numpy as np

from src.experiment import BUDGET_MS, evaluate_seed
from src.netsim import STABLE_LINK, NodeCfg, dynamic_link_sequence, make_trace


def test_dynamic_sequence_has_bursts_and_wander():
    seq = dynamic_link_sequence(2000, np.random.default_rng(0))
    bw = np.array([link.bw_mbps for link in seq])
    assert bw.min() < 5.0
    assert bw.max() > 70.0
    assert np.mean(bw < 20.0) > 0.05
    assert np.unique(np.round(bw, 3)).size > 2


def test_stable_trace_delay_is_constant():
    timing = {"stage1": 0.12, "exit1_head": 0.02, "stage2_with_final_exit": 0.13}
    tr = make_trace(50, STABLE_LINK, timing, NodeCfg(), np.random.default_rng(0))
    assert np.allclose(tr.t_link, tr.t_link[0])
    assert tr.t_link[0] > 0


def test_seed0_stable_comparable_and_dynamic_la_better():
    rows, _, _ = evaluate_seed(0)
    by = {(r["regime"], r["policy"]): r for r in rows}
    stable_fixed = by[("stable", "fixed")]["ontime_accuracy"]
    stable_la = by[("stable", "latency_aware")]["ontime_accuracy"]
    assert abs(stable_la - stable_fixed) < 0.01
    assert stable_la <= stable_fixed + 1e-9
    assert by[("stable", "fixed")]["deadline_hit_rate"] == 1.0
    assert by[("stable", "latency_aware")]["deadline_hit_rate"] == 1.0

    dyn_fixed = by[("dynamic", "fixed")]
    dyn_la = by[("dynamic", "latency_aware")]
    assert dyn_la["ontime_accuracy"] > dyn_fixed["ontime_accuracy"] + 0.02
    assert dyn_la["deadline_hit_rate"] > dyn_fixed["deadline_hit_rate"] + 0.02
    assert BUDGET_MS == 20.0
