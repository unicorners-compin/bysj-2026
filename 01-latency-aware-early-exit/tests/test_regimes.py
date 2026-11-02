"""Link regimes and the held-out seed-0 comparison across budgets."""
import numpy as np

from src.experiment import BUDGETS, REGIMES, evaluate_seed
from src.netsim import STABLE_LINK, NodeCfg, dynamic_link_sequence, make_trace, moderate_link_sequence


def test_dynamic_sequence_has_bursts_and_wander():
    seq = dynamic_link_sequence(2000, np.random.default_rng(0))
    bw = np.array([link.bw_mbps for link in seq])
    assert bw.min() < 5.0
    assert bw.max() > 70.0
    assert np.mean(bw < 20.0) > 0.05


def test_moderate_fluctuation_sits_between_stable_and_dynamic():
    moderate = np.array([link.bw_mbps for link in moderate_link_sequence(4000, np.random.default_rng(0))])
    dynamic = np.array([link.bw_mbps for link in dynamic_link_sequence(4000, np.random.default_rng(0))])
    assert moderate.min() > dynamic.min()
    assert np.mean(moderate < 20.0) < np.mean(dynamic < 20.0)
    assert moderate.max() > 40.0


def test_stable_trace_delay_is_constant():
    timing = {"stage1": 0.12, "exit1_head": 0.02, "stage2_with_final_exit": 0.13}
    tr = make_trace(50, STABLE_LINK, timing, NodeCfg(), np.random.default_rng(0))
    assert np.allclose(tr.t_link, tr.t_link[0])
    assert tr.t_link[0] > 0


def test_seed0_stable_and_loose_comparable_fluctuation_tight_la_better():
    rows, _, _ = evaluate_seed(0)
    by = {(r["regime"], r["budget_level"], r["policy"]): r for r in rows}
    assert set(REGIMES) == {"stable", "moderate", "dynamic"}
    assert [level for level, _ in BUDGETS] == ["tight", "mid", "loose"]

    for level, _ in BUDGETS:
        fixed = by[("stable", level, "fixed")]["ontime_accuracy"]
        aware = by[("stable", level, "latency_aware")]["ontime_accuracy"]
        assert aware <= fixed + 1e-9
        assert abs(aware - fixed) < 0.01
        assert by[("stable", level, "fixed")]["deadline_hit_rate"] == 1.0

    for regime in REGIMES:
        fixed = by[(regime, "loose", "fixed")]["ontime_accuracy"]
        aware = by[(regime, "loose", "latency_aware")]["ontime_accuracy"]
        assert aware <= fixed + 1e-9
        assert abs(aware - fixed) < 0.01

    for level in ("tight", "mid"):
        fixed = by[("dynamic", level, "fixed")]
        aware = by[("dynamic", level, "latency_aware")]
        assert aware["ontime_accuracy"] > fixed["ontime_accuracy"] + 0.02
        assert aware["deadline_hit_rate"] > fixed["deadline_hit_rate"] + 0.02

    fixed = by[("moderate", "tight", "fixed")]
    aware = by[("moderate", "tight", "latency_aware")]
    assert aware["ontime_accuracy"] > fixed["ontime_accuracy"] + 0.005
    assert aware["deadline_hit_rate"] > fixed["deadline_hit_rate"]
