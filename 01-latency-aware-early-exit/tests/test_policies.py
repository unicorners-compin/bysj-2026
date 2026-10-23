import numpy as np

from src.netsim import NETWORKS, NodeCfg, make_trace
from src.policies import Policy, run_policy


def _setup(n=200):
    rng = np.random.default_rng(0)
    timing = {"stage1": 0.1, "exit1_head": 0.01, "stage2_with_final_exit": 0.1}
    tr = make_trace(n, NETWORKS["poor"], timing, NodeCfg(), rng)
    conf = rng.uniform(0.3, 1.0, n)
    lab = rng.integers(0, 10, n)
    return tr, conf, lab, lab.copy(), lab.copy()


def test_full_sends_every_feature():
    tr, conf, p1, p2, lab = _setup()
    r = run_policy(Policy("full", "full"), conf, p1, p2, lab, tr, 1e9, NodeCfg())
    assert r["exit1_ratio"] == 0.0 and r["mean_tx_bytes"] == 6272


def test_feasibility_exits_when_budget_too_small():
    tr, conf, p1, p2, lab = _setup()
    r = run_policy(Policy("la", "dynamic"), conf, p1, p2, lab, tr, 10.0, NodeCfg())
    assert r["exit1_ratio"] == 1.0 and r["timeout_rate"] == 0.0
