from src.des import Fault, SimCfg, run
from src.task import COMBOS


def test_fault_free_digest_same_for_all_strategies():
    d = {run(COMBOS["A"], SimCfg(strategy=s))["digest"] for s in ("recompute", "per_stage", "selective")}
    assert len(d) == 1


def test_crash_recovery_is_consistent():
    ref = run(COMBOS["B"], SimCfg(strategy="per_stage"))
    for tm in ("during", "before_ack", "after_ack"):
        r = run(COMBOS["B"], SimCfg(strategy="per_stage"), Fault("fatal", 1, stage=1, timing=tm))
        assert r["completed"] and r["digest"] == ref["digest"]


def test_no_recovery_fails_on_crash():
    r = run(COMBOS["A"], SimCfg(policy="none"), Fault("fatal", 0, at=0.05))
    assert not r["completed"]
