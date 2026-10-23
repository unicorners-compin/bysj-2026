from src.fault import FaultDesc, FaultKind, plan_recovery, recover_until_ok


def test_transient_retries():
    f = FaultDesc(FaultKind.TRANSIENT, "nic", "timeout", True, 2)
    assert plan_recovery(f, 0).name == "retry"
    assert recover_until_ok(f).name == "isolate"


def test_degraded_fallback():
    f = FaultDesc(FaultKind.DEGRADED, "model", "drift", False)
    assert plan_recovery(f).name == "degrade"
