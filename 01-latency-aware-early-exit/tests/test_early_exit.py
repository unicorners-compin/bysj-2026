from src.early_exit import demo_stages, run_with_budget


def test_tight_budget_exits_early():
    r = run_with_budget(demo_stages(), budget_ms=10.0)
    assert r.stage == 1
    assert r.latency_ms <= 10.0 + 1e-6


def test_confidence_exits_before_last():
    r = run_with_budget(demo_stages(), budget_ms=100.0, confidence_threshold=0.85)
    assert r.stage == 3
    assert r.confidence >= 0.85
