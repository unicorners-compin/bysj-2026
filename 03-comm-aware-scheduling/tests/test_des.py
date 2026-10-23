from src.des import CONDS, POLICIES, NetCfg, simulate


def test_deterministic_and_joint_beats_nearest_under_high_load():
    cfg = NetCfg(n_tasks=1500)
    a = simulate(CONDS["C2"], cfg, POLICIES["joint"], 1)
    b = simulate(CONDS["C2"], cfg, POLICIES["joint"], 1)
    assert a["mean_ct_ms"] == b["mean_ct_ms"]
    n = simulate(CONDS["C2"], cfg, POLICIES["nearest"], 1)
    assert a["mean_ct_ms"] < n["mean_ct_ms"]
