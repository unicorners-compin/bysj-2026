"""Multi-process coordinator/worker prototype.

The kill test must fail before the runtime exists, and after it exists it checks
that SIGKILL of one worker still yields the fault-free aggregate and that a
recomputed stage is deduplicated by (task_id, stage_id).
"""
import os
import signal

import numpy as np

from src.mp_runtime import ConfirmLog, KillSpec, MpCfg, queued_bytes, run_job
from src.policy import should_save
from src.task import N_KEYS, Combo, Data, global_digest, make_partition, run_stage

LIGHT = Combo("L", 1200, 5.0, 10.0, 2.0, 16, "轻")
HEAVY = Combo("H", 200, 5.0, 10.0, 2.0, 16, "状态大", wire_payload=8192)


def test_should_save_recompute_per_stage_and_selective():
    bw, s0 = 50e6, 0.002
    nbytes = 1000
    persist = nbytes / bw + s0
    assert should_save("recompute", 1.0, persist * 10, nbytes, bw, s0) is False
    assert should_save("per_stage", 1.0, 0.0, nbytes, bw, s0) is True
    assert should_save("selective", 1.0, persist, nbytes, bw, s0) is True
    assert should_save("selective", 2.0, persist, nbytes, bw, s0) is False
    assert should_save("selective", 1.0, persist * 0.999, nbytes, bw, s0) is False


def test_confirm_log_dedups_task_stage():
    log = ConfirmLog()
    data = Data(np.arange(N_KEYS), np.ones(2 * N_KEYS), 0)
    assert log.accept("t000", 2, "final", data) == "new"
    assert log.accept("t000", 2, "final", data) == "dup"
    assert log.dup_intercepted == 1
    assert log.final_count == 1
    assert log.digest() == global_digest([data])
    assert np.allclose(log.total(), data.val)


def test_rejects_fewer_than_three_workers():
    try:
        run_job(LIGHT, MpCfg(n_workers=2, n_tasks=1, min_stage_s=0, timeout_s=5))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _sequential(combo, n_tasks, seed):
    partials = []
    for i in range(n_tasks):
        data = make_partition(i, combo, seed)
        for stage in range(3):
            data = run_stage(stage, data, combo)
        partials.append(data)
    return partials


def test_fault_free_matches_sequential_and_strategies():
    partials = _sequential(LIGHT, 3, 0)
    ref = global_digest(partials)
    ref_total = np.sum([p.val for p in partials], axis=0)
    digests = []
    for strategy in ("recompute", "per_stage", "selective"):
        result = run_job(LIGHT, MpCfg(strategy=strategy, n_workers=3, n_tasks=3, min_stage_s=0,
                                       data_seed=0, timeout_s=20))
        assert result["completed"] and result["final_count"] == 3
        assert result["digest"] == ref
        assert np.allclose(result["total"], ref_total)
        digests.append(result["digest"])
        assert result["assignments"]
        for item in result["assignments"]:
            assert item["task_id"].startswith("t")
            assert isinstance(item["stage_id"], int)
            assert item["input_ref"] in item["needed_refs"]
            assert item["checksum"]
    assert len(set(digests)) == 1


def test_state_transfer_counts_pickled_bytes_not_logical_pad():
    """Huge unallocated pad must not show up in the transfer metric."""
    padded = Combo("pad", 40, 5.0, 10.0, 2.0, 8_000_000, "仅逻辑填充")
    result = run_job(padded, MpCfg(strategy="per_stage", n_workers=3, n_tasks=1, min_stage_s=0,
                                   data_seed=0, timeout_s=20))
    data = make_partition(0, padded, 0)
    expect = 0
    logical = 0
    for stage in range(2):
        data = run_stage(stage, data, padded)
        expect += queued_bytes(data)
        logical += data.nbytes
    assert result["ckpt_bytes"] == expect
    assert logical > result["ckpt_bytes"] * 50
    assert result["ckpt_bytes"] < 100_000


def test_heavy_state_selective_keeps_less_than_per_stage():
    common = dict(n_workers=3, n_tasks=2, min_stage_s=0, data_seed=1, lam=1.0, timeout_s=20)
    light = run_job(LIGHT, MpCfg(strategy="per_stage", **common))
    recomputed = run_job(HEAVY, MpCfg(strategy="recompute", **common))
    per_stage = run_job(HEAVY, MpCfg(strategy="per_stage", **common))
    selective = run_job(HEAVY, MpCfg(strategy="selective", **common))
    assert recomputed["ckpt_bytes"] == 0
    assert selective["ckpt_bytes"] == 0
    assert per_stage["ckpt_bytes"] > light["ckpt_bytes"] * 5
    assert per_stage["ckpt_bytes"] < 20_000_000
    assert recomputed["digest"] == per_stage["digest"] == selective["digest"]


def test_kill_worker_output_equals_fault_free_and_dedup():
    """SIGKILL one worker mid-stage. Aggregate matches the fault-free run.

    recompute does not keep stage 0, so recovery runs that stage again and the
    coordinator must dedup the second (task_id, stage_id) confirmation.
    """
    common = dict(n_workers=3, n_tasks=3, min_stage_s=0.2, data_seed=2, lam=1.0, timeout_s=30,
                  hb_timeout=3.0)
    for strategy in ("recompute", "per_stage"):
        cfg = MpCfg(strategy=strategy, **common)
        fault_free = run_job(LIGHT, cfg)
        killed = run_job(LIGHT, cfg, kill=KillSpec(task_index=0, stage_id=1))
        assert killed["completed"]
        assert killed["final_count"] == common["n_tasks"]
        assert killed["digest"] == fault_free["digest"]
        assert np.allclose(killed["total"], fault_free["total"])
        assert killed["victim_exitcode"] == -signal.SIGKILL
        assert killed["victim_pid"] != killed["coordinator_pid"]
        assert killed["coordinator_pid"] == os.getpid()
        assert killed["killed_stage"] == 1
        assert killed["detect_reason"] == "exit"
        assert len(killed["confirmed_keys"]) == len(set(killed["confirmed_keys"]))
        if strategy == "recompute":
            assert killed["dup_intercepted"] >= 1
            assert killed["reexec_stages"] >= 1
            assert killed["duplicate_compute_s"] > 0
            task_id, stage_id = killed["confirmed_keys"][0]
            assert killed["log"].accept(task_id, stage_id, "final", None) == "dup"
            assert killed["log"].digest() == fault_free["digest"]
            assert killed["log"].final_count == common["n_tasks"]
