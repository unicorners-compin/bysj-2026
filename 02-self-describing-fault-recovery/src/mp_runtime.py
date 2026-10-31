"""Multiprocess prototype: 1 coordinator + >=3 worker processes.

The coordinator process dispatches self-describing stages over multiprocessing
queues. Each worker is a separate process. A fault is injected by SIGKILL of the
worker that is inside a chosen stage. The coordinator notices the death from the
process exit code (and, if a worker stops heartbeating while still alive, from a
heartbeat timeout) and reassigns the unfinished task from the last confirmed
checkpoint. Confirmed results are deduplicated by (task_id, stage_id).

Stage outputs are the same deterministic numpy functions as the single-process
simulator. Checkpoint decisions call ``policy.should_save`` (C_k >= lam * P_k).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from multiprocessing.reduction import ForkingPickler
from collections import defaultdict, deque
from dataclasses import dataclass
from queue import Empty
from typing import Dict, List, Optional, Tuple

import numpy as np

from .policy import should_save
from .task import N_KEYS, Combo, Data, TaskDesc, digest_data, global_digest, make_partition, run_stage, stage_cost_s

N_STAGES = 3


def queued_bytes(obj) -> int:
    """Length of the pickle Queue puts on the pipe for this object.

    multiprocessing uses ForkingPickler. The checkpoint body is what moves;
    unallocated simulator padding is not part of ``Data`` and is not counted.
    """
    return len(ForkingPickler.dumps(obj))


@dataclass(frozen=True)
class MpCfg:
    n_workers: int = 4
    n_tasks: int = 6
    strategy: str = "selective"
    lam: float = 1.0
    bw: float = 50e6
    s0: float = 0.002
    hb_interval: float = 0.1
    hb_timeout: float = 3.0
    data_seed: int = 0
    min_stage_s: float = 0.0
    timeout_s: float = 60.0


@dataclass(frozen=True)
class KillSpec:
    """SIGKILL the worker while it is inside this task's stage."""

    task_index: int
    stage_id: int


class ConfirmLog:
    """Coordinator's confirmed-state log. First (task_id, stage_id) wins."""

    def __init__(self):
        self.confirmed: Dict[Tuple[str, int], str] = {}
        self.partials: Dict[str, Data] = {}
        self.ckpt: Dict[Tuple[str, int], Data] = {}
        self.dup_intercepted = 0
        self.final_count = 0

    def accept(self, task_id: str, stage_id: int, kind: str, data: Optional[Data]) -> str:
        key = (task_id, int(stage_id))
        if key in self.confirmed:
            self.dup_intercepted += 1
            return "dup"
        if kind == "final" and data is None:
            raise ValueError("final result needs data")
        self.confirmed[key] = kind
        if kind == "ckpt" and data is not None:
            self.ckpt[key] = data
        if kind == "final":
            self.partials[task_id] = data
            self.final_count += 1
        return "new"

    def digest(self) -> str:
        if not self.partials:
            return ""
        return global_digest(list(self.partials.values()))

    def total(self) -> np.ndarray:
        tot = np.zeros(2 * N_KEYS)
        for part in self.partials.values():
            tot += part.val
        return tot


def _run_for(stage: int, data: Data, combo: Combo, min_stage_s: float) -> Data:
    """Real stage function, repeated so a SIGKILL can land inside the stage."""
    deadline = time.perf_counter() + min_stage_s
    out = run_stage(stage, data, combo)
    while time.perf_counter() < deadline:
        out = run_stage(stage, data, combo)
    return out


def _worker_main(wid, in_q, out_q, combo: Combo, cfg: MpCfg, cur_task, cur_stage, stage_t0, last_hb):
    def heartbeat():
        while True:
            last_hb[wid] = time.monotonic()
            time.sleep(cfg.hb_interval)

    threading.Thread(target=heartbeat, daemon=True).start()
    last_hb[wid] = time.monotonic()
    out_q.put(("ready", wid))
    try:
        while True:
            msg = in_q.get()
            if msg is None or msg[0] == "shutdown":
                out_q.put(("bye", wid))
                return
            if msg[0] != "assign":
                out_q.put(("worker_error", wid, f"unexpected {msg[0]}"))
                return
            desc: TaskDesc = msg[1]
            data: Data = msg[2]
            if desc.input_ref not in desc.needed_refs:
                out_q.put(("worker_error", wid, "needed_refs missing input_ref"))
                return
            if desc.checksum and digest_data(data) != desc.checksum:
                out_q.put(("worker_error", wid, "input checksum mismatch"))
                return
            stage = desc.stage_id
            acc = 0.0
            cur_task[wid] = int(desc.task_id[1:])
            while True:
                # Publish the clock before the stage id so the killer never
                # observes the stage without a start time.
                stage_t0[wid] = time.monotonic()
                cur_stage[wid] = stage
                t0 = time.perf_counter()
                out = _run_for(stage, data, combo, cfg.min_stage_s)
                dt = time.perf_counter() - t0
                cur_stage[wid] = -1
                cost = stage_cost_s(stage, len(data.key), combo)
                acc += cost
                if stage == N_STAGES - 1:
                    out_q.put(("final", wid, desc.task_id, stage, out, dt, cost))
                    ack = in_q.get()
                    if ack[0] == "shutdown":
                        out_q.put(("bye", wid))
                        return
                    break
                # P_k uses the bytes that would actually be pickled, not logical pad.
                persist = queued_bytes(out) if cfg.strategy == "selective" else 0
                if should_save(cfg.strategy, cfg.lam, acc, persist, cfg.bw, cfg.s0):
                    out_q.put(("ckpt", wid, desc.task_id, stage, out, dt, cost))
                    acc = 0.0
                else:
                    out_q.put(("progress", wid, desc.task_id, stage, dt, cost))
                ack = in_q.get()
                if ack[0] == "shutdown":
                    out_q.put(("bye", wid))
                    return
                if ack[1] != "continue":
                    break
                data = out
                stage += 1
            cur_task[wid] = -1
            cur_stage[wid] = -1
            out_q.put(("idle", wid))
    except Exception as exc:  # pragma: no cover - surfaced to the coordinator
        out_q.put(("worker_error", wid, repr(exc)))


def _make_desc(combo: Combo, task_id: str, stage: int, input_ref: str, data: Data,
               saved: List[int], attempt: int) -> TaskDesc:
    return TaskDesc(
        task_id=task_id,
        job_id="job0",
        stage_id=stage,
        stage_plan=combo.plan(),
        input_ref=input_ref,
        attempt=attempt,
        saved_stages=tuple(sorted(set(saved))),
        checksum=digest_data(data),
        needed_refs=(input_ref,),
    )


def run_job(combo: Combo, cfg: MpCfg, kill: Optional[KillSpec] = None) -> dict:
    if cfg.n_workers < 3:
        raise ValueError("need at least 3 worker processes (1 coordinator + >=3 workers)")
    if cfg.n_tasks < 1:
        raise ValueError("n_tasks must be positive")
    if cfg.strategy not in ("recompute", "per_stage", "selective"):
        raise ValueError(cfg.strategy)
    if kill is not None and not (0 <= kill.task_index < cfg.n_tasks and 0 <= kill.stage_id < N_STAGES):
        raise ValueError("kill target is outside this job")

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    in_qs = [ctx.Queue() for _ in range(cfg.n_workers)]
    cur_task = ctx.Array("i", [-1] * cfg.n_workers)
    cur_stage = ctx.Array("i", [-1] * cfg.n_workers)
    stage_t0 = ctx.Array("d", [0.0] * cfg.n_workers)
    last_hb = ctx.Array("d", [time.monotonic()] * cfg.n_workers)
    victim = ctx.Value("i", -1)
    kill_at = ctx.Value("d", 0.0)
    killed_stage_v = ctx.Value("i", -1)
    victim_pid_v = ctx.Value("i", 0)

    procs = [
        ctx.Process(
            target=_worker_main,
            args=(i, in_qs[i], out_q, combo, cfg, cur_task, cur_stage, stage_t0, last_hb),
            name=f"worker-{i}",
        )
        for i in range(cfg.n_workers)
    ]
    for proc in procs:
        proc.start()

    log = ConfirmLog()
    pending = deque()
    for i in range(cfg.n_tasks):
        task_id = f"t{i:03d}"
        data = make_partition(i, combo, cfg.data_seed)
        pending.append((_make_desc(combo, task_id, 0, f"orig:{task_id}", data, [], 0), data))
    inflight: Dict[int, str] = {}
    owner: Dict[str, int] = {}
    saved: Dict[str, List[int]] = defaultdict(list)
    ckpt_data: Dict[Tuple[str, int], Data] = {}
    attempts = {f"t{i:03d}": 0 for i in range(cfg.n_tasks)}
    done = set()
    dead = set()
    ready = set()
    assignments: List[dict] = []
    affected = f"t{kill.task_index:03d}" if kill is not None else ""
    affected_done_at = None
    st = dict(ckpt_bytes=0, recovery_load_bytes=0, reexec_stages=0, reexec_wall=0.0, stage_runs=0,
              lost_compute_s=0.0, detect_s=float("nan"), detect_reason="", victim_exit=None,
              from_ckpt=0, from_scratch=0)
    stop_killer = threading.Event()
    coordinator_pid = os.getpid()

    def _assign():
        for wid in range(cfg.n_workers):
            if not pending:
                return
            if wid in dead or wid in inflight or not procs[wid].is_alive():
                continue
            desc, data = pending.popleft()
            inflight[wid] = desc.task_id
            owner[desc.task_id] = wid
            assignments.append({
                "task_id": desc.task_id,
                "stage_id": desc.stage_id,
                "input_ref": desc.input_ref,
                "needed_refs": desc.needed_refs,
                "checksum": desc.checksum,
                "attempt": desc.attempt,
                "saved_stages": desc.saved_stages,
            })
            in_qs[wid].put(("assign", desc, data))
            if desc.input_ref.startswith("ckpt:"):
                st["recovery_load_bytes"] += queued_bytes(data)

    def _requeue(task_id: str):
        if task_id in done or task_id in inflight.values():
            return
        if any(desc.task_id == task_id for desc, _ in pending):
            return
        idx = int(task_id[1:])
        stages = saved[task_id]
        if stages:
            k = max(stages)
            data = ckpt_data[(task_id, k)]
            ref = f"ckpt:{task_id}:{k}"
            stage = k + 1
            st["from_ckpt"] += 1
        else:
            data = make_partition(idx, combo, cfg.data_seed)
            ref = f"orig:{task_id}"
            stage = 0
            st["from_scratch"] += 1
        if stage >= N_STAGES:
            return
        attempts[task_id] += 1
        pending.append((_make_desc(combo, task_id, stage, ref, data, stages, attempts[task_id]), data))

    def _handle_death(wid: int, reason: str):
        if wid in dead:
            return
        dead.add(wid)
        proc = procs[wid]
        if proc.is_alive():
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                pass
        proc.join(timeout=2)
        if victim.value == wid and st["victim_exit"] is None:
            st["victim_exit"] = proc.exitcode
            st["detect_reason"] = reason
            st["detect_s"] = time.monotonic() - kill_at.value
            st["lost_compute_s"] = max(0.0, kill_at.value - stage_t0[wid])
        task_id = inflight.pop(wid, None)
        if task_id and task_id not in done:
            _requeue(task_id)
        _assign()

    def _on_stage(msg):
        nonlocal affected_done_at
        kind = msg[0]
        wid, task_id, stage = msg[1], msg[2], msg[3]
        if kind == "progress":
            dt = msg[4]
            data = None
        elif kind == "ckpt":
            data, dt = msg[4], msg[5]
        else:
            data, dt = msg[4], msg[5]
        st["stage_runs"] += 1
        status = log.accept(task_id, stage, kind, data)
        if status == "dup":
            st["reexec_stages"] += 1
            st["reexec_wall"] += float(dt)
        elif kind == "ckpt":
            st["ckpt_bytes"] += queued_bytes(data)
            saved[task_id].append(stage)
            ckpt_data[(task_id, stage)] = data
        elif kind == "final":
            done.add(task_id)
            if task_id == affected and affected_done_at is None:
                affected_done_at = time.monotonic()
        if wid in dead or not procs[wid].is_alive():
            return
        in_qs[wid].put(("ack", "stop" if kind == "final" else "continue"))

    def _handle(msg):
        kind = msg[0]
        if kind == "ready":
            ready.add(msg[1])
            last_hb[msg[1]] = time.monotonic()
        elif kind == "idle":
            inflight.pop(msg[1], None)
            _assign()
        elif kind == "worker_error":
            raise RuntimeError(f"worker {msg[1]}: {msg[2]}")
        elif kind in ("progress", "ckpt", "final"):
            _on_stage(msg)
        elif kind in ("bye", "hb"):
            return
        else:
            raise RuntimeError(f"bad message {kind}")

    def _drain(timeout: float):
        try:
            msg = out_q.get(timeout=timeout)
        except Empty:
            return
        _handle(msg)
        while True:
            try:
                msg = out_q.get_nowait()
            except Empty:
                return
            _handle(msg)

    def _poll_failures():
        now = time.monotonic()
        for wid, proc in enumerate(procs):
            if wid in dead:
                continue
            if proc.exitcode is not None or not proc.is_alive():
                _handle_death(wid, "exit")
                continue
            if wid in ready and now - last_hb[wid] > cfg.hb_timeout:
                _handle_death(wid, "heartbeat")

    def _killer():
        if kill is None:
            return
        while not stop_killer.is_set():
            for wid, proc in enumerate(procs):
                if wid in dead or not proc.is_alive() or not proc.pid:
                    continue
                if cur_task[wid] == kill.task_index and cur_stage[wid] == kill.stage_id:
                    with victim.get_lock():
                        victim.value = wid
                        kill_at.value = time.monotonic()
                        killed_stage_v.value = int(cur_stage[wid])
                        victim_pid_v.value = int(proc.pid)
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
            time.sleep(0.001)

    killer = threading.Thread(target=_killer, name="fault-injector", daemon=True)
    t_run0 = None
    try:
        arm = time.monotonic() + min(15.0, cfg.timeout_s)
        while len(ready) < cfg.n_workers:
            _drain(0.05)
            _poll_failures()
            if time.monotonic() > arm:
                raise TimeoutError(f"workers not ready: {sorted(ready)}")
        killer.start()
        t_run0 = time.monotonic()
        _assign()
        deadline = t_run0 + cfg.timeout_s
        while len(done) < cfg.n_tasks:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"mp run timed out done={len(done)}/{cfg.n_tasks} inflight={dict(inflight)} dead={sorted(dead)}"
                )
            _drain(0.02)
            _poll_failures()
            _assign()
        if kill is not None and victim.value < 0:
            raise RuntimeError("fault injection did not kill a worker")
    finally:
        stop_killer.set()
        for wid, proc in enumerate(procs):
            if proc.is_alive():
                try:
                    in_qs[wid].put(("shutdown",))
                except Exception:
                    pass
        for proc in procs:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
        for q in [out_q, *in_qs]:
            q.close()
            q.cancel_join_thread()

    makespan = time.monotonic() - t_run0 if t_run0 is not None else float("nan")
    if kill is not None and affected_done_at is not None and kill_at.value > 0:
        recovery = affected_done_at - kill_at.value
    else:
        recovery = 0.0 if kill is None else float("nan")
    victim_wid = victim.value if victim.value >= 0 else None
    if victim_wid is not None and st["victim_exit"] is None:
        st["victim_exit"] = procs[victim_wid].exitcode
        if not st["detect_reason"]:
            st["detect_reason"] = "exit"
    return dict(
        completed=len(done) == cfg.n_tasks and log.final_count == cfg.n_tasks,
        digest=log.digest(),
        total=log.total(),
        final_count=log.final_count,
        dup_intercepted=log.dup_intercepted,
        ckpt_bytes=st["ckpt_bytes"],
        recovery_load_bytes=st["recovery_load_bytes"],
        state_transfer_bytes=st["ckpt_bytes"] + st["recovery_load_bytes"],
        stage_runs=st["stage_runs"],
        reexec_stages=st["reexec_stages"],
        lost_compute_s=st["lost_compute_s"],
        duplicate_compute_s=st["reexec_wall"] + st["lost_compute_s"],
        recovery_time_s=recovery,
        detect_s=st["detect_s"],
        detect_reason=st["detect_reason"],
        victim_wid=victim_wid,
        victim_pid=int(victim_pid_v.value) if victim_wid is not None else None,
        victim_exitcode=st["victim_exit"],
        killed_stage=int(killed_stage_v.value) if victim_wid is not None else None,
        makespan_s=makespan,
        confirmed_keys=list(log.confirmed.keys()),
        assignments=assignments,
        log=log,
        coordinator_pid=coordinator_pid,
        from_ckpt=st["from_ckpt"],
        from_scratch=st["from_scratch"],
        killed=victim_wid is not None,
    )
