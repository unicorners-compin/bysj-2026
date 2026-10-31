"""Event-driven simulation of 1 coordinator + N workers running self-describing tasks.

Single-process discrete-event simulation (simulated clock, not wall clock).  Stage
outputs are *really computed* (src.task) so that the final result digest of every
faulty run can be compared with a fault-free reference; only time is simulated.

State-saving strategies : recompute | per_stage | selective(lambda), via policy.should_save
Recovery policies       : none | blind_retry | self_describing
Fault kinds             : fatal (crash) | transient (retryable error) | degraded (slow node)
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fault import FaultDesc, FaultKind, plan_recovery
from .policy import should_save as policy_should_save
from .task import Combo, Data, TaskDesc, global_digest, make_partition, run_stage, stage_cost_s

N_STAGES = 3
DEDUP_ENTRY_BYTES = 64     # task_id + stage_id + digest + flags


@dataclass
class SimCfg:
    n_workers: int = 4
    n_tasks: int = 12
    hb: float = 0.5            # heartbeat interval (s)
    lease: float = 2.0         # lease timeout (s)
    bw: float = 50e6           # state transfer rate worker<->coordinator (B/s)
    s0: float = 0.002          # fixed cost per state save / load (s)
    msg_lat: float = 0.001     # control message latency (s)
    strategy: str = "selective"
    lam: float = 1.0
    dedup: bool = True
    policy: str = "self_describing"
    max_retries: int = 3
    backoff0: float = 0.25
    blind_delay: float = 0.1           # blind_retry: fixed re-dispatch delay, no backoff, always from scratch
    task_timeout_factor: float = 2.5   # blind_retry: re-dispatch if a task runs longer than this x expected
    slowdown: float = 6.0
    data_seed: int = 0


@dataclass
class Fault:
    kind: str                       # fatal | transient | degraded
    worker: int
    # Exp A positional trigger (fatal only): stage index + timing
    stage: Optional[int] = None
    timing: Optional[str] = None    # during | before_ack | after_ack
    frac: float = 0.5
    # Exp B time trigger
    at: Optional[float] = None
    duration: float = 0.0           # transient: how long the condition lasts


@dataclass
class Worker:
    wid: int
    alive: bool = True
    degraded: bool = False
    excluded: bool = False          # isolated / degraded nodes get no new work
    slow: float = 1.0
    op: int = 0                     # op counter, stale events are ignored
    busy: Optional[Tuple[str, int]] = None   # (task_id, exec_id)
    cur_start: float = 0.0
    cur_end: float = 0.0
    cur_stage: int = -1
    cur_cost: float = 0.0
    local: Optional[Data] = None
    desc: Optional[TaskDesc] = None
    fault_on_next_start: bool = False


class Sim:
    def __init__(self, combo: Combo, cfg: SimCfg, fault: Optional[Fault] = None, rng=None):
        self.c, self.cfg, self.fault = combo, cfg, fault
        self.rng = rng or np.random.default_rng(0)
        self.t = 0.0
        self.q: list = []
        self.seq = 0
        self.workers = [Worker(i) for i in range(cfg.n_workers)]
        self.pending: List[TaskDesc] = []
        self.ckpt: Dict[Tuple[str, int], Data] = {}
        self.dedup: Dict[Tuple[str, int], str] = {}
        self.partials: Dict[str, Data] = {}
        self.final_count = 0
        self.done: set = set()
        self.assigned: Dict[str, set] = {}       # task_id -> worker ids currently executing it
        self.exec_seq = 0
        self.retries: Dict[str, int] = {}
        self.failed = False
        self.m = dict(compute_s=0.0, wasted_s=0.0, ckpt_bytes=0, recovery_load_bytes=0, dup_intercepted=0,
                      dup_accepted=0, saves=0, detect_s=float("nan"), fault_time=float("nan"), actions=[],
                      retries=0, recover_from_ckpt=0, recover_from_scratch=0)
        self.fault_fired = False
        self.task_done_t: Dict[str, float] = {}
        self.affected: Optional[str] = None
        plan = combo.plan()
        for i in range(cfg.n_tasks):
            tid = f"t{i:03d}"
            d = make_partition(i, combo, cfg.data_seed)
            self.pending.append(TaskDesc(tid, "job0", 0, plan, f"orig:{tid}", 0, (), ""))
        self.n_in = {}   # expected stage-input sizes for timeout estimate
        self.expected_task_s = (combo.records_per_task * (combo.filter_cost_us + 0.54 * (combo.transform_cost_us + combo.aggregate_cost_us))) * 1e-6
        if fault is not None and fault.at is not None:
            self.push(fault.at, "fault_time", fault.worker)

    # ------------------------------------------------------------------ engine
    def push(self, t, kind, *payload):
        self.seq += 1
        heapq.heappush(self.q, (t, self.seq, kind, payload))

    def run(self, t_max=10_000.0):
        self.assign()
        while self.q and not self.failed:
            t, _, kind, payload = heapq.heappop(self.q)
            if t > t_max:
                break
            self.t = t
            getattr(self, "on_" + kind)(*payload)
            if len(self.done) == self.cfg.n_tasks:
                break
        self.makespan = self.t
        return self

    # ------------------------------------------------------------------ coordinator
    def assign(self):
        for w in self.workers:
            if not self.pending:
                return
            if w.alive and not w.excluded and w.busy is None:
                d = self.pending.pop(0)
                if d.task_id in self.done:
                    continue
                self.start_task(w, d)

    def start_task(self, w: Worker, d: TaskDesc):
        self.exec_seq += 1
        w.busy = (d.task_id, self.exec_seq)
        w.desc = d
        self.assigned.setdefault(d.task_id, set()).add(w.wid)
        w.op += 1
        if d.input_ref.startswith("ckpt:"):
            st = int(d.input_ref.split(":")[2])
            data = self.ckpt[(d.task_id, st)]
            self.m["recovery_load_bytes"] += data.nbytes
            delay = data.nbytes / self.cfg.bw + self.cfg.s0 + self.cfg.msg_lat
        else:
            data = make_partition(int(d.task_id[1:]), self.c, self.cfg.data_seed)
            delay = self.cfg.msg_lat
        w.local = data
        self.push(self.t + delay, "begin_stage", w.wid, w.op, d.stage_id, 0.0)
        if self.cfg.policy == "blind_retry":
            self.push(self.t + self.cfg.task_timeout_factor * self.expected_task_s + delay, "task_timeout",
                      d.task_id, w.wid, w.busy[1])

    def on_begin_stage(self, wid, op, stage, acc):
        w = self.workers[wid]
        if not w.alive or w.op != op:
            return
        # transient fault pending on this worker: the stage fails right at its start
        f = self.fault
        if f and f.kind == "transient" and self.fault_fired and wid == f.worker and self.t < f.at + f.duration \
                and (w.fault_on_next_start or True):
            if w.fault_on_next_start or self.t >= f.at:
                w.fault_on_next_start = False
                self.push(self.t + self.cfg.msg_lat, "error_report", wid, op, stage)
                return
        cost = stage_cost_s(stage, len(w.local.key), self.c) * w.slow
        w.cur_start, w.cur_end, w.cur_stage, w.cur_cost = self.t, self.t + cost, stage, cost
        self.push(w.cur_end, "stage_done", wid, op, stage, acc)
        if f and f.kind == "fatal" and not self.fault_fired and f.stage == stage and f.timing == "during" and wid == f.worker:
            self.schedule_crash(self.t + f.frac * cost)

    def on_stage_done(self, wid, op, stage, acc):
        w = self.workers[wid]
        if not w.alive or w.op != op:
            return
        self.m["compute_s"] += w.cur_cost
        w.cur_stage = -1
        out = run_stage(stage, w.local, self.c)
        w.local = out
        acc += w.cur_cost / max(w.slow, 1e-9) * w.slow
        last = stage == N_STAGES - 1
        if last:
            kind, size = "final", out.nbytes
        else:
            kind = "ckpt" if self.should_save(acc, out.nbytes) else "progress"
            size = out.nbytes if kind == "ckpt" else 64
        if kind in ("ckpt", "final"):
            delay = size / self.cfg.bw + self.cfg.s0
        else:
            delay = self.cfg.msg_lat
        if kind == "ckpt":
            self.m["ckpt_bytes"] += size
            self.m["saves"] += 1
            acc = 0.0
        self.push(self.t + delay, "msg_arrive", wid, op, w.busy, stage, kind, out, acc)

    def should_save(self, acc_compute, nbytes):
        return policy_should_save(self.cfg.strategy, self.cfg.lam, acc_compute, nbytes, self.cfg.bw, self.cfg.s0)

    def on_msg_arrive(self, wid, op, busy, stage, kind, out, acc):
        tid = busy[0]
        key = (tid, stage)
        dup = key in self.dedup
        if dup:
            self.m["dup_intercepted"] += 1
        if not dup or not self.cfg.dedup:
            if dup:
                self.m["dup_accepted"] += 1
            self.dedup[key] = "ok"
            if kind == "ckpt":
                self.ckpt[key] = out
            if kind == "final":
                self.final_count += 1
                self.task_done_t.setdefault(tid, self.t)
                self.partials.setdefault(tid, [])
                self.partials[tid].append(out)
                self.done.add(tid)
                self.pending = [p for p in self.pending if p.task_id != tid]
        f = self.fault
        if f and f.kind == "fatal" and not self.fault_fired and f.timing == "before_ack" and f.stage == stage and wid == f.worker:
            self.schedule_crash(self.t)       # crash before the ack reaches the worker
            return
        self.push(self.t + self.cfg.msg_lat, "ack", wid, op, stage, acc)

    def on_ack(self, wid, op, stage, acc):
        w = self.workers[wid]
        if not w.alive or w.op != op:
            return
        f = self.fault
        trigger = f and f.kind == "fatal" and not self.fault_fired and f.timing == "after_ack" and f.stage == stage and wid == f.worker
        if stage == N_STAGES - 1:
            self.finish_exec(w)
            if trigger:
                self.schedule_crash(self.t + 0.01)
            self.assign()
            return
        if trigger:
            nxt = stage_cost_s(stage + 1, len(w.local.key), self.c) * w.slow
            self.schedule_crash(self.t + f.frac * nxt)
        self.push(self.t, "begin_stage", wid, op, stage + 1, acc)

    def finish_exec(self, w: Worker):
        if w.busy:
            self.assigned.get(w.busy[0], set()).discard(w.wid)
        w.busy, w.local, w.desc = None, None, None

    # ------------------------------------------------------------------ faults
    def schedule_crash(self, t):
        self.fault_fired = True
        self.push(t, "crash", self.fault.worker)

    def on_fault_time(self, wid):
        f = self.fault
        self.fault_fired = True
        self.m["fault_time"] = self.t
        w = self.workers[wid]
        if w.busy is not None:
            self.affected = w.busy[0]
        if f.kind == "fatal":
            self.on_crash(wid)
        elif f.kind == "transient":
            if w.busy is not None and w.cur_stage >= 0:
                # running stage raises an error now
                self.m["wasted_s"] += self.t - w.cur_start
                self.m["compute_s"] += self.t - w.cur_start
                stage = w.cur_stage
                w.op += 1
                w.cur_stage = -1
                self.push(self.t + self.cfg.msg_lat, "error_report", wid, w.op, stage)
            else:
                w.fault_on_next_start = True
        elif f.kind == "degraded":
            w.slow = self.cfg.slowdown
            w.degraded = True
            if w.cur_stage >= 0:
                remaining = (w.cur_end - self.t) * self.cfg.slowdown
                done_part = self.t - w.cur_start
                w.op += 1
                w.cur_cost = done_part + remaining
                w.cur_end = self.t + remaining
                # re-schedule the in-flight stage with the new end time (same accumulated work)
                self.push(w.cur_end, "stage_done", wid, w.op, w.cur_stage, 0.0)
            if self.cfg.policy == "self_describing":
                # degradation is self-described in the next heartbeat
                t_hb = math.ceil(self.t / self.cfg.hb + 1e-9) * self.cfg.hb
                self.push(t_hb + self.cfg.msg_lat, "degraded_report", wid)

    def on_crash(self, wid):
        w = self.workers[wid]
        if not w.alive:
            return
        if math.isnan(self.m["fault_time"]):
            self.m["fault_time"] = self.t
        if w.busy is not None and self.affected is None:
            self.affected = w.busy[0]
        if w.cur_stage >= 0:
            part = self.t - w.cur_start
            self.m["compute_s"] += part
            self.m["wasted_s"] += part
        w.alive = False
        w.op += 1
        w.cur_stage = -1
        if self.cfg.policy == "none":
            self.m["actions"].append("fail")
            self.failed = True
            return
        last_hb = math.floor(self.t / self.cfg.hb) * self.cfg.hb
        self.push(last_hb + self.cfg.lease, "detect", wid)

    def on_detect(self, wid):
        w = self.workers[wid]
        self.m["detect_s"] = self.t - self.m["fault_time"]
        desc = FaultDesc(FaultKind.FATAL, f"worker{wid}", "lease expired", False, 0)
        tasks = [w.busy[0]] if w.busy else []
        self.assigned.get(tasks[0], set()).discard(wid) if tasks else None
        w.busy = None
        if self.cfg.policy == "blind_retry":
            for tid in tasks:
                self.blind_redispatch(tid)
        else:
            act = plan_recovery(desc)            # -> isolate
            w.excluded = True
            for tid in tasks:
                self.m["actions"].append("isolate_recover")
                self.recover(tid)
        self.assign()

    def recover(self, tid, front=True):
        if tid in self.done:
            return
        saved = [s for (t, s) in self.ckpt if t == tid]
        plan = self.c.plan()
        old = TaskDesc(tid, "job0", 0, plan, f"orig:{tid}")
        if saved:
            k = max(saved)
            d = old.next_attempt(k + 1, f"ckpt:{tid}:{k}")
            d = TaskDesc(**{**d.__dict__, "saved_stages": tuple(sorted(saved))})
            self.m["recover_from_ckpt"] += 1
        else:
            d = old.next_attempt(0, f"orig:{tid}")
            self.m["recover_from_scratch"] += 1
        if front:
            self.pending.insert(0, d)
        else:
            self.pending.append(d)

    def blind_redispatch(self, tid):
        n = self.retries.get(tid, 0)
        if tid in self.done:
            return
        if n >= self.cfg.max_retries:
            self.m["actions"].append("fail")
            self.failed = True
            return
        self.retries[tid] = n + 1
        self.m["retries"] += 1
        self.m["actions"].append("blind_retry")
        self.push(self.t + self.cfg.blind_delay, "requeue", TaskDesc(tid, "job0", 0, self.c.plan(), f"orig:{tid}", n + 1))

    def on_requeue(self, d):
        if d.task_id not in self.done:
            self.pending.insert(0, d)
            self.assign()

    def on_error_report(self, wid, op, stage):
        """A transient fault reported by the worker itself (self-describing error message)."""
        w = self.workers[wid]
        if not w.alive or w.busy is None:
            return
        tid = w.busy[0]
        pol = self.cfg.policy
        if pol == "none":
            self.m["actions"].append("fail")
            self.failed = True
            return
        if pol == "blind_retry":
            # immediately retry the whole task from scratch on whatever worker is idle (often the same)
            self.finish_exec(w)
            self.blind_redispatch(tid)
            self.assign()
            return
        n = self.retries.get(tid, 0)
        desc = FaultDesc(FaultKind.TRANSIENT, f"worker{wid}", "stage error", True, self.cfg.max_retries)
        act = plan_recovery(desc, n)
        if act.name == "retry":
            self.retries[tid] = n + 1
            self.m["retries"] += 1
            self.m["actions"].append("retry")
            w.op += 1
            # retry the failed stage locally (input still in worker memory) after exponential backoff
            self.push(self.t + self.cfg.backoff0 * (2 ** n), "begin_stage", wid, w.op, stage, 0.0)
        else:
            self.m["actions"].append("retry_exhausted_isolate")
            w.excluded = True
            self.finish_exec(w)
            w.op += 1
            self.recover(tid)
            self.assign()

    def on_degraded_report(self, wid):
        w = self.workers[wid]
        if not w.alive:
            return
        desc = FaultDesc(FaultKind.DEGRADED, f"worker{wid}", f"slowdown x{self.cfg.slowdown:g}", False, 0)
        act = plan_recovery(desc)   # -> degrade
        w.excluded = True
        self.m["detect_s"] = self.t - self.m["fault_time"]
        if w.busy is not None:
            tid = w.busy[0]
            if w.cur_stage >= 0:
                self.m["wasted_s"] += self.t - w.cur_start
                self.m["compute_s"] += self.t - w.cur_start
            # only migrate if a healthy worker will pick it up; otherwise keep running slowly
            healthy = [x for x in self.workers if x.alive and not x.excluded]
            if healthy:
                self.m["actions"].append("degrade_migrate")
                w.op += 1
                w.cur_stage = -1
                self.finish_exec(w)
                self.recover(tid)
            else:
                self.m["actions"].append("degrade_keep")
        else:
            self.m["actions"].append("degrade_keep")
        self.assign()

    def on_task_timeout(self, tid, wid, exec_id):
        if tid in self.done or self.failed:
            return
        w = self.workers[wid]
        if w.busy is None or w.busy[1] != exec_id:
            return
        # blind retry: launch a duplicate copy from scratch, the slow original keeps running
        self.blind_redispatch(tid)
        self.assign()

    # ------------------------------------------------------------------ results
    def result(self):
        ok_done = len(self.done) == self.cfg.n_tasks and not self.failed
        parts = [p for lst in self.partials.values() for p in lst]
        dig = global_digest(parts) if ok_done else ""
        store = sum(d.nbytes for d in self.ckpt.values())
        return dict(completed=ok_done, digest=dig, makespan=self.makespan, **{k: v for k, v in self.m.items() if k != "actions"},
                    actions="|".join(self.m["actions"]), ckpt_store_bytes=store,
                    dedup_bytes=len(self.dedup) * DEDUP_ENTRY_BYTES, final_results=self.final_count,
                    affected=self.affected or "", task_done_t=self.task_done_t)


def run(combo, cfg, fault=None, seed=0):
    return Sim(combo, cfg, fault, np.random.default_rng(seed)).run().result()
