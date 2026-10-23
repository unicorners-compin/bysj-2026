"""Discrete-event simulation of task scheduling over in-network compute nodes.

Topology (8 compute nodes): 2 ingress regions x 3 near nodes + 2 far nodes.
Path propagation delay: same region 1 ms, cross region 4 ms, near<->far 10 ms.
Bandwidth contention: every node has one ingress channel (near: near_bw, far: far_bw);
transfers to a node are serialised FIFO on that channel, then propagate.
Nodes: FCFS, one task at a time, finite queue (rejections count as not on time).
All times are SIMULATED milliseconds; wall-clock time is recorded separately.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

NEAR = list(range(6))
FAR = [6, 7]
REGION = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}


def prop_delay(i: int, j: int) -> float:
    if i == j:
        return 0.0
    if i in FAR or j in FAR:
        return 10.0
    return 1.0 if REGION[i] == REGION[j] else 4.0


@dataclass
class Cond:
    name: str
    rho: float            # offered load / total capacity
    near_bw_mbps: float
    far_bw_mbps: float
    desc: str = ""


CONDS = {
    "C1": Cond("C1", 0.3, 1000.0, 1000.0, "低负载、带宽充足"),
    "C2": Cond("C2", 0.65, 1000.0, 1000.0, "高负载、带宽充足"),
    "C3": Cond("C3", 0.65, 1000.0, 200.0, "高负载、远端带宽受限"),
}


@dataclass
class NetCfg:
    near_speed: float = 1.0
    far_speed: float = 4.0
    near_qlen: int = 20
    far_qlen: int = 60
    mean_work_ms: float = 20.0          # at speed 1
    data_median_mb: float = 0.5
    data_sigma: float = 1.0
    deadline_mult: float = 3.0          # deadline = mult * W/near_speed + slack
    deadline_slack_ms: float = 30.0
    n_tasks: int = 10000

    def speeds(self):
        return np.array([self.near_speed] * 6 + [self.far_speed] * 2)

    def qlens(self):
        return np.array([self.near_qlen] * 6 + [self.far_qlen] * 2)


@dataclass
class Policy:
    name: str
    kind: str                     # random | round_robin | nearest | least_load | score
    alpha: float = 1.0            # weight of communication term
    beta: float = 1.0             # weight of queueing term
    migrate: bool = False
    theta_ms: float = 10.0
    rel_min: float = 0.2
    check_ms: float = 40.0        # migration check period (2 x mean exec time)
    info_period_ms: float = 20.0  # >0: scheduler only sees load snapshots refreshed with this period


def gen_tasks(cond: Cond, cfg: NetCfg, seed: int):
    rng = np.random.default_rng(seed)
    cap = cfg.speeds().sum()
    lam = cond.rho * cap / cfg.mean_work_ms      # tasks per ms
    n = cfg.n_tasks
    arr = np.cumsum(rng.exponential(1.0 / lam, n))
    entry = rng.choice(NEAR, n)
    data = cfg.data_median_mb * np.exp(rng.normal(0, cfg.data_sigma, n))
    work = rng.exponential(cfg.mean_work_ms, n)
    dl = cfg.deadline_mult * work / cfg.near_speed + cfg.deadline_slack_ms
    return arr, entry, data, work, dl


class Sim:
    def __init__(self, cond: Cond, cfg: NetCfg, pol: Policy, seed: int):
        self.cond, self.cfg, self.p = cond, cfg, pol
        self.arr, self.entry, self.data, self.work, self.dl = gen_tasks(cond, cfg, seed)
        self.rng = np.random.default_rng(seed + 7919)      # policy-side randomness (random policy only)
        self.speed = cfg.speeds()
        self.qlen = cfg.qlens()
        self.bw = np.array([cond.near_bw_mbps] * 6 + [cond.far_bw_mbps] * 2)  # Mbps
        self.N = 8
        self.D = np.array([[prop_delay(i, j) for j in range(8)] for i in range(8)])
        self.chan_free = np.zeros(8)          # ingress channel free time
        self.chan_bytes = np.zeros(8)
        self.queue: List[List[int]] = [[] for _ in range(8)]
        self.running: List[Optional[int]] = [None] * 8
        self.run_end = np.zeros(8)
        self.inflight_work = np.zeros(8)      # work of tasks in transfer towards node
        self.queued_work = np.zeros(8)
        self.inflight_cnt = np.zeros(8, dtype=int)
        self.rr = 0
        self.ev: list = []
        self.seq = 0
        n = cfg.n_tasks
        self.node = np.full(n, -1)
        self.t_comm = np.zeros(n)
        self.t_start = np.full(n, np.nan)
        self.t_done = np.full(n, np.nan)
        self.rejected = np.zeros(n, dtype=bool)
        self.migrated = np.zeros(n, dtype=bool)
        self.mig_bytes = 0.0
        self.snap = None
        self.snap_t = -1e18

    def push(self, t, kind, *a):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, a))

    # ---------------------------------------------------------------- load view
    def backlog_ms(self, now):
        """Remaining work (in ms at the node's own speed) per node: in service + queued + in flight."""
        rem = np.maximum(self.run_end - now, 0.0)
        rem = np.where([r is not None for r in self.running], rem, 0.0)
        return rem + (self.queued_work + self.inflight_work) / self.speed

    def view(self, now):
        if self.p.info_period_ms <= 0:
            return self.backlog_ms(now), self.inflight_cnt + np.array([len(q) for q in self.queue]) + \
                np.array([r is not None for r in self.running]), np.maximum(self.chan_free - now, 0)
        if now - self.snap_t >= self.p.info_period_ms:
            self.snap_t = now - ((now - self.snap_t) % self.p.info_period_ms) if self.snap is not None else now
            self.snap = (self.backlog_ms(now), self.inflight_cnt + np.array([len(q) for q in self.queue]) +
                         np.array([r is not None for r in self.running]), np.maximum(self.chan_free - now, 0), now)
        b, c, ch, t0 = self.snap
        return np.maximum(b - (now - t0), 0), c, np.maximum(ch - (now - t0), 0)

    def comm_ms(self, k, src, dst, chan_wait):
        if src == dst:
            return np.zeros_like(chan_wait) if np.ndim(dst) else 0.0
        return chan_wait + self.data[k] * 8e3 / self.bw[dst] + self.D[src, dst]

    def choose(self, k, now):
        p, s = self.p, self.entry[k]
        full = np.array([len(self.queue[j]) + self.inflight_cnt[j] >= self.qlen[j] for j in range(8)])
        if p.kind == "random":
            return int(self.rng.integers(8))
        if p.kind == "round_robin":
            j = self.rr % 8
            self.rr += 1
            return j
        if p.kind == "nearest":
            return s       # entry node itself is always the nearest (zero transfer)
        backlog, cnt, chan = self.view(now)
        if p.kind == "least_load":
            c = np.where(full, 1e18, cnt + 1e-6 * np.arange(8))
            return int(np.argmin(c))
        js = np.arange(8)
        comm = np.where(js == s, 0.0, chan + self.data[k] * 8e3 / self.bw + self.D[s])
        score = p.alpha * comm + p.beta * backlog + self.work[k] / self.speed
        score = np.where(full, 1e18, score)
        return int(np.argmin(score))

    # ---------------------------------------------------------------- events
    def send(self, k, src, dst, now, migration=False):
        if src == dst:
            self.push(now, "arrive", k, dst)
            self.inflight_work[dst] += self.work[k]
            self.inflight_cnt[dst] += 1
            return
        start = max(now, self.chan_free[dst])
        ser = self.data[k] * 8e3 / self.bw[dst]
        self.chan_free[dst] = start + ser
        self.chan_bytes[dst] += self.data[k]
        t_arr = start + ser + self.D[src, dst]
        self.t_comm[k] += t_arr - now
        if migration:
            self.mig_bytes += self.data[k]
        self.inflight_work[dst] += self.work[k]
        self.inflight_cnt[dst] += 1
        self.push(t_arr, "arrive", k, dst)

    def on_task(self, k):
        now = self.arr[k]
        j = self.choose(k, now)
        if len(self.queue[j]) + self.inflight_cnt[j] >= self.qlen[j]:
            self.rejected[k] = True
            return
        self.node[k] = j
        self.send(k, self.entry[k], j, now)

    def on_arrive(self, k, j, now):
        self.inflight_work[j] -= self.work[k]
        self.inflight_cnt[j] -= 1
        self.node[k] = j
        if len(self.queue[j]) >= self.qlen[j]:
            self.rejected[k] = True
            return
        self.queue[j].append(k)
        self.queued_work[j] += self.work[k]
        self.try_start(j, now)

    def try_start(self, j, now):
        if self.running[j] is None and self.queue[j]:
            k = self.queue[j].pop(0)
            self.queued_work[j] -= self.work[k]
            self.running[j] = k
            self.t_start[k] = now
            self.run_end[j] = now + self.work[k] / self.speed[j]
            self.push(self.run_end[j], "done", k, j)

    def on_done(self, k, j, now):
        self.t_done[k] = now
        self.running[j] = None
        self.try_start(j, now)

    def on_check(self, now):
        p = self.p
        rem = np.where([r is not None for r in self.running], np.maximum(self.run_end - now, 0.0), 0.0)
        for j in range(8):
            ahead = rem[j]
            for k in list(self.queue[j]):
                if self.migrated[k]:
                    ahead += self.work[k] / self.speed[j]
                    continue
                t_stay = ahead + self.work[k] / self.speed[j]
                backlog, cnt, chan = self.view(now)
                best, best_t = None, t_stay
                for m in range(8):
                    if m == j or len(self.queue[m]) + self.inflight_cnt[m] >= self.qlen[m]:
                        continue
                    tm = chan[m] + self.data[k] * 8e3 / self.bw[m] + self.D[j, m] + backlog[m] + self.work[k] / self.speed[m]
                    if tm < best_t:
                        best, best_t = m, tm
                g = t_stay - best_t
                if best is not None and g > p.theta_ms and g / t_stay >= p.rel_min:
                    self.queue[j].remove(k)
                    self.queued_work[j] -= self.work[k]
                    self.migrated[k] = True
                    self.send(k, j, best, now, migration=True)
                else:
                    ahead += self.work[k] / self.speed[j]

    def run(self):
        w0 = time.perf_counter()
        n = self.cfg.n_tasks
        for k in range(n):
            self.push(self.arr[k], "task", k)
        if self.p.migrate:
            t = self.p.check_ms
            while t < self.arr[-1]:
                self.push(t, "check")
                t += self.p.check_ms
        while self.ev:
            t, _, kind, a = heapq.heappop(self.ev)
            if kind == "task":
                self.on_task(*a)
            elif kind == "arrive":
                self.on_arrive(a[0], a[1], t)
            elif kind == "done":
                self.on_done(a[0], a[1], t)
            else:
                self.on_check(t)
        self.wall_s = time.perf_counter() - w0
        self.end_t = t
        return self

    def metrics(self):
        ok = ~self.rejected & ~np.isnan(self.t_done)
        ct = self.t_done[ok] - self.arr[ok]
        on_time = np.zeros(len(ok), dtype=bool)
        on_time[ok] = ct <= self.dl[ok]
        horizon = self.end_t
        util = self.chan_bytes * 8e6 / (self.bw * 1e3 * horizon)     # MB -> bits over (Mbps * ms) capacity
        busy = np.bincount(self.node[ok], weights=self.work[ok] / self.speed[self.node[ok]], minlength=8) / horizon
        wait = self.t_start[ok] - self.arr[ok] - self.t_comm[ok]
        return dict(mean_ct_ms=float(ct.mean()), p50_ct_ms=float(np.percentile(ct, 50)),
                    p95_ct_ms=float(np.percentile(ct, 95)), p99_ct_ms=float(np.percentile(ct, 99)),
                    on_time_rate=float(on_time.mean()), reject_rate=float(self.rejected.mean()),
                    mean_comm_ms=float(self.t_comm[ok].mean()), mean_wait_ms=float(wait.mean()),
                    migrated_ratio=float(self.migrated.mean()),
                    mig_bytes_ratio=float(self.mig_bytes / max(self.data.sum(), 1e-9)),
                    far_ratio=float(np.isin(self.node[ok], FAR).mean()),
                    link_util_std=float(util.std()), link_util_max=float(util.max()),
                    node_util_std=float(busy.std()), node_util_max=float(busy.max()),
                    wall_s=float(self.wall_s), _ct=ct)


POLICIES = {
    "random": Policy("random", "random"),
    "round_robin": Policy("round_robin", "round_robin"),
    "nearest": Policy("nearest", "nearest"),
    "least_load": Policy("least_load", "least_load"),
    "comm_only": Policy("comm_only", "score", alpha=1.0, beta=0.0),
    "load_only": Policy("load_only", "score", alpha=0.0, beta=1.0),
    "joint": Policy("joint", "score", alpha=1.0, beta=1.0),
    "joint_migrate": Policy("joint_migrate", "score", alpha=1.0, beta=1.0, migrate=True),
}


def simulate(cond, cfg, pol, seed):
    return Sim(cond, cfg, pol, seed).run().metrics()
