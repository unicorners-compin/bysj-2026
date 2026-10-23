"""Exit policies: full inference, fixed threshold, and latency-aware dynamic threshold.

All policies are evaluated by ``run_policy`` on one pre-drawn Trace so that they face
identical delays.  The latency-aware policy keeps an exponentially weighted estimate of
the link delay (mean and std) that is updated only by samples that actually continue.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from .netsim import FEATURE_BYTES, NodeCfg, Trace


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str                 # full | fixed | dynamic
    tau: float = 0.9          # fixed threshold (fixed kind) / ignored
    tau_min: float = 0.5
    tau_max: float = 0.99
    gamma: float = 1.0
    margin: float = 1.0       # safety margin in units of estimated link std
    feasibility: bool = True  # budget feasibility check
    dynamic: bool = True      # rho-dependent threshold
    ewma: float = 0.1         # estimator smoothing; 0 => never update (static estimate)
    probe_every: int = 0      # send a probe feature after this many samples without observation (0 = off)


def run_policy(p: Policy, conf1, pred1, pred2, label, tr: Trace, budget: float, node: NodeCfg):
    n = len(label)
    cont = np.zeros(n, dtype=bool)
    probe = np.zeros(n, dtype=bool)
    if p.kind == "full":
        cont[:] = True
    elif p.kind == "fixed" and not p.feasibility:
        cont = conf1 < p.tau
    else:
        mu, var = tr.nominal_link, tr.nominal_std ** 2
        t2_hat = float(tr.t2.mean())
        t1, tl, c1 = tr.t1, tr.t_link, conf1
        a = p.ewma
        since = 0
        probe = np.zeros(n, dtype=bool)
        sqrt = math.sqrt
        for i in range(n):
            since += 1
            ti = t1[i] + node.t_dec_ms
            tcont = mu + t2_hat
            if p.feasibility and ti + tcont + p.margin * sqrt(var) > budget:
                # exit: continuing is predicted to miss the budget
                if p.probe_every and since >= p.probe_every and a > 0:
                    probe[i] = True  # feature sent in background only to refresh the estimate
                    diff = tl[i] - mu
                    mu += a * diff
                    var = (1 - a) * (var + a * diff * diff)
                    since = 0
                continue
            if p.dynamic:
                rho = (budget - ti - tcont) / budget
                rho = 0.0 if rho < 0 else (1.0 if rho > 1 else rho)
                thr = p.tau_min + (p.tau_max - p.tau_min) * rho ** p.gamma
            else:
                thr = p.tau
            if c1[i] < thr:
                cont[i] = True
                since = 0
                if a > 0:
                    x = tl[i]
                    diff = x - mu
                    mu += a * diff
                    var = (1 - a) * (var + a * diff * diff)
    lat = tr.t1 + node.t_dec_ms + np.where(cont, tr.t_link + tr.t2, 0.0)
    pred = np.where(cont, pred2, pred1)
    correct = pred == label
    timeout = lat > budget
    return {
        "accuracy": float(correct.mean()),
        "ontime_accuracy": float((correct & ~timeout).mean()),
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "timeout_rate": float(timeout.mean()),
        "mean_tx_bytes": float((cont | probe).mean() * FEATURE_BYTES),
        "exit1_ratio": float(1 - cont.mean()),
        "_cont": cont,
        "_lat": lat,
    }
