"""Full experiment for the latency-aware early-exit thesis (Python teaching prototype).

Inputs : figs/exit_outputs_seed{s}.npz, figs/compute_timing.csv (produced by src.train
         from a real MNIST-trained 2-exit CNN) -- only numpy/matplotlib needed here.
Network: simulated (src.netsim); NOT a real network, NOT DPDK/hardware.

Run from topic root:  python3 -m src.experiment
Outputs: figs/*.csv and figs/fig*.png (see README).
"""
from __future__ import annotations

import csv
from dataclasses import replace
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .netsim import NET_CN, NETWORKS, FEATURE_BYTES, Link, NodeCfg, make_trace
from .policies import Policy, run_policy

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
SEEDS = (0, 1, 2)
BUDGET_LEVELS = {"tight": 10.0, "mid": 30.0, "loose": 80.0}   # absolute budgets in ms (application requirement)
BUDGET_CN = {"tight": "紧", "mid": "中", "loose": "松"}
TAU_GRID = [round(x, 3) for x in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97,
                                  0.98, 0.985, 0.99, 0.995, 0.998, 0.999)]
ACC_TOL = 0.01          # fixed threshold: val accuracy may drop at most 1 pp vs full
TIMEOUT_CAP = 0.05      # latency-aware: val timeout rate cap used in parameter selection
LA_GRID = [dict(tau_min=a, tau_max=b, gamma=g, margin=m)
           for a in (0.5, 0.6, 0.7, 0.8, 0.9)
           for b in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
           for g in (0.5, 1.0, 2.0)
           for m in (0.0, 1.0, 2.0) if a < b]
NODE = NodeCfg()
NOTE = "Python teaching prototype: real MNIST CNN outputs + simulated network; not hardware/DPDK"


def setup_font():
    from matplotlib import font_manager
    p = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(p).exists():
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def load_seed(s):
    z = np.load(FIGS / f"exit_outputs_seed{s}.npz")
    split = lambda pre: {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
    return split("val_"), split("test_")


def load_timing():
    t = defaultdict(dict)
    with open(FIGS / "compute_timing.csv") as f:
        for r in csv.DictReader(f):
            t[int(r["seed"])][r["part"]] = float(r["median_ms"])
    return t


def write_csv(name, rows):
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(FIGS / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def rnd(r, nd=5):
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in r.items()}


def trace_for(seed, net, n, split, timing, k=2):
    # independent stream per (seed, net, split, K) so that every policy sees the same delays
    code = {"good": 1, "medium": 2, "poor": 3}[net] * 10 + (1 if split == "val" else 2) + 100 * k
    rng = np.random.default_rng(seed * 1000 + code)
    return make_trace(n, NETWORKS[net], timing[seed], NODE, rng, k_nodes=k)


def ev(policy, d, tr, budget):
    return run_policy(policy, d["conf1"], d["pred1"], d["pred2"], d["label"], tr, budget, NODE)


def select_fixed(val, trv):
    full = ev(Policy("full", "full"), val, trv, 1e9)
    ok = [t for t in TAU_GRID
          if ev(Policy("fixed", "fixed", tau=t, feasibility=False), val, trv, 1e9)["accuracy"] >= full["accuracy"] - ACC_TOL]
    return min(ok) if ok else TAU_GRID[-1]


def la_policy(prm, name="latency_aware"):
    return Policy(name, "dynamic", **prm)


def select_la(val, trv, budget):
    scored = []
    for prm in LA_GRID:
        r = ev(la_policy(prm), val, trv, budget)
        scored.append((prm, r))
    feas = [x for x in scored if x[1]["timeout_rate"] <= TIMEOUT_CAP]
    if feas:
        best = max(feas, key=lambda x: (round(x[1]["accuracy"], 4), -x[1]["mean_latency_ms"]))
    else:
        best = min(scored, key=lambda x: (x[1]["timeout_rate"], -x[1]["accuracy"]))
    return best[0], scored


def main():
    setup_font()
    timing = load_timing()
    data = {s: load_seed(s) for s in SEEDS}
    per_seed, params_rows, model_rows = [], [], []
    shared_pool = defaultdict(list)   # combo index -> list of val ontime-acc over configs
    selected = {}
    for s in SEEDS:
        val, test = data[s]
        model_rows.append({"seed": s,
                           "val_acc_exit1": float((val["pred1"] == val["label"]).mean()),
                           "val_acc_exit2": float((val["pred2"] == val["label"]).mean()),
                           "test_acc_exit1": float((test["pred1"] == test["label"]).mean()),
                           "test_acc_exit2": float((test["pred2"] == test["label"]).mean()),
                           "t_stage1_ms": timing[s]["stage1"], "t_exit1_ms": timing[s]["exit1_head"],
                           "t_stage2_ms": timing[s]["stage2_with_final_exit"]})
        for net in NETWORKS:
            trv = trace_for(s, net, len(val["label"]), "val", timing)
            trt = trace_for(s, net, len(test["label"]), "test", timing)
            base = ev(Policy("full", "full"), val, trv, 1e9)["mean_latency_ms"]
            tau_fix = select_fixed(val, trv)
            for bl, B in BUDGET_LEVELS.items():
                prm, scored = select_la(val, trv, B)
                for i, (_, r) in enumerate(scored):
                    shared_pool[i].append(r["ontime_accuracy"])
                selected[(s, net, bl)] = (B, tau_fix, prm, trt)
                params_rows.append({"seed": s, "network": net, "budget_level": bl, "budget_ms": round(B, 4),
                                    "base_full_val_ms": round(base, 4), "tau_fixed": tau_fix, **prm})
    shared_idx = max(shared_pool, key=lambda i: np.mean(shared_pool[i]))
    shared = LA_GRID[shared_idx]
    print("shared LA params:", shared)

    for (s, net, bl), (B, tau_fix, prm, trt) in selected.items():
        test = data[s][1]
        pols = [
            Policy("full", "full"),
            Policy("fixed", "fixed", tau=tau_fix, feasibility=False),
            Policy("conf_only", "fixed", tau=prm["tau_max"], feasibility=False),
            Policy("feas_only", "fixed", tau=tau_fix, feasibility=True, dynamic=False, margin=prm["margin"]),
            la_policy(prm, "latency_aware"),
            la_policy(shared, "latency_aware_shared"),
        ]
        for p in pols:
            r = ev(p, test, trt, B)
            per_seed.append(rnd({"seed": s, "network": net, "budget_level": bl, "budget_ms": B,
                                 "policy": p.name, **r}))
    write_csv("main_results_per_seed.csv", per_seed)
    write_csv("params_selected.csv", params_rows + [{"seed": "shared", "network": "all", "budget_level": "all",
                                                     "budget_ms": "", "base_full_val_ms": "", "tau_fixed": "", **shared}])
    write_csv("model_accuracy.csv", [rnd(r) for r in model_rows])

    # summary mean/std over seeds
    metrics = ["accuracy", "ontime_accuracy", "mean_latency_ms", "p95_latency_ms", "timeout_rate", "mean_tx_bytes", "exit1_ratio"]
    groups = defaultdict(list)
    for r in per_seed:
        groups[(r["network"], r["budget_level"], r["policy"])].append(r)
    summary = []
    for (net, bl, pol), rs in groups.items():
        row = {"network": net, "budget_level": bl, "policy": pol, "budget_ms": round(np.mean([x["budget_ms"] for x in rs]), 3)}
        for m in metrics:
            v = np.array([x[m] for x in rs])
            row[m + "_mean"] = round(float(v.mean()), 5)
            row[m + "_std"] = round(float(v.std(ddof=1)), 5)
        summary.append(row)
    write_csv("main_results_summary.csv", summary)
    S = {(r["network"], r["budget_level"], r["policy"]): r for r in summary}

    # ---- budget sweep (shared LA params, fixed tau per seed) -----------------------
    factors = [4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100]  # absolute budgets (ms)
    sweep = []
    for s in SEEDS:
        val, test = data[s]
        for net in NETWORKS:
            trv = trace_for(s, net, len(val["label"]), "val", timing)
            trt = trace_for(s, net, len(test["label"]), "test", timing)
            base = ev(Policy("full", "full"), val, trv, 1e9)["mean_latency_ms"]
            tau_fix = selected[(s, net, "mid")][1]
            for f in factors:
                for p in (Policy("full", "full"), Policy("fixed", "fixed", tau=tau_fix, feasibility=False),
                          la_policy(shared, "latency_aware_shared")):
                    r = ev(p, test, trt, float(f))
                    sweep.append(rnd({"seed": s, "network": net, "budget_ms": f, "policy": p.name, **r}))
    write_csv("budget_sweep.csv", sweep)

    # ---- trade-off curves at mid budget --------------------------------------------
    trade = []
    for s in SEEDS:
        val, test = data[s]
        for net in NETWORKS:
            B, tau_fix, prm, trt = selected[(s, net, "mid")]
            for t in TAU_GRID:
                r = ev(Policy("fixed", "fixed", tau=t, feasibility=False), test, trt, B)
                trade.append(rnd({"seed": s, "network": net, "policy": "fixed", "param": t, **r}))
            for tm in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
                for tmin in (0.5, 0.7, 0.9):
                    if tmin >= tm:
                        continue
                    pp = dict(prm, tau_max=tm, tau_min=tmin)
                    r = ev(la_policy(pp), test, trt, B)
                    trade.append(rnd({"seed": s, "network": net, "policy": "latency_aware",
                                      "param": f"{tmin}/{tm}", **r}))
            r = ev(Policy("full", "full"), test, trt, B)
            trade.append(rnd({"seed": s, "network": net, "policy": "full", "param": "", **r}))
    write_csv("tradeoff_mid_budget.csv", trade)

    # ---- K=3 (relay) at medium network, all budgets -------------------------------
    k3 = []
    for s in SEEDS:
        val, test = data[s]
        trv3 = trace_for(s, "medium", len(val["label"]), "val", timing, k=3)
        trt3 = trace_for(s, "medium", len(test["label"]), "test", timing, k=3)
        base3 = ev(Policy("full", "full"), val, trv3, 1e9)["mean_latency_ms"]
        tau_fix = selected[(s, "medium", "mid")][1]
        for bl, B in BUDGET_LEVELS.items():
            prm, _ = select_la(val, trv3, B)
            for p in (Policy("full", "full"), Policy("fixed", "fixed", tau=tau_fix, feasibility=False), la_policy(prm)):
                r = ev(p, test, trt3, B)
                k3.append(rnd({"seed": s, "K": 3, "network": "medium", "budget_level": bl, "budget_ms": B, "policy": p.name, **r}))
    write_csv("k3_relay_results.csv", k3)

    # ---- time-varying network (Gilbert-Elliott between medium and poor) ------------
    dyn = []
    dyn_series = {}
    for s in SEEDS:
        val, test = data[s]
        n = len(test["label"])
        rng = np.random.default_rng(9000 + s)
        state, states = 0, []
        for _ in range(n):
            if rng.random() < 0.004:
                state = 1 - state
            states.append(state)
        seq = [NETWORKS["medium"] if st == 0 else NETWORKS["poor"] for st in states]
        tr = make_trace(n, NETWORKS["medium"], timing[s], NODE, rng, link_seq=seq)
        trv = trace_for(s, "medium", len(val["label"]), "val", timing)
        base = ev(Policy("full", "full"), val, trv, 1e9)["mean_latency_ms"]
        B = BUDGET_LEVELS["mid"]
        prm = selected[(s, "medium", "mid")][2]
        tau_fix = selected[(s, "medium", "mid")][1]
        variants = [Policy("fixed", "fixed", tau=tau_fix, feasibility=False),
                    replace(la_policy(prm), name="la_static_estimate", ewma=0.0),
                    la_policy(prm, "la_ewma"),
                    replace(la_policy(prm), name="la_ewma_probe", probe_every=20)]
        for p in variants:
            r = ev(p, test, tr, B)
            dyn.append(rnd({"seed": s, "budget_ms": B, "bad_state_fraction": float(np.mean(states)), "policy": p.name, **r}))
            if s == 0:
                dyn_series[p.name] = (r["_lat"] > B).astype(float)
        if s == 0:
            dyn_series["_states"] = np.array(states)
            dyn_series["_B"] = B
    write_csv("dynamic_network_results.csv", dyn)

    # ---- calibration / confidence of exit 1 (test, all seeds) ----------------------
    cal = []
    bins = np.linspace(0, 1, 11)
    for s in SEEDS:
        test = data[s][1]
        c, ok = test["conf1"], test["pred1"] == test["label"]
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (c > lo) & (c <= hi)
            if m.sum():
                ece += m.mean() * abs(ok[m].mean() - c[m].mean())
                cal.append({"seed": s, "bin_lo": round(lo, 1), "bin_hi": round(hi, 1), "count": int(m.sum()),
                            "mean_conf": round(float(c[m].mean()), 5), "accuracy": round(float(ok[m].mean()), 5)})
        cal.append({"seed": s, "bin_lo": "ECE", "bin_hi": "", "count": len(c), "mean_conf": round(float(ece), 5), "accuracy": ""})
    write_csv("exit1_calibration.csv", cal)

    # ---- confidence measure comparison (max-prob vs margin vs entropy), no network ---
    cm = []
    for s in SEEDS:
        test = data[s][1]
        ok1 = test["pred1"] == test["label"]; ok2 = test["pred2"] == test["label"]
        for meas, score in (("max_prob", test["conf1"]), ("margin", test["margin1"]), ("neg_entropy", -test["entropy1"])):
            for q in (0.3, 0.4, 0.5, 0.6, 0.7):
                thr = np.quantile(score, 1 - q)
                ex = score >= thr
                cm.append(rnd({"seed": s, "measure": meas, "target_exit_ratio": q, "exit1_ratio": float(ex.mean()),
                               "accuracy": float(np.where(ex, ok1, ok2).mean()),
                               "exit1_subset_accuracy": float(ok1[ex].mean())}))
    write_csv("confidence_measure_comparison.csv", cm)

    make_figs(S, sweep, trade, per_seed, dyn_series, cal, model_rows)
    print("done;", NOTE)


POL_CN = {"full": "完整推理", "fixed": "固定阈值", "conf_only": "仅考虑置信度", "feas_only": "仅预算可行性",
          "latency_aware": "时延感知(逐条件调参)", "latency_aware_shared": "时延感知(共用参数)"}
COL = {"full": "#6b7280", "fixed": "#2563eb", "conf_only": "#a855f7", "feas_only": "#f59e0b",
       "latency_aware": "#dc2626", "latency_aware_shared": "#16a34a"}
FOOT = "教学原型：MNIST 实训 CNN 出口输出 + 单机网络仿真，非硬件/DPDK 实测"


def _save(fig, name):
    fig.text(0.99, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="#6b7280")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)


def make_figs(S, sweep, trade, per_seed, dyn_series, cal, model_rows):
    # fig1 training curves
    log = list(csv.DictReader(open(FIGS / "train_log.csv")))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, lab, c in (("val_acc_exit1", "中间出口", "#f59e0b"), ("val_acc_exit2", "最终出口", "#2563eb")):
        eps = sorted({int(r["epoch"]) for r in log})
        arr = np.array([[float(r[key]) for r in log if int(r["epoch"]) == e] for e in eps])
        ax.errorbar(eps, arr.mean(1) * 100, yerr=arr.std(1, ddof=1) * 100, marker="o", capsize=3, label=lab, color=c)
    ax.set_xlabel("训练轮次 (epoch)"); ax.set_ylabel("验证集准确率 (%)"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("两出口 CNN 在 MNIST 验证集上的训练曲线（3 个种子均值±标准差）")
    _save(fig, "fig1_training_curves.png")

    # fig2 exit1 confidence histogram + reliability
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    data0 = load_seed(0)[1]
    ok = data0["pred1"] == data0["label"]
    a1.hist(data0["conf1"][ok], bins=40, alpha=.7, label="中间出口预测正确", color="#16a34a")
    a1.hist(data0["conf1"][~ok], bins=40, alpha=.7, label="中间出口预测错误", color="#dc2626")
    a1.set_xlabel("中间出口最大类别概率 c1"); a1.set_ylabel("样本数"); a1.legend(); a1.set_title("测试集置信度分布（种子0）")
    rows = [r for r in cal if r["bin_lo"] != "ECE"]
    for s, mk in zip(SEEDS, "osd"):
        rr = [r for r in rows if r["seed"] == s]
        a2.plot([r["mean_conf"] for r in rr], [r["accuracy"] for r in rr], marker=mk, label=f"种子{s}")
    a2.plot([0, 1], [0, 1], "k--", lw=1, label="理想校准")
    eces = [r["mean_conf"] for r in cal if r["bin_lo"] == "ECE"]
    a2.set_title(f"中间出口可靠性图（ECE 均值 {np.mean(eces):.3f}）")
    a2.set_xlabel("分箱平均置信度"); a2.set_ylabel("分箱准确率"); a2.legend(); a2.grid(alpha=.3)
    _save(fig, "fig2_exit1_confidence.png")

    # fig3 trade-off
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, net in zip(axs, NETWORKS):
        for pol, mk in (("fixed", "o-"), ("latency_aware", "s")):
            rr = [r for r in trade if r["network"] == net and r["policy"] == pol]
            params = sorted({r["param"] for r in rr}, key=str)
            xs = [np.mean([r["mean_latency_ms"] for r in rr if r["param"] == p]) for p in params]
            ys = [np.mean([r["ontime_accuracy"] for r in rr if r["param"] == p]) * 100 for p in params]
            if pol == "fixed":
                ax.plot(xs, ys, mk, color=COL[pol], ms=4, label="固定阈值 τ 扫描")
            else:
                ax.scatter(xs, ys, marker="s", color=COL[pol], s=22, label="时延感知 (τ_min, τ_max) 扫描", zorder=3)
        rr = [r for r in trade if r["network"] == net and r["policy"] == "full"]
        ax.scatter([np.mean([r["mean_latency_ms"] for r in rr])], [np.mean([r["ontime_accuracy"] for r in rr]) * 100],
                   marker="*", s=160, color=COL["full"], label="完整推理", zorder=4)
        ax.set_title(f"{NET_CN[net]}网络，中预算"); ax.set_xlabel("平均仿真时延 (ms)"); ax.set_ylabel("按时正确率 (%)")
        ax.grid(alpha=.3)
    axs[0].legend(fontsize=8)
    _save(fig, "fig3_tradeoff_mid_budget.png")

    # fig4 timeout vs budget factor
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, net in zip(axs, NETWORKS):
        for pol in ("full", "fixed", "latency_aware_shared"):
            rr = [r for r in sweep if r["network"] == net and r["policy"] == pol]
            fs = sorted({r["budget_ms"] for r in rr})
            m = [np.mean([r["timeout_rate"] for r in rr if r["budget_ms"] == f]) * 100 for f in fs]
            ax.plot(fs, m, marker=".", color=COL[pol], label=POL_CN[pol])
        for f in BUDGET_LEVELS.values():
            ax.axvline(f, color="#d1d5db", ls=":")
        ax.set_title(f"{NET_CN[net]}网络"); ax.set_xlabel("时延预算 B (ms，对数坐标)"); ax.set_xscale("log"); ax.grid(alpha=.3)
    axs[0].set_ylabel("超时率 (%)"); axs[0].legend(fontsize=8)
    _save(fig, "fig4_timeout_vs_budget.png")

    # fig5 exit stage distribution
    fig, ax = plt.subplots(figsize=(11, 4.2))
    labels, x = [], 0
    for net in NETWORKS:
        for bl in BUDGET_LEVELS:
            for j, pol in enumerate(("fixed", "latency_aware")):
                r = S[(net, bl, pol)]
                e1 = r["exit1_ratio_mean"] * 100
                ax.bar(x + j * 0.38, e1, 0.36, color="#f59e0b" if j == 0 else "#fb7185")
                ax.bar(x + j * 0.38, 100 - e1, 0.36, bottom=e1, color="#93c5fd" if j == 0 else "#60a5fa")
            labels.append((x + 0.19, f"{NET_CN[net]}/{BUDGET_CN[bl]}"))
            x += 1
    ax.set_xticks([l[0] for l in labels]); ax.set_xticklabels([l[1] for l in labels])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#f59e0b", label="固定阈值：中间出口退出"), Patch(color="#93c5fd", label="固定阈值：最终出口"),
                       Patch(color="#fb7185", label="时延感知：中间出口退出"), Patch(color="#60a5fa", label="时延感知：最终出口")],
              fontsize=8, ncol=2, loc="upper right")
    ax.set_ylabel("样本比例 (%)"); ax.set_ylim(0, 125); ax.set_title("退出位置分布（左柱固定阈值，右柱时延感知；网络/预算档）")
    _save(fig, "fig5_exit_stage_distribution.png")

    # fig6 ablation
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    pols = ("fixed", "conf_only", "feas_only", "latency_aware")
    cfgs = (("medium", "mid"), ("poor", "tight"), ("good", "loose"))
    w = 0.2
    for ax, met, lab in ((axs[0], "timeout_rate", "超时率 (%)"), (axs[1], "accuracy", "准确率 (%)")):
        for j, pol in enumerate(pols):
            v = [S[(n, b, pol)][met + "_mean"] * 100 for n, b in cfgs]
            e = [S[(n, b, pol)][met + "_std"] * 100 for n, b in cfgs]
            ax.bar(np.arange(3) + (j - 1.5) * w, v, w, yerr=e, capsize=2, color=COL[pol], label=POL_CN[pol])
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{NET_CN[n]}网络/{BUDGET_CN[b]}预算" for n, b in cfgs])
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=.3)
    axs[1].set_ylim(80, 100)
    axs[0].legend(fontsize=8)
    axs[0].set_title("消融：超时率"); axs[1].set_title("消融：整体准确率")
    _save(fig, "fig6_ablation.png")

    # fig7 dynamic network timeout (rolling)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    st = dyn_series["_states"]
    ax.fill_between(np.arange(len(st)), 0, st * 100, color="#fee2e2", step="mid", label="较差状态区间")
    k = 200
    for name, lab, c in (("fixed", "固定阈值", COL["fixed"]), ("la_static_estimate", "时延感知-静态估计", "#a855f7"),
                         ("la_ewma", "时延感知-EWMA", COL["latency_aware"]), ("la_ewma_probe", "时延感知-EWMA+探测", COL["latency_aware_shared"])):
        y = np.convolve(dyn_series[name], np.ones(k) / k, mode="same") * 100
        ax.plot(y, color=c, lw=1.2, label=lab)
    ax.set_xlabel("测试样本序号（按到达顺序）"); ax.set_ylabel("滑动超时率 (%)，窗口200")
    ax.set_title("时变网络（一般/较差两状态切换）下的滑动超时率，种子0"); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=.3)
    _save(fig, "fig7_dynamic_network.png")


if __name__ == "__main__":
    main()
