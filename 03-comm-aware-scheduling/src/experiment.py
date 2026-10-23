"""Scheduling experiments (Python teaching prototype, discrete-event simulation).

Main   : 3 conditions (C1/C2/C3) x 8 policies x 5 seeds x 10000 tasks
Sweeps : offered load, far-link bandwidth, migration threshold theta, load-info staleness
Tuning : theta chosen on separate tuning seeds / a different load point only.
All times are simulated milliseconds (NOT wall clock, NOT real network / DPDK).

Run from topic root:  python3 -m src.experiment
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .des import CONDS, POLICIES, Cond, NetCfg, Sim, simulate

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
SEEDS = (1, 2, 3, 4, 5)
SWEEP_SEEDS = (1, 2, 3)
TUNE_SEEDS = (101, 102, 103)
TUNE_COND = Cond("TUNE", 0.6, 1000.0, 300.0, "调参场景")
THETAS = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
ORDER = ("random", "round_robin", "nearest", "least_load", "comm_only", "load_only", "joint", "joint_migrate")
CN = {"random": "随机", "round_robin": "轮询", "nearest": "最近节点", "least_load": "最小负载(任务数)",
      "comm_only": "仅通信感知(β=0)", "load_only": "仅负载感知(α=0)", "joint": "联合感知不迁移", "joint_migrate": "联合感知+迁移"}
COL = {"random": "#9ca3af", "round_robin": "#6b7280", "nearest": "#f59e0b", "least_load": "#2563eb",
       "comm_only": "#a855f7", "load_only": "#06b6d4", "joint": "#f43f5e", "joint_migrate": "#16a34a"}
FOOT = "教学原型：Python 离散事件仿真（仿真时间 ms），非真实网络/硬件/DPDK"
METRICS = ("mean_ct_ms", "p50_ct_ms", "p95_ct_ms", "p99_ct_ms", "on_time_rate", "reject_rate", "mean_comm_ms", "mean_wait_ms",
           "migrated_ratio", "mig_bytes_ratio", "far_ratio", "link_util_std", "link_util_max", "node_util_std", "node_util_max", "wall_s")


def setup_font():
    from matplotlib import font_manager
    p = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(p).exists():
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def write_csv(name, rows):
    with open(FIGS / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def clean(m):
    return {k: round(float(v), 5) for k, v in m.items() if not k.startswith("_")}


def summarize(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    out = []
    for k, rs in g.items():
        row = dict(zip(keys, k))
        row["n_seeds"] = len(rs)
        for m in METRICS:
            v = np.array([r[m] for r in rs])
            row[m + "_mean"] = round(float(v.mean()), 5)
            row[m + "_std"] = round(float(v.std(ddof=1)), 5) if len(v) > 1 else 0.0
        out.append(row)
    return out


def tune_theta(cfg):
    rows = []
    for th in THETAS:
        ms = [simulate(TUNE_COND, cfg, replace(POLICIES["joint_migrate"], theta_ms=th), s) for s in TUNE_SEEDS]
        rows.append({"theta_ms": th, "mean_ct_ms": round(float(np.mean([m["mean_ct_ms"] for m in ms])), 4),
                     "p95_ct_ms": round(float(np.mean([m["p95_ct_ms"] for m in ms])), 4),
                     "on_time_rate": round(float(np.mean([m["on_time_rate"] for m in ms])), 5),
                     "migrated_ratio": round(float(np.mean([m["migrated_ratio"] for m in ms])), 5)})
    best = min(rows, key=lambda r: (round(r["mean_ct_ms"], 1), r["migrated_ratio"]))["theta_ms"]
    return best, rows


def main():
    setup_font()
    FIGS.mkdir(exist_ok=True)
    cfg = NetCfg()
    theta, tune_rows = tune_theta(cfg)
    print("theta selected:", theta)
    write_csv("theta_tuning.csv", tune_rows)
    pols = {k: (replace(v, theta_ms=theta) if v.migrate else v) for k, v in POLICIES.items()}

    main_rows, cts = [], {}
    for cname, cond in CONDS.items():
        for pname in ORDER:
            for s in SEEDS:
                m = simulate(cond, cfg, pols[pname], s)
                if s == SEEDS[0]:
                    cts[(cname, pname)] = m["_ct"]
                main_rows.append({"condition": cname, "policy": pname, "seed": s, **clean(m)})
    write_csv("main_results_per_seed.csv", main_rows)
    summ = summarize(main_rows, ["condition", "policy"])
    write_csv("main_results_summary.csv", summ)
    S = {(r["condition"], r["policy"]): r for r in summ}

    # sweeps
    sweep_pols = ("round_robin", "nearest", "least_load", "load_only", "joint", "joint_migrate")
    load_rows = []
    for rho in (0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        cond = Cond(f"rho{rho}", rho, 1000.0, 1000.0)
        for p in sweep_pols:
            for s in SWEEP_SEEDS:
                load_rows.append({"rho": rho, "policy": p, "seed": s, **clean(simulate(cond, cfg, pols[p], s))})
    write_csv("load_sweep.csv", load_rows)
    bw_rows = []
    for bw in (50.0, 100.0, 200.0, 300.0, 500.0, 1000.0):
        cond = Cond(f"bw{bw}", 0.65, 1000.0, bw)
        for p in sweep_pols:
            for s in SWEEP_SEEDS:
                bw_rows.append({"far_bw_mbps": bw, "policy": p, "seed": s, **clean(simulate(cond, cfg, pols[p], s))})
    write_csv("bandwidth_sweep.csv", bw_rows)
    th_rows = []
    for cname in ("C2", "C3"):
        for th in THETAS + (1e9,):
            for s in SWEEP_SEEDS:
                p = replace(pols["joint_migrate"], theta_ms=th) if th < 1e9 else pols["joint"]
                th_rows.append({"condition": cname, "theta_ms": th if th < 1e9 else "no_migration", "seed": s,
                                **clean(simulate(CONDS[cname], cfg, p, s))})
    write_csv("theta_sweep.csv", th_rows)
    st_rows = []
    for per in (0.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        for p in ("load_only", "joint", "joint_migrate"):
            for s in SWEEP_SEEDS:
                st_rows.append({"info_period_ms": per, "policy": p, "seed": s,
                                **clean(simulate(CONDS["C3"], cfg, replace(pols[p], info_period_ms=per), s))})
    write_csv("staleness_sweep.csv", st_rows)
    figs(S, cts, load_rows, bw_rows, th_rows, st_rows, theta)
    print("done")


def _save(fig, name):
    fig.text(0.99, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="#6b7280")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)


def mean_by(rows, xkey, p, metric):
    xs = sorted({r[xkey] for r in rows if r["policy"] == p})
    return xs, [np.mean([r[metric] for r in rows if r["policy"] == p and r[xkey] == x]) for x in xs]


def figs(S, cts, load_rows, bw_rows, th_rows, st_rows, theta):
    conds = list(CONDS)
    # fig1 mean and P95
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    w = 0.1
    for ax, met, lab in ((axs[0], "mean_ct_ms", "平均完成时间 (ms，对数)"), (axs[1], "p95_ct_ms", "P95 完成时间 (ms，对数)")):
        for j, p in enumerate(ORDER):
            v = [S[(c, p)][met + "_mean"] for c in conds]
            e = [S[(c, p)][met + "_std"] for c in conds]
            ax.bar(np.arange(3) + (j - 3.5) * w, v, w, yerr=e, capsize=1.5, color=COL[p], label=CN[p])
        ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels([f"{c}\n{CONDS[c].desc}" for c in conds])
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=.3)
    axs[0].legend(fontsize=7, ncol=2)
    _save(fig, "fig1_mean_p95_by_condition.png")

    # fig2 boxplot of completion times (C2, C3; seed 1)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, c in zip(axs, ("C2", "C3")):
        data = [cts[(c, p)] for p in ORDER]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, whis=(5, 95))
        for patch, p in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COL[p]); patch.set_alpha(.6)
        ax.set_yscale("log"); ax.set_xticks(range(1, 9)); ax.set_xticklabels([CN[p] for p in ORDER], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{c}（{CONDS[c].desc}）任务完成时间分布，种子1，须线=P5/P95"); ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("完成时间 (ms，对数)")
    _save(fig, "fig2_completion_time_boxplot.png")

    # fig3 on-time rate
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for j, p in enumerate(ORDER):
        v = [S[(c, p)]["on_time_rate_mean"] * 100 for c in conds]
        e = [S[(c, p)]["on_time_rate_std"] * 100 for c in conds]
        ax.bar(np.arange(3) + (j - 3.5) * w, v, w, yerr=e, capsize=1.5, color=COL[p], label=CN[p])
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{c}：{CONDS[c].desc}" for c in conds]); ax.set_ylabel("按时完成率 (%)")
    ax.legend(fontsize=7, ncol=4, loc="lower left"); ax.grid(axis="y", alpha=.3)
    _save(fig, "fig3_on_time_rate.png")

    # fig4 balance: link util std & node util std
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, met, lab in ((axs[0], "link_util_std", "入口链路利用率标准差（通信负载均衡，越低越均衡）"),
                         (axs[1], "node_util_std", "节点计算利用率标准差（越低越均衡）")):
        for j, p in enumerate(ORDER):
            v = [S[(c, p)][met + "_mean"] for c in conds]
            ax.bar(np.arange(3) + (j - 3.5) * w, v, w, color=COL[p], label=CN[p])
        ax.set_xticks(range(3)); ax.set_xticklabels(conds); ax.set_title(lab, fontsize=10); ax.grid(axis="y", alpha=.3)
    axs[0].legend(fontsize=7, ncol=2)
    _save(fig, "fig4_comm_and_node_balance.png")

    # fig5 load sweep
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for p in ("round_robin", "nearest", "least_load", "load_only", "joint", "joint_migrate"):
        x, y = mean_by(load_rows, "rho", p, "mean_ct_ms"); axs[0].plot(x, y, marker="o", ms=3, color=COL[p], label=CN[p])
        x, y = mean_by(load_rows, "rho", p, "p95_ct_ms"); axs[1].plot(x, y, marker="o", ms=3, color=COL[p], label=CN[p])
    for ax, lab in zip(axs, ("平均完成时间 (ms，对数)", "P95 完成时间 (ms，对数)")):
        ax.set_yscale("log"); ax.set_xlabel("系统负载 ρ（带宽均为 1 Gbps）"); ax.set_ylabel(lab); ax.grid(alpha=.3)
    axs[0].legend(fontsize=7)
    _save(fig, "fig5_load_sweep.png")

    # fig6 bandwidth sweep
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for p in ("round_robin", "nearest", "least_load", "load_only", "joint", "joint_migrate"):
        x, y = mean_by(bw_rows, "far_bw_mbps", p, "on_time_rate"); ax.plot(x, np.array(y) * 100, marker="o", ms=3, color=COL[p], label=CN[p])
    ax.set_xscale("log"); ax.set_xlabel("远端节点链路带宽 (Mbps，对数；ρ=0.65)"); ax.set_ylabel("按时完成率 (%)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    _save(fig, "fig6_bandwidth_sweep.png")

    # fig7 theta sweep
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, c in zip(axs, ("C2", "C3")):
        rr = [r for r in th_rows if r["condition"] == c and r["theta_ms"] != "no_migration"]
        ths = sorted({r["theta_ms"] for r in rr})
        ct = [np.mean([r["mean_ct_ms"] for r in rr if r["theta_ms"] == t]) for t in ths]
        mg = [np.mean([r["migrated_ratio"] for r in rr if r["theta_ms"] == t]) * 100 for t in ths]
        base = np.mean([r["mean_ct_ms"] for r in th_rows if r["condition"] == c and r["theta_ms"] == "no_migration"])
        xs = [max(t, 0.5) for t in ths]
        ax.plot(xs, ct, "o-", color="#16a34a", label="联合感知+迁移：平均完成时间")
        ax.axhline(base, color="#f43f5e", ls="--", label="联合感知不迁移")
        ax2 = ax.twinx(); ax2.bar(xs, mg, width=[x * 0.3 for x in xs], alpha=.25, color="#2563eb", label="迁移任务比例")
        ax.set_xscale("log"); ax.set_xlabel("迁移收益阈值 θ (ms，对数；θ=0 画在 0.5 处)"); ax.set_ylabel("平均完成时间 (ms)"); ax2.set_ylabel("迁移任务比例 (%)")
        ax.axvline(max(theta, 0.5), color="#9ca3af", ls=":")
        ax.set_title(f"{c}：迁移阈值影响（调参所选 θ={theta:g} ms）")
        l1, b1 = ax.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels(); ax.legend(l1 + l2, b1 + b2, fontsize=7)
    _save(fig, "fig7_theta_sweep.png")

    # fig8 staleness
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for p in ("load_only", "joint", "joint_migrate"):
        x, y = mean_by(st_rows, "info_period_ms", p, "mean_ct_ms"); axs[0].plot([max(v, 1) for v in x], y, marker="o", color=COL[p], label=CN[p])
        x, y = mean_by(st_rows, "info_period_ms", p, "on_time_rate"); axs[1].plot([max(v, 1) for v in x], np.array(y) * 100, marker="o", color=COL[p], label=CN[p])
    for ax, lab in zip(axs, ("平均完成时间 (ms)", "按时完成率 (%)")):
        ax.set_xscale("log"); ax.set_xlabel("负载信息更新周期 (ms，对数；0 画在 1 处)"); ax.set_ylabel(lab); ax.grid(alpha=.3)
    axs[0].legend(fontsize=8); axs[0].set_title("C3 条件下负载信息滞后的影响")
    _save(fig, "fig8_staleness_sweep.png")


if __name__ == "__main__":
    main()
