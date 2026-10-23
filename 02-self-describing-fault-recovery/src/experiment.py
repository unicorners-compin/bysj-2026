"""Fault-recovery experiments (Python teaching prototype, event-driven simulation).

Exp A  state-saving strategies (recompute / per_stage / selective) under a single
       worker crash: 4 size combos x 3 stages x 3 fault timings x 5 repetitions.
Exp B  self-describing fault handling: fault kind (fatal / transient / degraded) x
       recovery policy (none / blind_retry / self_describing), 200 trials each.
All stage outputs are really computed and every completed run is checked against the
fault-free reference digest.  Times are simulated seconds, NOT wall clock, NOT DPDK.

Run from topic root:  python3 -m src.experiment
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .des import Fault, SimCfg, run
from .task import COMBOS, TUNE_COMBO

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
STRATS = ("recompute", "per_stage", "selective")
STRAT_CN = {"recompute": "整体重算", "per_stage": "每阶段保存", "selective": "选择性保存"}
TIMINGS = ("during", "before_ack", "after_ack")
TIMING_CN = {"during": "阶段执行中", "before_ack": "结果发出后确认前", "after_ack": "确认后"}
STAGE_CN = ["过滤", "变换", "聚合"]
KINDS = ("fatal", "transient", "degraded")
KIND_CN = {"fatal": "致命(进程崩溃)", "transient": "瞬态(可重试错误)", "degraded": "性能退化(慢节点)"}
POLS = ("none", "blind_retry", "self_describing")
POL_CN = {"none": "无恢复", "blind_retry": "盲目重试", "self_describing": "自描述策略化恢复"}
LAMS = (0.25, 0.5, 1.0, 2.0, 4.0)
REPS = 5
N_B = 200
DEADLINE_FACTOR = 1.5
FOOT = "教学原型：单机事件驱动仿真（阶段计算真实执行，时间为仿真秒），非硬件/DPDK/生产遥测"


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


def r5(x):
    return round(float(x), 5)


def crash_faults(rep_seed):
    rng = np.random.default_rng(rep_seed)
    out = []
    for st in range(3):
        for tm in TIMINGS:
            for rep in range(REPS):
                out.append((st, tm, rep, Fault("fatal", worker=int((rep + st) % 4), stage=st, timing=tm,
                                               frac=float(rng.uniform(0.2, 0.8)))))
    return out


def tune_lambda():
    rows = []
    for lam in LAMS:
        cfg = SimCfg(strategy="selective", lam=lam, data_seed=99)
        ff = run(TUNE_COMBO, cfg)
        mks = [run(TUNE_COMBO, cfg, f)["makespan"] for _, _, _, f in crash_faults(777)]
        rows.append({"lambda": lam, "ff_makespan_s": r5(ff["makespan"]), "mean_fault_makespan_s": r5(np.mean(mks)),
                     "ff_ckpt_MB": r5(ff["ckpt_bytes"] / 1e6), "saves": ff["saves"]})
    best = min(rows, key=lambda r: (r["mean_fault_makespan_s"], r["ff_ckpt_MB"]))["lambda"]
    return best, rows


def exp_a(lam):
    rows, ff_rows = [], []
    for cname, combo in COMBOS.items():
        refs = {}
        for st in STRATS:
            cfg = SimCfg(strategy=st, lam=lam)
            ff = run(combo, cfg)
            refs[st] = ff
            ff_rows.append({"combo": cname, "strategy": st, "ff_makespan_s": r5(ff["makespan"]),
                            "ff_ckpt_MB": r5(ff["ckpt_bytes"] / 1e6), "saves": ff["saves"],
                            "store_MB": r5(ff["ckpt_store_bytes"] / 1e6), "dedup_KB": r5(ff["dedup_bytes"] / 1e3),
                            "compute_s": r5(ff["compute_s"]), "digest": ff["digest"]})
        digests = {refs[s]["digest"] for s in STRATS}
        assert len(digests) == 1, "fault-free digests differ between strategies"
        for st in STRATS:
            cfg = SimCfg(strategy=st, lam=lam)
            ff = refs[st]
            for stage, tm, rep, f in crash_faults(1000 + ord(cname)):
                r = run(combo, cfg, f)
                aff = r["affected"]
                delay = (r["task_done_t"].get(aff, np.nan) - ff["task_done_t"].get(aff, np.nan)) if aff else 0.0
                rows.append({"combo": cname, "strategy": st, "fault_stage": stage, "timing": tm, "rep": rep,
                             "worker": f.worker, "completed": int(r["completed"]),
                             "consistent": int(r["digest"] == ff["digest"]),
                             "makespan_s": r5(r["makespan"]), "makespan_delta_s": r5(r["makespan"] - ff["makespan"]),
                             "affected_task": aff, "recovery_time_s": r5(delay),
                             "detect_s": r5(r["detect_s"]), "lost_compute_s": r5(r["wasted_s"]),
                             "recompute_s": r5(r["compute_s"] - ff["compute_s"]),
                             "normal_ckpt_MB": r5(r["ckpt_bytes"] / 1e6), "recovery_load_MB": r5(r["recovery_load_bytes"] / 1e6),
                             "store_MB": r5(r["ckpt_store_bytes"] / 1e6), "dedup_KB": r5(r["dedup_bytes"] / 1e3),
                             "dup_intercepted": r["dup_intercepted"], "from_ckpt": r["recover_from_ckpt"],
                             "from_scratch": r["recover_from_scratch"]})
    return rows, ff_rows


def exp_b(lam):
    combo = COMBOS["B"]
    rows = []
    base_cfg = dict(strategy="selective", lam=lam)
    ff = run(combo, SimCfg(**base_cfg))
    deadline = DEADLINE_FACTOR * ff["makespan"]
    for kind in KINDS:
        rng = np.random.default_rng({"fatal": 11, "transient": 22, "degraded": 33}[kind])
        faults = [Fault(kind, int(rng.integers(4)), at=float(rng.uniform(0.1, 0.8) * ff["makespan"]),
                        duration=float(rng.exponential(0.8))) for _ in range(N_B)]
        variants = [(p, True) for p in POLS]
        if kind == "degraded":
            variants.append(("blind_retry", False))
        for pol, dedup in variants:
            cfg = SimCfg(policy=pol, dedup=dedup, **base_cfg)
            for i, f in enumerate(faults):
                r = run(combo, cfg, f, seed=i)
                ok = bool(r["completed"]) and r["digest"] == ff["digest"]
                acts = [a for a in r["actions"].split("|") if a]
                rows.append({"kind": kind, "policy": pol if dedup else pol + "_no_dedup", "trial": i,
                             "fault_worker": f.worker, "fault_at_s": r5(f.at), "transient_duration_s": r5(f.duration),
                             "completed": int(r["completed"]), "consistent": int(ok),
                             "on_time": int(ok and r["makespan"] <= deadline),
                             "makespan_s": r5(r["makespan"]), "recovery_latency_s": r5(r["makespan"] - ff["makespan"]) if r["completed"] else "",
                             "retries": r["retries"], "final_action": acts[-1] if acts else "none_needed",
                             "actions": "|".join(acts) if acts else "none_needed",
                             "lost_compute_s": r5(r["wasted_s"]), "recompute_s": r5(r["compute_s"] - ff["compute_s"]) if r["completed"] else "",
                             "dup_intercepted": r["dup_intercepted"], "dup_accepted": r["dup_accepted"],
                             "detect_s": r5(r["detect_s"]) if r["detect_s"] == r["detect_s"] else ""})
    return rows, ff, deadline


def agg(rows, keys, metrics):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    out = []
    for k, rs in g.items():
        row = dict(zip(keys, k))
        row["n"] = len(rs)
        for m in metrics:
            v = np.array([float(x[m]) for x in rs if x[m] != ""])
            v = v[~np.isnan(v)]
            row[m + "_mean"] = r5(v.mean()) if len(v) else ""
            row[m + "_std"] = r5(v.std(ddof=1)) if len(v) > 1 else ""
        out.append(row)
    return out


def _save(fig, name):
    fig.text(0.99, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="#6b7280")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)


COL = {"recompute": "#6b7280", "per_stage": "#2563eb", "selective": "#dc2626",
       "none": "#6b7280", "blind_retry": "#f59e0b", "self_describing": "#16a34a", "blind_retry_no_dedup": "#a855f7"}


def figs(a_rows, ff_rows, b_rows, tune_rows, lam):
    combos = list(COMBOS)
    # fig1 recovery time per combo x strategy (all stages/timings)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, tm in zip(axs, TIMINGS):
        for j, st in enumerate(STRATS):
            v = [np.mean([r["recovery_time_s"] for r in a_rows if r["combo"] == c and r["strategy"] == st and r["timing"] == tm]) for c in combos]
            e = [np.std([r["recovery_time_s"] for r in a_rows if r["combo"] == c and r["strategy"] == st and r["timing"] == tm], ddof=1) for c in combos]
            ax.bar(np.arange(4) + (j - 1) * 0.27, v, 0.27, yerr=e, capsize=2, color=COL[st], label=STRAT_CN[st])
        ax.set_xticks(range(4)); ax.set_xticklabels([f"组合{c}\n{COMBOS[c].desc}" for c in combos], fontsize=8)
        ax.set_title(f"故障时机：{TIMING_CN[tm]}"); ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("受影响任务恢复时间 (仿真秒)"); axs[0].legend(fontsize=8)
    _save(fig, "fig1_recovery_time_by_combo.png")

    # fig2 fault-free overhead: state transfer and makespan
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for j, st in enumerate(STRATS):
        v = [next(r for r in ff_rows if r["combo"] == c and r["strategy"] == st)["ff_ckpt_MB"] for c in combos]
        m = [next(r for r in ff_rows if r["combo"] == c and r["strategy"] == st)["ff_makespan_s"] for c in combos]
        a1.bar(np.arange(4) + (j - 1) * 0.27, v, 0.27, color=COL[st], label=STRAT_CN[st])
        a2.bar(np.arange(4) + (j - 1) * 0.27, m, 0.27, color=COL[st], label=STRAT_CN[st])
    for a, t, yl in ((a1, "无故障运行的状态传输量", "MB（对数坐标）"), (a2, "无故障作业完成时间", "仿真秒")):
        a.set_xticks(range(4)); a.set_xticklabels([f"组合{c}" for c in combos]); a.set_title(t); a.set_ylabel(yl); a.grid(axis="y", alpha=.3)
    a1.set_yscale("symlog", linthresh=1); a1.legend(fontsize=8)
    _save(fig, "fig2_fault_free_overhead.png")

    # fig3 recompute by fault stage x timing for combos B and D
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, c in zip(axs, ("B", "D")):
        labels = [f"{STAGE_CN[s]}\n{TIMING_CN[t][:4]}" for s in range(3) for t in TIMINGS]
        for j, st in enumerate(STRATS):
            v = [np.mean([r["recompute_s"] for r in a_rows if r["combo"] == c and r["strategy"] == st and r["fault_stage"] == s and r["timing"] == t])
                 for s in range(3) for t in TIMINGS]
            ax.bar(np.arange(9) + (j - 1) * 0.27, v, 0.27, color=COL[st], label=STRAT_CN[st])
        ax.set_xticks(range(9)); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"组合{c}（{COMBOS[c].desc}）：重复计算量"); ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("重复计算耗时 (仿真秒)"); axs[0].legend(fontsize=8)
    _save(fig, "fig3_recompute_by_timing.png")

    # fig4 action distribution (Exp B)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    act_names = ["none_needed", "retry", "retry_exhausted_isolate", "isolate_recover", "degrade_migrate", "degrade_keep", "blind_retry", "fail"]
    act_cn = {"none_needed": "无需动作", "retry": "退避重试后恢复", "retry_exhausted_isolate": "重试耗尽→隔离迁移",
              "isolate_recover": "隔离+检查点恢复", "degrade_migrate": "降级(迁移出慢节点)", "degrade_keep": "降级(原地继续)",
              "blind_retry": "盲目重试(从头)", "fail": "作业失败"}
    colors = plt.cm.tab10(np.linspace(0, 1, len(act_names)))
    x, ticks = 0, []
    for kind in KINDS:
        for pol in POLS:
            rs = [r for r in b_rows if r["kind"] == kind and r["policy"] == pol]
            cnt = Counter(r["final_action"] for r in rs)
            bottom = 0
            for a, col in zip(act_names, colors):
                h = cnt.get(a, 0) / len(rs) * 100
                ax.bar(x, h, 0.7, bottom=bottom, color=col, label=act_cn[a] if x == 0 or a not in [l.get_label() for l in ax.patches] else None)
                bottom += h
            ticks.append((x, f"{ {'fatal': '致命', 'transient': '瞬态', 'degraded': '性能退化'}[kind]}\n{ {'none': '无恢复', 'blind_retry': '盲目重试', 'self_describing': '自描述'}[pol]}"))
            x += 1
        x += 0.5
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=act_cn[a]) for a, c in zip(act_names, colors)], fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], fontsize=8)
    ax.set_ylabel("试验占比 (%)，按最终动作")
    _save(fig, "fig4_action_distribution.png")

    # fig5 recovery latency CDF
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, kind in zip(axs, KINDS):
        for pol in POLS:
            v = np.sort([r["recovery_latency_s"] for r in b_rows if r["kind"] == kind and r["policy"] == pol and r["consistent"]])
            if len(v):
                n_all = len([r for r in b_rows if r["kind"] == kind and r["policy"] == pol])
                ax.step(v, np.arange(1, len(v) + 1) / n_all, where="post", color=COL[pol], label=POL_CN[pol])
        ax.set_title(KIND_CN[kind]); ax.set_xlabel("作业完成时间增量 (仿真秒)"); ax.grid(alpha=.3); ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("累计比例（以全部试验为分母，未完成不计入）"); axs[0].legend(fontsize=8)
    _save(fig, "fig5_recovery_latency_cdf.png")

    # fig6 retry histogram (transient)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, pol in enumerate(("blind_retry", "self_describing")):
        rs = [r for r in b_rows if r["kind"] == "transient" and r["policy"] == pol]
        cnt = Counter(min(r["retries"], 6) for r in rs)
        xs = np.arange(0, 7)
        ax.bar(xs + (j - 0.5) * 0.38, [cnt.get(k, 0) / len(rs) * 100 for k in xs], 0.38, color=COL[pol], label=POL_CN[pol])
    ax.set_xticks(range(7)); ax.set_xticklabels([str(i) for i in range(6)] + ["≥6"])
    ax.set_xlabel("单次试验内的重试次数"); ax.set_ylabel("试验占比 (%)"); ax.legend(); ax.grid(axis="y", alpha=.3)
    ax.set_title("瞬态故障下的重试次数分布")
    _save(fig, "fig6_transient_retry_histogram.png")

    # fig7 completion rate
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, pol in enumerate(POLS):
        v = [np.mean([r["on_time"] for r in b_rows if r["kind"] == k and r["policy"] == pol]) * 100 for k in KINDS]
        c = [np.mean([r["consistent"] for r in b_rows if r["kind"] == k and r["policy"] == pol]) * 100 for k in KINDS]
        ax.bar(np.arange(3) + (j - 1) * 0.27, c, 0.27, color=COL[pol], alpha=.35)
        ax.bar(np.arange(3) + (j - 1) * 0.27, v, 0.27, color=COL[pol], label=POL_CN[pol])
    ax.set_xticks(range(3)); ax.set_xticklabels([KIND_CN[k] for k in KINDS]); ax.set_ylabel("比例 (%)")
    ax.set_title(f"作业完成率：深色=截止时间({DEADLINE_FACTOR}×无故障完成时间)内正确完成，浅色=最终正确完成")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    _save(fig, "fig7_completion_rate.png")

    # fig8 lambda tuning
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot([r["lambda"] for r in tune_rows], [r["mean_fault_makespan_s"] for r in tune_rows], "o-", color="#dc2626", label="含故障平均完成时间")
    ax1.plot([r["lambda"] for r in tune_rows], [r["ff_makespan_s"] for r in tune_rows], "s--", color="#2563eb", label="无故障完成时间")
    ax2 = ax1.twinx()
    ax2.bar([r["lambda"] for r in tune_rows], [r["ff_ckpt_MB"] for r in tune_rows], width=[l * 0.3 for l in LAMS], alpha=.25, color="#16a34a", label="无故障状态传输量")
    ax1.set_xscale("log", base=2); ax1.set_xlabel("λ（对数坐标）"); ax1.set_ylabel("仿真秒"); ax2.set_ylabel("MB")
    ax1.axvline(lam, color="#9ca3af", ls=":")
    l1, b1 = ax1.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, fontsize=8); ax1.set_title(f"调参配置上的 λ 选择（选定 λ={lam:g}）"); ax1.grid(alpha=.3)
    _save(fig, "fig8_lambda_tuning.png")


def main():
    setup_font()
    FIGS.mkdir(exist_ok=True)
    lam, tune_rows = tune_lambda()
    print("selected lambda", lam)
    write_csv("lambda_tuning.csv", tune_rows)
    a_rows, ff_rows = exp_a(lam)
    write_csv("expA_runs.csv", a_rows)
    write_csv("expA_fault_free.csv", ff_rows)
    mets = ["completed", "consistent", "recovery_time_s", "makespan_delta_s", "detect_s", "lost_compute_s", "recompute_s",
            "normal_ckpt_MB", "recovery_load_MB", "store_MB", "dup_intercepted"]
    write_csv("expA_summary_by_combo_strategy_timing.csv", agg(a_rows, ["combo", "strategy", "timing"], mets))
    write_csv("expA_summary_by_combo_strategy.csv", agg(a_rows, ["combo", "strategy"], mets))
    write_csv("expA_summary_by_combo_strategy_stage.csv", agg(a_rows, ["combo", "strategy", "fault_stage"], mets))
    b_rows, ffb, deadline = exp_b(lam)
    write_csv("expB_runs.csv", b_rows)
    bm = ["completed", "consistent", "on_time", "recovery_latency_s", "retries", "lost_compute_s", "recompute_s", "dup_intercepted", "dup_accepted", "detect_s"]
    summ = agg(b_rows, ["kind", "policy"], bm)
    for r in summ:
        r["deadline_s"] = r5(deadline)
        r["ff_makespan_s"] = r5(ffb["makespan"])
    write_csv("expB_summary.csv", summ)
    act = []
    for (k, p), cnt in sorted(Counter((r["kind"], r["policy"], ) for r in b_rows).items()):
        c2 = Counter(r["final_action"] for r in b_rows if r["kind"] == k and r["policy"] == p)
        for a, n in sorted(c2.items()):
            act.append({"kind": k, "policy": p, "final_action": a, "count": n, "ratio": r5(n / cnt)})
    write_csv("expB_action_distribution.csv", act)
    rh = []
    for p in ("blind_retry", "self_describing"):
        rs = [r for r in b_rows if r["kind"] == "transient" and r["policy"] == p]
        for k, n in sorted(Counter(r["retries"] for r in rs).items()):
            rh.append({"policy": p, "retries": k, "count": n, "ratio": r5(n / len(rs))})
    write_csv("expB_transient_retry_histogram.csv", rh)
    figs(a_rows, ff_rows, b_rows, tune_rows, lam)
    print("done")


if __name__ == "__main__":
    main()
