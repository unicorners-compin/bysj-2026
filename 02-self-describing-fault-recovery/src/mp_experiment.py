"""Multiprocess fault-recovery experiment (wall clock, real processes).

Compares whole-job recompute, save-every-stage, and selective saving
(C_k >= lambda * P_k) when one worker is SIGKILLed mid-run. The coordinator
stays up. Digests are checked against a fault-free run of the same code and
against a sequential execution of the three stage functions.

Run from the topic directory:  python3 -m src.mp_experiment
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .mp_runtime import KillSpec, MpCfg, run_job
from .task import Combo, global_digest, make_partition, run_stage

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
STRATS = ("recompute", "per_stage", "selective")
STRAT_CN = {"recompute": "整体重算", "per_stage": "每阶段保存", "selective": "选择性保存"}
LAM = 1.0
N_WORKERS = 4
N_TASKS = 3
REPS = 4
MIN_STAGE_S = 0.08
# Heavy checkpoints carry 2048 real bytes per surviving record (~4 MB each).
# Three tasks × two saved stages stay in the tens of MB once pickled.
COMBOS = {
    "light": Combo("light", 4000, 5.0, 10.0, 2.0, 16, "状态小", wire_payload=0),
    "heavy": Combo("heavy", 4000, 5.0, 10.0, 2.0, 16, "状态大", wire_payload=2048),
}
FOOT = "多进程原型：1 个协调进程 + 4 个工作进程，运行中 SIGKILL 一个工作进程；阶段为真实 numpy，恢复时间为墙钟"


def setup_font():
    from matplotlib import font_manager
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def write_csv(name, rows):
    with open(FIGS / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def r5(x):
    return round(float(x), 5)


def sequential_digest(combo: Combo) -> str:
    partials = []
    for i in range(N_TASKS):
        data = make_partition(i, combo, 0)
        for stage in range(3):
            data = run_stage(stage, data, combo)
        partials.append(data)
    return global_digest(partials)


def _cfg(strategy: str) -> MpCfg:
    return MpCfg(
        n_workers=N_WORKERS,
        n_tasks=N_TASKS,
        strategy=strategy,
        lam=LAM,
        data_seed=0,
        min_stage_s=MIN_STAGE_S,
        timeout_s=40,
        hb_timeout=3.0,
    )


def _base(combo, strategy, result, consistent):
    return {
        "combo": combo,
        "strategy": strategy,
        "completed": int(result["completed"]),
        "consistent": int(consistent),
        "digest": result["digest"],
        "makespan_s": r5(result["makespan_s"]),
        "ckpt_MB": r5(result["ckpt_bytes"] / 1e6),
        "recovery_load_MB": r5(result["recovery_load_bytes"] / 1e6),
        "state_transfer_MB": r5(result["state_transfer_bytes"] / 1e6),
        "duplicate_compute_s": r5(result["duplicate_compute_s"]),
        "reexec_stages": result["reexec_stages"],
        "stage_runs": result["stage_runs"],
        "dup_intercepted": result["dup_intercepted"],
        "final_count": result["final_count"],
    }


def run_suite():
    fault_rows, free_rows = [], []
    for name, combo in COMBOS.items():
        ref = sequential_digest(combo)
        for strategy in STRATS:
            cfg = _cfg(strategy)
            free = run_job(combo, cfg)
            ok = free["completed"] and free["digest"] == ref
            free_rows.append(_base(name, strategy, free, ok))
            for rep in range(REPS):
                killed = run_job(combo, cfg, kill=KillSpec(task_index=0, stage_id=1))
                consistent = int(killed["completed"] and killed["digest"] == free["digest"] == ref)
                fault_rows.append({
                    **_base(name, strategy, killed, consistent),
                    "rep": rep,
                    "kill_task": 0,
                    "kill_stage": 1,
                    "recovery_time_s": r5(killed["recovery_time_s"]),
                    "lost_compute_s": r5(killed["lost_compute_s"]),
                    "detect_s": r5(killed["detect_s"]),
                    "detect_reason": killed["detect_reason"],
                    "victim_exitcode": killed["victim_exitcode"],
                    "from_ckpt": killed["from_ckpt"],
                    "from_scratch": killed["from_scratch"],
                    "ff_makespan_s": r5(free["makespan_s"]),
                    "makespan_delta_s": r5(killed["makespan_s"] - free["makespan_s"]),
                })
    return fault_rows, free_rows


def _group(rows, combo, strategy):
    return [r for r in rows if r["combo"] == combo and r["strategy"] == strategy]


def summarize(fault_rows, free_rows):
    out = []
    for name in COMBOS:
        for strategy in STRATS:
            rs = _group(fault_rows, name, strategy)
            free = _group(free_rows, name, strategy)[0]

            def mean_std(key):
                v = np.array([float(r[key]) for r in rs], dtype=float)
                std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                return r5(v.mean()), r5(std)

            rec_m, rec_s = mean_std("recovery_time_s")
            dup_m, dup_s = mean_std("duplicate_compute_s")
            st_m, st_s = mean_std("state_transfer_MB")
            out.append({
                "combo": name,
                "strategy": strategy,
                "n": len(rs),
                "completion_rate": r5(np.mean([r["consistent"] for r in rs])),
                "recovery_time_s_mean": rec_m,
                "recovery_time_s_std": rec_s,
                "duplicate_compute_s_mean": dup_m,
                "duplicate_compute_s_std": dup_s,
                "state_transfer_MB_mean": st_m,
                "state_transfer_MB_std": st_s,
                "ckpt_MB_mean": mean_std("ckpt_MB")[0],
                "dup_intercepted_mean": mean_std("dup_intercepted")[0],
                "reexec_stages_mean": mean_std("reexec_stages")[0],
                "ff_state_transfer_MB": free["state_transfer_MB"],
                "ff_makespan_s": free["makespan_s"],
            })
    return out


def _save(fig, name):
    handles, labels = [], []
    for ax in fig.axes:
        found_h, found_l = ax.get_legend_handles_labels()
        if found_h:
            handles, labels = found_h, found_l
            break
    fig.tight_layout()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14),
                   frameon=False, fontsize=8)
    fig.text(0.5, -0.24, FOOT, ha="center", va="top", fontsize=7, color="#6b7280")
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figs(summary):
    combos = list(COMBOS)
    labels = [f"{c}\n{COMBOS[c].desc}" for c in combos]
    x = np.arange(len(combos))
    colors = {"recompute": "#6b7280", "per_stage": "#2563eb", "selective": "#dc2626"}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for j, strategy in enumerate(STRATS):
        means, stds = [], []
        for combo in combos:
            row = next(r for r in summary if r["combo"] == combo and r["strategy"] == strategy)
            means.append(row["recovery_time_s_mean"])
            stds.append(row["recovery_time_s_std"])
        ax.bar(x + (j - 1) * 0.25, means, 0.25, yerr=stds, capsize=3, color=colors[strategy], label=STRAT_CN[strategy])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("恢复时间 (墙钟秒)")
    ax.set_title("杀死一个工作进程后，受影响任务的恢复时间")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_mp1_recovery_time.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels = (
        (axes[0], "duplicate_compute_s_mean", "duplicate_compute_s_std", "重复计算 (墙钟秒)", "重复计算（重做阶段 + 被杀死的部分）"),
        (axes[1], "state_transfer_MB_mean", "state_transfer_MB_std", "状态传输量 (MB)", "状态传输量（检查点 + 恢复装载，序列化字节）"),
    )
    for ax, key, err, ylabel, title in panels:
        for j, strategy in enumerate(STRATS):
            means, stds = [], []
            for combo in combos:
                row = next(r for r in summary if r["combo"] == combo and r["strategy"] == strategy)
                means.append(row[key])
                stds.append(row[err])
            ax.bar(x + (j - 1) * 0.25, means, 0.25, yerr=stds, capsize=3, color=colors[strategy], label=STRAT_CN[strategy])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_yscale("symlog", linthresh=1)
    _save(fig, "fig_mp2_duplicate_and_transfer.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for j, strategy in enumerate(STRATS):
        vals = [next(r for r in summary if r["combo"] == c and r["strategy"] == strategy)["completion_rate"] * 100 for c in combos]
        ax.bar(x + (j - 1) * 0.25, vals, 0.25, color=colors[strategy], label=STRAT_CN[strategy])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("完成率 (%)")
    ax.set_title("最终摘要与无故障运行一致的比例")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_mp3_completion_rate.png")


def main():
    setup_font()
    FIGS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    fault_rows, free_rows = run_suite()
    wall = time.perf_counter() - t0
    summary = summarize(fault_rows, free_rows)
    write_csv("mp_runs.csv", fault_rows)
    write_csv("mp_fault_free.csv", free_rows)
    write_csv("mp_summary.csv", summary)
    write_csv("mp_meta.csv", [{
        "wall_s": r5(wall),
        "n_fault_runs": len(fault_rows),
        "n_workers": N_WORKERS,
        "n_tasks": N_TASKS,
        "reps": REPS,
        "lambda": LAM,
        "min_stage_s": MIN_STAGE_S,
        "consistent_runs": int(sum(r["consistent"] for r in fault_rows)),
    }])
    figs(summary)
    print(f"wall_s {wall:.2f}")
    for row in summary:
        print(
            f"{row['combo']:5} {row['strategy']:10} completion={row['completion_rate']:.2f} "
            f"recovery={row['recovery_time_s_mean']:.3f}s dup={row['duplicate_compute_s_mean']:.3f}s "
            f"transfer={row['state_transfer_MB_mean']:.3f}MB"
        )
    if any(not r["consistent"] for r in fault_rows):
        raise SystemExit("a fault run did not match the fault-free digest")


if __name__ == "__main__":
    main()
