"""Three link regimes × three latency budgets for latency-aware early exit.

Exit confidences and predictions come from the committed MNIST two-exit CNN
(src.train). Stage times come from figs/compute_timing.csv. The link is not a
network interface: bandwidth, delay and jitter are turned into a delay trace
inside this process (src.netsim).

Regimes: stable (fixed bandwidth and delay), moderate fluctuation, strong
fluctuation (wander, congestion bursts, jitter). Budgets: 12 / 20 / 80 ms.
Fixed-threshold and latency-aware parameters are chosen on the validation split
of the same regime and budget (objective: on-time accuracy, then accuracy) and
scored on the test split. Run from the topic directory:  python3 -m src.experiment
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .netsim import (
    DYNAMIC_PRIOR,
    MODERATE_PRIOR,
    STABLE_LINK,
    NodeCfg,
    dynamic_link_sequence,
    make_trace,
    moderate_link_sequence,
)
from .policies import Policy, run_policy

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
SEEDS = (0, 1, 2)
BUDGETS = (("tight", 12.0), ("mid", 20.0), ("loose", 80.0))
REGIMES = ("stable", "moderate", "dynamic")
NODE = NodeCfg()
TAU_GRID = [round(float(x), 3) for x in np.linspace(0.5, 0.995, 16)]
LA_GRID = [
    dict(tau_min=tmin, tau_max=tmax, gamma=g, margin=m, ewma=a, probe_every=1)
    for tmin in (0.5, 0.7, 0.85)
    for tmax in (0.9, 0.97, 0.995)
    for g in (1.0, 2.0)
    for m in (0.0, 1.0)
    for a in (0.4, 0.8)
    if tmin < tmax
]
FOOT = "教学原型：MNIST 两出口 CNN 的出口输出 + Python 进程内链路仿真"


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


def load_seed(seed: int):
    z = np.load(FIGS / f"exit_outputs_seed{seed}.npz")

    def split(pre):
        return {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}

    return split("val_"), split("test_")


def load_timing():
    timing = defaultdict(dict)
    with open(FIGS / "compute_timing.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            timing[int(row["seed"])][row["part"]] = float(row["median_ms"])
    return timing


def write_csv(name, rows):
    rows = [{k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows]
    with open(FIGS / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _public(metrics):
    out = {}
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        out[key] = round(float(value), 5) if isinstance(value, float) else value
    out["deadline_hit_rate"] = round(1.0 - metrics["timeout_rate"], 5)
    return out


def trace_for(regime: str, n: int, timing: dict, rng: np.random.Generator):
    if regime == "stable":
        return make_trace(n, STABLE_LINK, timing, NODE, rng)
    if regime == "moderate":
        seq = moderate_link_sequence(n, rng)
        return make_trace(n, MODERATE_PRIOR, timing, NODE, rng, link_seq=seq)
    if regime == "dynamic":
        seq = dynamic_link_sequence(n, rng)
        return make_trace(n, DYNAMIC_PRIOR, timing, NODE, rng, link_seq=seq)
    raise ValueError(regime)


def _better(metrics, best):
    key = (round(metrics["ontime_accuracy"], 5), round(metrics["accuracy"], 5), -metrics["timeout_rate"])
    return best is None or key > best[0]


def select_fixed(data, tr, budget):
    best = None
    for tau in TAU_GRID:
        policy = Policy("fixed", "fixed", tau=tau, feasibility=False)
        metrics = run_policy(policy, data["conf1"], data["pred1"], data["pred2"], data["label"], tr, budget, NODE)
        if _better(metrics, best):
            best = ((round(metrics["ontime_accuracy"], 5), round(metrics["accuracy"], 5), -metrics["timeout_rate"]), tau)
    return best[1]


def select_la(data, tr, budget):
    best = None
    for prm in LA_GRID:
        policy = Policy("latency_aware", "dynamic", **prm)
        metrics = run_policy(policy, data["conf1"], data["pred1"], data["pred2"], data["label"], tr, budget, NODE)
        if _better(metrics, best):
            best = ((round(metrics["ontime_accuracy"], 5), round(metrics["accuracy"], 5), -metrics["timeout_rate"]), prm)
    return best[1]


def score(policy, data, tr, budget):
    return run_policy(policy, data["conf1"], data["pred1"], data["pred2"], data["label"], tr, budget, NODE)


def evaluate_seed(seed: int, timing: dict | None = None):
    """Tune on validation and score on test for every regime and budget."""
    if timing is None:
        timing = load_timing()[seed]
    val, test = load_seed(seed)
    rows = []
    series = None
    chosen = {}
    streams = {"stable": (10 + seed, 110 + seed), "moderate": (30 + seed, 310 + seed), "dynamic": (20 + seed, 210 + seed)}
    for regime in REGIMES:
        val_stream, test_stream = streams[regime]
        tr_val = trace_for(regime, len(val["label"]), timing, np.random.default_rng(val_stream))
        tr_test = trace_for(regime, len(test["label"]), timing, np.random.default_rng(test_stream))
        for level, budget in BUDGETS:
            tau = select_fixed(val, tr_val, budget)
            prm = select_la(val, tr_val, budget)
            chosen[(regime, level)] = {"tau_fixed": tau, **prm}
            policies = [
                Policy("full", "full"),
                Policy("fixed", "fixed", tau=tau, feasibility=False),
                Policy("latency_aware", "dynamic", **prm),
            ]
            for policy in policies:
                metrics = score(policy, test, tr_test, budget)
                rows.append({
                    "seed": seed,
                    "regime": regime,
                    "budget_level": level,
                    "budget_ms": budget,
                    "policy": policy.name,
                    "tau_fixed": tau if policy.name == "fixed" else "",
                    **_public(metrics),
                })
                if regime == "dynamic" and level == "tight" and seed == 0:
                    if series is None:
                        burst = dynamic_link_sequence(len(test["label"]), np.random.default_rng(test_stream))
                        series = {
                            "burst": np.array([link.bw_mbps < 20.0 for link in burst], dtype=float),
                            "budget_ms": budget,
                        }
                    series[policy.name] = (metrics["_lat"] > budget).astype(float)
    return rows, chosen, series


def summarize(rows):
    metrics = ["accuracy", "ontime_accuracy", "deadline_hit_rate", "timeout_rate", "mean_latency_ms", "exit1_ratio"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["regime"], row["budget_level"], row["policy"])].append(row)
    out = []
    for (regime, level, policy), items in groups.items():
        rec = {"regime": regime, "budget_level": level, "policy": policy,
               "budget_ms": items[0]["budget_ms"], "n_seeds": len(items)}
        for metric in metrics:
            values = np.array([item[metric] for item in items], dtype=float)
            rec[metric + "_mean"] = round(float(values.mean()), 5)
            rec[metric + "_std"] = round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 5)
        out.append(rec)
    return out


def _save(fig, name):
    fig.text(0.5, -0.26, FOOT, ha="center", va="top", fontsize=7, color="#6b7280")
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_figs(summary, series):
    order = ("full", "fixed", "latency_aware")
    names = {"full": "完整推理", "fixed": "固定阈值", "latency_aware": "时延感知"}
    colors = {"full": "#6b7280", "fixed": "#2563eb", "latency_aware": "#dc2626"}
    regime_cn = {"stable": "稳定", "moderate": "中等波动", "dynamic": "强波动"}
    level_cn = {"tight": "紧\n12 ms", "mid": "中\n20 ms", "loose": "松\n80 ms"}
    lookup = {(r["regime"], r["budget_level"], r["policy"]): r for r in summary}
    levels = [level for level, _ in BUDGETS]

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), sharey="row")
    x = np.arange(len(levels))
    for col, regime in enumerate(REGIMES):
        for row, (metric, title) in enumerate((("ontime_accuracy", "按时正确率 (%)"), ("deadline_hit_rate", "截止命中率 (%)"))):
            ax = axes[row, col]
            for j, policy in enumerate(order):
                means = [lookup[(regime, level, policy)][metric + "_mean"] * 100 for level in levels]
                stds = [lookup[(regime, level, policy)][metric + "_std"] * 100 for level in levels]
                ax.bar(x + (j - 1) * 0.25, means, 0.25, yerr=stds, capsize=2, color=colors[policy], label=names[policy])
            ax.set_xticks(x)
            ax.set_xticklabels([level_cn[level] for level in levels])
            ax.set_ylim(0, 112)
            ax.grid(axis="y", alpha=0.3)
            if row == 0:
                ax.set_title(regime_cn[regime])
            if col == 0:
                ax.set_ylabel(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)
    fig.suptitle("验证集选参后的测试集结果（3 个种子均值）")
    _save(fig, "fig_regime_budget.png")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    burst = series["burst"]
    ax.fill_between(np.arange(len(burst)), 0, burst * 100, step="mid", color="#fee2e2", label="拥塞突发")
    window = 200
    kernel = np.ones(window) / window
    for policy, color in (("fixed", colors["fixed"]), ("latency_aware", colors["latency_aware"])):
        rolled = np.convolve(series[policy], kernel, mode="same") * 100
        ax.plot(rolled, color=color, lw=1.2, label=names[policy])
    ax.set_xlabel("测试样本序号")
    ax.set_ylabel("滑动超时率 (%)，窗口 200")
    ax.set_title(f"强波动网络、紧预算 {series['budget_ms']:.0f} ms 的滑动超时率（种子 0）")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    _save(fig, "fig_dynamic_timeout_trace.png")


def main():
    setup_font()
    FIGS.mkdir(exist_ok=True)
    timing = load_timing()
    rows, params = [], []
    series = None
    for seed in SEEDS:
        seed_rows, chosen, seed_series = evaluate_seed(seed, timing[seed])
        rows.extend(seed_rows)
        for (regime, level), prm in chosen.items():
            budget = dict(BUDGETS)[level]
            params.append({"seed": seed, "regime": regime, "budget_level": level, "budget_ms": budget, **prm})
        if seed_series is not None:
            series = seed_series
    summary = summarize(rows)
    write_csv("regime_budget_per_seed.csv", rows)
    write_csv("regime_budget_summary.csv", summary)
    write_csv("params_selected.csv", params)
    make_figs(summary, series)
    for row in summary:
        print(
            f"{row['regime']:9} {row['budget_level']:6} {row['policy']:16} "
            f"ontime={row['ontime_accuracy_mean']:.4f} hit={row['deadline_hit_rate_mean']:.4f} "
            f"acc={row['accuracy_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
