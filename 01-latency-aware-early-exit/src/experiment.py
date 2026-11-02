"""Stable vs dynamic link experiment for latency-aware early exit.

Exit confidences and predictions come from the committed MNIST two-exit CNN
(src.train). Stage times come from figs/compute_timing.csv. The link is not a
network interface: bandwidth, delay and jitter are turned into a delay trace
inside this process (src.netsim).

Stable regime: one bandwidth and one propagation delay, no jitter.
Dynamic regime: slow bandwidth wander, congestion bursts, and per-sample jitter.

Fixed threshold and latency-aware parameters are chosen on the validation split
of the same regime (objective: on-time accuracy, then accuracy) and scored on
the test split. Run from the topic directory:  python3 -m src.experiment
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .netsim import DYNAMIC_PRIOR, STABLE_LINK, NodeCfg, dynamic_link_sequence, make_trace
from .policies import Policy, run_policy

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
SEEDS = (0, 1, 2)
BUDGET_MS = 20.0
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
    seq = dynamic_link_sequence(n, rng)
    return make_trace(n, DYNAMIC_PRIOR, timing, NODE, rng, link_seq=seq)


def _better(metrics, best):
    key = (round(metrics["ontime_accuracy"], 5), round(metrics["accuracy"], 5), -metrics["timeout_rate"])
    return best is None or key > best[0]


def select_fixed(data, tr):
    best = None
    for tau in TAU_GRID:
        policy = Policy("fixed", "fixed", tau=tau, feasibility=False)
        metrics = run_policy(policy, data["conf1"], data["pred1"], data["pred2"], data["label"], tr, BUDGET_MS, NODE)
        if _better(metrics, best):
            best = ((round(metrics["ontime_accuracy"], 5), round(metrics["accuracy"], 5), -metrics["timeout_rate"]), tau)
    return best[1]


def select_la(data, tr):
    best = None
    for prm in LA_GRID:
        policy = Policy("latency_aware", "dynamic", **prm)
        metrics = run_policy(policy, data["conf1"], data["pred1"], data["pred2"], data["label"], tr, BUDGET_MS, NODE)
        if _better(metrics, best):
            best = ((round(metrics["ontime_accuracy"], 5), round(metrics["accuracy"], 5), -metrics["timeout_rate"]), prm)
    return best[1]


def score(policy, data, tr):
    metrics = run_policy(policy, data["conf1"], data["pred1"], data["pred2"], data["label"], tr, BUDGET_MS, NODE)
    return metrics


def evaluate_seed(seed: int, timing: dict | None = None):
    """Tune on validation and score on test for both regimes. Returns row dicts and the dynamic test series."""
    if timing is None:
        timing = load_timing()[seed]
    val, test = load_seed(seed)
    rows = []
    series = None
    chosen = {}
    for regime, val_stream, test_stream in (
        ("stable", 10 + seed, 110 + seed),
        ("dynamic", 20 + seed, 210 + seed),
    ):
        tr_val = trace_for(regime, len(val["label"]), timing, np.random.default_rng(val_stream))
        tr_test = trace_for(regime, len(test["label"]), timing, np.random.default_rng(test_stream))
        tau = select_fixed(val, tr_val)
        prm = select_la(val, tr_val)
        chosen[regime] = {"tau_fixed": tau, **prm}
        policies = [
            Policy("full", "full"),
            Policy("fixed", "fixed", tau=tau, feasibility=False),
            Policy("latency_aware", "dynamic", **prm),
        ]
        for policy in policies:
            metrics = score(policy, test, tr_test)
            rows.append({
                "seed": seed,
                "regime": regime,
                "budget_ms": BUDGET_MS,
                "policy": policy.name,
                "tau_fixed": tau if policy.name == "fixed" else "",
                **_public(metrics),
            })
            if regime == "dynamic" and seed == 0:
                if series is None:
                    series = {
                        "burst": np.array([link.bw_mbps < 20.0 for link in dynamic_link_sequence(len(test["label"]), np.random.default_rng(test_stream))], dtype=float),
                        "budget_ms": BUDGET_MS,
                    }
                series[policy.name] = (metrics["_lat"] > BUDGET_MS).astype(float)
    return rows, chosen, series


def summarize(rows):
    metrics = ["accuracy", "ontime_accuracy", "deadline_hit_rate", "timeout_rate", "mean_latency_ms", "exit1_ratio"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["regime"], row["policy"])].append(row)
    out = []
    for (regime, policy), items in groups.items():
        rec = {"regime": regime, "policy": policy, "budget_ms": BUDGET_MS, "n_seeds": len(items)}
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
    regimes = ("stable", "dynamic")
    regime_cn = {"stable": "稳定网络", "dynamic": "波动网络"}
    lookup = {(r["regime"], r["policy"]): r for r in summary}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    x = np.arange(len(regimes))
    for ax, metric, title, ylab in (
        (axes[0], "ontime_accuracy", "按时正确率", "按时正确率 (%)"),
        (axes[1], "deadline_hit_rate", "截止时间命中率", "命中率 (%)"),
    ):
        for j, policy in enumerate(order):
            means = [lookup[(regime, policy)][metric + "_mean"] * 100 for regime in regimes]
            stds = [lookup[(regime, policy)][metric + "_std"] * 100 for regime in regimes]
            ax.bar(x + (j - 1) * 0.25, means, 0.25, yerr=stds, capsize=3, color=colors[policy], label=names[policy])
        ax.set_xticks(x)
        ax.set_xticklabels([regime_cn[r] for r in regimes])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=8)
    fig.suptitle(f"预算 {BUDGET_MS:.0f} ms，验证集选参后在测试集上的 3 个种子均值")
    _save(fig, "fig_stable_vs_dynamic.png")

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
    ax.set_title("波动网络下的滑动超时率（种子 0）")
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
        for regime, prm in chosen.items():
            params.append({"seed": seed, "regime": regime, "budget_ms": BUDGET_MS, **prm})
        if seed_series is not None:
            series = seed_series
    summary = summarize(rows)
    write_csv("stable_vs_dynamic_per_seed.csv", rows)
    write_csv("stable_vs_dynamic_summary.csv", summary)
    write_csv("params_selected.csv", params)
    make_figs(summary, series)
    for row in summary:
        print(
            f"{row['regime']:8} {row['policy']:16} "
            f"ontime={row['ontime_accuracy_mean']:.4f} hit={row['deadline_hit_rate_mean']:.4f} "
            f"acc={row['accuracy_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
