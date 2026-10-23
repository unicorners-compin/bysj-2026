"""Baseline (CPU-only) vs communication-aware placement (toy / simulated).

Run from topic root:
  python3 -m src.experiment

Writes CSV + PNG under figs/. Undergrad Python sims — not cluster traces.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .scheduler import Node, place_tasks

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
NOTE = "toy simulated (Python); not real cluster traces"


def _setup_font() -> None:
    from matplotlib import font_manager
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
                prop = font_manager.FontProperties(fname=path)
                plt.rcParams["font.family"] = prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue
    # Fallback: English-friendly DejaVu (CJK glyphs may be missing)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False



def _ensure_figs() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)


def random_nodes(rng: random.Random, n: int = 5) -> list:
    return [
        Node(
            name=f"n{i}",
            cpu_load=rng.uniform(0.05, 0.85),
            comm_load=rng.uniform(0.05, 0.90),
        )
        for i in range(n)
    ]


def load_balance_metric(placements, nodes_after_cpu, nodes_after_comm) -> float:
    """Std of final comm loads (lower = more balanced on network)."""
    return float(np.std(nodes_after_comm))


def simulate_policy(alpha: float, beta: float, n_trials: int, n_tasks: int, seed: int):
    """Place with (alpha, beta); evaluate with a fixed true cost (0.4 cpu + 0.6 comm)."""
    from .scheduler import score_node

    eval_alpha, eval_beta = 0.4, 0.6
    rng = random.Random(seed)
    avg_scores = []
    balance = []
    for _ in range(n_trials):
        nodes = random_nodes(rng)
        tasks = [f"t{i}" for i in range(n_tasks)]
        loads = {n.name: [n.cpu_load, n.comm_load] for n in nodes}
        # Greedy with policy weights, but score each pick under eval weights
        trial_scores = []
        for task in tasks:
            best_name = None
            best_policy = float("inf")
            best_eval = float("inf")
            for name, (cpu, comm) in loads.items():
                pol = score_node(Node(name, cpu, comm), alpha=alpha, beta=beta)
                ev = score_node(Node(name, cpu, comm), alpha=eval_alpha, beta=eval_beta)
                if pol < best_policy:
                    best_policy = pol
                    best_name = name
                    best_eval = ev
            assert best_name is not None
            trial_scores.append(best_eval)
            cpu, comm = loads[best_name]
            loads[best_name] = (min(1.0, cpu + 0.15), min(1.0, comm + 0.2))
        avg_scores.append(float(np.mean(trial_scores)))
        final_comm = [v[1] for v in loads.values()]
        balance.append(float(np.std(final_comm)))
    return {
        "mean_avg_score": float(np.mean(avg_scores)),
        "std_avg_score": float(np.std(avg_scores)),
        "mean_comm_load_std": float(np.mean(balance)),
        "std_comm_load_std": float(np.std(balance)),
        "eval_alpha": eval_alpha,
        "eval_beta": eval_beta,
    }


def run_comparison(n_trials: int = 200, n_tasks: int = 8, seed: int = 99) -> list:
    policies = [
        ("cpu_only", 1.0, 0.0),
        ("comm_aware", 0.4, 0.6),
        ("balanced", 0.5, 0.5),
    ]
    rows = []
    for name, a, b in policies:
        m = simulate_policy(a, b, n_trials, n_tasks, seed)
        rows.append(
            {
                "policy": name,
                "alpha": a,
                "beta": b,
                "n_trials": n_trials,
                "n_tasks": n_tasks,
                **m,
                "note": NOTE,
            }
        )
    return rows


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_scores(rows: list) -> Path:
    labels = [r["policy"] for r in rows]
    means = [r["mean_avg_score"] for r in rows]
    errs = [r["std_avg_score"] for r in rows]
    colors = ["#94a3b8", "#2563eb", "#059669"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, color=colors[: len(labels)], capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("统一评价得分 0.4·CPU+0.6·Comm（越低越好）")
    ax.set_title("策略对比：统一评价得分\n（Python 玩具仿真，非集群轨迹）")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "policy_avg_score.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_balance(rows: list) -> Path:
    labels = [r["policy"] for r in rows]
    means = [r["mean_comm_load_std"] for r in rows]
    errs = [r["std_comm_load_std"] for r in rows]
    colors = ["#94a3b8", "#2563eb", "#059669"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, color=colors[: len(labels)], capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("放置后通信负载标准差（越低越均衡）")
    ax.set_title("策略对比：通信负载均衡\n（Python 玩具仿真）")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "policy_comm_balance.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    _setup_font()
    _ensure_figs()
    rows = run_comparison()
    _write_csv(FIGS / "policy_comparison.csv", rows)
    p1 = plot_scores(rows)
    p2 = plot_balance(rows)
    print(f"wrote {FIGS / 'policy_comparison.csv'}")
    for r in rows:
        print(
            f"  {r['policy']}: mean_score={r['mean_avg_score']:.3f} "
            f"comm_std={r['mean_comm_load_std']:.3f}"
        )
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print(f"NOTE: {NOTE}")


if __name__ == "__main__":
    main()
