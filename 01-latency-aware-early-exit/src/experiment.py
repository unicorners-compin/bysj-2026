"""Parameter sweeps for latency-aware early exit (toy / simulated numbers).

Run from topic root:
  python3 -m src.experiment

Writes CSV + PNG under figs/. All values are undergrad-friendly Python
simulations — not DPDK or hardware measurements.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .early_exit import random_stages, run_with_budget

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
NOTE = "toy simulated (Python); not hardware / DPDK"


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


def sweep_budget(n_trials: int = 200, seed: int = 42):
    """Vary latency budget; record success (conf>=0.85) and exit stage."""
    rng = random.Random(seed)
    budgets = [10, 15, 20, 25, 30, 40, 50, 70, 100]
    threshold = 0.85
    rows = []
    for budget in budgets:
        stages_hit = []
        success = 0
        latencies = []
        for _ in range(n_trials):
            r = run_with_budget(
                random_stages(rng),
                budget_ms=float(budget),
                confidence_threshold=threshold,
            )
            stages_hit.append(r.stage)
            latencies.append(r.latency_ms)
            if r.confidence >= threshold:
                success += 1
        rows.append(
            {
                "budget_ms": budget,
                "success_rate": success / n_trials,
                "mean_exit_stage": float(np.mean(stages_hit)),
                "mean_latency_ms": float(np.mean(latencies)),
                "n_trials": n_trials,
                "note": NOTE,
            }
        )
    return rows


def sweep_confidence(n_trials: int = 200, seed: int = 7):
    """Vary confidence threshold at fixed budget; record latency."""
    rng = random.Random(seed)
    thresholds = [0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95]
    budget = 50.0
    rows = []
    for thr in thresholds:
        latencies = []
        stages = []
        for _ in range(n_trials):
            r = run_with_budget(
                random_stages(rng),
                budget_ms=budget,
                confidence_threshold=thr,
            )
            latencies.append(r.latency_ms)
            stages.append(r.stage)
        rows.append(
            {
                "confidence_threshold": thr,
                "budget_ms": budget,
                "mean_latency_ms": float(np.mean(latencies)),
                "mean_exit_stage": float(np.mean(stages)),
                "n_trials": n_trials,
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


def plot_budget(rows: list) -> Path:
    budgets = [r["budget_ms"] for r in rows]
    success = [r["success_rate"] for r in rows]
    stages = [r["mean_exit_stage"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(budgets, success, "o-", color="#2563eb", label="成功率 (conf≥0.85)")
    ax1.set_xlabel("时延预算 (ms)")
    ax1.set_ylabel("成功率", color="#2563eb")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(budgets, stages, "s--", color="#dc2626", label="平均退出级数")
    ax2.set_ylabel("平均退出级数", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title("时延预算扫描：成功率与退出级数\n（Python 玩具仿真，非实测）")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGS / "budget_vs_success_stage.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_confidence(rows: list) -> Path:
    thr = [r["confidence_threshold"] for r in rows]
    lat = [r["mean_latency_ms"] for r in rows]
    stages = [r["mean_exit_stage"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(thr, lat, "o-", color="#059669", label="平均时延")
    ax1.set_xlabel("置信度阈值")
    ax1.set_ylabel("平均时延 (ms)", color="#059669")
    ax1.tick_params(axis="y", labelcolor="#059669")

    ax2 = ax1.twinx()
    ax2.plot(thr, stages, "s--", color="#7c3aed", label="平均退出级数")
    ax2.set_ylabel("平均退出级数", color="#7c3aed")
    ax2.tick_params(axis="y", labelcolor="#7c3aed")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("置信度阈值扫描：时延与退出级数\n（budget=50ms，Python 玩具仿真）")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGS / "confidence_vs_latency.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    _setup_font()
    _ensure_figs()
    budget_rows = sweep_budget()
    conf_rows = sweep_confidence()
    _write_csv(FIGS / "budget_sweep.csv", budget_rows)
    _write_csv(FIGS / "confidence_sweep.csv", conf_rows)
    p1 = plot_budget(budget_rows)
    p2 = plot_confidence(conf_rows)
    print(f"wrote {FIGS / 'budget_sweep.csv'}")
    print(f"wrote {FIGS / 'confidence_sweep.csv'}")
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print(f"NOTE: {NOTE}")


if __name__ == "__main__":
    main()
