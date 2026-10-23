"""Monte-Carlo fault recovery experiment (toy / simulated numbers).

Run from topic root:
  python3 -m src.experiment

Writes CSV + PNG under figs/. Undergrad Python sims — not production telemetry.
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fault import FaultDesc, FaultKind, plan_recovery

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
NOTE = "toy simulated (Python); not production telemetry"


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


def simulate_one(fault: FaultDesc, rng: random.Random) -> dict:
    """Simulate recovery attempts; transient may succeed mid-retry (toy)."""
    retries = 0
    final_action = None
    for attempt in range(fault.max_retries + 1):
        action = plan_recovery(fault, attempt)
        if action.name == "retry":
            retries += 1
            # Toy: 55% chance transient clears after a retry
            if fault.kind == FaultKind.TRANSIENT and rng.random() < 0.55:
                final_action = "recovered_after_retry"
                break
            continue
        final_action = action.name
        break
    if final_action is None:
        final_action = "isolate"
    return {
        "kind": fault.kind.value,
        "component": fault.component,
        "retries": retries,
        "final_action": final_action,
        "note": NOTE,
    }


def run_batch(n_per_kind: int = 300, seed: int = 123) -> list:
    rng = random.Random(seed)
    components = ["link-a", "nic-0", "classifier", "switch-1", "queue-b"]
    rows = []
    for kind in FaultKind:
        for i in range(n_per_kind):
            if kind == FaultKind.TRANSIENT:
                fault = FaultDesc(
                    kind, rng.choice(components), "timeout/loss", True, max_retries=rng.choice([2, 3])
                )
            elif kind == FaultKind.DEGRADED:
                fault = FaultDesc(
                    kind, rng.choice(components), "low confidence", False, max_retries=0
                )
            else:
                fault = FaultDesc(
                    kind, rng.choice(components), "hard failure", False, max_retries=0
                )
            rows.append(simulate_one(fault, rng))
    return rows


def summarize(rows: list) -> tuple[list, list]:
    """Action distribution by kind; retry histogram for transient."""
    action_rows = []
    by_kind: dict = defaultdict(Counter)
    for r in rows:
        by_kind[r["kind"]][r["final_action"]] += 1
    for kind, ctr in sorted(by_kind.items()):
        total = sum(ctr.values())
        for action, cnt in sorted(ctr.items()):
            action_rows.append(
                {
                    "kind": kind,
                    "final_action": action,
                    "count": cnt,
                    "rate": cnt / total,
                    "note": NOTE,
                }
            )

    retry_rows = []
    transient = [r for r in rows if r["kind"] == "transient"]
    hist = Counter(r["retries"] for r in transient)
    for k in sorted(hist):
        retry_rows.append(
            {
                "retries": k,
                "count": hist[k],
                "rate": hist[k] / max(1, len(transient)),
                "note": NOTE,
            }
        )
    return action_rows, retry_rows


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_actions(action_rows: list) -> Path:
    kinds = sorted({r["kind"] for r in action_rows})
    actions = sorted({r["final_action"] for r in action_rows})
    x = np.arange(len(kinds))
    width = 0.8 / max(1, len(actions))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2563eb", "#059669", "#dc2626", "#f59e0b", "#7c3aed"]
    for i, action in enumerate(actions):
        rates = []
        for k in kinds:
            match = [r["rate"] for r in action_rows if r["kind"] == k and r["final_action"] == action]
            rates.append(match[0] if match else 0.0)
        ax.bar(x + i * width, rates, width, label=action, color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(actions) - 1) / 2)
    ax.set_xticklabels(kinds)
    ax.set_ylabel("占比")
    ax.set_xlabel("故障类型")
    ax.set_ylim(0, 1.05)
    ax.set_title("恢复动作分布（按故障类型）\n（Python 玩具仿真，非生产遥测）")
    ax.legend(title="最终动作")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "recovery_action_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_retries(retry_rows: list) -> Path:
    xs = [r["retries"] for r in retry_rows]
    ys = [r["count"] for r in retry_rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(xs, ys, color="#2563eb", width=0.6)
    ax.set_xlabel("重试次数（transient）")
    ax.set_ylabel("样本数")
    ax.set_title("瞬时故障重试次数分布\n（Python 玩具仿真）")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "transient_retry_histogram.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    _setup_font()
    _ensure_figs()
    raw = run_batch()
    _write_csv(FIGS / "fault_trials.csv", raw)
    action_rows, retry_rows = summarize(raw)
    _write_csv(FIGS / "action_distribution.csv", action_rows)
    _write_csv(FIGS / "retry_histogram.csv", retry_rows)
    p1 = plot_actions(action_rows)
    p2 = plot_retries(retry_rows)
    print(f"wrote {FIGS / 'fault_trials.csv'} ({len(raw)} rows)")
    print(f"wrote {FIGS / 'action_distribution.csv'}")
    print(f"wrote {FIGS / 'retry_histogram.csv'}")
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print(f"NOTE: {NOTE}")


if __name__ == "__main__":
    main()
