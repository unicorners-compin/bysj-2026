"""CLI demo for latency-aware early exit."""
from __future__ import annotations

from .early_exit import demo_stages, run_with_budget


def main() -> None:
    for budget in (10.0, 25.0, 50.0):
        r = run_with_budget(demo_stages(), budget_ms=budget)
        print(
            f"budget={budget:4.0f}ms -> stage={r.stage} "
            f"conf={r.confidence:.2f} latency={r.latency_ms:.1f}ms out={r.output}"
        )


if __name__ == "__main__":
    main()
