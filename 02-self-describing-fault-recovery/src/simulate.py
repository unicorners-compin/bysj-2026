from .fault import FaultDesc, FaultKind, recover_until_ok, plan_recovery


def main() -> None:
    cases = [
        FaultDesc(FaultKind.TRANSIENT, "link-a", "packet loss", True, 2),
        FaultDesc(FaultKind.DEGRADED, "classifier", "low confidence", False),
        FaultDesc(FaultKind.FATAL, "switch-1", "ASIC hang", False),
    ]
    for f in cases:
        first = plan_recovery(f, 0)
        final = recover_until_ok(f)
        print(f"{f.kind.value:10} first={first.name:8} final={final.name:8} | {final.detail}")


if __name__ == "__main__":
    main()
