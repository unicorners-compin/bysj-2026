"""Self-describing faults and recovery policies."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FaultKind(str, Enum):
    TRANSIENT = "transient"
    DEGRADED = "degraded"
    FATAL = "fatal"


@dataclass(frozen=True)
class FaultDesc:
    kind: FaultKind
    component: str
    message: str
    retryable: bool
    max_retries: int = 3


@dataclass
class RecoveryAction:
    name: str
    detail: str


def plan_recovery(fault: FaultDesc, attempt: int = 0) -> RecoveryAction:
    if fault.kind == FaultKind.TRANSIENT and fault.retryable and attempt < fault.max_retries:
        return RecoveryAction("retry", f"retry {attempt + 1}/{fault.max_retries} on {fault.component}")
    if fault.kind == FaultKind.DEGRADED:
        return RecoveryAction("degrade", f"serve fallback for {fault.component}")
    return RecoveryAction("isolate", f"isolate {fault.component}: {fault.message}")


def recover_until_ok(fault: FaultDesc) -> RecoveryAction:
    """Simulate attempts until a non-retry action is chosen."""
    action: Optional[RecoveryAction] = None
    for attempt in range(fault.max_retries + 1):
        action = plan_recovery(fault, attempt)
        if action.name != "retry":
            return action
    assert action is not None
    return action
