"""Small CNN with one intermediate exit and one final exit (thesis Table 4-1)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage1(nn.Module):
    """Runs on node 1 (near-end). Output feature: 32x7x7 float32 = 6272 bytes."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)  # 16x14x14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)  # 32x7x7
        return x


class Exit1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 10)

    def forward(self, h):
        return self.fc(F.adaptive_avg_pool2d(h, 1).flatten(1))


class Stage2(nn.Module):
    """Runs on node 2 (far-end) and ends with the final exit."""

    def __init__(self) -> None:
        super().__init__()
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, h):
        x = F.relu(self.c3(h))                    # 64x7x7
        x = F.max_pool2d(F.relu(self.c4(x)), 2)   # 64x3x3
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


class EarlyExitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = Stage1()
        self.exit1 = Exit1()
        self.stage2 = Stage2()

    def forward(self, x):
        h = self.stage1(x)
        return self.exit1(h), self.stage2(h), h


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
