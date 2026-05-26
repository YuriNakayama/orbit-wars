"""Value head for the reinforce/case1 PPO baseline.

A 2-layer MLP on the bottleneck context vector. Kept separate from the policy
heads so BC weights (which only cover backbone + per_planet head) load
cleanly via `strict=False`; the value head is the only newly initialized
sub-module on warm-start.
"""

from __future__ import annotations

import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(ctx).squeeze(-1)
        return out


class ShipLogStd(nn.Module):
    """State-independent log-σ for the Gaussian ships head.

    BC supplied only the *mean* (`ship_pred`) as a SmoothL1 regression. PPO
    needs a stochastic policy, so we add a learnable scalar log-σ. Starting
    value defaults to ~0.5 (σ ≈ 1.6 in log1p-ships space), which gives the
    optimizer enough exploration without overwhelming the BC mean prior.
    """

    def __init__(self, init: float = 0.5) -> None:
        super().__init__()
        self.log_std = nn.Parameter(torch.full((1,), float(init)))

    def forward(self) -> torch.Tensor:
        return self.log_std


__all__ = ["ValueHead", "ShipLogStd"]
