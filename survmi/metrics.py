from __future__ import annotations

import torch


def concordance_index(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    """Harrell's C-index for survival risk scores.

    Higher risk is interpreted as shorter survival.
    """

    risk = risk.detach().float().reshape(-1).cpu()
    time = time.detach().float().reshape(-1).cpu()
    event = event.detach().float().reshape(-1).cpu()

    concordant = 0.0
    comparable = 0.0
    n = risk.numel()
    for i in range(n):
        if event[i] <= 0:
            continue
        for j in range(n):
            if time[i] >= time[j]:
                continue
            comparable += 1.0
            if risk[i] > risk[j]:
                concordant += 1.0
            elif risk[i] == risk[j]:
                concordant += 0.5

    if comparable == 0:
        return float("nan")
    return concordant / comparable


class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

