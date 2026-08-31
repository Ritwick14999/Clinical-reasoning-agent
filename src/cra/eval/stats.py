"""Statistics: bootstrap confidence intervals and McNemar's test.

Every headline number in docs/DESIGN.md is required to carry a CI
("accuracy deltas without CIs are not claims"); paired model/ablation
comparisons use McNemar's test rather than a plain difference in
proportions, since the same question set is answered under both conditions.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    n_boot: int


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of ``values`` (e.g. a 0/1
    correctness vector). Deterministic for a given seed, so a report is
    reproducible byte-for-byte, not just approximately similar on rerun."""
    n = len(values)
    if n == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"), n_boot=n_boot)

    point = sum(values) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()

    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return BootstrapCI(point=point, lo=means[lo_idx], hi=means[hi_idx], n_boot=n_boot)


@dataclass
class McNemarResult:
    b: int  # correct under A, incorrect under B
    c: int  # incorrect under A, correct under B
    p_value: float


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """Exact (binomial) McNemar's test over paired per-question outcomes for
    two conditions on the *same* question set (e.g. two models, or an
    ablation on/off)."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired sequences must be the same length")

    b = sum(1 for a, bb in zip(correct_a, correct_b, strict=True) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b, strict=True) if not a and bb)
    n = b + c
    if n == 0:
        return McNemarResult(b=b, c=c, p_value=1.0)

    k = min(b, c)
    p_value = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) * (0.5**n))
    return McNemarResult(b=b, c=c, p_value=p_value)
