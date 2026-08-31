"""Tool-use metrics against the ``expected_tools`` oracle.

The oracle itself is a rule-based, imprecise stand-in for "which tools this
case actually needed" (``cra.data.expected_tools``'s module docstring
explains why). Report these numbers as "precision/recall against the
oracle", never as "precision/recall against ground truth" -- the oracle's own
precision/recall needs a hand audit before either claim is trustworthy
(docs/HANDOFF.md decision 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from cra.data.expected_tools import ALWAYS_PERMITTED, resolve_expected_tools
from cra.types import Trace


@dataclass
class ToolMetrics:
    n_traces_with_expected_tools: int
    precision: float | None
    recall: float | None


def compute_tool_metrics(traces: list[Trace], expected_fn=resolve_expected_tools) -> ToolMetrics:
    graded = [t for t in traces if expected_fn(t.question)]
    if not graded:
        return ToolMetrics(n_traces_with_expected_tools=0, precision=None, recall=None)

    precisions: list[float] = []
    recalls: list[float] = []
    for t in graded:
        expected = set(expected_fn(t.question))
        # Always-permitted tools (retrieval) are excluded from the precision
        # denominator: calling one is instructed behaviour, not imprecision.
        used = set(t.tool_names) - (ALWAYS_PERMITTED - expected)
        if used:
            precisions.append(len(used & expected) / len(used))
        recalls.append(len(used & expected) / len(expected))

    return ToolMetrics(
        n_traces_with_expected_tools=len(graded),
        precision=(sum(precisions) / len(precisions)) if precisions else None,
        recall=sum(recalls) / len(recalls),
    )
