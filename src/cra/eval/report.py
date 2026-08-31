"""Aggregate a headline report from committed traces: accuracy with bootstrap
CIs, retrieval/tool metrics, and the failure-mode breakdown -- Phase 3's
actual output. Pure function of ``(traces, records)``: no model calls, no
network. Any entailment grading was already baked into each
``TraceEvalRecord`` upstream, by whichever checker ``build_records`` was
given (possibly none).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from cra.eval.records import TraceEvalRecord
from cra.eval.retrieval_metrics import RetrievalMetrics, compute_retrieval_metrics
from cra.eval.stats import BootstrapCI, bootstrap_ci
from cra.eval.tool_metrics import ToolMetrics, compute_tool_metrics
from cra.types import Trace


@dataclass
class DatasetSummary:
    experiment_id: str
    model_id: str
    dataset: str
    n: int
    n_answered: int
    accuracy: BootstrapCI
    retrieval: RetrievalMetrics
    tools: ToolMetrics
    failure_mode_counts: dict[str | None, int]


def summarize(traces: list[Trace], records: list[TraceEvalRecord]) -> list[DatasetSummary]:
    """One row per (experiment_id, model_id, dataset) present in ``traces``."""
    keys = sorted({(t.experiment_id, t.model_id, t.question.dataset) for t in traces})
    summaries = []
    for experiment_id, model_id, dataset in keys:
        t_group = [
            t for t in traces
            if t.experiment_id == experiment_id and t.model_id == model_id
            and t.question.dataset == dataset
        ]
        r_group = [
            r for r in records
            if r.experiment_id == experiment_id and r.model_id == model_id and r.dataset == dataset
        ]
        correctness = [1.0 if t.is_correct else 0.0 for t in t_group if t.is_correct is not None]

        summaries.append(
            DatasetSummary(
                experiment_id=experiment_id,
                model_id=model_id,
                dataset=dataset,
                n=len(t_group),
                n_answered=len(correctness),
                accuracy=bootstrap_ci(correctness),
                retrieval=compute_retrieval_metrics(t_group),
                tools=compute_tool_metrics(t_group),
                failure_mode_counts=dict(Counter(r.failure_mode for r in r_group)),
            )
        )
    return summaries


def render_markdown_table(summaries: list[DatasetSummary]) -> str:
    lines = [
        "| Experiment | Model | Dataset | N | Answered | Accuracy (95% CI) | Hit@k | Tool P/R |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        acc = f"{s.accuracy.point:.1%} [{s.accuracy.lo:.1%}, {s.accuracy.hi:.1%}]"
        hit = f"{s.retrieval.hit_at_k:.1%}" if s.retrieval.hit_at_k is not None else "n/a"
        # Always show the denominator: a rate over one or two gradable traces
        # reads as a catastrophic result when it is really an absent oracle.
        if s.tools.precision is not None:
            pr = (
                f"{s.tools.precision:.1%} / {s.tools.recall:.1%} "
                f"(n={s.tools.n_traces_with_expected_tools})"
            )
        elif s.tools.n_traces_with_expected_tools:
            pr = f"n/a (n={s.tools.n_traces_with_expected_tools}, no tools used)"
        else:
            pr = "n/a (no expectations)"
        lines.append(
            f"| {s.experiment_id} | {s.model_id} | {s.dataset} | {s.n} | {s.n_answered} "
            f"| {acc} | {hit} | {pr} |"
        )
    return "\n".join(lines)


_LABEL_ORDER = [
    "no_answer", "retrieval_failure", "tool_misuse",
    "reasoning_failure", "unsupported_claim", "correct_grounded", None,
]


def render_failure_mode_table(summaries: list[DatasetSummary]) -> str:
    header = ["Experiment", "Model", "Dataset", "N"] + [
        (label or "correct_ungraded") for label in _LABEL_ORDER
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for s in summaries:
        row = [s.experiment_id, s.model_id, s.dataset, str(s.n)]
        for label in _LABEL_ORDER:
            count = s.failure_mode_counts.get(label, 0)
            row.append(f"{count} ({count / s.n:.0%})" if s.n else "0")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
