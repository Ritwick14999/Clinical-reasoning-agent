"""Orchestrates the eval stage: traces -> records -> report + figure.

``build_records`` (and therefore ``run_eval``) is the pure, no-network path:
entailment stays ungraded unless an :class:`~cra.eval.entailment.base.EntailmentChecker`
is passed in explicitly, which the default ``cra eval`` CLI command never
does -- it wires in :class:`~cra.eval.entailment.nli.NLIEntailmentChecker`
only when ``--nli`` is passed, and otherwise leaves correct-answer traces
honestly ungraded rather than silently calling a model.

``cross_judge_traces`` is the separate, explicitly live-model path -- see
``cra.eval.entailment.judge``'s module docstring for why it's kept apart.
"""

from __future__ import annotations

import random
from pathlib import Path

from cra.eval.entailment.base import EntailmentChecker
from cra.eval.failure_modes import classify_trace
from cra.eval.records import TraceEvalRecord
from cra.eval.report import (
    DatasetSummary,
    render_failure_mode_table,
    render_markdown_table,
    summarize,
)
from cra.llm.base import LLMClient
from cra.trace_io import read_trace_dir
from cra.types import Trace
from cra.viz.plots import plot_failure_modes


def build_records(
    traces: list[Trace], entailment: EntailmentChecker | None = None
) -> list[TraceEvalRecord]:
    return [classify_trace(t, entailment=entailment) for t in traces]


def run_eval(
    trace_dirs: list[str],
    out_dir: str = "results",
    entailment: EntailmentChecker | None = None,
) -> tuple[list[Trace], list[TraceEvalRecord], list[DatasetSummary]]:
    traces: list[Trace] = []
    for d in trace_dirs:
        traces.extend(read_trace_dir(d))
    if not traces:
        raise RuntimeError(f"no traces found in {trace_dirs}")

    records = build_records(traces, entailment=entailment)
    summaries = summarize(traces, records)

    out_root = Path(out_dir)
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "headline.md").write_text(render_markdown_table(summaries), encoding="utf-8")
    (tables_dir / "failure_modes.md").write_text(render_failure_mode_table(summaries), encoding="utf-8")

    plot_failure_modes(records, out_root / "figures" / "failure_modes.png")

    return traces, records, summaries


def cross_judge_traces(
    traces_by_model: dict[str, list[Trace]],
    llms_by_model: dict[str, LLMClient],
    sample_n: int = 40,
    seed: int = 12345,
) -> list[TraceEvalRecord]:
    """Judges each model's traces with a *different* model's :class:`LLMJudge`
    -- never its own. With exactly two models (this project's case) each
    judges the other's sample; with more, each is judged round-robin by the
    next model in sorted order. See ``cra.eval.entailment.judge``'s module
    docstring for why this design avoids the same-family judge-circularity
    docs/DESIGN.md Sec 7 flags.
    """
    from cra.eval.entailment.judge import LLMJudge

    model_ids = sorted(traces_by_model)
    if len(model_ids) < 2:
        raise ValueError("cross-model judging needs at least two models' traces")

    records: list[TraceEvalRecord] = []
    for i, target_model in enumerate(model_ids):
        judge_model = model_ids[(i + 1) % len(model_ids)]
        checker = LLMJudge(llms_by_model[judge_model])

        candidates = [t for t in traces_by_model[target_model] if t.final is not None]
        rng = random.Random(seed)
        rng.shuffle(candidates)

        for t in candidates[:sample_n]:
            records.append(classify_trace(t, entailment=checker))
    return records
