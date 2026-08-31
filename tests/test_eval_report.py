"""Integration coverage for the eval orchestration: traces on disk -> tables
+ figure, and the blind-annotation round trip. No network, no model calls --
the whole point of the default `cra eval` path.
"""

from __future__ import annotations

from cra.eval.annotate import read_annotations, sample_for_annotation, write_annotation_template
from cra.eval.run import build_records, run_eval
from cra.trace_io import write_traces
from cra.types import FinalAnswer, Question, Trace


def _trace(qid, dataset, gold_answer, answer, experiment_id="exp1", model_id="m1"):
    q = Question(qid=qid, dataset=dataset, split="dev", question="Q?", gold_answer=gold_answer)
    return Trace(
        question=q, experiment_id=experiment_id, model_id=model_id,
        final=FinalAnswer(answer=answer, justification="Because."),
        terminated_by="model",
    )


def _make_traces():
    return [
        _trace("1", "pubmedqa", "yes", "yes"),
        _trace("2", "pubmedqa", "yes", "no"),
        _trace("3", "medqa", "A", "A"),
        _trace("4", "medqa", "A", "B"),
    ]


def test_run_eval_writes_tables_and_figure(tmp_path):
    trace_dir = tmp_path / "traces" / "exp1"
    write_traces(_make_traces(), trace_dir / "traces.jsonl.gz")

    out_dir = tmp_path / "results"
    traces, records, summaries = run_eval([str(trace_dir)], out_dir=str(out_dir))

    assert len(traces) == 4
    assert len(records) == 4
    assert {s.dataset for s in summaries} == {"pubmedqa", "medqa"}

    assert (out_dir / "tables" / "headline.md").exists()
    assert (out_dir / "tables" / "failure_modes.md").exists()
    assert (out_dir / "figures" / "failure_modes.png").exists()

    table = (out_dir / "tables" / "headline.md").read_text(encoding="utf-8")
    assert "pubmedqa" in table and "medqa" in table


def test_run_eval_raises_on_empty_trace_dir(tmp_path):
    import pytest

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match="no traces"):
        run_eval([str(empty_dir)])


def test_annotation_round_trip(tmp_path):
    traces = _make_traces()
    trace_dir = tmp_path / "traces" / "exp1"
    write_traces(traces, trace_dir / "traces.jsonl.gz")

    records = build_records(traces)
    sample = sample_for_annotation(records, n=2, seed=1)
    assert len(sample) == 2

    csv_path, transcript_path = write_annotation_template(sample, [str(trace_dir)], tmp_path / "sample.csv")
    assert csv_path.exists()
    assert transcript_path.exists()
    assert "trace_id=" in transcript_path.read_text(encoding="utf-8")

    # Simulate a human filling in the human_label column for the first row.
    import csv as csv_module

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = list(csv_module.DictReader(fh))
    reader[0]["human_label"] = "reasoning_failure"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=["trace_id", "dataset", "qid", "model_id", "human_label"])
        writer.writeheader()
        writer.writerows(reader)

    annotations = read_annotations(csv_path)
    assert len(annotations) == 1
    assert list(annotations.values())[0] == "reasoning_failure"
