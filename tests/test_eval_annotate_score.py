"""score_annotations: joining completed human labels back against the
classifier's predictions to compute kappa -- the step that closes the loop
docs/DESIGN.md Sec 8 asks for ("a second independent pass... would let us
report inter-annotator kappa").
"""

from __future__ import annotations

from cra.eval.annotate import score_annotations
from cra.eval.records import ToolUseAssessment, TraceEvalRecord


def _record(trace_id, failure_mode) -> TraceEvalRecord:
    return TraceEvalRecord(
        trace_id=trace_id, experiment_id="e", model_id="m", dataset="pubmedqa",
        split="dev", qid=trace_id, is_correct=True, retrieval_gold_available=True,
        retrieval_hit=True, tool_use=ToolUseAssessment(), failure_mode=failure_mode,
        hallucinated=None,
    )


def test_perfect_agreement():
    records = [_record("a", "reasoning_failure"), _record("b", "correct_grounded")]
    annotations = {"a": "reasoning_failure", "b": "correct_grounded"}
    score = score_annotations(records, annotations)
    assert score.n_scored == 2
    assert score.agreement is not None
    assert score.agreement.kappa == 1.0


def test_ungraded_records_are_excluded_but_counted():
    records = [_record("a", "reasoning_failure"), _record("b", None)]
    annotations = {"a": "reasoning_failure", "b": "correct_grounded"}
    score = score_annotations(records, annotations)
    assert score.n_scored == 1
    assert score.n_ungraded_skipped == 1
    assert score.agreement is None  # fewer than 2 gradable pairs


def test_missing_human_labels_are_counted_not_silently_dropped():
    records = [_record("a", "reasoning_failure"), _record("b", "correct_grounded")]
    annotations = {"a": "reasoning_failure"}  # "b" never got labeled
    score = score_annotations(records, annotations)
    assert score.n_missing == 1
    assert score.n_scored == 1


def test_disagreement_produces_kappa_below_one():
    records = [
        _record("a", "reasoning_failure"), _record("b", "correct_grounded"),
        _record("c", "reasoning_failure"), _record("d", "correct_grounded"),
    ]
    annotations = {"a": "reasoning_failure", "b": "reasoning_failure", "c": "correct_grounded", "d": "correct_grounded"}
    score = score_annotations(records, annotations)
    assert score.agreement is not None
    assert score.agreement.kappa < 1.0
