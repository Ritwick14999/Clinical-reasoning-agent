from __future__ import annotations

from cra.eval.retrieval_metrics import compute_retrieval_metrics
from cra.eval.tool_metrics import compute_tool_metrics
from cra.types import Evidence, Question, ToolCallRecord, Trace


def _trace(gold_source_ids=None, evidence=None, expected_tools=None, tool_calls=None, dataset="pubmedqa"):
    q = Question(
        qid="q", dataset=dataset, split="dev", question="Q?", gold_answer="yes",
        gold_source_ids=gold_source_ids or [], expected_tools=expected_tools or [],
    )
    return Trace(question=q, evidence=evidence or [], tool_calls=tool_calls or [])


def test_retrieval_metrics_no_gold_traces():
    m = compute_retrieval_metrics([_trace(dataset="medqa")])
    assert m.n_with_gold == 0
    assert m.hit_at_k is None
    assert m.mrr is None


def test_retrieval_metrics_hit_at_first_rank():
    ev = [Evidence(evidence_id="E1", kind="passage", source_tool="s", text="t", source_id="111")]
    m = compute_retrieval_metrics([_trace(gold_source_ids=["111"], evidence=ev)])
    assert m.n_with_gold == 1
    assert m.hit_at_k == 1.0
    assert m.mrr == 1.0


def test_retrieval_metrics_miss():
    ev = [Evidence(evidence_id="E1", kind="passage", source_tool="s", text="t", source_id="222")]
    m = compute_retrieval_metrics([_trace(gold_source_ids=["111"], evidence=ev)])
    assert m.hit_at_k == 0.0
    assert m.mrr == 0.0


def test_retrieval_metrics_rank_two_halves_mrr():
    ev = [
        Evidence(evidence_id="E1", kind="passage", source_tool="s", text="t", source_id="222"),
        Evidence(evidence_id="E2", kind="passage", source_tool="s", text="t", source_id="111"),
    ]
    m = compute_retrieval_metrics([_trace(gold_source_ids=["111"], evidence=ev)])
    assert m.hit_at_k == 1.0
    assert m.mrr == 0.5


def test_tool_metrics_no_expected_tools():
    m = compute_tool_metrics([_trace()])
    assert m.n_traces_with_expected_tools == 0
    assert m.precision is None


def test_tool_metrics_perfect_precision_recall():
    tc = ToolCallRecord(index=0, step=0, name="calc_a", args={}, ok=True, output="x")
    m = compute_tool_metrics([_trace(expected_tools=["calc_a"], tool_calls=[tc])])
    assert m.precision == 1.0
    assert m.recall == 1.0


def test_tool_metrics_never_called_zero_recall_none_precision():
    m = compute_tool_metrics([_trace(expected_tools=["calc_a"], tool_calls=[])])
    assert m.recall == 0.0
    assert m.precision is None  # no calls made at all -- precision is undefined, not 0
