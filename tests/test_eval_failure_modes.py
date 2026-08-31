"""The failure-mode taxonomy is the core contribution -- every rule gets its
own test, plus the two deliberate departures from the design prose (no_answer
as its own bucket; correct-but-ungraded staying None rather than defaulting
to correct_grounded).
"""

from __future__ import annotations

from cra.eval.entailment.base import EntailmentLabel
from cra.eval.failure_modes import assess_tool_use, classify_trace, retrieval_hit
from cra.types import Evidence, FinalAnswer, Question, ToolCallRecord, Trace


class FakeEntailment:
    """Deterministic checker keyed on non-overlapping markers, so a claim
    like "unsupported" can't accidentally match "supported" as a substring."""

    name = "fake"

    def check(self, claim: str, evidence_texts: list[str]) -> EntailmentLabel:
        if "ENTAILED_MARKER" in claim:
            return "entailed"
        if "CONTRADICTED_MARKER" in claim:
            return "contradicted"
        return "not_addressed"


def _question(dataset="pubmedqa", gold_source_ids=None, expected_tools=None, gold_answer="yes"):
    return Question(
        qid="q1",
        dataset=dataset,
        split="dev",
        question="Does X help Y?",
        gold_answer=gold_answer,
        gold_source_ids=gold_source_ids or [],
        expected_tools=expected_tools or [],
    )


def _trace(question, final=None, evidence=None, tool_calls=None, terminated_by="model") -> Trace:
    return Trace(
        question=question,
        model_id="test-model",
        experiment_id="test-exp",
        final=final,
        evidence=evidence or [],
        tool_calls=tool_calls or [],
        terminated_by=terminated_by,
    )


def test_no_answer_is_its_own_bucket_not_reasoning_failure():
    trace = _trace(_question(), final=None, terminated_by="unparseable")
    record = classify_trace(trace)
    assert record.is_correct is None
    assert record.failure_mode == "no_answer"


def test_r1_retrieval_failure_when_gold_missed():
    q = _question(gold_source_ids=["999"])
    ev = [Evidence(evidence_id="E1", kind="passage", source_tool="search_literature", text="t", source_id="111")]
    trace = _trace(q, final=FinalAnswer(answer="no", justification="x"), evidence=ev)
    record = classify_trace(trace)
    assert record.is_correct is False  # gold_answer yes, answered no
    assert record.retrieval_gold_available is True
    assert record.retrieval_hit is False
    assert record.failure_mode == "retrieval_failure"


def test_r1_does_not_fire_when_gold_hit_even_if_wrong():
    q = _question(gold_source_ids=["111"])
    ev = [Evidence(evidence_id="E1", kind="passage", source_tool="search_literature", text="t", source_id="111")]
    trace = _trace(q, final=FinalAnswer(answer="no", justification="x"), evidence=ev)
    record = classify_trace(trace)
    assert record.retrieval_hit is True
    assert record.failure_mode != "retrieval_failure"


def test_r2_tool_misuse_required_tool_never_called():
    q = _question(dataset="medqa", expected_tools=["calc_cha2ds2_vasc"], gold_answer="B")
    trace = _trace(q, final=FinalAnswer(answer="A", justification="x"))
    record = classify_trace(trace)
    assert record.retrieval_gold_available is False  # medqa: R1 structurally can't fire
    assert record.failure_mode == "tool_misuse"
    assert "required_tool_never_called" in record.tool_use.reasons


def test_r3_reasoning_failure_when_tools_used_correctly_but_still_wrong():
    q = _question(dataset="medqa", expected_tools=["calc_cha2ds2_vasc"], gold_answer="B")
    tc = ToolCallRecord(index=0, step=0, name="calc_cha2ds2_vasc", args={}, ok=True, output="5")
    trace = _trace(q, final=FinalAnswer(answer="A", justification="x"), tool_calls=[tc])
    record = classify_trace(trace)
    assert record.tool_use.reasons == []
    assert record.failure_mode == "reasoning_failure"


def test_correct_answer_ungraded_without_entailment_checker():
    q = _question(gold_answer="yes")
    trace = _trace(q, final=FinalAnswer(answer="yes", justification="Some claim here."))
    record = classify_trace(trace)  # no entailment checker passed
    assert record.is_correct is True
    assert record.failure_mode is None  # not correct_grounded by default
    assert record.hallucinated is None
    assert record.claims and record.claims[0].label is None


def test_r5_correct_grounded_when_all_claims_entailed():
    q = _question(gold_answer="yes")
    trace = _trace(q, final=FinalAnswer(answer="yes", justification="This claim is ENTAILED_MARKER."))
    record = classify_trace(trace, entailment=FakeEntailment())
    assert record.failure_mode == "correct_grounded"
    assert record.hallucinated is False


def test_r4_unsupported_claim_when_a_claim_is_not_addressed():
    q = _question(gold_answer="yes")
    trace = _trace(
        q, final=FinalAnswer(answer="yes", justification="This claim is ENTAILED_MARKER. This other bit is unrelated.")
    )
    record = classify_trace(trace, entailment=FakeEntailment())
    assert record.failure_mode == "unsupported_claim"
    assert record.hallucinated is True


def test_hallucination_flag_is_independent_of_primary_label_for_wrong_answers():
    """A wrong answer can still hallucinate -- the flag must not vanish just
    because the primary label is already R1/R2/R3."""
    q = _question(gold_source_ids=["999"], gold_answer="yes")
    ev = [Evidence(evidence_id="E1", kind="passage", source_tool="search_literature", text="t", source_id="111")]
    trace = _trace(
        q, final=FinalAnswer(answer="no", justification="This claim is CONTRADICTED_MARKER."), evidence=ev
    )
    record = classify_trace(trace, entailment=FakeEntailment())
    assert record.failure_mode == "retrieval_failure"  # R1 still wins
    assert record.hallucinated is True  # but the flag is not lost


def test_assess_tool_use_unnecessary_call():
    q = _question(expected_tools=["calc_a"])
    tc = ToolCallRecord(index=0, step=0, name="calc_b", args={}, ok=True, output="x")
    trace = _trace(q, tool_calls=[tc])
    assessment = assess_tool_use(trace)
    assert "required_tool_never_called" in assessment.reasons
    assert "unnecessary_call" in assessment.reasons
    assert "wrong_tool" in assessment.reasons  # used tools share nothing with expected


def test_assess_tool_use_malformed_call():
    q = _question()
    tc = ToolCallRecord(
        index=0, step=0, name="calc_a", args={}, ok=False, output="err",
        error="malformed_call: bad json",
    )
    trace = _trace(q, tool_calls=[tc])
    assert "malformed_call" in assess_tool_use(trace).reasons


def test_retrieval_hit_none_when_no_gold_source():
    trace = _trace(_question(dataset="medqa", gold_source_ids=[]))
    assert retrieval_hit(trace) is None
