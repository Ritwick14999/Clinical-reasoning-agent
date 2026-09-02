"""Paired ablation comparisons.

The pinned property is the one that makes a tools-vs-closed-book comparison mean
anything: a closed-book arm retrieves nothing, so scoring its claims against its
own (empty) evidence would return 100% unsupported by construction and measure
the ablation rather than the agent.
"""

from __future__ import annotations

import pytest

from cra.eval.ablation import compare, evidence_for_grounding, mcnemar, summarise_arm
from cra.types import Evidence, FinalAnswer, Question, Trace


class FakeEntailment:
    """Entails a claim when any premise shares a distinctive word with it."""

    name = "fake"

    def check(self, claim, evidence_texts):
        words = {w for w in claim.lower().split() if len(w) > 4}
        for text in evidence_texts:
            if words & {w for w in text.lower().split() if len(w) > 4}:
                return "entailed"
        return "not_addressed"


def _question(qid: str, gold: str = "yes") -> Question:
    return Question(
        qid=qid, dataset="pubmedqa", split="dev", question="Does treatment help?",
        gold_answer=gold,
    )


def _trace(qid, answer, justification, passages=(), experiment_id="exp", budget=5):
    return Trace(
        question=_question(qid),
        experiment_id=experiment_id,
        tool_budget=budget,
        final=FinalAnswer(answer=answer, justification=justification),
        evidence=[
            Evidence(evidence_id=f"E{i+1}", kind="passage", source_tool="search_literature",
                     text=text, source_id=str(i))
            for i, text in enumerate(passages)
        ],
    )


class TestEvidenceBorrowing:
    def test_a_trace_uses_its_own_evidence(self):
        donor = _trace("q1", "yes", "j", passages=["donor passage"])
        own = _trace("q1", "yes", "j", passages=["own passage"])
        premises, borrowed = evidence_for_grounding(own, donor)
        assert premises == ["own passage"]
        assert borrowed is False

    def test_an_empty_trace_borrows_from_its_pair(self):
        donor = _trace("q1", "yes", "j", passages=["donor passage"])
        closed = _trace("q1", "yes", "j", passages=[])
        premises, borrowed = evidence_for_grounding(closed, donor)
        assert premises == ["donor passage"]
        assert borrowed is True

    def test_borrowing_takes_passages_only_not_tool_outputs(self):
        """A calculator result from another episode was computed from different
        arguments and says nothing about this one."""
        donor = _trace("q1", "yes", "j", passages=["donor passage"])
        donor.evidence.append(
            Evidence(evidence_id="T1", kind="tool_output", source_tool="calc_meld", text="MELD 24")
        )
        premises, _ = evidence_for_grounding(_trace("q1", "yes", "j"), donor)
        assert premises == ["donor passage"]

    def test_no_donor_leaves_the_premise_set_empty(self):
        premises, borrowed = evidence_for_grounding(_trace("q1", "yes", "j"), None)
        assert premises == []
        assert borrowed is False


class TestSummarise:
    def test_closed_book_without_borrowing_is_trivially_unsupported(self):
        """The failure mode this module exists to avoid."""
        traces = [_trace("q1", "yes", "Treatment reduces mortality substantially.")]
        summary = summarise_arm(traces, FakeEntailment())
        assert summary.grounding_rate_claims == 0.0
        assert summary.traces_with_unsupported_claim == 1

    def test_borrowed_evidence_makes_grounding_measurable(self):
        donor = _trace("q1", "yes", "j", passages=["Treatment reduces mortality substantially."])
        traces = [_trace("q1", "yes", "Treatment reduces mortality substantially.")]
        summary = summarise_arm(
            traces, FakeEntailment(), donors={("dev", "pubmedqa", "q1"): donor}
        )
        assert summary.grounding_rate_claims == 1.0
        assert summary.traces_with_unsupported_claim == 0

    def test_unanswered_traces_are_excluded_from_accuracy(self):
        answered = _trace("q1", "yes", "j")
        unanswered = Trace(question=_question("q2"), final=None)
        summary = summarise_arm([answered, unanswered], FakeEntailment())
        assert summary.n == 2
        assert summary.n_answered == 1
        assert summary.accuracy == 1.0


class TestMcNemar:
    def test_no_discordant_pairs_gives_no_p_value(self):
        assert mcnemar(0, 0) is None

    def test_symmetric_discordance_is_not_significant(self):
        assert mcnemar(10, 10) > 0.9

    def test_lopsided_discordance_is_significant(self):
        assert mcnemar(20, 2) < 0.001

    def test_small_counts_use_the_exact_test(self):
        """Discordant counts here are often tiny, where chi-square is unsound."""
        assert 0.0 < mcnemar(1, 0) <= 1.0


class TestCompare:
    def _arms(self):
        baseline = [
            _trace("q1", "yes", "Treatment reduces mortality substantially.",
                   passages=["Treatment reduces mortality substantially."], experiment_id="base"),
            _trace("q2", "no", "Unrelated assertion about surgery.",
                   passages=["Treatment reduces mortality."], experiment_id="base"),
        ]
        variant = [
            _trace("q1", "yes", "Treatment reduces mortality substantially.",
                   experiment_id="var", budget=0),
            _trace("q2", "yes", "Another unrelated assertion.", experiment_id="var", budget=0),
        ]
        return baseline, variant

    def test_pairs_on_shared_questions(self):
        baseline, variant = self._arms()
        result = compare(baseline, variant, FakeEntailment())
        assert result.n_paired == 2

    def test_mcnemar_counts_the_discordant_direction(self):
        baseline, variant = self._arms()
        result = compare(baseline, variant, FakeEntailment())
        # q2: baseline answered "no" (wrong), variant "yes" (right) -> c
        assert (result.b, result.c) == (0, 1)

    def test_borrowing_is_reported_in_the_notes(self):
        baseline, variant = self._arms()
        result = compare(baseline, variant, FakeEntailment())
        assert any("retrieved nothing" in n for n in result.notes)

    def test_unshared_questions_are_reported(self):
        baseline, variant = self._arms()
        variant.append(_trace("q3", "yes", "j", experiment_id="var"))
        result = compare(baseline, variant, FakeEntailment())
        assert result.n_paired == 2
        assert any("shared question" in n for n in result.notes)

    def test_disabling_borrowing_restores_the_naive_comparison(self):
        baseline, variant = self._arms()
        result = compare(baseline, variant, FakeEntailment(), borrow_evidence=False)
        assert result.variant.grounding_rate_claims == 0.0


@pytest.mark.parametrize("borrow", [True, False])
def test_baseline_is_never_borrowed_against(borrow):
    """The baseline has its own evidence; only the evidence-less arm borrows."""
    baseline, variant = TestCompare()._arms()
    result = compare(baseline, variant, FakeEntailment(), borrow_evidence=borrow)
    assert result.baseline.claims_total > 0
