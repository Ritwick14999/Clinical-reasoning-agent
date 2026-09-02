"""Demo rendering and helpers.

Gradio is an optional extra, so everything here exercises the pure functions and
never imports gradio -- the demo must not be able to break the default test run.
"""

from __future__ import annotations

import pytest

from cra.demo.app import (
    DEMO_CORPUS,
    _load_checker,
    _parse_options,
    _render_grounding,
    _render_trace,
)
from cra.types import Evidence, FinalAnswer, Question, ToolCallRecord, Trace


def _trace(answer="B", justification="Aspirin is less effective than warfarin.",
           gold="B", evidence=(), tool_calls=()):
    return Trace(
        question=Question(
            qid="d", dataset="medqa", split="dev", question="Which strategy?",
            options={"A": "Aspirin", "B": "Anticoagulation"}, gold_answer=gold,
        ),
        final=FinalAnswer(answer=answer, justification=justification),
        evidence=list(evidence),
        tool_calls=list(tool_calls),
    )


class TestParseOptions:
    def test_parses_letter_paren_form(self):
        assert _parse_options("A) foo\nB) bar") == {"A": "foo", "B": "bar"}

    @pytest.mark.parametrize("sep", [")", ".", ":"])
    def test_accepts_common_separators(self, sep):
        assert _parse_options(f"A{sep} foo") == {"A": "foo"}

    def test_lowercase_letters_are_upcased(self):
        assert _parse_options("a) foo") == {"A": "foo"}

    def test_blank_input_is_none_not_empty_dict(self):
        """None means 'free-text answer'; an empty dict would render an empty list."""
        assert _parse_options("") is None
        assert _parse_options("   \n  ") is None

    def test_prose_lines_are_ignored(self):
        assert _parse_options("This is not an option\nA) but this is") == {"A": "but this is"}


class TestFallbackChecker:
    def test_reports_itself_as_a_fallback(self):
        backend = _load_checker(force_keyword=True)
        assert "fallback" in backend.name
        assert "not the NLI checker" in backend.note

    def test_overlapping_claim_is_entailed(self):
        backend = _load_checker(force_keyword=True)
        assert backend.check(
            "Aspirin is substantially less effective than warfarin for stroke prevention",
            [DEMO_CORPUS[1].text],
        ) == "entailed"

    def test_unrelated_claim_is_not_addressed(self):
        backend = _load_checker(force_keyword=True)
        assert backend.check(
            "Metformin should be held before contrast imaging", [DEMO_CORPUS[1].text]
        ) == "not_addressed"

    def test_no_evidence_means_nothing_is_entailed(self):
        backend = _load_checker(force_keyword=True)
        assert backend.check("Anything at all here", []) == "not_addressed"


class TestRendering:
    def test_trace_without_tools_says_so(self):
        assert "No tools were called" in _render_trace(_trace())

    def test_failed_tool_call_shows_its_error(self):
        call = ToolCallRecord(
            index=0, step=0, name="calc_meld", args={"inr": -1}, ok=False,
            output="ERROR", error="schema_violation: at inr",
        )
        rendered = _render_trace(_trace(tool_calls=[call]))
        assert "schema_violation" in rendered
        assert "failed" in rendered

    def test_refused_call_is_distinguished_from_a_failed_one(self):
        call = ToolCallRecord(
            index=0, step=0, name="search_literature", args={}, ok=False,
            output="", error="budget_exhausted", executed=False,
        )
        assert "refused" in _render_trace(_trace(tool_calls=[call]))

    def test_evidence_is_shown_with_its_id_and_source(self):
        ev = Evidence(
            evidence_id="E1", kind="passage", source_tool="search_literature",
            text=DEMO_CORPUS[0].text, title="Risk stratification", source_id="21873455",
        )
        rendered = _render_trace(_trace(evidence=[ev]))
        assert "[E1]" in rendered
        assert "21873455" in rendered

    def test_html_in_a_question_is_escaped(self):
        call = ToolCallRecord(
            index=0, step=0, name="<script>alert(1)</script>", args={}, ok=True, output="",
        )
        rendered = _render_trace(_trace(tool_calls=[call]))
        assert "<script>" not in rendered
        assert "&lt;script&gt;" in rendered


class TestGroundingPanel:
    def _backend(self):
        return _load_checker(force_keyword=True)

    def test_unparseable_answer_is_reported(self):
        trace = _trace()
        trace.final = None
        assert "no parseable answer" in _render_grounding(trace, self._backend())

    def test_empty_evidence_is_called_out_as_a_construction(self):
        """Otherwise a closed-book episode reads as a 100% hallucination rate."""
        rendered = _render_grounding(_trace(), self._backend())
        assert "unsupported by construction" in rendered

    def test_right_answer_with_unsupported_claims_is_highlighted(self):
        """The case the whole project is about must be visible, not buried."""
        trace = _trace(
            answer="B", gold="B",
            justification="Randomised trials have established this as standard of care.",
        )
        rendered = _render_grounding(trace, self._backend())
        assert "Right answer, ungrounded justification" in rendered

    def test_grounded_answer_is_not_flagged(self):
        ev = Evidence(
            evidence_id="E1", kind="passage", source_tool="search_literature",
            text=DEMO_CORPUS[1].text,
        )
        trace = _trace(
            justification=(
                "Aspirin monotherapy is substantially less effective than warfarin for "
                "stroke prevention in atrial fibrillation."
            ),
            evidence=[ev],
        )
        rendered = _render_grounding(trace, self._backend())
        assert "Right answer, ungrounded justification" not in rendered
        assert "entailed" in rendered

    def test_citation_framing_shows_what_was_actually_tested(self):
        ev = Evidence(
            evidence_id="E1", kind="passage", source_tool="search_literature",
            text=DEMO_CORPUS[1].text,
        )
        trace = _trace(
            justification="E1 states that aspirin monotherapy is substantially less effective.",
            evidence=[ev],
        )
        assert "tested as:" in _render_grounding(trace, self._backend())
