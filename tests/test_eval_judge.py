"""LLMJudge: label parsing and the max_tokens regression.

Regression coverage for a real bug: qwen3:8b is a "thinking" model that
spends tokens on internal reasoning before any visible output. With
max_tokens=8, every judge call returned empty text and silently fell back to
"not_addressed", producing a spuriously ~100% hallucination rate on a real
40-trace sample that was measuring this bug, not the agent -- caught only by
manually inspecting raw judge output against an implausible headline number.
"""

from __future__ import annotations

from cra.eval.entailment.judge import LLMJudge
from cra.llm.base import LLMResponse
from cra.llm.mock import ScriptedClient


def test_parse_label_exact_match():
    assert LLMJudge._parse_label("entailed") == "entailed"
    assert LLMJudge._parse_label("Contradicted.") == "contradicted"
    assert LLMJudge._parse_label('"not_addressed"') == "not_addressed"


def test_parse_label_searches_within_wrapper_text():
    """A model that ignores the 'one word only' instruction but still states
    the label inside a sentence should still resolve correctly."""
    assert LLMJudge._parse_label("The label is entailed based on E1.") == "entailed"


def test_parse_label_empty_text_falls_back_conservatively():
    assert LLMJudge._parse_label("") == "not_addressed"
    assert LLMJudge._parse_label("   ") == "not_addressed"


def test_parse_label_gibberish_falls_back_conservatively():
    assert LLMJudge._parse_label("I cannot determine this.") == "not_addressed"


def test_check_short_circuits_with_no_evidence():
    llm = ScriptedClient([LLMResponse(text="entailed")])
    judge = LLMJudge(llm)
    assert judge.check("some claim", []) == "not_addressed"
    assert llm.calls == []  # no model call made -- nothing to judge against


def test_check_passes_generous_max_tokens_not_the_old_tight_budget():
    """The actual regression: max_tokens must be large enough for a
    thinking model to get past its reasoning tokens before answering."""
    llm = ScriptedClient([LLMResponse(text="entailed")])
    judge = LLMJudge(llm)
    judge.check("claim", ["passage"])
    # Measured against the real model: ~600 output tokens were consumed on
    # internal reasoning for a single claim before qwen3 ever emitted the
    # label. 300 was tried and was still not enough; require real headroom.
    assert judge.max_tokens >= 1000


def test_check_uses_parsed_label_from_real_response_shape():
    llm = ScriptedClient([LLMResponse(text="")])  # simulates the thinking-cutoff bug
    judge = LLMJudge(llm)
    assert judge.check("claim", ["passage"]) == "not_addressed"
