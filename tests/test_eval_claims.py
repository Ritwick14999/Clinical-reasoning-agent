from __future__ import annotations

from cra.eval.claims import extract_claims


def test_splits_into_sentences():
    text = "The score is 5. Anticoagulation is recommended."
    assert extract_claims(text) == ["The score is 5.", "Anticoagulation is recommended."]


def test_filters_questions():
    text = "Is this the right approach? Anticoagulation is recommended."
    assert extract_claims(text) == ["Anticoagulation is recommended."]


def test_filters_hedge_only_sentences():
    text = "It is unclear. Anticoagulation is recommended."
    assert extract_claims(text) == ["Anticoagulation is recommended."]


def test_filters_bare_answer_restatement():
    text = "E1 supports oral anticoagulation. On that basis the best option is B."
    assert extract_claims(text) == ["E1 supports oral anticoagulation."]


def test_empty_justification():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_short_fragment_dropped():
    assert extract_claims("Yes.") == []
