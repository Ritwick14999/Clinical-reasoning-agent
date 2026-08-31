from __future__ import annotations

import pytest

from cra.eval.claims import (
    extract_claims,
    is_meta_evidence_statement,
    strip_citation_framing,
)


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


class TestCitationFraming:
    """Citing a source must not read as hallucination.

    Measured before this existed: claims mentioning an evidence ID were graded
    ``entailed`` 14% of the time against 36% for claims without one. The agent
    is instructed to cite, so that gap was measurement error.
    """

    @pytest.mark.parametrize(
        "written,proposition",
        [
            ("E1 states that aspirin is less effective.", "Aspirin is less effective."),
            ("According to E1, warfarin raises bleeding risk.", "Warfarin raises bleeding risk."),
            ("Per E1 and E2, the score is high.", "The score is high."),
            ("Margins are clearly separated, as shown in E1.", "Margins are clearly separated."),
            ("Aspirin is substantially less effective (E2).", "Aspirin is substantially less effective."),
            ("T1 shows that her score is 5.", "Her score is 5."),
        ],
    )
    def test_framing_is_stripped_to_the_proposition(self, written, proposition):
        assert strip_citation_framing(written) == proposition

    def test_uncited_claims_are_untouched(self):
        claim = "Randomised trials have established this as the standard of care."
        assert strip_citation_framing(claim) == claim

    def test_stripping_never_empties_a_claim(self):
        """A sentence that is nothing but a citation keeps its original text."""
        assert strip_citation_framing("E1.") == "E1."
        assert strip_citation_framing("As shown in E1.") == "As shown in E1."

    def test_negation_survives_stripping(self):
        """The wrapper goes; the proposition's meaning must not change."""
        out = strip_citation_framing("E1 states that aspirin is not effective.")
        assert "not effective" in out


class TestMetaEvidenceStatements:
    """Commentary about the evidence is not a claim about the world.

    An agent noting that a retrieved passage is off-topic is doing the right
    thing -- the opposite of what ``unsupported_claim`` exists to catch.
    """

    @pytest.mark.parametrize(
        "sentence",
        [
            "E2 is unrelated to resuscitation devices.",
            "E2 is irrelevant as it focuses on Dutch patients rather than U.S. cohorts.",
            "The other passages do not address this question.",
            "E3 is about laparoscopic adrenalectomy.",
            "The remaining evidence does not discuss this drug.",
        ],
    )
    def test_detected(self, sentence):
        assert is_meta_evidence_statement(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            "Aspirin is less effective than warfarin.",
            "Chemical shift MRI can demonstrate lesion margins.",
            "The patient is at high risk of stroke.",
        ],
    )
    def test_clinical_claims_are_not_meta(self, sentence):
        assert not is_meta_evidence_statement(sentence)

    def test_excluded_from_extraction(self):
        justification = (
            "Aspirin is less effective than warfarin. "
            "E2 is unrelated to anticoagulation. "
            "The patient should receive oral anticoagulation."
        )
        claims = extract_claims(justification)
        assert len(claims) == 2
        assert not any("unrelated" in c for c in claims)
