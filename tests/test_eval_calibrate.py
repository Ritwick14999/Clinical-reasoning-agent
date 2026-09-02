"""Threshold calibration.

The property that matters most here is non-circularity: the sweep must fit on
one annotation pass and report on a disjoint one, or it is tuning the classifier
to its own validation set. The rest pins that the sweep applies exactly the
production decision rule rather than a reimplementation that could drift.
"""

from __future__ import annotations

import pytest

from cra.eval.calibrate import (
    DEFAULT_GRID,
    ScoredClaim,
    ScoredTrace,
    calibrate,
    kappa_at,
    render_report,
)


def _correct(trace_id: str, *entail_scores: float) -> ScoredTrace:
    return ScoredTrace(
        trace_id=trace_id,
        is_correct=True,
        fixed_label=None,
        claims=[ScoredClaim(entail=e, contradict=0.0) for e in entail_scores],
    )


def _fixed(trace_id: str, label: str) -> ScoredTrace:
    return ScoredTrace(trace_id=trace_id, is_correct=False, fixed_label=label)


class TestLabelAt:
    def test_all_claims_above_the_cut_is_grounded(self):
        assert _correct("t", 0.9, 0.8).label_at(0.5) == "correct_grounded"

    def test_one_claim_below_the_cut_makes_it_unsupported(self):
        assert _correct("t", 0.9, 0.2).label_at(0.5) == "unsupported_claim"

    def test_the_cut_is_strict(self):
        """A score exactly at the threshold is not entailed: check() uses `>`."""
        assert _correct("t", 0.5).label_at(0.5) == "unsupported_claim"
        assert _correct("t", 0.51).label_at(0.5) == "correct_grounded"

    def test_raising_the_cut_can_only_add_unsupported_labels(self):
        trace = _correct("t", 0.6)
        assert trace.label_at(0.5) == "correct_grounded"
        assert trace.label_at(0.7) == "unsupported_claim"

    def test_no_claims_is_grounded(self):
        """Matches how failure_modes treats an empty claim list."""
        assert _correct("t").label_at(0.5) == "correct_grounded"

    @pytest.mark.parametrize(
        "label", ["retrieval_failure", "tool_misuse", "reasoning_failure", "no_answer"]
    )
    def test_threshold_independent_labels_never_move(self, label):
        trace = _fixed("t", label)
        assert {trace.label_at(t) for t in DEFAULT_GRID} == {label}


class TestKappaAt:
    def test_unlabelled_traces_are_skipped(self):
        scored = [_correct("a", 0.9), _correct("b", 0.9)]
        _, n, _ = kappa_at(scored, {"a": "correct_grounded"}, 0.5)
        assert n == 1

    def test_perfect_single_class_agreement_is_not_reported_as_zero(self):
        """cohen_kappa_score is undefined with no expected disagreement; a naive
        call returns 0.0, which would read as chance agreement."""
        scored = [_correct("a", 0.9), _correct("b", 0.9)]
        human = {"a": "correct_grounded", "b": "correct_grounded"}
        k, n, agreement = kappa_at(scored, human, 0.5)
        assert (k, n, agreement) == (1.0, 2, 1.0)

    def test_no_overlap_yields_no_kappa(self):
        k, n, _ = kappa_at([_correct("a", 0.9)], {"zzz": "correct_grounded"}, 0.5)
        assert k is None
        assert n == 0


class TestCalibrate:
    def _sets(self):
        # Dev: a cut near 0.4 separates these correctly.
        dev = [_correct("d1", 0.45), _correct("d2", 0.35), _fixed("d3", "reasoning_failure")]
        dev_human = {
            "d1": "correct_grounded",
            "d2": "unsupported_claim",
            "d3": "reasoning_failure",
        }
        held = [_correct("h1", 0.44), _correct("h2", 0.30), _fixed("h3", "retrieval_failure")]
        held_human = {
            "h1": "correct_grounded",
            "h2": "unsupported_claim",
            "h3": "retrieval_failure",
        }
        return dev, dev_human, held, held_human

    def test_finds_a_cut_that_beats_the_default(self):
        dev, dev_human, held, held_human = self._sets()
        result = calibrate(dev, dev_human, held, held_human)
        assert result.dev_kappa >= result.default_dev_kappa
        assert 0.35 <= result.best_entail <= 0.45

    def test_reports_on_the_held_out_set(self):
        dev, dev_human, held, held_human = self._sets()
        result = calibrate(dev, dev_human, held, held_human)
        assert result.heldout_n == 3
        assert result.heldout_kappa is not None

    def test_fitting_never_touches_the_held_out_labels(self):
        """Changing only the held-out set must not move the chosen cut."""
        dev, dev_human, held, held_human = self._sets()
        chosen = calibrate(dev, dev_human, held, held_human).best_entail
        other = [_correct("h1", 0.99), _correct("h2", 0.99)]
        other_human = {"h1": "unsupported_claim", "h2": "unsupported_claim"}
        assert calibrate(dev, dev_human, other, other_human).best_entail == chosen

    def test_ties_break_toward_the_lower_cut(self):
        """A lower cut calls more claims entailed, so it cannot inflate a
        finding about how often claims are unsupported."""
        dev = [_fixed("d1", "reasoning_failure")]  # kappa identical at every cut
        human = {"d1": "reasoning_failure"}
        result = calibrate(dev, human, dev, human)
        assert result.best_entail == min(DEFAULT_GRID)

    def test_grid_excludes_cuts_that_can_never_fire(self):
        assert max(DEFAULT_GRID) < 1.0

    def test_report_names_both_sets_and_the_unswept_threshold(self):
        dev, dev_human, held, held_human = self._sets()
        text = render_report(calibrate(dev, dev_human, held, held_human))
        assert "held-out" in text
        assert "contradiction threshold is not swept" in text
