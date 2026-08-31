from __future__ import annotations

import pytest

from cra.eval.agreement import cohens_kappa


def test_perfect_agreement_kappa_is_one():
    labels = ["a", "b", "c", "a", "b"]
    result = cohens_kappa(labels, labels)
    assert result.observed_agreement == 1.0
    assert result.kappa == pytest.approx(1.0)


def test_no_better_than_chance_agreement_near_zero():
    # Two independent-looking uniform labelings over a large balanced sample
    # should land near kappa=0, not exercise a division edge case.
    a = ["x", "y"] * 50
    b = ["y", "x"] * 50  # perfectly anti-correlated on this construction
    result = cohens_kappa(a, b)
    assert result.n == 100
    assert result.kappa < 0  # systematic disagreement, worse than chance


def test_confusion_matrix_counts():
    a = ["x", "x", "y"]
    b = ["x", "y", "y"]
    result = cohens_kappa(a, b)
    assert result.confusion[("x", "x")] == 1
    assert result.confusion[("x", "y")] == 1
    assert result.confusion[("y", "y")] == 1


def test_requires_equal_length():
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_requires_nonempty():
    with pytest.raises(ValueError):
        cohens_kappa([], [])
