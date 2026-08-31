from __future__ import annotations

import pytest

from cra.eval.stats import bootstrap_ci, mcnemar_test


def test_bootstrap_ci_point_estimate():
    values = [1.0] * 8 + [0.0] * 2  # 80% accuracy
    ci = bootstrap_ci(values, n_boot=1000, seed=1)
    assert ci.point == 0.8
    assert ci.lo <= ci.point <= ci.hi


def test_bootstrap_ci_deterministic_for_same_seed():
    values = [1.0, 0.0, 1.0, 1.0, 0.0]
    a = bootstrap_ci(values, seed=42)
    b = bootstrap_ci(values, seed=42)
    assert a == b


def test_bootstrap_ci_empty_is_nan():
    ci = bootstrap_ci([])
    assert ci.point != ci.point  # NaN != NaN


def test_bootstrap_ci_all_correct_is_a_point_mass():
    ci = bootstrap_ci([1.0] * 20, n_boot=500, seed=1)
    assert ci.point == 1.0
    assert ci.lo == 1.0
    assert ci.hi == 1.0


def test_mcnemar_identical_sequences_no_discordant_pairs():
    a = [True, False, True, True]
    result = mcnemar_test(a, a)
    assert result.b == 0
    assert result.c == 0
    assert result.p_value == 1.0


def test_mcnemar_all_discordant_one_direction():
    a = [True, True, True, True]
    b = [False, False, False, False]
    result = mcnemar_test(a, b)
    assert result.b == 4
    assert result.c == 0
    assert result.p_value < 0.2  # exact binomial, small n -- not below 0.05, but clearly skewed


def test_mcnemar_requires_equal_length():
    with pytest.raises(ValueError):
        mcnemar_test([True], [True, False])
