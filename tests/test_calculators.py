"""Worked examples for the clinical calculators.

Expected values are computed by hand from the published formulae, not captured
from a previous run of this code -- a regression test against your own bug is
worth nothing.
"""

from __future__ import annotations

import math

import pytest

from cra.tools.calculators.scores import (
    CalculatorInputError,
    anion_gap,
    cha2ds2_vasc,
    meld,
    wells_pe,
)


class TestCha2ds2Vasc:
    def test_maximum_score_is_nine(self):
        r = cha2ds2_vasc(
            age=80, sex="female", congestive_heart_failure=True, hypertension=True,
            diabetes=True, stroke_tia_thromboembolism=True, vascular_disease=True,
        )
        assert r.score == 9

    def test_young_healthy_man_scores_zero(self):
        assert cha2ds2_vasc(age=40, sex="male").score == 0

    def test_age_bands(self):
        assert cha2ds2_vasc(age=64, sex="male").score == 0
        assert cha2ds2_vasc(age=65, sex="male").score == 1
        assert cha2ds2_vasc(age=74, sex="male").score == 1
        assert cha2ds2_vasc(age=75, sex="male").score == 2

    def test_stroke_history_is_worth_two(self):
        assert cha2ds2_vasc(age=50, sex="male", stroke_tia_thromboembolism=True).score == 2

    def test_female_sex_alone_is_low_risk(self):
        """The sex point alone must not read as anticoagulation-eligible."""
        r = cha2ds2_vasc(age=50, sex="female")
        assert r.score == 1
        assert "Low risk" in r.interpretation

    def test_sex_synonyms_accepted(self):
        assert cha2ds2_vasc(age=50, sex="F").score == cha2ds2_vasc(age=50, sex="female").score

    @pytest.mark.parametrize("bad", ["", "other", "unknown", None])
    def test_rejects_unknown_sex(self, bad):
        with pytest.raises(CalculatorInputError):
            cha2ds2_vasc(age=50, sex=bad)

    @pytest.mark.parametrize("bad", [-1, 200, 55.5, "eighty"])
    def test_rejects_bad_age(self, bad):
        with pytest.raises(CalculatorInputError):
            cha2ds2_vasc(age=bad, sex="male")


class TestWellsPE:
    def test_all_criteria(self):
        r = wells_pe(
            clinical_signs_of_dvt=True, pe_most_likely_diagnosis=True, heart_rate_over_100=True,
            immobilization_or_surgery=True, previous_pe_or_dvt=True, hemoptysis=True,
            malignancy=True,
        )
        assert r.score == pytest.approx(12.5)

    def test_no_criteria(self):
        r = wells_pe()
        assert r.score == 0
        assert "Low probability" in r.interpretation
        assert "PE unlikely" in r.interpretation

    def test_three_tier_boundaries(self):
        assert "Low probability" in wells_pe(heart_rate_over_100=True).interpretation  # 1.5
        assert "Moderate" in wells_pe(previous_pe_or_dvt=True, hemoptysis=True).interpretation  # 2.5
        assert "High probability" in wells_pe(
            clinical_signs_of_dvt=True, pe_most_likely_diagnosis=True, hemoptysis=True
        ).interpretation  # 7.0

    def test_two_tier_boundary_at_four(self):
        # 3.0 + 1.5 = 4.5 -> likely; 3.0 + 1.0 = 4.0 -> unlikely (boundary is inclusive)
        assert "PE likely" in wells_pe(
            clinical_signs_of_dvt=True, heart_rate_over_100=True
        ).interpretation
        assert "PE unlikely" in wells_pe(
            clinical_signs_of_dvt=True, hemoptysis=True
        ).interpretation


class TestMeld:
    def test_worked_example(self):
        # 3.78*ln(3.1) + 11.2*ln(1.8) + 9.57*ln(2.0) + 6.43 = 23.92 -> 24
        r = meld(bilirubin_mg_dl=3.1, inr=1.8, creatinine_mg_dl=2.0)
        assert r.score == 24

    def test_values_below_one_are_floored(self):
        """All three labs at 0.5 floor to 1.0, so the score is the constant, floored to 6."""
        assert meld(bilirubin_mg_dl=0.5, inr=0.5, creatinine_mg_dl=0.5).score == 6

    def test_creatinine_capped_at_four(self):
        assert (
            meld(bilirubin_mg_dl=2.0, inr=1.5, creatinine_mg_dl=8.0).score
            == meld(bilirubin_mg_dl=2.0, inr=1.5, creatinine_mg_dl=4.0).score
        )

    def test_dialysis_forces_creatinine_to_four(self):
        with_dialysis = meld(bilirubin_mg_dl=2.0, inr=1.5, creatinine_mg_dl=1.0, dialysis=True)
        equivalent = meld(bilirubin_mg_dl=2.0, inr=1.5, creatinine_mg_dl=4.0)
        assert with_dialysis.score == equivalent.score
        assert any("dialysis" in w for w in with_dialysis.warnings)

    def test_score_is_bounded(self):
        assert meld(bilirubin_mg_dl=1.0, inr=1.0, creatinine_mg_dl=1.0).score == 6
        assert meld(bilirubin_mg_dl=50, inr=10, creatinine_mg_dl=4.0).score == 40

    def test_meld_na_worked_example(self):
        # MELD 24, Na 128: 24 + 1.32*9 - 0.033*24*9 = 28.75 -> 29
        r = meld(bilirubin_mg_dl=3.1, inr=1.8, creatinine_mg_dl=2.0, sodium_meq_l=128)
        assert r.score == 29
        assert r.name == "MELD-Na"

    def test_sodium_adjustment_skipped_at_or_below_eleven(self):
        r = meld(bilirubin_mg_dl=1.0, inr=1.0, creatinine_mg_dl=1.0, sodium_meq_l=125)
        assert r.score == 6
        assert any("not applied" in w for w in r.warnings)

    def test_sodium_is_bounded(self):
        low = meld(bilirubin_mg_dl=3.1, inr=1.8, creatinine_mg_dl=2.0, sodium_meq_l=120)
        at_bound = meld(bilirubin_mg_dl=3.1, inr=1.8, creatinine_mg_dl=2.0, sodium_meq_l=125)
        assert low.score == at_bound.score

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bilirubin_mg_dl": 0, "inr": 1.5, "creatinine_mg_dl": 1.0},
            {"bilirubin_mg_dl": -3, "inr": 1.5, "creatinine_mg_dl": 1.0},
            {"bilirubin_mg_dl": 2, "inr": 1.5, "creatinine_mg_dl": 1.0, "sodium_meq_l": 40},
            {"bilirubin_mg_dl": "high", "inr": 1.5, "creatinine_mg_dl": 1.0},
            {"bilirubin_mg_dl": math.nan, "inr": 1.5, "creatinine_mg_dl": 1.0},
        ],
    )
    def test_rejects_invalid_input(self, kwargs):
        with pytest.raises(CalculatorInputError):
            meld(**kwargs)


class TestAnionGap:
    def test_basic(self):
        r = anion_gap(sodium_meq_l=140, chloride_meq_l=104, bicarbonate_meq_l=24)
        assert r.score == 12
        assert "within the normal range" in r.interpretation

    def test_elevated(self):
        r = anion_gap(sodium_meq_l=140, chloride_meq_l=100, bicarbonate_meq_l=10)
        assert r.score == 30
        assert "elevated" in r.interpretation

    def test_potassium_inclusion_changes_gap_and_reference(self):
        r = anion_gap(
            sodium_meq_l=140, chloride_meq_l=104, bicarbonate_meq_l=24, potassium_meq_l=4.0
        )
        assert r.score == 16
        assert "10-16" in r.interpretation

    def test_albumin_correction(self):
        # AG 12, albumin 2.0 -> 12 + 2.5*2 = 17
        r = anion_gap(
            sodium_meq_l=140, chloride_meq_l=104, bicarbonate_meq_l=24, albumin_g_dl=2.0
        )
        assert r.score == 17
        assert "elevated" in r.interpretation

    def test_albumin_correction_unmasks_a_normal_looking_gap(self):
        r = anion_gap(
            sodium_meq_l=140, chloride_meq_l=104, bicarbonate_meq_l=24, albumin_g_dl=2.0
        )
        assert any("masking" in w for w in r.warnings)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sodium_meq_l": 400, "chloride_meq_l": 104, "bicarbonate_meq_l": 24},
            {"sodium_meq_l": 140, "chloride_meq_l": -5, "bicarbonate_meq_l": 24},
            {"sodium_meq_l": 140, "chloride_meq_l": 104, "bicarbonate_meq_l": 0},
            {"sodium_meq_l": 140, "chloride_meq_l": 104, "bicarbonate_meq_l": 24,
             "potassium_meq_l": 40},
            {"sodium_meq_l": 140, "chloride_meq_l": 104, "bicarbonate_meq_l": 24,
             "albumin_g_dl": 20},
            {"sodium_meq_l": None, "chloride_meq_l": 104, "bicarbonate_meq_l": 24},
        ],
    )
    def test_rejects_nonphysiological_input(self, kwargs):
        with pytest.raises(CalculatorInputError):
            anion_gap(**kwargs)


def test_calculators_are_pure():
    """Same inputs, same output, no accumulated state."""
    a = cha2ds2_vasc(age=70, sex="female", hypertension=True)
    b = cha2ds2_vasc(age=70, sex="female", hypertension=True)
    assert a.score == b.score
    assert a.render() == b.render()
