"""Clinical risk scores as pure, total functions.

No model in the loop and no hidden state: given the same inputs these return
the same output forever. That is what lets the evaluator assert, without
hedging, that a wrong score is the agent's argument error rather than a tool
defect -- which is the precondition for the ``tool_misuse`` label meaning
anything.

Formulae follow the original publications; each function names its source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CalcResult:
    name: str
    score: float
    interpretation: str
    components: list[tuple[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reference: str = ""
    # Additive point scores collapse their zero-scoring criteria into one line;
    # formulae whose terms are signed magnitudes list every term.
    collapse_zero_components: bool = True

    def render(self) -> str:
        lines = [f"{self.name}: {self.score:g}", f"Interpretation: {self.interpretation}"]
        if self.components:
            lines.append("Components:")
            shown = [c for c in self.components if c[1] or not self.collapse_zero_components]
            lines += [f"  - {label}: {value:+g}" for label, value in shown]
            if self.collapse_zero_components:
                zero = [label for label, value in self.components if not value]
                if zero:
                    lines.append(f"  (contributing 0: {', '.join(zero)})")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        if self.reference:
            lines.append(f"Reference: {self.reference}")
        return "\n".join(lines)


class CalculatorInputError(ValueError):
    """Raised for inputs that are outside physiological range or contradictory."""


# --------------------------------------------------------------------------
# CHA2DS2-VASc
# --------------------------------------------------------------------------

def cha2ds2_vasc(
    age: int,
    sex: str,
    congestive_heart_failure: bool = False,
    hypertension: bool = False,
    diabetes: bool = False,
    stroke_tia_thromboembolism: bool = False,
    vascular_disease: bool = False,
) -> CalcResult:
    """CHA2DS2-VASc stroke risk in non-valvular atrial fibrillation.

    Lip et al., Chest 2010;137(2):263-72. Maximum score 9.
    """
    if not isinstance(age, int) or age < 0 or age > 130:
        raise CalculatorInputError(f"age must be an integer in 0-130, got {age!r}")
    sex_n = str(sex).strip().lower()
    if sex_n in ("f", "female", "woman"):
        female = True
    elif sex_n in ("m", "male", "man"):
        female = False
    else:
        raise CalculatorInputError(f"sex must be 'male' or 'female', got {sex!r}")

    if age >= 75:
        age_pts, age_label = 2.0, "Age >=75"
    elif age >= 65:
        age_pts, age_label = 1.0, "Age 65-74"
    else:
        age_pts, age_label = 0.0, "Age <65"

    components = [
        ("Congestive heart failure / LV dysfunction", 1.0 * congestive_heart_failure),
        ("Hypertension", 1.0 * hypertension),
        (age_label, age_pts),
        ("Diabetes mellitus", 1.0 * diabetes),
        ("Prior stroke / TIA / thromboembolism", 2.0 * stroke_tia_thromboembolism),
        ("Vascular disease (prior MI, PAD, aortic plaque)", 1.0 * vascular_disease),
        ("Female sex", 1.0 * female),
    ]
    score = sum(p for _, p in components)

    # Guideline thresholds are sex-specific: the sex point alone does not make a
    # woman anticoagulation-eligible.
    non_sex = score - (1.0 if female else 0.0)
    if non_sex == 0:
        interp = "Low risk. Guidelines generally recommend no antithrombotic therapy."
    elif non_sex == 1:
        interp = (
            "Intermediate risk. Oral anticoagulation may be considered "
            "(shared decision-making)."
        )
    else:
        interp = "High risk. Oral anticoagulation is generally recommended."
    interp += f" (Score {score:g}; {non_sex:g} excluding the sex-category point.)"

    return CalcResult(
        name="CHA2DS2-VASc",
        score=score,
        interpretation=interp,
        components=components,
        reference="Lip GYH et al., Chest 2010;137(2):263-72",
    )


# --------------------------------------------------------------------------
# Wells' criteria for pulmonary embolism
# --------------------------------------------------------------------------

def wells_pe(
    clinical_signs_of_dvt: bool = False,
    pe_most_likely_diagnosis: bool = False,
    heart_rate_over_100: bool = False,
    immobilization_or_surgery: bool = False,
    previous_pe_or_dvt: bool = False,
    hemoptysis: bool = False,
    malignancy: bool = False,
) -> CalcResult:
    """Wells' criteria for pulmonary embolism.

    Wells PS et al., Thromb Haemost 2000;83(3):416-20. Both the traditional
    three-tier and the two-tier (modified) stratifications are reported, since
    clinical questions use either.
    """
    components = [
        ("Clinical signs/symptoms of DVT", 3.0 * clinical_signs_of_dvt),
        ("PE is the most likely diagnosis (or equally likely)", 3.0 * pe_most_likely_diagnosis),
        ("Heart rate > 100 bpm", 1.5 * heart_rate_over_100),
        ("Immobilization >=3 days or surgery in previous 4 weeks", 1.5 * immobilization_or_surgery),
        ("Previous objectively diagnosed PE or DVT", 1.5 * previous_pe_or_dvt),
        ("Hemoptysis", 1.0 * hemoptysis),
        ("Malignancy (treated within 6 months or palliative)", 1.0 * malignancy),
    ]
    score = sum(p for _, p in components)

    if score < 2:
        three_tier = "Low probability (~1.3-12.1%)"
    elif score <= 6:
        three_tier = "Moderate probability (~16.2-40.5%)"
    else:
        three_tier = "High probability (~37.5-91%)"
    two_tier = "PE unlikely" if score <= 4 else "PE likely"

    return CalcResult(
        name="Wells' criteria for PE",
        score=score,
        interpretation=(
            f"Three-tier: {three_tier}. Two-tier (modified): {two_tier}. "
            "'PE unlikely' supports a d-dimer-first strategy; 'PE likely' supports "
            "proceeding to CT pulmonary angiography."
        ),
        components=components,
        reference="Wells PS et al., Thromb Haemost 2000;83(3):416-20",
    )


# --------------------------------------------------------------------------
# MELD / MELD-Na
# --------------------------------------------------------------------------

def meld(
    bilirubin_mg_dl: float,
    inr: float,
    creatinine_mg_dl: float,
    dialysis: bool = False,
    sodium_meq_l: float | None = None,
) -> CalcResult:
    """MELD, and MELD-Na when sodium is supplied.

    MELD(i) = 3.78*ln(bilirubin) + 11.2*ln(INR) + 9.57*ln(creatinine) + 6.43
    (Kamath et al., Hepatology 2001). Lab values below 1.0 are floored to 1.0;
    creatinine is capped at 4.0 and forced to 4.0 after >=2 dialysis sessions
    in the past week. MELD-Na follows UNOS (Kim et al., NEJM 2008), applied
    only when MELD > 11, with sodium bounded to 125-137.
    """
    for label, value in (
        ("bilirubin_mg_dl", bilirubin_mg_dl),
        ("inr", inr),
        ("creatinine_mg_dl", creatinine_mg_dl),
    ):
        if value is None or not isinstance(value, int | float) or math.isnan(float(value)):
            raise CalculatorInputError(f"{label} must be a number, got {value!r}")
        if float(value) <= 0:
            raise CalculatorInputError(f"{label} must be positive, got {value!r}")

    warnings: list[str] = []
    bili = max(float(bilirubin_mg_dl), 1.0)
    inr_v = max(float(inr), 1.0)
    creat = max(float(creatinine_mg_dl), 1.0)
    if dialysis:
        if creat < 4.0:
            warnings.append("Creatinine set to 4.0 mg/dL because dialysis was reported.")
        creat = 4.0
    elif creat > 4.0:
        warnings.append("Creatinine capped at 4.0 mg/dL per the MELD definition.")
        creat = 4.0

    raw = 3.78 * math.log(bili) + 11.2 * math.log(inr_v) + 9.57 * math.log(creat) + 6.43
    score = min(40.0, max(6.0, round(raw)))
    name = "MELD"
    components = [
        (f"ln(bilirubin={bili:g}) x 3.78", round(3.78 * math.log(bili), 2)),
        (f"ln(INR={inr_v:g}) x 11.2", round(11.2 * math.log(inr_v), 2)),
        (f"ln(creatinine={creat:g}) x 9.57", round(9.57 * math.log(creat), 2)),
        ("constant", 6.43),
    ]

    if sodium_meq_l is not None:
        na = float(sodium_meq_l)
        if not 100 <= na <= 180:
            raise CalculatorInputError(
                f"sodium_meq_l must be in 100-180 mEq/L, got {sodium_meq_l!r}"
            )
        na_b = min(137.0, max(125.0, na))
        if na_b != na:
            warnings.append(f"Sodium bounded from {na:g} to {na_b:g} mEq/L per the UNOS formula.")
        if score > 11:
            adjusted = score + 1.32 * (137 - na_b) - (0.033 * score * (137 - na_b))
            score = min(40.0, max(6.0, round(adjusted)))
            components.append((f"sodium adjustment (Na={na_b:g})", round(adjusted - raw, 2)))
        else:
            warnings.append("MELD <= 11, so the sodium adjustment is not applied (UNOS rule).")
        name = "MELD-Na"

    if score >= 30:
        band = "very high 3-month mortality (~52.6% at 30-39)"
    elif score >= 20:
        band = "high 3-month mortality (~19.6% at 20-29)"
    elif score >= 10:
        band = "moderate 3-month mortality (~6.0% at 10-19)"
    else:
        band = "low 3-month mortality (~1.9% at <10)"

    return CalcResult(
        name=name,
        score=score,
        interpretation=f"{name} {score:g}: {band}. Used for liver transplant allocation priority.",
        components=components,
        warnings=warnings,
        reference="Kamath PS et al., Hepatology 2001;33(2):464-70; Kim WR et al., NEJM 2008;359:1018-26",
    )


# --------------------------------------------------------------------------
# Anion gap
# --------------------------------------------------------------------------

def anion_gap(
    sodium_meq_l: float,
    chloride_meq_l: float,
    bicarbonate_meq_l: float,
    potassium_meq_l: float | None = None,
    albumin_g_dl: float | None = None,
) -> CalcResult:
    """Serum anion gap, with optional potassium and albumin correction.

    AG = Na - (Cl + HCO3); when potassium is supplied, AG = (Na + K) - (Cl + HCO3).
    Hypoalbuminaemia masks a raised gap, so the albumin-corrected gap
    (Figge et al., J Lab Clin Med 1998) is reported when albumin is given:
    corrected AG = AG + 2.5 * (4.0 - albumin).
    """
    for label, value, lo, hi in (
        ("sodium_meq_l", sodium_meq_l, 90, 190),
        ("chloride_meq_l", chloride_meq_l, 50, 160),
        ("bicarbonate_meq_l", bicarbonate_meq_l, 1, 60),
    ):
        if value is None or not isinstance(value, int | float):
            raise CalculatorInputError(f"{label} must be a number, got {value!r}")
        if not lo <= float(value) <= hi:
            raise CalculatorInputError(
                f"{label}={value!r} is outside the physiological range {lo}-{hi} mEq/L"
            )

    na, cl, hco3 = float(sodium_meq_l), float(chloride_meq_l), float(bicarbonate_meq_l)
    components = [("Sodium", na), ("Chloride", -cl), ("Bicarbonate", -hco3)]
    gap = na - (cl + hco3)
    include_k = potassium_meq_l is not None
    if include_k:
        k = float(potassium_meq_l)
        if not 0.5 <= k <= 12:
            raise CalculatorInputError(
                f"potassium_meq_l={potassium_meq_l!r} is outside the physiological range 0.5-12"
            )
        gap += k
        components.insert(1, ("Potassium", k))

    # Reference ranges differ depending on whether potassium is included.
    upper = 16.0 if include_k else 12.0
    lower = 10.0 if include_k else 8.0
    warnings: list[str] = []

    corrected = None
    if albumin_g_dl is not None:
        alb = float(albumin_g_dl)
        if not 0.5 <= alb <= 7.0:
            raise CalculatorInputError(
                f"albumin_g_dl={albumin_g_dl!r} is outside the physiological range 0.5-7.0"
            )
        corrected = gap + 2.5 * (4.0 - alb)
        components.append((f"albumin correction (albumin={alb:g} g/dL)", corrected - gap))
        if alb < 3.5 and gap <= upper < corrected:
            warnings.append(
                "The uncorrected gap is normal but the albumin-corrected gap is raised; "
                "hypoalbuminaemia was masking an elevated anion gap."
            )

    effective = corrected if corrected is not None else gap
    if effective > upper:
        state = "elevated -- consider a high anion gap metabolic acidosis (GOLDMARK/MUDPILES)"
    elif effective < lower:
        state = "low -- consider hypoalbuminaemia, paraproteinaemia, or a laboratory artefact"
    else:
        state = "within the normal range"

    label = "Anion gap (K included)" if include_k else "Anion gap"
    interp = f"{label} = {gap:g} mEq/L (reference {lower:g}-{upper:g}); {state}."
    if corrected is not None:
        interp = (
            f"{label} = {gap:g} mEq/L; albumin-corrected = {corrected:g} mEq/L "
            f"(reference {lower:g}-{upper:g}); {state}."
        )

    return CalcResult(
        name=label,
        score=round(effective, 2),
        interpretation=interp,
        components=components,
        warnings=warnings,
        collapse_zero_components=False,
        reference="Figge J et al., J Lab Clin Med 1998;131(6):563-72 (albumin correction)",
    )
