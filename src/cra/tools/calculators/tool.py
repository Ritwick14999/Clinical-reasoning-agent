"""Tool wrappers around the pure score functions.

The wrapper only translates: JSON arguments in, rendered text plus citable
evidence out. All clinical logic stays in :mod:`cra.tools.calculators.scores`
so it can be tested without any agent machinery.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from cra.tools.base import EvidenceDraft, ToolResult
from cra.tools.calculators import scores
from cra.tools.calculators.scores import CalcResult, CalculatorInputError

_BOOL = {"type": "boolean", "default": False}


def _b(desc: str) -> dict[str, Any]:
    return {**_BOOL, "description": desc}


class CalculatorTool:
    """Adapts a ``CalcResult``-returning function into the Tool protocol."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        fn: Callable[..., CalcResult],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._fn = fn

    def run(self, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            result = self._fn(**kwargs)
        except CalculatorInputError as exc:
            # A rejected argument is a first-class observation: the agent gets
            # the message back and the trace records a recoverable tool error.
            return ToolResult(
                ok=False,
                output=f"ERROR: invalid input for {self.name}: {exc}",
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except TypeError as exc:
            return ToolResult(
                ok=False,
                output=f"ERROR: wrong arguments for {self.name}: {exc}",
                error=f"argument mismatch: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        rendered = result.render()
        return ToolResult(
            ok=True,
            output=rendered,
            evidence=[
                EvidenceDraft(
                    kind="tool_output",
                    text=rendered,
                    title=f"{result.name} (computed)",
                    source_id=f"{self.name}:{sorted(kwargs.items())}",
                    metadata={"tool": self.name, "score": result.score, "args": kwargs},
                )
            ],
            latency_ms=(time.perf_counter() - started) * 1000,
        )


CHA2DS2_VASC = CalculatorTool(
    name="calc_cha2ds2_vasc",
    description=(
        "Compute the CHA2DS2-VASc stroke-risk score for a patient with non-valvular atrial "
        "fibrillation, and return the guideline anticoagulation implication. Use this whenever a "
        "case describes atrial fibrillation and gives enough history to score it. Do not estimate "
        "the score yourself."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "age": {"type": "integer", "description": "Patient age in years."},
            "sex": {
                "type": "string",
                "enum": ["male", "female"],
                "description": "Biological sex; female adds one point (sex category).",
            },
            "congestive_heart_failure": _b("History of heart failure or LV systolic dysfunction."),
            "hypertension": _b("History of hypertension or current antihypertensive treatment."),
            "diabetes": _b("Diabetes mellitus."),
            "stroke_tia_thromboembolism": _b(
                "Prior stroke, TIA, or systemic thromboembolism (worth 2 points)."
            ),
            "vascular_disease": _b("Prior MI, peripheral artery disease, or aortic plaque."),
        },
        "required": ["age", "sex"],
        "additionalProperties": False,
    },
    fn=scores.cha2ds2_vasc,
)

WELLS_PE = CalculatorTool(
    name="calc_wells_pe",
    description=(
        "Compute Wells' criteria for pulmonary embolism and return both the three-tier and "
        "two-tier risk stratification, which determine whether to start with d-dimer or go "
        "straight to CT pulmonary angiography. Use when a case raises suspicion of PE."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "clinical_signs_of_dvt": _b("Leg swelling and pain on deep vein palpation (3 points)."),
            "pe_most_likely_diagnosis": _b(
                "PE is the most likely diagnosis, or at least as likely as the alternatives "
                "(3 points)."
            ),
            "heart_rate_over_100": _b("Heart rate above 100 bpm (1.5 points)."),
            "immobilization_or_surgery": _b(
                "Immobilisation for 3 or more days, or surgery in the previous 4 weeks "
                "(1.5 points)."
            ),
            "previous_pe_or_dvt": _b("Previous objectively diagnosed PE or DVT (1.5 points)."),
            "hemoptysis": _b("Haemoptysis (1 point)."),
            "malignancy": _b("Malignancy treated within 6 months, or palliative (1 point)."),
        },
        "required": [],
        "additionalProperties": False,
    },
    fn=scores.wells_pe,
)

MELD = CalculatorTool(
    name="calc_meld",
    description=(
        "Compute MELD (or MELD-Na when a sodium value is supplied) for chronic liver disease "
        "severity and transplant priority. Requires bilirubin, INR and creatinine. The function "
        "applies the official flooring, capping and dialysis rules; pass raw lab values."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "bilirubin_mg_dl": {"type": "number", "description": "Total bilirubin in mg/dL."},
            "inr": {"type": "number", "description": "International normalised ratio."},
            "creatinine_mg_dl": {"type": "number", "description": "Serum creatinine in mg/dL."},
            "dialysis": _b(
                "Two or more haemodialysis sessions in the past week, or 24 hours of CVVHD; "
                "forces creatinine to 4.0."
            ),
            "sodium_meq_l": {
                "type": "number",
                "description": "Serum sodium in mEq/L. Supply it to compute MELD-Na instead.",
            },
        },
        "required": ["bilirubin_mg_dl", "inr", "creatinine_mg_dl"],
        "additionalProperties": False,
    },
    fn=scores.meld,
)

ANION_GAP = CalculatorTool(
    name="calc_anion_gap",
    description=(
        "Compute the serum anion gap from electrolytes, optionally including potassium and "
        "applying the albumin correction. Use for any acid-base case that supplies sodium, "
        "chloride and bicarbonate; hypoalbuminaemia can hide a raised gap, so pass albumin "
        "when it is available."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "sodium_meq_l": {"type": "number", "description": "Serum sodium in mEq/L."},
            "chloride_meq_l": {"type": "number", "description": "Serum chloride in mEq/L."},
            "bicarbonate_meq_l": {
                "type": "number",
                "description": "Serum bicarbonate (or total CO2) in mEq/L.",
            },
            "potassium_meq_l": {
                "type": "number",
                "description": "Serum potassium in mEq/L. Including it raises the reference range.",
            },
            "albumin_g_dl": {
                "type": "number",
                "description": "Serum albumin in g/dL, for the albumin-corrected gap.",
            },
        },
        "required": ["sodium_meq_l", "chloride_meq_l", "bicarbonate_meq_l"],
        "additionalProperties": False,
    },
    fn=scores.anion_gap,
)

ALL_CALCULATORS = [CHA2DS2_VASC, WELLS_PE, MELD, ANION_GAP]
