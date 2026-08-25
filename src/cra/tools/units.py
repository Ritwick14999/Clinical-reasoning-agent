"""Laboratory unit conversion between conventional and SI units."""

from __future__ import annotations

import time
from typing import Any

from cra.tools.base import EvidenceDraft, ToolResult

# analyte -> (conventional unit, SI unit, multiply conventional by this to get SI)
CONVERSIONS: dict[str, tuple[str, str, float]] = {
    "glucose": ("mg/dL", "mmol/L", 0.0555),
    "creatinine": ("mg/dL", "umol/L", 88.4),
    "urea nitrogen": ("mg/dL", "mmol/L", 0.357),
    "bun": ("mg/dL", "mmol/L", 0.357),
    "total bilirubin": ("mg/dL", "umol/L", 17.1),
    "bilirubin": ("mg/dL", "umol/L", 17.1),
    "calcium": ("mg/dL", "mmol/L", 0.25),
    "phosphate": ("mg/dL", "mmol/L", 0.323),
    "magnesium": ("mg/dL", "mmol/L", 0.411),
    "albumin": ("g/dL", "g/L", 10.0),
    "total protein": ("g/dL", "g/L", 10.0),
    "hemoglobin": ("g/dL", "g/L", 10.0),
    "haemoglobin": ("g/dL", "g/L", 10.0),
    "total cholesterol": ("mg/dL", "mmol/L", 0.0259),
    "cholesterol": ("mg/dL", "mmol/L", 0.0259),
    "ldl cholesterol": ("mg/dL", "mmol/L", 0.0259),
    "hdl cholesterol": ("mg/dL", "mmol/L", 0.0259),
    "triglycerides": ("mg/dL", "mmol/L", 0.0113),
    "uric acid": ("mg/dL", "umol/L", 59.48),
    "lactate": ("mg/dL", "mmol/L", 0.111),
    "ammonia": ("ug/dL", "umol/L", 0.587),
    "iron": ("ug/dL", "umol/L", 0.179),
    "ferritin": ("ng/mL", "ug/L", 1.0),
    "tsh": ("uIU/mL", "mIU/L", 1.0),
    "free t4": ("ng/dL", "pmol/L", 12.87),
    "cortisol": ("ug/dL", "nmol/L", 27.59),
    "testosterone": ("ng/dL", "nmol/L", 0.0347),
    "digoxin": ("ng/mL", "nmol/L", 1.281),
    "lithium": ("mEq/L", "mmol/L", 1.0),
    "phenytoin": ("ug/mL", "umol/L", 3.964),
    "theophylline": ("ug/mL", "umol/L", 5.55),
}

_ALIASES = {"umol/l": "umol/L", "mmol/l": "mmol/L", "g/l": "g/L", "mg/dl": "mg/dL",
            "g/dl": "g/dL", "ug/dl": "ug/dL", "ng/ml": "ng/mL", "ug/ml": "ug/mL",
            "nmol/l": "nmol/L", "pmol/l": "pmol/L", "meq/l": "mEq/L", "ug/l": "ug/L",
            "miu/l": "mIU/L", "uiu/ml": "uIU/mL", "µmol/l": "umol/L", "μmol/l": "umol/L"}


def _canon_unit(unit: str) -> str:
    u = str(unit).strip().replace("µ", "u").replace("μ", "u")
    return _ALIASES.get(u.lower(), u)


class UnitConversionTool:
    name = "convert_lab_units"
    description = (
        "Convert a laboratory value between conventional (US) and SI units, for example "
        "creatinine mg/dL to umol/L or glucose mg/dL to mmol/L. Use this rather than converting "
        "from memory when a case reports a lab in unfamiliar units, especially before passing "
        "the value to a calculator that expects a specific unit."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "analyte": {
                "type": "string",
                "description": f"Analyte name, e.g. one of: {', '.join(sorted(CONVERSIONS)[:12])}...",
            },
            "value": {"type": "number", "description": "The numeric value to convert."},
            "from_unit": {"type": "string", "description": "Unit of the supplied value."},
            "to_unit": {"type": "string", "description": "Desired unit."},
        },
        "required": ["analyte", "value", "from_unit", "to_unit"],
        "additionalProperties": False,
    }

    def run(
        self,
        analyte: str | None = None,
        value: float | None = None,
        from_unit: str | None = None,
        to_unit: str | None = None,
        **_: Any,
    ) -> ToolResult:
        started = time.perf_counter()
        key = str(analyte or "").strip().lower()
        if key not in CONVERSIONS:
            return ToolResult.failure(
                f"unknown analyte {analyte!r}. Supported analytes: {', '.join(sorted(CONVERSIONS))}"
            )
        if not isinstance(value, int | float):
            return ToolResult.failure(f"'value' must be a number, got {value!r}")

        conv_unit, si_unit, factor = CONVERSIONS[key]
        src, dst = _canon_unit(from_unit or ""), _canon_unit(to_unit or "")
        if src == conv_unit and dst == si_unit:
            result = float(value) * factor
        elif src == si_unit and dst == conv_unit:
            result = float(value) / factor
        elif src == dst and src in (conv_unit, si_unit):
            result = float(value)
        else:
            return ToolResult.failure(
                f"cannot convert {analyte} from {from_unit!r} to {to_unit!r}; "
                f"supported units are {conv_unit} and {si_unit}"
            )

        rounded = round(result, 4 if abs(result) < 1 else 2)
        output = (
            f"{analyte}: {value:g} {src} = {rounded:g} {dst} "
            f"(conversion factor {conv_unit} -> {si_unit}: x{factor:g})"
        )
        return ToolResult(
            ok=True,
            output=output,
            evidence=[
                EvidenceDraft(
                    kind="tool_output",
                    text=output,
                    title=f"Unit conversion: {analyte}",
                    source_id=f"convert_lab_units:{key}:{value}:{src}:{dst}",
                    metadata={"analyte": key, "result": rounded, "unit": dst},
                )
            ],
            latency_ms=(time.perf_counter() - started) * 1000,
        )
