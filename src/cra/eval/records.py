"""Per-trace evaluation records -- the output of the eval stage.

One :class:`TraceEvalRecord` per :class:`~cra.types.Trace`. Building these is
what ``cra eval`` does; ``report.py`` and ``viz/plots.py`` are pure
aggregations over a list of these plus nothing else, which is what keeps them
"a pure function of traces" in the sense CLAUDE.md's rollout/eval separation
means it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cra.types import Dataset, Split

EntailmentLabel = Literal["entailed", "contradicted", "not_addressed"]

# Ordered R1 -> R5 in docs/DESIGN.md Sec 6, plus two additions this module
# makes explicit rather than silently folding into an existing bucket -- see
# failure_modes.py's module docstring for why each exists:
#   * "no_answer" -- CLAUDE.md: "a missing answer is distinct from a wrong
#     answer... never collapse them."
#   * the primary Literal excludes "ungraded": a correct trace with no
#     entailment pass run yet has ``failure_mode = None``, not a sixth label,
#     so "ungraded" can never be mistaken for a real taxonomy bucket anywhere
#     labels get compared, counted or plotted.
FailureMode = Literal[
    "no_answer",
    "retrieval_failure",
    "tool_misuse",
    "reasoning_failure",
    "unsupported_claim",
    "correct_grounded",
]

TOOL_MISUSE_REASONS = (
    "required_tool_never_called",
    "unnecessary_call",
    "wrong_tool",
    "malformed_call",
)


class ClaimRecord(BaseModel):
    text: str
    label: EntailmentLabel | None = None  # None = not graded (no checker run)
    checker: str | None = None  # e.g. "nli", "judge:qwen3:8b"


class ToolUseAssessment(BaseModel):
    expected: list[str] = Field(default_factory=list)
    used: list[str] = Field(default_factory=list)
    # Subset of TOOL_MISUSE_REASONS; non-empty + an incorrect answer is what
    # fires the R2 (tool_misuse) label.
    reasons: list[str] = Field(default_factory=list)


class TraceEvalRecord(BaseModel):
    # ``model_id`` is a deliberate field name; silence pydantic's protected-namespace warning.
    model_config = ConfigDict(protected_namespaces=())

    trace_id: str
    experiment_id: str
    model_id: str
    dataset: Dataset
    split: Split
    qid: str

    is_correct: bool | None
    retrieval_gold_available: bool
    # None when retrieval_gold_available is False (MedQA, the trap set) --
    # distinct from False, which means gold was available and missed.
    retrieval_hit: bool | None
    tool_use: ToolUseAssessment

    # None means "correct answer, grounding not yet assessed" -- see the
    # FailureMode docstring above. Never "correct_grounded" by default.
    failure_mode: FailureMode | None
    # None means "not graded" (no EntailmentChecker was supplied when this
    # record was built). Computed independently of failure_mode so a
    # hallucination inside a *wrong* answer is never invisible to the rate.
    hallucinated: bool | None
    claims: list[ClaimRecord] = Field(default_factory=list)
