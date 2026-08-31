"""Core data types.

The :class:`Trace` is the interface between the two pipeline stages: rollout
(expensive, stochastic, needs model credentials) and evaluation (a pure
function of traces, no network). Everything the evaluator needs must therefore
be *inlined* in the trace -- notably the full text of every retrieved passage
and tool output. A trace must remain interpretable years later, with the
retrieval index long since rebuilt.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

Dataset = Literal["medqa", "pubmedqa", "trapset"]
Split = Literal["dev", "test"]
TerminatedBy = Literal["model", "step_limit", "unparseable", "error"]
EvidenceKind = Literal["passage", "tool_output"]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Evidence(BaseModel):
    """A citable unit the agent has seen.

    ``evidence_id`` is stable within a trace and is what the agent is asked to
    cite: ``E1, E2, ...`` for retrieved passages, ``T1, T2, ...`` for tool
    outputs (a computed risk score can legitimately ground a claim).
    """

    evidence_id: str
    kind: EvidenceKind
    source_tool: str
    text: str
    title: str | None = None
    source_id: str | None = None  # PMID for passages; tool name+args digest otherwise
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        head = f"[{self.evidence_id}]"
        if self.title:
            head += f" {self.title}"
        if self.source_id:
            head += f" (source: {self.source_id})"
        return f"{head}\n{self.text}"


class ToolCallRecord(BaseModel):
    """One dispatched tool call. Failures are recorded, never raised."""

    index: int
    step: int
    name: str
    args: dict[str, Any]
    ok: bool
    output: str
    error: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    # False when the budget was already spent and the call was refused rather
    # than run. Refused calls still need a recorded result -- tool-use APIs
    # reject a turn whose tool_use blocks have no matching tool_result -- but
    # they must not consume budget or count as tool use by the agent.
    executed: bool = True


class Step(BaseModel):
    index: int
    kind: Literal["model", "repair", "forced_final"]
    text: str | None = None
    thinking: str | None = None
    n_tool_calls: int = 0
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0


class FinalAnswer(BaseModel):
    answer: str
    justification: str
    citations: list[str] = Field(default_factory=list)
    raw: str = ""


class Question(BaseModel):
    """A benchmark item, plus the annotations the evaluator needs.

    ``gold_source_ids`` is what makes the ``retrieval_failure`` label decidable;
    it is populated for PubMedQA (the linked abstract's PMID) and empty for
    MedQA, which has no gold source document. ``expected_tools`` is the
    rule-derived oracle behind the ``tool_misuse`` label -- fallible by
    construction, and audited separately.
    """

    qid: str
    dataset: Dataset
    split: Split
    question: str
    options: dict[str, str] | None = None
    gold_answer: str
    gold_source_ids: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_gold_source(self) -> bool:
        return bool(self.gold_source_ids)


class Trace(BaseModel):
    """A complete episode. Self-contained by design."""

    # ``model_id`` is a deliberate field name; silence pydantic's protected-namespace warning.
    model_config = ConfigDict(protected_namespaces=())

    schema_version: str = SCHEMA_VERSION
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    experiment_id: str = "adhoc"
    question: Question

    # Provenance: everything needed to explain how this trace came to exist.
    model_id: str = "unknown"
    provider: str = "unknown"
    config_hash: str = ""
    git_sha: str = ""
    seed: int | None = None
    temperature: float = 0.0
    tool_budget: int = 0
    max_steps: int = 0
    agent_mode: str = "function_calling"
    # 0 means "not applicable" (mock/anthropic providers have no fixed local
    # context window). Set by the rollout runner's preflight check for Ollama
    # runs, so a truncated episode is distinguishable from a normal one after
    # the fact -- see docs/HANDOFF.md, "The context-overflow hazard".
    context_length: int = 0
    created_at: float = Field(default_factory=time.time)

    steps: list[Step] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    final: FinalAnswer | None = None

    terminated_by: TerminatedBy = "model"
    repair_used: bool = False
    budget_exhausted_at_step: int | None = None
    wall_time_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None

    # -- convenience accessors used throughout the evaluator ---------------

    @property
    def executed_tool_calls(self) -> list[ToolCallRecord]:
        return [tc for tc in self.tool_calls if tc.executed]

    @property
    def n_tool_calls(self) -> int:
        """Budget-consuming calls. A malformed or failing call still counts:
        the agent spent the opportunity."""
        return len(self.executed_tool_calls)

    @property
    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.executed_tool_calls]

    @property
    def retrieved_source_ids(self) -> list[str]:
        return [e.source_id for e in self.evidence if e.kind == "passage" and e.source_id]

    def evidence_by_id(self, evidence_id: str) -> Evidence | None:
        return next((e for e in self.evidence if e.evidence_id == evidence_id), None)

    @property
    def is_correct(self) -> bool | None:
        """None when the agent produced no parseable answer -- distinct from wrong."""
        if self.final is None:
            return None
        return normalize_answer(self.final.answer) == normalize_answer(self.question.gold_answer)


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison.

    Handles the two answer spaces in use: MCQ letters (``A``-``E``, tolerating
    ``(A)``, ``A.``, ``Option A``) and PubMedQA's ``yes``/``no``/``maybe``.
    """
    a = (answer or "").strip().lower()
    a = a.removeprefix("option").strip()
    a = a.strip("()[].: \t\n\"'")
    if a in {"yes", "no", "maybe"}:
        return a
    if len(a) >= 1 and a[0] in "abcde" and (len(a) == 1 or not a[1].isalnum()):
        return a[0]
    return a
