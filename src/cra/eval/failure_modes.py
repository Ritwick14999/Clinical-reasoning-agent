"""The failure-mode taxonomy: docs/DESIGN.md Sec 6, R1 -> R5.

    R1 retrieval_failure   gold source known AND gold not in retrieved evidence AND wrong
    R2 tool_misuse         not R1 AND wrong AND tool use looks wrong (see assess_tool_use)
    R3 reasoning_failure   not R1 AND not R2 AND wrong (right evidence, right tools, wrong conclusion)
    R4 unsupported_claim   correct answer, but >=1 justification claim not entailed
    R5 correct_grounded    correct answer, every claim entailed

This module is stricter than the design prose in two places, both to honour
invariants documented elsewhere rather than because the taxonomy needed
rethinking:

* **A trace with no parseable final answer gets its own ``no_answer`` label**,
  not ``reasoning_failure``. ``Trace.is_correct`` already returns ``None`` for
  exactly this case, and CLAUDE.md is explicit: "a missing answer is distinct
  from a wrong answer... never collapse them." R1-R3 as written in the design
  doc implicitly assume an answer exists ("answer incorrect"); a trace with no
  answer at all doesn't satisfy that precondition, so it needs a place that
  isn't R3.
* **R4 vs R5 require an entailment pass.** When no
  :class:`~cra.eval.entailment.base.EntailmentChecker` is supplied, a correct
  trace's ``failure_mode`` is left as ``None`` -- "not yet gradable" -- rather
  than defaulting to ``correct_grounded``, which would silently assert a
  grounding check that never ran. Every incorrect-answer label (R1-R3) is
  always resolved with no entailment needed at all: only the correct/wrong
  answer's *justification quality* needs the model-independent NLI pass.

Hallucination is tracked as an independent boolean, over every graded trace
(right or wrong), never derived from ``failure_mode`` alone -- so a
hallucinated claim inside a wrong (R1/R2/R3) answer is never invisible to the
rate, exactly as docs/DESIGN.md Sec 6 requires.
"""

from __future__ import annotations

from cra.data.expected_tools import ALWAYS_PERMITTED, resolve_expected_tools
from cra.eval.claims import extract_claims, strip_citation_framing
from cra.eval.entailment.base import EntailmentChecker
from cra.eval.records import ClaimRecord, FailureMode, ToolUseAssessment, TraceEvalRecord
from cra.types import Trace

_MALFORMED_PREFIXES = ("malformed_call", "schema_violation", "unknown_tool")


def _has_malformed_call(trace: Trace) -> bool:
    return any(tc.error and tc.error.startswith(_MALFORMED_PREFIXES) for tc in trace.tool_calls)


def _recovered_from_malformed(trace: Trace) -> bool:
    """True when every malformed call was followed by a successful call.

    "Followed by" is ordered, not merely co-present: a tool that succeeded
    *before* failing was not a recovery. A call to a tool that does not exist
    counts as recovered if any later call succeeded at all, since there is no
    same-named tool to retry.
    """
    calls = sorted(trace.tool_calls, key=lambda tc: (tc.step, tc.index))
    for i, tc in enumerate(calls):
        if not (tc.error and tc.error.startswith(_MALFORMED_PREFIXES)):
            continue
        later = calls[i + 1 :]
        if tc.error.startswith("unknown_tool"):
            if not any(later_call.ok for later_call in later):
                return False
        elif not any(later_call.ok and later_call.name == tc.name for later_call in later):
            return False
    return True


def assess_tool_use(trace: Trace, expected_fn=resolve_expected_tools) -> ToolUseAssessment:
    """Compares actual tool use against the ``expected_tools`` oracle.

    The oracle is a keyword-regex stand-in for "which tools this case
    actually needed" (see ``cra.data.expected_tools``'s docstring) -- treat
    ``reasons`` as evidence for an audit, not as ground truth about the
    agent's competence.
    """
    expected = list(expected_fn(trace.question))
    used = trace.tool_names
    expected_set, used_set = set(expected), set(used)
    reasons: list[str] = []

    if expected_set - used_set:
        reasons.append("required_tool_never_called")
    # Retrieval is always permitted, so it can never make a call "unnecessary".
    if (used_set - expected_set) - ALWAYS_PERMITTED and expected_set:
        reasons.append("unnecessary_call")
    if expected_set and used_set and not (used_set & expected_set):
        reasons.append("wrong_tool")
    if _has_malformed_call(trace):
        # A malformed call the agent retried successfully is a formatting slip,
        # not the cause of a wrong answer. Both are recorded; only the
        # unrecovered form can capture the primary label.
        reasons.append(
            "malformed_call_recovered" if _recovered_from_malformed(trace) else "malformed_call"
        )

    return ToolUseAssessment(expected=expected, used=used, reasons=reasons)


def retrieval_hit(trace: Trace) -> bool | None:
    """None when this dataset has no gold source (MedQA, the trap set) --
    distinct from False, which means gold existed and was missed."""
    if not trace.question.has_gold_source:
        return None
    gold = set(trace.question.gold_source_ids)
    return bool(gold & set(trace.retrieved_source_ids))


def retrieved_nothing(trace: Trace) -> bool:
    """True when retrieval was available to the agent but produced no passage.

    R1's gold-PMID test only works where a gold source exists, which excludes
    MedQA entirely. But "correct evidence existed but was not retrieved" -- the
    taxonomy's own wording -- is satisfied just as plainly when the agent
    retrieved nothing at all, and that is decidable without gold provenance.
    Human annotation flagged exactly these cases as retrieval failures while
    the classifier, unable to fire R1, was calling them reasoning failures.

    Gated on the tool budget so the closed-book ablation, where retrieval is
    withheld by design, is not relabelled as failing at it.
    """
    if trace.tool_budget <= 0:
        return False
    return not any(e.kind == "passage" for e in trace.evidence)


def _grade_claims(
    trace: Trace, entailment: EntailmentChecker | None
) -> tuple[list[ClaimRecord], bool | None]:
    if trace.final is None:
        return [], None

    texts = extract_claims(trace.final.justification)
    evidence_texts = [e.text for e in trace.evidence]

    if entailment is None:
        return [ClaimRecord(text=t) for t in texts], None

    # The citation wrapper is stripped before the claim becomes a hypothesis:
    # asking whether an abstract entails "E1 states that X" is structurally
    # unanswerable, and penalised the agent for citing as instructed. See
    # cra.eval.claims for the measurement that motivated this.
    claims = []
    for t in texts:
        hypothesis = strip_citation_framing(t)
        claims.append(
            ClaimRecord(
                text=t,
                tested_text=hypothesis if hypothesis != t else None,
                label=entailment.check(hypothesis, evidence_texts),
                checker=entailment.name,
            )
        )
    hallucinated = any(c.label in ("contradicted", "not_addressed") for c in claims)
    return claims, hallucinated


def classify_trace(
    trace: Trace,
    entailment: EntailmentChecker | None = None,
    expected_fn=resolve_expected_tools,
) -> TraceEvalRecord:
    tool_use = assess_tool_use(trace, expected_fn=expected_fn)
    hit = retrieval_hit(trace)
    gold_available = trace.question.has_gold_source

    failure_mode: FailureMode | None
    nothing_retrieved = retrieved_nothing(trace)
    retrieval_failure_reason: str | None = None

    if trace.is_correct is None:
        failure_mode = "no_answer"
    elif not trace.is_correct:
        if (gold_available and hit is False) or nothing_retrieved:
            failure_mode = "retrieval_failure"
            retrieval_failure_reason = (
                "gold_not_retrieved" if (gold_available and hit is False) else "nothing_retrieved"
            )
        elif tool_use.blocking_reasons:
            failure_mode = "tool_misuse"
        else:
            failure_mode = "reasoning_failure"
    else:
        failure_mode = None  # correct; resolved to R4/R5 below if graded

    claims, hallucinated = _grade_claims(trace, entailment)
    if trace.is_correct and hallucinated is not None:
        failure_mode = "unsupported_claim" if hallucinated else "correct_grounded"

    return TraceEvalRecord(
        trace_id=trace.trace_id,
        experiment_id=trace.experiment_id,
        model_id=trace.model_id,
        dataset=trace.question.dataset,
        split=trace.question.split,
        qid=trace.question.qid,
        is_correct=trace.is_correct,
        retrieval_failure_reason=retrieval_failure_reason,
        retrieval_gold_available=gold_available,
        retrieval_hit=hit,
        tool_use=tool_use,
        failure_mode=failure_mode,
        hallucinated=hallucinated,
        claims=claims,
    )
