"""Calibrating the entailment thresholds against human labels.

The thresholds decide where an entailment probability becomes a verdict, and
they were left at their 0.5 default through every result so far. That is a real
gap: the headline grounding number depends on them, and an uncalibrated cut
could be systematically strict (overstating unsupported claims) or lax
(understating them).

The reason it was skipped is worth stating, because it dictates the design
here. Tuning thresholds to maximise agreement with the same labels the kappa is
then reported against is circular -- it fits the classifier to its own
validation set and inflates the number while measuring nothing. The project
already has the structure that resolves this: two disjoint annotation passes.
So the sweep fits on the **development** pass and reports on the **held-out**
pass, exactly as a model would be fitted on train and reported on test.

Thresholds only affect traces whose answer was *correct*: those are the ones
routed to R4 (``unsupported_claim``) versus R5 (``correct_grounded``) by the
entailment result. Incorrect answers are labelled by retrieval and tool
evidence with no entailment involved, so they are unaffected by any cut and
contribute the same agreement at every candidate. They are still included in
the reported kappa, because that is the kappa the project reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cra.eval.claims import extract_claims, strip_citation_framing
from cra.eval.entailment.nli import NLIEntailmentChecker
from cra.eval.failure_modes import assess_tool_use, retrieval_hit, retrieved_nothing
from cra.types import Trace

# Coarse enough to sweep quickly, fine enough to locate a sensible cut. Values
# at or above 1.0 are excluded: the rule is a strict `>` comparison, so a cut of
# 1.0 can never fire.
DEFAULT_GRID = [round(0.05 * i, 2) for i in range(1, 20)]


@dataclass
class ScoredClaim:
    """A claim's cached probabilities, so a sweep never re-runs the model."""

    entail: float
    contradict: float


@dataclass
class ScoredTrace:
    """Everything about a trace that a threshold sweep needs, precomputed.

    The threshold-independent part of the label -- whether the answer was
    correct, and which upstream cause applies when it was not -- is resolved
    once. Only the correct-answer branch is re-decided per candidate.
    """

    trace_id: str
    is_correct: bool | None
    fixed_label: str | None  # label when it does not depend on thresholds
    claims: list[ScoredClaim] = field(default_factory=list)

    def label_at(self, entail_threshold: float) -> str | None:
        """The label this trace receives at a given entailment cut.

        Only the entailment threshold appears. A claim counts as unsupported
        when it is *not* entailed, and the contradiction threshold merely
        decides whether such a claim is reported as ``contradicted`` or
        ``not_addressed`` -- both of which set the hallucination flag, so
        neither changes the label. Sweeping it would produce a grid of
        identical rows.
        """
        if self.fixed_label is not None:
            return self.fixed_label
        # No extracted claims means nothing to contradict the evidence, which
        # matches how failure_modes treats an empty claim list.
        if any(not (c.entail > entail_threshold) for c in self.claims):
            return "unsupported_claim"
        return "correct_grounded"


def _fixed_label(trace: Trace) -> str | None:
    """The label when thresholds cannot change it, else None."""
    if trace.is_correct is None:
        return "no_answer"
    if trace.is_correct:
        return None  # R4 vs R5 depends on the entailment cut
    gold_available = trace.question.has_gold_source
    if (gold_available and retrieval_hit(trace) is False) or retrieved_nothing(trace):
        return "retrieval_failure"
    if assess_tool_use(trace).blocking_reasons:
        return "tool_misuse"
    return "reasoning_failure"


def score_traces(
    traces: list[Trace], checker: NLIEntailmentChecker, trace_ids: set[str] | None = None
) -> list[ScoredTrace]:
    """Precompute per-claim probabilities once for every trace of interest."""
    out: list[ScoredTrace] = []
    for trace in traces:
        if trace_ids is not None and trace.trace_id not in trace_ids:
            continue
        fixed = _fixed_label(trace)
        claims: list[ScoredClaim] = []
        if fixed is None and trace.final is not None:
            premises = [e.text for e in trace.evidence]
            for claim in extract_claims(trace.final.justification):
                entail, contradict = checker.score(strip_citation_framing(claim), premises)
                claims.append(ScoredClaim(entail=entail, contradict=contradict))
        out.append(
            ScoredTrace(
                trace_id=trace.trace_id,
                is_correct=trace.is_correct,
                fixed_label=fixed,
                claims=claims,
            )
        )
    return out


def kappa_at(
    scored: list[ScoredTrace],
    human: dict[str, str],
    entail_threshold: float,
) -> tuple[float | None, int, float]:
    """``(kappa, n, observed_agreement)`` at one entailment cut."""
    from sklearn.metrics import cohen_kappa_score

    auto: list[str] = []
    ref: list[str] = []
    for st in scored:
        label = human.get(st.trace_id)
        if label is None:
            continue
        predicted = st.label_at(entail_threshold)
        if predicted is None:
            continue
        auto.append(predicted)
        ref.append(label)
    if not auto:
        return None, 0, 0.0
    agreement = sum(a == b for a, b in zip(auto, ref, strict=False)) / len(auto)
    if len(set(auto)) == 1 and len(set(ref)) == 1 and auto[0] == ref[0]:
        # Degenerate: perfect agreement on a single class. kappa is undefined
        # there (no expected disagreement), so report the agreement instead of
        # a misleading 0.0.
        return 1.0, len(auto), agreement
    return float(cohen_kappa_score(ref, auto)), len(auto), agreement


@dataclass
class CalibrationResult:
    best_entail: float
    dev_kappa: float
    dev_n: int
    heldout_kappa: float | None
    heldout_n: int
    default_entail: float
    default_dev_kappa: float
    default_heldout_kappa: float | None
    grid: list[tuple[float, float]] = field(default_factory=list)


def calibrate(
    dev_scored: list[ScoredTrace],
    dev_human: dict[str, str],
    heldout_scored: list[ScoredTrace],
    heldout_human: dict[str, str],
    grid: list[float] | None = None,
    default_entail: float = 0.5,
) -> CalibrationResult:
    """Fit the entailment cut on the dev pass, report it on the held-out pass."""
    grid = grid or DEFAULT_GRID

    scored_grid: list[tuple[float, float]] = []
    best_kappa, best_te = -2.0, default_entail
    for te in grid:
        k, _, _ = kappa_at(dev_scored, dev_human, te)
        if k is None:
            continue
        scored_grid.append((te, k))
        # Ties break toward the LOWER cut. A lower cut calls more claims
        # entailed, so it is the conservative choice for a project whose
        # headline is how often claims are *not* supported: it cannot inflate
        # the finding.
        if k > best_kappa or (k == best_kappa and te < best_te):
            best_kappa, best_te = k, te

    dev_k, dev_n, _ = kappa_at(dev_scored, dev_human, best_te)
    held_k, held_n, _ = kappa_at(heldout_scored, heldout_human, best_te)
    def_dev_k, _, _ = kappa_at(dev_scored, dev_human, default_entail)
    def_held_k, _, _ = kappa_at(heldout_scored, heldout_human, default_entail)

    return CalibrationResult(
        best_entail=best_te,
        dev_kappa=dev_k if dev_k is not None else 0.0,
        dev_n=dev_n,
        heldout_kappa=held_k,
        heldout_n=held_n,
        default_entail=default_entail,
        default_dev_kappa=def_dev_k if def_dev_k is not None else 0.0,
        default_heldout_kappa=def_held_k,
        grid=scored_grid,
    )


def render_report(result: CalibrationResult) -> str:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "| Entailment cut | Fitted on dev (kappa) | Reported on held-out (kappa) |",
        "|---|---|---|",
        f"| default, entail > {result.default_entail} | {result.default_dev_kappa:.3f} "
        f"| {fmt(result.default_heldout_kappa)} |",
        f"| calibrated, entail > {result.best_entail} | {result.dev_kappa:.3f} "
        f"| {fmt(result.heldout_kappa)} |",
        "",
        f"Fitted on {result.dev_n} development labels; reported on {result.heldout_n} "
        "held-out labels the sweep never saw.",
        "",
        "The contradiction threshold is not swept: a claim counts as unsupported when it "
        "is not entailed, so that threshold only decides whether such a claim is reported "
        "as contradicted or not-addressed and cannot change any label.",
    ]
    return "\n".join(lines)
