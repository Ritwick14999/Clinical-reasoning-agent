"""Paired ablation comparisons between two experiments over the same questions.

Two problems make a naive tools-vs-closed-book comparison meaningless, and both
are handled here.

**Grounding needs a premise set the closed-book arm does not have.** A
closed-book agent retrieves nothing, so every claim it makes is trivially
``not_addressed`` and its hallucination rate is 100% by construction --
comparing that against the tool-using arm measures the ablation, not the agent.
The question worth asking is whether the claims would be supported *if one went
and looked*. So closed-book claims are scored against the evidence the paired
tool-using run retrieved for the same question. That evidence is already inlined
in committed traces, so the evaluation stays a pure function of traces with no
retrieval index, network or model involved.

**Accuracy differences need a paired test.** The two arms answer the same
questions, so McNemar's test on the discordant pairs is the right instrument;
an unpaired interval comparison throws away the pairing and understates power.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cra.eval.claims import extract_claims, strip_citation_framing
from cra.eval.entailment.base import EntailmentChecker
from cra.types import Trace


@dataclass
class ArmSummary:
    experiment_id: str
    n: int
    n_answered: int
    accuracy: float | None
    traces_with_unsupported_claim: int
    claims_total: int
    claims_entailed: int
    claims_not_addressed: int
    claims_contradicted: int

    @property
    def hallucination_rate_traces(self) -> float | None:
        """Share of graded traces carrying at least one unsupported claim."""
        return self.traces_with_unsupported_claim / self.n if self.n else None

    @property
    def grounding_rate_claims(self) -> float | None:
        """Share of claims entailed by the evidence they are scored against."""
        return self.claims_entailed / self.claims_total if self.claims_total else None


@dataclass
class PairedAblation:
    baseline: ArmSummary
    variant: ArmSummary
    n_paired: int
    # McNemar on correctness: b = baseline right & variant wrong, c = the reverse.
    b: int
    c: int
    p_value: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def accuracy_delta(self) -> float | None:
        if self.baseline.accuracy is None or self.variant.accuracy is None:
            return None
        return self.variant.accuracy - self.baseline.accuracy

    @property
    def hallucination_delta(self) -> float | None:
        a, b = self.baseline.hallucination_rate_traces, self.variant.hallucination_rate_traces
        return None if a is None or b is None else b - a


def _key(trace: Trace) -> tuple[str, str, str]:
    return (trace.question.split, trace.question.dataset, trace.question.qid)


def evidence_for_grounding(
    trace: Trace, donor: Trace | None, use_donor_when_empty: bool = True
) -> tuple[list[str], bool]:
    """Passages to score this trace's claims against, and whether they were borrowed.

    A trace with retrieved passages is scored against its own. One with none --
    the closed-book arm, or an agent that simply never searched -- is scored
    against the paired run's evidence for the same question, so "did it assert
    things the literature does not support" stays answerable. Tool outputs are
    excluded when borrowing: a calculator result from a different episode was
    computed from different arguments and is not evidence about this one.
    """
    own = [e.text for e in trace.evidence if e.kind == "passage"]
    if own or not use_donor_when_empty or donor is None:
        return own, False
    return [e.text for e in donor.evidence if e.kind == "passage"], True


def summarise_arm(
    traces: list[Trace],
    entailment: EntailmentChecker,
    donors: dict[tuple[str, str, str], Trace] | None = None,
) -> ArmSummary:
    answered = [t for t in traces if t.final is not None]
    correct = [t for t in answered if t.is_correct]
    entailed = not_addressed = contradicted = 0
    flagged = 0

    for trace in traces:
        if trace.final is None:
            continue
        donor = (donors or {}).get(_key(trace))
        premises, _ = evidence_for_grounding(trace, donor)
        labels = []
        for claim in extract_claims(trace.final.justification):
            label = entailment.check(strip_citation_framing(claim), premises)
            labels.append(label)
            if label == "entailed":
                entailed += 1
            elif label == "contradicted":
                contradicted += 1
            else:
                not_addressed += 1
        if any(x in ("contradicted", "not_addressed") for x in labels):
            flagged += 1

    return ArmSummary(
        experiment_id=traces[0].experiment_id if traces else "?",
        n=len(traces),
        n_answered=len(answered),
        accuracy=(len(correct) / len(answered)) if answered else None,
        traces_with_unsupported_claim=flagged,
        claims_total=entailed + not_addressed + contradicted,
        claims_entailed=entailed,
        claims_not_addressed=not_addressed,
        claims_contradicted=contradicted,
    )


def mcnemar(b: int, c: int) -> float | None:
    """Exact two-sided McNemar p-value on the discordant pairs.

    The exact binomial form is used rather than the chi-square approximation:
    discordant counts here are often small, where the approximation is unsound.
    """
    from scipy import stats

    n = b + c
    if n == 0:
        return None
    return float(stats.binomtest(b, n, 0.5).pvalue)


def compare(
    baseline_traces: list[Trace],
    variant_traces: list[Trace],
    entailment: EntailmentChecker,
    borrow_evidence: bool = True,
) -> PairedAblation:
    """Compare two arms over the questions they share."""
    base_by_key = {_key(t): t for t in baseline_traces}
    var_by_key = {_key(t): t for t in variant_traces}
    shared = sorted(set(base_by_key) & set(var_by_key))

    notes: list[str] = []
    if len(shared) < max(len(base_by_key), len(var_by_key)):
        notes.append(
            f"paired on {len(shared)} shared question(s); "
            f"baseline had {len(base_by_key)}, variant {len(var_by_key)}"
        )

    base = [base_by_key[k] for k in shared]
    var = [var_by_key[k] for k in shared]

    donors = base_by_key if borrow_evidence else None
    borrowed = sum(
        1 for t in var if evidence_for_grounding(t, base_by_key.get(_key(t)))[1]
    )
    if borrowed:
        notes.append(
            f"{borrowed} variant trace(s) retrieved nothing and were scored against the "
            "baseline's evidence for the same question"
        )

    b = c = 0
    for key in shared:
        bc, vc = base_by_key[key].is_correct, var_by_key[key].is_correct
        if bc and vc is False:
            b += 1
        elif bc is False and vc:
            c += 1

    return PairedAblation(
        baseline=summarise_arm(base, entailment),
        variant=summarise_arm(var, entailment, donors=donors),
        n_paired=len(shared),
        b=b,
        c=c,
        p_value=mcnemar(b, c),
        notes=notes,
    )


def render_table(comparisons: list[tuple[str, PairedAblation]]) -> str:
    """Markdown table, one row per ablation arm."""
    lines = [
        "| Ablation | N paired | Accuracy | delta | McNemar p | Traces w/ unsupported claim | delta |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, cmp in comparisons:
        base_acc = f"{cmp.baseline.accuracy:.1%}" if cmp.baseline.accuracy is not None else "n/a"
        var_acc = f"{cmp.variant.accuracy:.1%}" if cmp.variant.accuracy is not None else "n/a"
        d_acc = f"{cmp.accuracy_delta:+.1%}" if cmp.accuracy_delta is not None else "n/a"
        # A tiny p-value stays legible as tiny; rounding it to 0.000 would read
        # as exactly zero, which no test reports.
        p = (
            "n/a"
            if cmp.p_value is None
            else (f"{cmp.p_value:.3f}" if cmp.p_value >= 0.001 else f"{cmp.p_value:.1e}")
        )
        base_h = cmp.baseline.hallucination_rate_traces
        var_h = cmp.variant.hallucination_rate_traces
        h = f"{base_h:.1%} -> {var_h:.1%}" if base_h is not None and var_h is not None else "n/a"
        d_h = f"{cmp.hallucination_delta:+.1%}" if cmp.hallucination_delta is not None else "n/a"
        lines.append(
            f"| {name} | {cmp.n_paired} | {base_acc} -> {var_acc} | {d_acc} | {p} | {h} | {d_h} |"
        )
    return "\n".join(lines)
