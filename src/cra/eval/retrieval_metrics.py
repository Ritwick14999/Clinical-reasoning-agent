"""Retrieval metrics: Hit@k and MRR against the gold source (PubMedQA only).

Only meaningful for traces whose ``Question.has_gold_source`` is true --
MedQA and the trap set have none, so those traces are excluded from the
denominator rather than counted as retrieval failures. That's a distinct,
R1-gated concept in ``failure_modes.py``; conflating "no gold to check
against" with "missed the gold" would misreport MedQA's retrieval as worse
than it was actually measured to be.

Rank is approximated as first-occurrence position in ``Trace.evidence`` --
the order evidence IDs were assigned, which follows retriever rank order
within a single search call. This is an approximation when an episode makes
multiple searches with different queries (see ``docs/HANDOFF.md``), not a
literal per-query MRR.
"""

from __future__ import annotations

from dataclasses import dataclass

from cra.types import Trace


@dataclass
class RetrievalMetrics:
    n_with_gold: int
    hit_at_k: float | None
    mrr: float | None


def _rank_of_gold(trace: Trace) -> int | None:
    gold = set(trace.question.gold_source_ids)
    for rank, source_id in enumerate(trace.retrieved_source_ids, start=1):
        if source_id in gold:
            return rank
    return None


def compute_retrieval_metrics(traces: list[Trace]) -> RetrievalMetrics:
    graded = [t for t in traces if t.question.has_gold_source]
    if not graded:
        return RetrievalMetrics(n_with_gold=0, hit_at_k=None, mrr=None)

    hits = 0
    reciprocal_ranks = []
    for t in graded:
        rank = _rank_of_gold(t)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return RetrievalMetrics(
        n_with_gold=len(graded),
        hit_at_k=hits / len(graded),
        mrr=sum(reciprocal_ranks) / len(graded),
    )
