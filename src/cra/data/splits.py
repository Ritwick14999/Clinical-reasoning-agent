"""Dataset split access with ``CRA_ALLOW_TEST`` discipline, and seeded sampling.

``test`` is for final numbers only (``CLAUDE.md``, "No test-set peeking").
That rule is enforced here, once, rather than trusted to discipline
elsewhere: this is the one gate every experiment config's question loading
passes through.
"""

from __future__ import annotations

import os
import random

from cra.data.medqa import load_medqa
from cra.data.pubmedqa import load_pubmedqa
from cra.data.trapset import load_trapset
from cra.types import Dataset, Question, Split

_LOADERS = {
    "pubmedqa": load_pubmedqa,
    "medqa": load_medqa,
    "trapset": load_trapset,
}


def _check_test_allowed(split: Split) -> None:
    if split == "test" and os.environ.get("CRA_ALLOW_TEST") != "1":
        raise RuntimeError(
            "split='test' requires CRA_ALLOW_TEST=1. All development and threshold "
            "calibration happens on 'dev'; test is touched once, for final numbers only "
            "(see CLAUDE.md, 'No test-set peeking')."
        )


def load_dataset(dataset: Dataset) -> list[Question]:
    return _LOADERS[dataset]()


def get_split(dataset: Dataset, split: Split) -> list[Question]:
    _check_test_allowed(split)
    return [q for q in load_dataset(dataset) if q.split == split]


def sample_stratified(
    questions: list[Question], n: int | None, seed: int = 12345
) -> list[Question]:
    """A seeded, gold-answer-stratified subsample, capped at ``n``.

    Stratifying by ``gold_answer`` keeps the label balance close to the full
    split's, so a 300-question headline run is not accidentally skewed by
    sampling variance in that balance. ``n`` is a per-call cap, not a
    per-dataset one -- callers apply it once per dataset (see
    ``docs/HANDOFF.md`` decision 3: "300 questions per dataset").
    """
    if n is None or n >= len(questions):
        return list(questions)
    if n <= 0:
        return []

    groups: dict[str, list[Question]] = {}
    for q in questions:
        groups.setdefault(q.gold_answer, []).append(q)

    rng = random.Random(seed)
    ordered_keys = sorted(groups)
    for key in ordered_keys:
        groups[key] = sorted(groups[key], key=lambda q: q.qid)
        rng.shuffle(groups[key])

    quota = n / len(questions)
    picked: list[Question] = []
    for key in ordered_keys:
        take = round(len(groups[key]) * quota)
        picked.extend(groups[key][:take])

    # Rounding can land one under or over target; true up from/into the
    # largest group so every other group's proportion stays untouched.
    largest = max(ordered_keys, key=lambda k: len(groups[k]))
    picked_qids = {q.qid for q in picked}
    if len(picked) < n:
        leftover = [q for q in groups[largest] if q.qid not in picked_qids]
        picked.extend(leftover[: n - len(picked)])
    elif len(picked) > n:
        picked = picked[:n]

    picked.sort(key=lambda q: q.qid)
    rng.shuffle(picked)
    return picked
