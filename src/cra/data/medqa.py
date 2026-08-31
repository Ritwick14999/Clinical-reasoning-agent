"""Parse the MedQA (USMLE, 4-option) test split from the MIRAGE benchmark file.

MIRAGE ships only the official USMLE test partition (1 273 items); no
independent MedQA ``dev`` split is reachable from the sources verified in
``docs/DESIGN.md`` Sec 0 (HuggingFace is blocked). A seeded,
answer-stratified split carves ``dev`` out of it instead, rather than
treating all 1 273 items as safe for threshold calibration.

**Stated limitation**: MedQA's ``dev`` questions here are not an
independently curated set -- they are a reproducible re-partition of the
official test set. Do not read too much into dev/test differences reported
for this dataset; ``split: test`` still touches the full official item pool
under the usual ``CRA_ALLOW_TEST`` discipline.

MedQA has no linked source document, so ``gold_source_ids`` is always empty
and the ``retrieval_failure`` label (R1) never fires for this dataset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from cra.data.expected_tools import expected_tools_for
from cra.types import Question

DEFAULT_MIRAGE = Path("data/raw/mirage/benchmark.json")
DEV_FRACTION = 0.2
SPLIT_SEED = 12345


def _dev_keys(data: dict, dev_fraction: float, seed: int) -> set[str]:
    """Answer-stratified so the split doesn't skew the A/B/C/D balance."""
    by_answer: dict[str, list[str]] = {}
    for key, item in data.items():
        by_answer.setdefault(item["answer"], []).append(key)

    rng = random.Random(seed)
    dev_keys: set[str] = set()
    for answer in sorted(by_answer):
        keys = sorted(by_answer[answer])  # deterministic order before shuffling
        rng.shuffle(keys)
        n_dev = round(len(keys) * dev_fraction)
        dev_keys.update(keys[:n_dev])
    return dev_keys


def load_medqa(
    mirage_path: str | Path = DEFAULT_MIRAGE,
    dev_fraction: float = DEV_FRACTION,
    seed: int = SPLIT_SEED,
) -> list[Question]:
    mirage_path = Path(mirage_path)
    if not mirage_path.exists():
        raise FileNotFoundError(
            f"{mirage_path} not found. Run `python tasks.py data` to download it first."
        )
    data = json.loads(mirage_path.read_text(encoding="utf-8"))["medqa"]
    dev_keys = _dev_keys(data, dev_fraction, seed)

    questions = []
    for key, item in sorted(data.items()):
        question_text = item["question"].strip()
        questions.append(
            Question(
                qid=f"medqa-{key}",
                dataset="medqa",
                split="dev" if key in dev_keys else "test",
                question=question_text,
                options=dict(item["options"]),
                gold_answer=item["answer"],
                gold_source_ids=[],
                expected_tools=expected_tools_for(question_text),
            )
        )
    return questions
