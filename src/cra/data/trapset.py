"""Loader for the hand-curated trap set.

Each item (``data/trapset/trapset_v1.jsonl``) pairs a clinical vignette with a
plausible-sounding distractor that the retrieval corpus does not support, plus
a recorded rationale explaining why. A model that answers confidently from the
distractor -- or invents supporting evidence for it -- is directly observable
against the rationale, rather than inferred from aggregate accuracy alone.

The trap set has no gold source document (``gold_source_ids`` is always
empty, like MedQA): these are hand-written vignettes, not linked to a
retrievable abstract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cra.data.expected_tools import expected_tools_for
from cra.types import Question

DEFAULT_PATH = Path("data/trapset/trapset_v1.jsonl")


def _parse_line(item: dict[str, Any]) -> Question:
    question_text = item["question"].strip()
    return Question(
        qid=item["qid"],
        dataset="trapset",
        split=item.get("split", "dev"),
        question=question_text,
        options=item.get("options"),
        gold_answer=item["gold_answer"],
        gold_source_ids=[],
        expected_tools=item.get("expected_tools") or expected_tools_for(question_text),
        metadata={
            "trap": item.get("trap", ""),
            "rationale": item.get("rationale", ""),
        },
    )


def load_trapset(path: str | Path = DEFAULT_PATH) -> list[Question]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    questions = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            questions.append(_parse_line(item))
    return questions
