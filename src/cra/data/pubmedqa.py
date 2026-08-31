"""Parse PubMedQA (PQA-L) into :class:`Question` objects, with dev/test split.

Both splits come from the same file, ``ori_pqal.json`` (1 000 expert-annotated
items, keyed by PMID). Split membership is decided by the *official*
``test_ground_truth.json``: a PMID that appears there is ``test``; every other
PMID is ``dev``. No re-split is invented here, unlike ``medqa.py`` -- the
official partition is directly reachable from the allowed sources.

Only the bare question is exposed as ``Question.question``; the source
abstract (``CONTEXTS``) is deliberately *not* inlined. It becomes a retrieval
corpus document instead (see ``cra.retrieval.corpus``), keyed by the same
PMID, so the agent must retrieve it -- which is what makes ``retrieval_failure``
decidable at all. Inlining the abstract would turn this into a closed-book
reading-comprehension task and make retrieval moot.
"""

from __future__ import annotations

import json
from pathlib import Path

from cra.data.expected_tools import expected_tools_for
from cra.types import Question

DEFAULT_ORI = Path("data/raw/pubmedqa/ori_pqal.json")
DEFAULT_TEST_GT = Path("data/raw/pubmedqa/test_ground_truth.json")

_VALID_ANSWERS = ("yes", "no", "maybe")


def load_pubmedqa(
    ori_path: str | Path = DEFAULT_ORI, test_gt_path: str | Path = DEFAULT_TEST_GT
) -> list[Question]:
    ori_path, test_gt_path = Path(ori_path), Path(test_gt_path)
    if not ori_path.exists():
        raise FileNotFoundError(
            f"{ori_path} not found. Run `python tasks.py data` to download it first."
        )
    data = json.loads(ori_path.read_text(encoding="utf-8"))
    test_gt = json.loads(test_gt_path.read_text(encoding="utf-8")) if test_gt_path.exists() else {}

    questions = []
    for pmid, item in sorted(data.items()):
        gold = item.get("final_decision")
        if gold not in _VALID_ANSWERS:
            continue

        official = test_gt.get(pmid)
        if official is not None and official != gold:
            # Trust the official label on divergence -- the point of cross-
            # checking is to catch drift between the two sources, not to
            # silently prefer one and hide the disagreement.
            gold = official

        question_text = item.get("QUESTION", "").strip()
        questions.append(
            Question(
                qid=str(pmid),
                dataset="pubmedqa",
                split="test" if pmid in test_gt else "dev",
                question=question_text,
                gold_answer=gold,
                gold_source_ids=[str(pmid)],
                expected_tools=expected_tools_for(question_text),
                metadata={"year": item.get("YEAR"), "meshes": item.get("MESHES", [])},
            )
        )
    return questions
