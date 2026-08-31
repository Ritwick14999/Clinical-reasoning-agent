"""Retrieval corpus construction and I/O.

The corpus is built from PubMedQA abstracts: each PQA-L item's ``CONTEXTS``
becomes one document keyed by its PMID, which is what makes ``Hit@k`` (and
therefore ``retrieval_failure``) measurable against ``Question.gold_source_ids``.
The corpus is written once to a jsonl file and re-read by whichever retriever
backend an experiment configures, so BM25 and dense retrieval search exactly
the same documents.

Honest limitation (see ``docs/DESIGN.md`` Sec 5): this corpus is topically
matched to PubMedQA but only partially to MedQA, so MedQA retrieval will be
weak. That is reported as a finding, not hidden.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_corpus_from_pubmedqa(raw_path: str | Path) -> list[CorpusDoc]:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run `python tasks.py data` first.")
    data = json.loads(raw_path.read_text(encoding="utf-8"))

    docs = []
    for pmid, item in sorted(data.items()):
        contexts = item.get("CONTEXTS") or []
        text = "\n\n".join(c.strip() for c in contexts if c and c.strip())
        if not text:
            continue
        docs.append(
            CorpusDoc(
                doc_id=str(pmid),
                text=text,
                metadata={
                    "labels": item.get("LABELS", []),
                    "year": item.get("YEAR"),
                    "meshes": item.get("MESHES", []),
                },
            )
        )
    return docs


def write_corpus_jsonl(docs: list[CorpusDoc], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    return path


def read_corpus_jsonl(path: str | Path) -> list[CorpusDoc]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python tasks.py index` to build the corpus first."
        )
    docs = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            docs.append(CorpusDoc(**json.loads(line)))
    return docs
