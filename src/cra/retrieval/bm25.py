"""BM25 retriever over the local corpus.

The offline default: no embeddings, no torch, cheap enough at this scale
(~1000 documents) to tokenize and rank in-process at experiment start, so no
persisted index format is needed. Implements the same ``Retriever`` protocol
as ``InMemoryRetriever`` (``cra.tools.retrieval``), so ``SearchLiteratureTool``
needs no special-casing per backend.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from cra.retrieval.corpus import CorpusDoc, read_corpus_jsonl
from cra.tools.retrieval import RetrievedPassage

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, docs: list[CorpusDoc]) -> None:
        if not docs:
            raise ValueError("BM25Retriever needs at least one document")
        self.docs = docs
        tokenized = [_tokenize(f"{d.title or ''} {d.text}") for d in docs]
        self._bm25 = BM25Okapi(tokenized)

    @classmethod
    def from_corpus_file(cls, path: str | Path) -> BM25Retriever:
        return cls(read_corpus_jsonl(path))

    def search(self, query: str, k: int = 5) -> list[RetrievedPassage]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self.docs)), key=lambda i: -scores[i])[:k]
        return [
            RetrievedPassage(
                doc_id=self.docs[i].doc_id,
                text=self.docs[i].text,
                title=self.docs[i].title,
                score=float(scores[i]),
                metadata=self.docs[i].metadata,
            )
            for i in ranked
            if scores[i] > 0
        ]
