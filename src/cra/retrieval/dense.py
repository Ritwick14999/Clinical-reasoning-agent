"""Dense (FAISS + sentence-transformers) retriever, behind the ``dense`` extra.

Optional because it pulls torch. Same ``Retriever`` protocol as
``BM25Retriever``, so an experiment config can switch ``retrieval.backend``
without touching the agent or tool layer. Import errors are deferred to
construction time and re-raised with the install hint, so importing this
module (e.g. for type references) never requires the extra.
"""

from __future__ import annotations

from pathlib import Path

from cra.retrieval.corpus import CorpusDoc, read_corpus_jsonl
from cra.tools.retrieval import RetrievedPassage

_MISSING_EXTRA = (
    "Dense retrieval requires the 'dense' extra. Install with: python tasks.py setup --extras dev,dense (not a bare pip -- it may belong to a different Python, and a uv-created venv has no pip) "
    "(pulls sentence-transformers, faiss-cpu and torch)."
)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    name = "dense"

    def __init__(self, docs: list[CorpusDoc], model_name: str = DEFAULT_MODEL) -> None:
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(_MISSING_EXTRA) from exc

        if not docs:
            raise ValueError("DenseRetriever needs at least one document")

        self.docs = docs
        self._np = np
        self._model = SentenceTransformer(model_name)
        texts = [f"{d.title or ''} {d.text}".strip() for d in docs]
        embeddings = np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    @classmethod
    def from_corpus_file(cls, path: str | Path, model_name: str = DEFAULT_MODEL) -> DenseRetriever:
        return cls(read_corpus_jsonl(path), model_name=model_name)

    def search(self, query: str, k: int = 5) -> list[RetrievedPassage]:
        query_emb = self._np.asarray(
            self._model.encode([query], normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        scores, indices = self._index.search(query_emb, min(k, len(self.docs)))
        out = []
        for score, i in zip(scores[0], indices[0], strict=True):
            if i < 0:
                continue
            d = self.docs[i]
            out.append(
                RetrievedPassage(
                    doc_id=d.doc_id, text=d.text, title=d.title, score=float(score), metadata=d.metadata
                )
            )
        return out
