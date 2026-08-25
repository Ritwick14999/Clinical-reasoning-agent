"""Literature search tool.

The tool holds no retrieval logic of its own -- it delegates to a
:class:`Retriever`, so the same tool serves a BM25 index, a dense FAISS index,
or a live PubMed client without the agent noticing. Concrete retrievers live
in :mod:`cra.retrieval`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cra.tools.base import EvidenceDraft, ToolResult

MAX_K = 10
SNIPPET_CHARS = 1400


@dataclass
class RetrievedPassage:
    doc_id: str
    text: str
    title: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[RetrievedPassage]: ...


class InMemoryRetriever:
    """Trivial substring-scoring retriever, for tests and the keyless demo."""

    name = "in_memory"

    def __init__(self, passages: list[RetrievedPassage]) -> None:
        self.passages = passages

    def search(self, query: str, k: int = 5) -> list[RetrievedPassage]:
        terms = {t for t in query.lower().split() if len(t) > 3}
        scored = []
        for p in self.passages:
            blob = f"{p.title or ''} {p.text}".lower()
            score = sum(blob.count(t) for t in terms)
            if score:
                scored.append((score, p))
        scored.sort(key=lambda sp: -sp[0])
        return [
            RetrievedPassage(p.doc_id, p.text, p.title, float(s), p.metadata)
            for s, p in scored[:k]
        ]


class SearchLiteratureTool:
    name = "search_literature"
    description = (
        "Search the biomedical literature corpus for passages relevant to a clinical question. "
        "Returns the top-k passages with their source identifier (PMID where available), title "
        "and text. Search with the clinical concepts at issue rather than the whole question, "
        "and cite the returned passage IDs in your justification. Passages not returned here "
        "are not available to you."
    )

    def __init__(self, retriever: Retriever, default_k: int = 5) -> None:
        self.retriever = retriever
        self.default_k = default_k
        self.input_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query: the key clinical concepts, not the full stem.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_K,
                    "default": default_k,
                    "description": f"Number of passages to return (1-{MAX_K}).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def run(self, query: str | None = None, k: int | None = None, **_: Any) -> ToolResult:
        started = time.perf_counter()
        if not isinstance(query, str) or not query.strip():
            return ToolResult.failure(f"'query' must be a non-empty string, got {query!r}")
        k_eff = self.default_k if k is None else int(k)
        if not 1 <= k_eff <= MAX_K:
            return ToolResult.failure(f"'k' must be between 1 and {MAX_K}, got {k!r}")

        passages = self.retriever.search(query.strip(), k=k_eff)
        latency_ms = (time.perf_counter() - started) * 1000
        if not passages:
            return ToolResult(
                ok=True,
                output=(
                    f"No passages matched the query {query!r}. The corpus contains nothing "
                    "relevant to these terms; do not assume evidence exists for this point."
                ),
                latency_ms=latency_ms,
            )

        drafts = []
        for p in passages:
            text = p.text.strip()
            truncated = text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else "")
            drafts.append(
                EvidenceDraft(
                    kind="passage",
                    text=truncated,
                    title=p.title,
                    source_id=p.doc_id,
                    score=p.score,
                    metadata={**p.metadata, "retriever": self.retriever.name, "query": query},
                )
            )
        # The passage text reaches the model via the evidence blocks the registry
        # appends once trace-global IDs have been assigned.
        return ToolResult(
            ok=True,
            output=f"Retrieved {len(passages)} passage(s) for {query!r}.",
            evidence=drafts,
            latency_ms=latency_ms,
        )
