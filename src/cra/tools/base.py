"""Tool interface and the evidence store.

Every tool returns the same envelope, and a tool failure is a *result*, never
an exception. If a bad argument crashed the episode we would destroy exactly
the traces the tool-misuse analysis exists to study.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cra.llm.base import ToolSpec
from cra.types import Evidence, EvidenceKind


@dataclass
class EvidenceDraft:
    """Evidence as a tool produces it, before trace-global IDs are assigned."""

    kind: EvidenceKind
    text: str
    title: str | None = None
    source_id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    output: str
    evidence: list[EvidenceDraft] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0

    @classmethod
    def failure(cls, error: str) -> ToolResult:
        return cls(ok=False, output=f"ERROR: {error}", error=error)


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    def run(self, **kwargs: Any) -> ToolResult: ...


def spec_of(tool: Tool) -> ToolSpec:
    return ToolSpec(name=tool.name, description=tool.description, input_schema=tool.input_schema)


class EvidenceStore:
    """Append-only, deduplicated, with stable per-trace IDs.

    Passages get ``E1, E2, ...`` and tool outputs ``T1, T2, ...``. The agent is
    told to cite these IDs, and the grounding check resolves claims against
    them. Deduplication matters: the same abstract retrieved by two queries
    must keep one ID, or citation metrics double-count.
    """

    def __init__(self) -> None:
        self.items: list[Evidence] = []
        self._by_fingerprint: dict[str, Evidence] = {}
        self._counters: dict[str, int] = {"passage": 0, "tool_output": 0}

    @staticmethod
    def _fingerprint(draft: EvidenceDraft) -> str:
        basis = f"{draft.kind}|{draft.source_id or ''}|{draft.text.strip()}"
        return hashlib.sha1(basis.encode()).hexdigest()

    def add(self, drafts: list[EvidenceDraft], source_tool: str) -> list[Evidence]:
        added: list[Evidence] = []
        for draft in drafts:
            fp = self._fingerprint(draft)
            existing = self._by_fingerprint.get(fp)
            if existing is not None:
                added.append(existing)
                continue
            prefix = "E" if draft.kind == "passage" else "T"
            self._counters[draft.kind] += 1
            item = Evidence(
                evidence_id=f"{prefix}{self._counters[draft.kind]}",
                kind=draft.kind,
                source_tool=source_tool,
                text=draft.text,
                title=draft.title,
                source_id=draft.source_id,
                score=draft.score,
                metadata=draft.metadata,
            )
            self._by_fingerprint[fp] = item
            self.items.append(item)
            added.append(item)
        return added
