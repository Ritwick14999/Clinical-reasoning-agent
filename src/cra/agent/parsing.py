"""Parsing the model's final answer.

Deliberately forgiving about packaging (code fences, stray prose) and strict
about content. A model that cannot produce the contract is exhibiting a real
failure that the trace must record, so parse failures return an explanatory
message rather than guessing.
"""

from __future__ import annotations

import json
import re

from cra.types import FinalAnswer, normalize_answer

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_EVIDENCE_ID = re.compile(r"\b([ET]\d+)\b")


def _candidate_blobs(text: str) -> list[str]:
    """JSON candidates, most-likely first."""
    stripped = text.strip()
    out = [stripped]
    out += [m.group(1).strip() for m in _FENCE.finditer(text)]
    # Balanced-brace scan: tolerates prose on either side of the object.
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(text[start : i + 1])
    seen, unique = set(), []
    for blob in out:
        if blob and blob not in seen:
            seen.add(blob)
            unique.append(blob)
    return unique


def parse_final(text: str, allowed: list[str] | None = None) -> tuple[FinalAnswer | None, str]:
    """Return ``(final, "")`` or ``(None, reason)``."""
    if not text or not text.strip():
        return None, "the reply was empty"

    obj = None
    for blob in _candidate_blobs(text):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            obj = parsed
            break
    if obj is None:
        return None, "no JSON object with an 'answer' field was found in the reply"

    raw_answer = obj.get("answer")
    if raw_answer is None or not str(raw_answer).strip():
        return None, "'answer' was empty"
    answer = str(raw_answer).strip()

    if allowed:
        normalized = normalize_answer(answer)
        allowed_norm = {normalize_answer(a): a for a in allowed}
        if normalized not in allowed_norm:
            return None, (
                f"'answer' was {answer!r}, which is not one of the allowed answers "
                f"({', '.join(allowed)})"
            )
        answer = allowed_norm[normalized]

    justification = str(obj.get("justification") or "").strip()

    citations = obj.get("citations", [])
    if isinstance(citations, str):
        citations = _EVIDENCE_ID.findall(citations)
    elif isinstance(citations, list):
        citations = [str(c).strip() for c in citations if str(c).strip()]
    else:
        citations = []
    # Keep only well-formed IDs, de-duplicated, order preserved.
    clean, seen = [], set()
    for c in citations:
        match = _EVIDENCE_ID.search(c)
        cid = match.group(1) if match else None
        if cid and cid not in seen:
            seen.add(cid)
            clean.append(cid)

    return FinalAnswer(answer=answer, justification=justification, citations=clean, raw=text), ""
