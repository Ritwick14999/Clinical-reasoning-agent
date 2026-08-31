"""Claim extraction: v1, rule-based.

Splits a justification into sentences and filters out non-assertions --
questions, hedge-only fragments -- so what's left is roughly "one checkable
factual claim per sentence", the contract ``agent/prompts.py``'s
``FINAL_ANSWER_CONTRACT`` asks the model to follow.

**v2 (LLM-based extraction) is not built.** docs/DESIGN.md Sec 7 asks for
agreement between v1 and v2 to be reported before v1 alone is trusted; that
comparison doesn't exist yet, so treat claim boundaries as an approximation,
not ground truth, same as the ``expected_tools`` oracle.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_QUESTION_RE = re.compile(r"\?\s*$")
# Sentences that carry no independently-checkable factual content: hedges,
# and bare restatements of the final answer letter/word with no reasoning.
_HEDGE_ONLY_RE = re.compile(
    r"^(it is (unclear|uncertain|not clear)\.?|"
    r"the evidence is (insufficient|limited)\.?|"
    r"i am not sure\.?|"
    r"further (research|studies?|investigation) (is|are) needed\.?|"
    r"(on (that|this) basis,?\s*)?(the|my)?\s*(best )?(answer|option) is [^.]{1,20}\.?)$",
    re.IGNORECASE,
)
_MIN_CLAIM_CHARS = 8


def extract_claims(justification: str) -> list[str]:
    """One claim per sentence, non-assertions filtered out. Empty input -> []."""
    if not justification or not justification.strip():
        return []

    sentences = _SENTENCE_SPLIT_RE.split(justification.strip())
    claims = []
    for raw in sentences:
        sentence = raw.strip()
        if len(sentence) < _MIN_CLAIM_CHARS:
            continue
        if _QUESTION_RE.search(sentence):
            continue
        if _HEDGE_ONLY_RE.match(sentence):
            continue
        claims.append(sentence)
    return claims
