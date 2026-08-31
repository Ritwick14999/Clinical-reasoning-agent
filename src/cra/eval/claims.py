"""Claim extraction: v1, rule-based.

Splits a justification into sentences and filters out non-assertions --
questions, hedge-only fragments, and statements *about* the evidence rather
than about the world -- so what's left is roughly "one checkable factual claim
per sentence", the contract ``agent/prompts.py``'s ``FINAL_ANSWER_CONTRACT``
asks the model to follow.

Two normalisations exist because the agent is *instructed* to cite its
sources, and a naive entailment check punishes it for complying. Measured over
the committed traces before these were added: claims mentioning an evidence ID
were scored ``entailed`` only 14% of the time against 36% for claims without
one, and ``contradicted`` 12% against 3%. Citing a source is not evidence of
hallucination, so that gap was measurement error, not signal.

* :func:`strip_citation_framing` removes the citation wrapper before the claim
  becomes an entailment hypothesis. "E1 states that X" asks an NLI model
  whether an abstract entails a statement about what E1 says; the answer is
  structurally no, whatever the abstract contains. The proposition under test
  is X.
* Meta-evidence statements ("E2 is unrelated to resuscitation devices") are
  dropped at extraction. They carry no claim about the world, and they are
  usually the agent correctly noticing bad retrieval -- the opposite of the
  behaviour ``unsupported_claim`` is meant to catch.

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

# One or more evidence IDs: "E1", "E1 and E2", "E1, T2 and E3".
_IDS = r"[ET]\d+(?:\s*(?:,|and)\s*[ET]\d+)*"

_REPORTING_VERB = (
    r"states?|shows?|reports?|indicates?|demonstrates?|suggests?|finds?|found|notes?|"
    r"describes?|confirms?|establishes?|addresses|supports?|documents?|observes?"
)

# Leading frames: "According to E1, X" / "E1 states that X" / "Per E1 and E2: X".
_LEAD_FRAMES = [
    re.compile(
        r"^(?:according to|per|based on|as (?:stated|shown|reported|noted|described|"
        r"demonstrated|documented) in|from)\s+" + _IDS + r"\s*[,:]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:the (?:evidence|passage|abstract|study|source) in\s+)?" + _IDS
        + r"\s+(?:" + _REPORTING_VERB + r")\s+(?:that\s+)?",
        re.IGNORECASE,
    ),
]

# Trailing frames: "..., as shown in E1." / "... (E1, E2)." / "... per E3."
_TRAIL_FRAMES = [
    re.compile(
        r"\s*[,;(]?\s*(?:as\s+)?(?:" + _REPORTING_VERB + r"|stated|shown|reported|noted|"
        r"described|demonstrated|documented|supported|per|according to|see)\s*"
        r"(?:in|by)?\s*" + _IDS + r"\s*\)?\s*([.!?])?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\s*\(\s*" + _IDS + r"\s*\)\s*([.!?])?\s*$", re.IGNORECASE),
]

# Statements about the evidence itself rather than about the world.
_META_EVIDENCE_RE = re.compile(
    r"^(?:"
    r"(?:the\s+)?(?:other\s+|remaining\s+)?(?:passages?|evidence|abstracts?|sources?|studies)"
    r"|" + _IDS +
    r")\b[^.]*?\b(?:"
    r"(?:is|are|was|were)\s+(?:not\s+)?(?:un)?related|"
    r"(?:is|are)\s+irrelevant|(?:is|are)\s+not\s+relevant|(?:is|are)\s+off-?topic|"
    r"do(?:es)?\s+not\s+(?:address|discuss|mention|relate|cover|speak)|"
    r"(?:is|are)\s+about\s+|(?:focus(?:es)?|focused)\s+on\s+|(?:is|are)\s+concerned\s+with"
    r")",
    re.IGNORECASE,
)


# A residue ending in a function word means stripping cut into the sentence
# rather than off it ("As shown in E1." -> "As shown in ."), so the original
# stands.
_DANGLING_TAIL_RE = re.compile(
    r"\b(?:in|of|by|to|from|per|and|with|as|for|on|at|that|the|a|an"
    r"|shown|stated|reported|noted|described|demonstrated)\s*[.!?]?$",
    re.IGNORECASE,
)


def is_meta_evidence_statement(sentence: str) -> bool:
    """True for commentary on the evidence rather than a claim about the world."""
    return bool(_META_EVIDENCE_RE.match(sentence.strip()))


def strip_citation_framing(claim: str) -> str:
    """Reduce a cited claim to the proposition it asserts.

    ``"E1 states that aspirin is less effective."`` -> ``"aspirin is less effective."``
    Returns the claim unchanged if stripping would leave nothing checkable.
    """
    text = claim.strip()
    for pattern in _LEAD_FRAMES:
        stripped = pattern.sub("", text, count=1)
        if stripped != text and len(stripped.strip()) >= _MIN_CLAIM_CHARS:
            text = stripped.strip()
            break
    for pattern in _TRAIL_FRAMES:
        stripped = pattern.sub(lambda m: m.group(1) or "", text, count=1)
        if stripped != text and len(stripped.strip()) >= _MIN_CLAIM_CHARS:
            text = stripped.strip()
            break
    # Any surviving bare IDs are references, not content.
    text = re.sub(r"\s*\b" + _IDS + r"\b\s*", " ", text).strip()
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:")
    if len(text) < _MIN_CLAIM_CHARS or _DANGLING_TAIL_RE.search(text):
        return claim.strip()
    return text[0].upper() + text[1:] if text else claim.strip()


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
        if is_meta_evidence_statement(sentence):
            continue
        claims.append(sentence)
    return claims
