"""Rule-based ``expected_tools`` oracle.

Detects, from the raw question text alone, whether a case supplies the inputs
a specific tool needs -- so the ``tool_misuse`` label (R2 in the failure-mode
taxonomy, ``docs/DESIGN.md`` Sec 6) has something to compare the agent's
actual tool use against.

This is deliberately the weakest link in the taxonomy: a keyword rule cannot
capture clinical judgement, and both false positives (inputs present but not
actually decision-relevant) and false negatives (paraphrased inputs the regex
misses) are expected. Hand-audit it on a sample and report its precision and
recall as a stated limitation, per ``docs/HANDOFF.md`` Sec 2a -- never present
this as ground truth.
"""

from __future__ import annotations

import re

from cra.tools.drugs import BRAND_TO_GENERIC, DRUG_CLASSES, INTERACTIONS

_AF_RE = re.compile(r"\batrial fibrillation\b|\bnon-?valvular af\b", re.IGNORECASE)
_PE_RE = re.compile(r"\bpulmonary embol", re.IGNORECASE)
_DVT_RE = re.compile(r"\bdeep vein thrombosis\b|\bdvt\b", re.IGNORECASE)
_DYSPNEA_RE = re.compile(
    r"\bdyspnea\b|\bdyspnoea\b|\bshortness of breath\b|\btachypnea\b", re.IGNORECASE
)
_LIVER_RE = re.compile(
    r"\bcirrhosis\b|\bliver (disease|failure|transplant)\b|\bhepatic\b", re.IGNORECASE
)
_BILIRUBIN_RE = re.compile(r"\bbilirubin\b", re.IGNORECASE)
_CREATININE_RE = re.compile(r"\bcreatinine\b", re.IGNORECASE)
_SODIUM_RE = re.compile(r"\bsodium\b|\bna\+?\s*[:=]?\s*\d", re.IGNORECASE)
_CHLORIDE_RE = re.compile(r"\bchloride\b", re.IGNORECASE)
_BICARB_RE = re.compile(r"\bbicarbonate\b|\bco2\b|\banion gap\b", re.IGNORECASE)

# Every drug name the interaction table can recognise, generic or brand,
# mapped to its canonical (generic) form so two mentions of the same drug
# under different names still count as one.
_DRUG_VOCAB: dict[str, str] = {}
for _members in DRUG_CLASSES.values():
    for _name in _members:
        _DRUG_VOCAB.setdefault(_name, _name)
for _brand, _generic in BRAND_TO_GENERIC.items():
    _DRUG_VOCAB.setdefault(_brand, _generic)
for _rule in INTERACTIONS:
    for _side in (_rule["a"], _rule["b"]):
        if not _side.startswith("@"):
            _DRUG_VOCAB.setdefault(_side, _side)

_DRUG_PATTERNS = {
    name: re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE) for name in _DRUG_VOCAB
}


def _mentioned_drugs(text: str) -> set[str]:
    return {generic for name, generic in _DRUG_VOCAB.items() if _DRUG_PATTERNS[name].search(text)}


# Retrieval is instructed unconditionally by the system prompt ("search the
# literature before making a factual claim you cannot support from the question
# itself"), so calling it is never evidence of misuse. It is excluded from the
# "unnecessary call" test and from the tool-precision denominator; otherwise the
# agent is penalised for following its own instructions.
ALWAYS_PERMITTED = frozenset({"search_literature"})


def expected_tools_for(text: str, dataset: str | None = None) -> list[str]:
    """Deterministic oracle. See module docstring.

    ``dataset`` adds dataset-level expectations that keyword rules cannot see.
    PubMedQA asks whether the published literature supports a claim, and ships
    a specific gold abstract, so answering it without retrieving anything is a
    tool failure by construction. Without this, 299 of 300 PubMedQA questions
    carried no expectation at all and the tool metrics there were vacuous.
    """
    tools: list[str] = []
    if dataset == "pubmedqa":
        tools.append("search_literature")

    if _AF_RE.search(text):
        tools.append("calc_cha2ds2_vasc")

    if _PE_RE.search(text) or (_DVT_RE.search(text) and _DYSPNEA_RE.search(text)):
        tools.append("calc_wells_pe")

    if _LIVER_RE.search(text) and _BILIRUBIN_RE.search(text) and _CREATININE_RE.search(text):
        tools.append("calc_meld")

    if _SODIUM_RE.search(text) and _CHLORIDE_RE.search(text) and _BICARB_RE.search(text):
        tools.append("calc_anion_gap")

    if len(_mentioned_drugs(text)) >= 2:
        tools.append("check_drug_interactions")

    return tools


def resolve_expected_tools(question) -> list[str]:
    """The oracle's verdict for a question, recomputed at evaluation time.

    Traces record whatever the oracle said when the rollout ran, but the oracle
    is a deterministic annotation over the question text, not something the
    model produced. Recomputing it here means an oracle fix applies to every
    committed trace immediately, with no re-rollout -- the same property that
    lets the rest of the evaluation be improved and re-run for free.

    This does not violate the "no model calls in eval" rule: it is a keyword
    regex over text already inlined in the trace, not a re-derivation of
    anything the model did.
    """
    return expected_tools_for(question.question, dataset=question.dataset)
