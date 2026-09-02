"""LLM-as-judge entailment checker.

Scores each claim against its evidence using the rubric in ``rubric.md`` and
any :class:`~cra.llm.base.LLMClient` already in the codebase -- the mock, for
tests, or a real Ollama model for a live pass.

**Not part of the default, "pure function of traces" eval path.**
docs/CONVENTIONS.md's rollout/eval separation exists specifically so eval can be
improved and re-run at zero cost; a checker that calls a model on every run
breaks that. So this class is invoked as a separate, explicitly-labelled
step (``cra judge-check``), never by ``cra eval``, which uses
:mod:`cra.eval.entailment.nli` instead -- the actual zero-network,
zero-marginal-cost checker docs/DESIGN.md Sec 7 calls for as the primary
instrument.

**Judge-circularity, and how this project sidesteps it.** docs/DESIGN.md
flags that a judge from the same model family as the agent it's grading is a
weaker check (self-grading dressed up as validation). Rather than accept that
weakness, or reach for a third paid model, :func:`cra.eval.run.cross_judge_traces`
has each of this project's two evaluated models judge *only the other's*
traces -- qwen3 never judges qwen3. Neither judge is fully independent of the
population of agents being studied, but "graded by a different model" is a
materially different, and separately reportable, failure mode than "graded
by itself."
"""

from __future__ import annotations

from pathlib import Path

from cra.eval.entailment.base import EntailmentLabel
from cra.llm.base import LLMClient, system, user

RUBRIC_PATH = Path(__file__).parent / "rubric.md"
_VALID_LABELS: set[EntailmentLabel] = {"entailed", "contradicted", "not_addressed"}


def _load_rubric() -> str:
    return RUBRIC_PATH.read_text(encoding="utf-8")


class LLMJudge:
    def __init__(
        self,
        llm: LLMClient,
        name: str | None = None,
        max_evidence_chars: int = 6000,
        max_tokens: int = 2048,
    ) -> None:
        self.llm = llm
        self.name = name or f"judge:{llm.model_id}"
        self.max_evidence_chars = max_evidence_chars
        # Generous even though the rubric asks for one word: a "thinking"
        # model (qwen3 in particular) spends tokens on internal reasoning
        # before any visible output -- measured at ~600 tokens for a single
        # claim/evidence pair through Ollama's OpenAI-compatible endpoint,
        # which does not expose that reasoning separately (`resp.thinking`
        # stays empty; it's silently part of the token budget). A tight
        # budget (300 was still not enough) cuts generation off before the
        # label is ever reached, and every call then falls back to
        # "not_addressed" -- which produced a spuriously ~100% hallucination
        # rate on a real sample, caught only by manually inspecting raw
        # judge output against an implausible headline number.
        self.max_tokens = max_tokens
        self._rubric = _load_rubric()

    def check(self, claim: str, evidence_texts: list[str]) -> EntailmentLabel:
        if not evidence_texts:
            return "not_addressed"

        evidence_blob = "\n\n---\n\n".join(evidence_texts)[: self.max_evidence_chars]
        prompt = f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_blob}"
        resp = self.llm.chat(
            [system(self._rubric), user(prompt)],
            tools=[],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return self._parse_label(resp.text)

    @staticmethod
    def _parse_label(raw_text: str) -> EntailmentLabel:
        text = (raw_text or "").strip().lower()
        stripped = text.strip(". \n\"'")
        if stripped in _VALID_LABELS:
            return stripped  # type: ignore[return-value]
        # A model that didn't follow the "one word only" instruction exactly
        # (common even outside the thinking-token issue above) may still
        # state the label inside a short sentence -- search for it rather
        # than requiring an exact match, before falling back to the
        # conservative "not proven" default.
        for candidate in _VALID_LABELS:
            if candidate in text:
                return candidate  # type: ignore[return-value]
        return "not_addressed"
