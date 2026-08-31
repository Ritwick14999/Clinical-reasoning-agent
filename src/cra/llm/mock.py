"""Deterministic clients that need no credentials and no network.

Two of them, for two different jobs:

``ScriptedClient``
    Exact turn-by-turn control. Used by unit tests to drive the agent loop
    down specific branches (budget exhaustion, malformed final answer, tool
    errors) without any inference.

``HeuristicMockClient``
    A deterministic pseudo-agent that calls a tool and then answers. It powers
    the credential-free demo and the end-to-end smoke tests. It is *not* a
    model: numbers produced from it are never reported as results.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable

from cra.llm.base import LLMResponse, Message, ToolCall, ToolSpec
from cra.types import Usage


def _seeded_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


class ScriptedClient:
    """Replays a fixed list of responses, in order."""

    def __init__(
        self,
        responses: Iterable[LLMResponse],
        model_id: str = "mock-scripted",
        provider: str = "mock",
    ) -> None:
        self._responses = list(responses)
        self._i = 0
        self.model_id = model_id
        self.provider = provider
        self.calls: list[list[Message]] = []
        self.exposed_tools: list[list[str]] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.exposed_tools.append([t.name for t in (tools or [])])
        if self._i >= len(self._responses):
            raise AssertionError(
                f"ScriptedClient exhausted after {len(self._responses)} responses; "
                "the loop asked for another turn than the test scripted."
            )
        resp = self._responses[self._i]
        self._i += 1
        return resp


class HeuristicMockClient:
    """A deterministic stand-in agent.

    Behaviour is a pure function of (seed, question text), so traces are
    reproducible. It knows nothing about gold answers -- it picks an option
    pseudo-randomly -- which keeps it honest as a demo backend.
    """

    def __init__(
        self,
        model_id: str = "mock-heuristic",
        provider: str = "mock",
        use_tools: bool = True,
        n_tool_calls: int = 1,
        unsupported_claim: bool = False,
    ) -> None:
        self.model_id = model_id
        self.provider = provider
        self.use_tools = use_tools
        self.n_tool_calls = n_tool_calls
        self.unsupported_claim = unsupported_claim

    @staticmethod
    def _question_text(messages: list[Message]) -> str:
        for m in messages:
            if m.role == "user":
                return m.content
        return ""

    @staticmethod
    def _options(question: str) -> list[str]:
        """The allowed answers for this episode.

        Reads the "Allowed answers: ..." line ``prompts.render_question``
        always appends, rather than scanning for lettered option lines --
        PubMedQA questions have no options and would otherwise fall through
        to a hardcoded ``["A"]``, which is not a valid yes/no/maybe answer.
        """
        match = re.search(r"^Allowed answers:\s*(.+)$", question, flags=re.MULTILINE)
        if match:
            return [a.strip() for a in match.group(1).split(",") if a.strip()]
        return re.findall(r"^\s*([A-E])[).:]\s", question, flags=re.MULTILINE) or ["A"]

    @staticmethod
    def _seen_evidence_ids(messages: list[Message]) -> list[str]:
        ids: list[str] = []
        for m in messages:
            if m.role == "tool":
                ids.extend(re.findall(r"\[([ET]\d+)\]", m.content))
        return ids

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> LLMResponse:
        question = self._question_text(messages)
        rng = _seeded_rng(seed, question, self.model_id)
        n_prior_tool_turns = sum(1 for m in messages if m.role == "tool")
        tools = tools or []

        if self.use_tools and tools and n_prior_tool_turns < self.n_tool_calls:
            spec = tools[0]
            args = self._synth_args(spec, question)
            return LLMResponse(
                text="",
                thinking=f"Consulting {spec.name} before answering.",
                tool_calls=[ToolCall(id=f"call_{n_prior_tool_turns}", name=spec.name, args=args)],
                usage=Usage(input_tokens=len(question) // 4, output_tokens=24),
                stop_reason="tool_use",
            )

        options = self._options(question)
        pick = rng.choice(options)
        cited = self._seen_evidence_ids(messages)[:2]
        sentences = []
        if cited:
            sentences.append(f"The retrieved evidence in {cited[0]} addresses this question.")
        else:
            sentences.append("No external evidence was consulted for this answer.")
        if self.unsupported_claim:
            sentences.append(
                "Randomised trials have established this as the standard of care in all settings."
            )
        sentences.append(f"On that basis the best option is {pick}.")
        final = {
            "answer": pick,
            "justification": " ".join(sentences),
            "citations": cited,
        }
        return LLMResponse(
            text=json.dumps(final),
            thinking="Weighing the options against what was retrieved.",
            usage=Usage(input_tokens=len(question) // 4, output_tokens=64),
            stop_reason="end_turn",
        )

    @staticmethod
    def _synth_args(spec: ToolSpec, question: str) -> dict:
        """Fill the tool's required fields with something schema-valid."""
        props = spec.input_schema.get("properties", {})
        required = spec.input_schema.get("required", list(props))
        args: dict = {}
        for name in required:
            schema = props.get(name, {})
            kind = schema.get("type", "string")
            if kind == "string":
                args[name] = " ".join(question.split()[:12]) if "quer" in name else "unknown"
            elif kind in ("integer", "number"):
                args[name] = schema.get("default", 1)
            elif kind == "boolean":
                args[name] = False
            elif kind == "array":
                args[name] = []
            else:
                args[name] = None
        return args
