"""Provider-agnostic LLM interface.

Everything upstream of this module (the agent loop, the tools, the evaluator)
is written against :class:`LLMClient` and never imports a vendor SDK. That is
what lets the same code path drive an API model, a locally served open-weight
model, and a deterministic mock with no network at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cra.types import Usage


@dataclass(frozen=True)
class ToolSpec:
    """A tool as advertised to the model. Vendor-neutral; adapters translate."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    # Set when the model emitted something that could not be parsed into args.
    parse_error: str | None = None


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    # Retained verbatim so adapters can replay provider-native turn structure.
    raw: Any = None


@dataclass
class LLMResponse:
    text: str = ""
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    stop_reason: str | None = None
    raw: Any = None


@runtime_checkable
class LLMClient(Protocol):
    """The whole provider surface the agent depends on."""

    model_id: str
    provider: str

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> LLMResponse: ...


def system(content: str) -> Message:
    return Message(role="system", content=content)


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(resp: LLMResponse) -> Message:
    return Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls, raw=resp.raw)


def tool_result(tool_call_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=tool_call_id)
