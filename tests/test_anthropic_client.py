"""Anthropic adapter message translation.

The adapter cannot be exercised against the live API here, but the part most
likely to be wrong is pure: turning this project's provider-neutral messages
into the shape the Messages API requires. Two of its rules are easy to get
wrong and fail only at request time, so they are pinned here.
"""

from __future__ import annotations

import sys
import types

import pytest

from cra.llm.base import LLMResponse, Message, ToolCall, assistant, system, tool_result, user


@pytest.fixture
def to_api(monkeypatch):
    """The translation function, importable without the anthropic package."""
    if "anthropic" not in sys.modules:
        stub = types.ModuleType("anthropic")
        stub.Anthropic = object
        stub.RateLimitError = type("RateLimitError", (Exception,), {})
        stub.APIConnectionError = type("APIConnectionError", (Exception,), {})
        stub.InternalServerError = type("InternalServerError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "anthropic", stub)
    from cra.llm.anthropic_client import AnthropicClient

    return AnthropicClient._to_api


def test_system_prompt_is_split_out_of_the_message_list(to_api):
    """The Messages API takes the system prompt as a top-level parameter, not a turn."""
    prompt, messages = to_api([system("be careful"), user("a question")])
    assert prompt == "be careful"
    assert [m["role"] for m in messages] == ["user"]


def test_multiple_system_messages_are_joined(to_api):
    prompt, _ = to_api([system("first"), system("second"), user("q")])
    assert prompt == "first\n\nsecond"


def test_tool_calls_become_tool_use_blocks(to_api):
    resp = LLMResponse(text="thinking out loud", tool_calls=[ToolCall("c1", "search", {"q": "x"})])
    _, messages = to_api([user("q"), assistant(resp)])
    blocks = messages[-1]["content"]
    assert [b["type"] for b in blocks] == ["text", "tool_use"]
    assert blocks[1] == {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "x"}}


def test_an_assistant_turn_is_never_empty(to_api):
    """The API rejects an assistant turn with no content blocks at all."""
    _, messages = to_api([user("q"), assistant(LLMResponse(text=""))])
    assert messages[-1]["content"], "an empty block list would be rejected at request time"


def test_consecutive_tool_results_merge_into_one_user_turn(to_api):
    """Parallel tool calls return several results, and the API requires them in a
    single user turn rather than one turn each."""
    resp = LLMResponse(
        text="",
        tool_calls=[ToolCall("c1", "search", {}), ToolCall("c2", "calc", {})],
    )
    _, messages = to_api(
        [user("q"), assistant(resp), tool_result("c1", "first"), tool_result("c2", "second")]
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    blocks = messages[-1]["content"]
    assert [b["tool_use_id"] for b in blocks] == ["c1", "c2"]


def test_a_tool_result_after_a_plain_user_turn_starts_its_own_turn(to_api):
    """Merging into a plain-text user turn would corrupt it: that turn's content
    is a string, not a block list."""
    _, messages = to_api([user("plain text"), tool_result("c1", "result")])
    assert messages[0]["content"] == "plain text"
    assert isinstance(messages[1]["content"], list)


def test_every_tool_use_has_a_matching_tool_result(to_api):
    """The API rejects a turn whose tool_use blocks are unanswered, which is why
    a budget-refused call still emits a result."""
    resp = LLMResponse(text="", tool_calls=[ToolCall("c1", "search", {})])
    _, messages = to_api([user("q"), assistant(resp), tool_result("c1", "refused")])
    emitted = {
        b["id"] for m in messages if isinstance(m["content"], list)
        for b in m["content"] if b.get("type") == "tool_use"
    }
    answered = {
        b["tool_use_id"] for m in messages if isinstance(m["content"], list)
        for b in m["content"] if b.get("type") == "tool_result"
    }
    assert emitted == answered == {"c1"}


def test_missing_credentials_fail_with_an_actionable_message(monkeypatch):
    stub = types.ModuleType("anthropic")
    stub.Anthropic = object
    stub.RateLimitError = type("RateLimitError", (Exception,), {})
    stub.APIConnectionError = type("APIConnectionError", (Exception,), {})
    stub.InternalServerError = type("InternalServerError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", stub)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from cra.llm.anthropic_client import AnthropicClient

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(api_key=None)


def test_round_trip_preserves_turn_order(to_api):
    """A multi-step episode must reach the API in the order it happened."""
    messages: list[Message] = [system("s"), user("q1")]
    messages.append(assistant(LLMResponse(text="", tool_calls=[ToolCall("c1", "search", {})])))
    messages.append(tool_result("c1", "evidence"))
    messages.append(assistant(LLMResponse(text="final")))
    _, api = to_api(messages)
    assert [m["role"] for m in api] == ["user", "assistant", "user", "assistant"]
