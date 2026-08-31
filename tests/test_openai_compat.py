"""OpenAICompatClient._to_api: translating internal Message objects into the
OpenAI-compatible wire format.

Regression coverage for a real bug found running qwen3:8b against a live
Ollama server: an assistant turn with neither text nor a tool call (an empty
decode -- not exotic at temperature=0) produced ``content: null`` with no
``tool_calls``, which Ollama's server rejects outright once that message is
replayed as history on the next turn ("invalid message content type: <nil>").
This crashed 50 real episodes in one session before being caught.

``_to_api`` is a ``@staticmethod``, so it's tested directly with no live
client, no network, and no mocking of the ``openai`` SDK.
"""

from __future__ import annotations

from cra.llm.base import LLMResponse, Message, ToolCall
from cra.llm.openai_compat import OpenAICompatClient


def _assistant_message(resp: LLMResponse) -> Message:
    return Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls, raw=resp.raw)


def test_empty_assistant_turn_sends_empty_string_not_null():
    """The bug: a completely empty model response (no text, no tool call)."""
    msg = _assistant_message(LLMResponse(text="", tool_calls=[]))
    api_msg = OpenAICompatClient._to_api([msg])[0]
    assert api_msg["content"] == ""
    assert api_msg["content"] is not None
    assert "tool_calls" not in api_msg


def test_assistant_turn_with_text_and_no_tool_calls():
    msg = _assistant_message(LLMResponse(text="hello", tool_calls=[]))
    api_msg = OpenAICompatClient._to_api([msg])[0]
    assert api_msg["content"] == "hello"


def test_assistant_turn_with_tool_calls_uses_null_content():
    """When tool_calls carries the turn, content: null is correct and expected."""
    call = ToolCall(id="c1", name="search_literature", args={"query": "x"})
    msg = _assistant_message(LLMResponse(text="", tool_calls=[call]))
    api_msg = OpenAICompatClient._to_api([msg])[0]
    assert api_msg["content"] is None
    assert api_msg["tool_calls"][0]["function"]["name"] == "search_literature"


def test_system_and_user_messages_pass_through():
    messages = [Message(role="system", content="sys"), Message(role="user", content="usr")]
    api = OpenAICompatClient._to_api(messages)
    assert api == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_tool_result_message():
    msg = Message(role="tool", content="observation text", tool_call_id="c1")
    api_msg = OpenAICompatClient._to_api([msg])[0]
    assert api_msg == {"role": "tool", "tool_call_id": "c1", "content": "observation text"}
