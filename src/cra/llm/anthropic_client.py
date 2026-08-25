"""Anthropic adapter (native tool use).

Not exercised in the build environment, which has no credentials, so it is
written defensively and covered by tests that replay recorded fixtures through
the cache rather than by live calls.
"""

from __future__ import annotations

import os
import time
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cra.llm.base import LLMResponse, Message, ToolCall, ToolSpec
from cra.llm.cache import ResponseCache, cache_key
from cra.types import Usage


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        model_id: str = "claude-sonnet-5",
        api_key: str | None = None,
        cache: ResponseCache | None = None,
        max_retries: int = 4,
        timeout: float = 120.0,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The anthropic package is required for this provider. "
                "Install the optional extra: pip install -e '.[api]'"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or run with a mock model config "
                "(configs/model/mock.yaml) which needs no credentials."
            )
        self.model_id = model_id
        self.cache = cache
        self.max_retries = max_retries
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=0)
        self._exc = anthropic

    # -- message translation ------------------------------------------------

    @staticmethod
    def _to_api(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Split off the system prompt and fold tool results into user turns."""
        system_prompt = "\n\n".join(m.content for m in messages if m.role == "system")
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args}
                    )
                # An assistant turn with no blocks at all is rejected by the API.
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "."}]})
            elif m.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
                # Consecutive tool results must be merged into one user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return system_prompt, out

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> LLMResponse:
        key = cache_key(self.provider, self.model_id, messages, tools, temperature, max_tokens, seed)
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        system_prompt, api_messages = self._to_api(messages)
        payload: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]

        started = time.perf_counter()
        raw = self._call(payload)
        latency_ms = (time.perf_counter() - started) * 1000

        text_parts, thinking_parts, tool_calls = [], [], []
        for block in raw.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", ""))
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input)))

        resp = LLMResponse(
            text="\n".join(text_parts).strip(),
            thinking="\n".join(thinking_parts).strip() or None,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=raw.usage.input_tokens, output_tokens=raw.usage.output_tokens
            ),
            latency_ms=latency_ms,
            stop_reason=raw.stop_reason,
        )
        if self.cache:
            self.cache.put(key, resp)
        return resp

    def _call(self, payload: dict[str, Any]):
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=32),
            retry=retry_if_exception_type(
                (self._exc.RateLimitError, self._exc.APIConnectionError, self._exc.InternalServerError)
            ),
            reraise=True,
        )
        def _inner():
            return self._client.messages.create(**payload)

        return _inner()
