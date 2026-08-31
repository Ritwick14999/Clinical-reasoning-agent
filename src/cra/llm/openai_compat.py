"""OpenAI-compatible adapter.

One adapter, three backends: OpenAI itself, a local vLLM server, and Ollama --
all of which speak the same chat-completions schema. That is how the
open-weight half of the model comparison runs without a second code path.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from cra.llm.base import LLMResponse, Message, ToolCall, ToolSpec
from cra.llm.cache import ResponseCache, cache_key
from cra.types import Usage


class OpenAICompatClient:
    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str = "openai_compat",
        cache: ResponseCache | None = None,
        max_retries: int = 4,
        timeout: float = 180.0,
    ) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The openai package is required for this provider. "
                "Install the optional extra: pip install -e '.[api]'"
            ) from exc

        self.model_id = model_id
        self.provider = provider
        self.cache = cache
        self.max_retries = max_retries
        # Local servers ignore the key but the SDK still insists on one.
        key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed-for-local-server"
        self._client = openai.OpenAI(
            api_key=key,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout,
            max_retries=0,
        )

    @staticmethod
    def _to_api(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role in ("system", "user"):
                out.append({"role": m.role, "content": m.content})
            elif m.role == "assistant":
                # `content: null` is only valid when tool_calls carries the
                # turn instead. A model that returns neither text nor a tool
                # call (which happens; an empty decode at temperature=0 is
                # not exotic) would otherwise produce content=None with no
                # tool_calls -- Ollama rejects that as "invalid message
                # content type: <nil>" on the *next* turn, once this message
                # is replayed as history, crashing an episode that had
                # already survived the actual empty response via the repair
                # round. Empty string is a valid, accepted content instead.
                content = None if m.tool_calls else (m.content or "")
                msg: dict[str, Any] = {"role": "assistant", "content": content}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                        }
                        for tc in m.tool_calls
                    ]
                out.append(msg)
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        return out

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

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._to_api(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

        started = time.perf_counter()
        raw = self._call(payload)
        latency_ms = (time.perf_counter() - started) * 1000
        choice = raw.choices[0].message

        tool_calls = []
        for tc in choice.tool_calls or []:
            # Open-weight models emit malformed JSON arguments often enough that
            # this must be recorded rather than raised: a broken call is data
            # for the tool-misuse analysis, not a crash.
            try:
                args = json.loads(tc.function.arguments or "{}")
                err = None
            except json.JSONDecodeError as exc:
                args, err = {}, f"unparseable arguments: {exc}"
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, args=args, parse_error=err)
            )

        usage = getattr(raw, "usage", None)
        resp = LLMResponse(
            text=(choice.content or "").strip(),
            thinking=getattr(choice, "reasoning_content", None),
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
            stop_reason=raw.choices[0].finish_reason,
        )
        if self.cache:
            self.cache.put(key, resp)
        return resp

    def _call(self, payload: dict[str, Any]):
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=32),
            reraise=True,
        )
        def _inner():
            return self._client.chat.completions.create(**payload)

        return _inner()
