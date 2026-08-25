"""Content-addressed on-disk response cache.

Two jobs. First, reruns of a rollout are free and byte-identical, which is what
makes an expensive stochastic pipeline auditable. Second, a cached run is a
recorded fixture: an adapter that talks to a live API can be exercised offline.

The key covers everything that could change a response -- provider, model,
decoding parameters, the full message list, and the advertised tool schemas.
Miss the tool schemas and a cache hit could return a call to a tool that is no
longer exposed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from cra.llm.base import LLMResponse, Message, ToolCall, ToolSpec
from cra.types import Usage


def cache_key(
    provider: str,
    model_id: str,
    messages: list[Message],
    tools: list[ToolSpec] | None,
    temperature: float,
    max_tokens: int,
    seed: int | None,
) -> str:
    payload = {
        "provider": provider,
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tool_call_id": m.tool_call_id,
                "tool_calls": [{"name": tc.name, "args": tc.args} for tc in m.tool_calls],
            }
            for m in messages
        ],
        "tools": [asdict(t) for t in (tools or [])],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


class ResponseCache:
    def __init__(self, root: str | Path = ".cache/llm", enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Shard by prefix: flat directories with 100k entries are painful to work with.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        d = json.loads(path.read_text())
        return LLMResponse(
            text=d["text"],
            thinking=d.get("thinking"),
            tool_calls=[ToolCall(**tc) for tc in d.get("tool_calls", [])],
            usage=Usage(**d.get("usage", {})),
            latency_ms=d.get("latency_ms", 0.0),
            stop_reason=d.get("stop_reason"),
        )

    def put(self, key: str, resp: LLMResponse) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": resp.text,
            "thinking": resp.thinking,
            "tool_calls": [asdict(tc) for tc in resp.tool_calls],
            "usage": resp.usage.model_dump(),
            "latency_ms": resp.latency_ms,
            "stop_reason": resp.stop_reason,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=0))
        tmp.replace(path)  # atomic: a killed run must not leave a truncated entry
