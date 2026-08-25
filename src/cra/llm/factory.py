"""Build an LLM client from a :class:`ModelConfig`."""

from __future__ import annotations

import os

from cra.config import ModelConfig
from cra.llm.base import LLMClient
from cra.llm.cache import ResponseCache


def build_llm(cfg: ModelConfig) -> LLMClient:
    cache = ResponseCache(cfg.cache_dir, enabled=cfg.use_cache) if cfg.use_cache else None

    if cfg.provider == "mock":
        from cra.llm.mock import HeuristicMockClient

        return HeuristicMockClient(model_id=cfg.model_id, **cfg.mock)

    if cfg.provider == "anthropic":
        from cra.llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            model_id=cfg.model_id,
            api_key=os.environ.get(cfg.api_key_env or "ANTHROPIC_API_KEY"),
            cache=cache,
        )

    if cfg.provider == "openai_compat":
        from cra.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient(
            model_id=cfg.model_id,
            base_url=cfg.base_url,
            api_key=os.environ.get(cfg.api_key_env) if cfg.api_key_env else None,
            provider=cfg.provider,
            cache=cache,
        )

    raise ValueError(
        f"unknown provider {cfg.provider!r}; expected one of: mock, anthropic, openai_compat"
    )
