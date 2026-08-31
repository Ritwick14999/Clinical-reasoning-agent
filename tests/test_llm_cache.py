"""ResponseCache: content-addressing and Windows-safe encoding.

Regression coverage for a real bug: ``Path.write_text``/``read_text`` default
to the locale encoding, which is cp1252 on Windows and cannot encode ordinary
model output (en dashes, curly quotes). Without an explicit ``encoding="utf-8"``,
caching a real response crashes *after* the expensive provider call already
happened, silently losing 69 real Ollama episodes in one session before this
was caught and fixed.
"""

from __future__ import annotations

from cra.llm.base import LLMResponse
from cra.llm.cache import ResponseCache
from cra.types import Usage


def test_cache_roundtrips_non_ascii_content(tmp_path):
    cache = ResponseCache(tmp_path)
    key = "k1"
    resp = LLMResponse(
        text="The study found no difference – “not significant” overall.",
        thinking="Weighing ± uncertainty in the results → leaning toward 'no'.",
        usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason="end_turn",
    )
    cache.put(key, resp)  # must not raise UnicodeEncodeError on Windows

    hit = cache.get(key)
    assert hit is not None
    assert hit.text == resp.text
    assert hit.thinking == resp.thinking


def test_cache_miss_then_hit():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cache = ResponseCache(d)
        assert cache.get("missing") is None
        assert cache.misses == 1

        resp = LLMResponse(text="ok")
        cache.put("present", resp)
        assert cache.get("present") is not None
        assert cache.hits == 1


def test_cache_disabled_never_touches_disk(tmp_path):
    cache = ResponseCache(tmp_path / "unused", enabled=False)
    cache.put("k", LLMResponse(text="x"))
    assert cache.get("k") is None
    assert not (tmp_path / "unused").exists()
