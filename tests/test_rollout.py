"""Rollout runner: preflight check, question loading, and resumability.

Fully offline: uses the mock LLM client, the committed trap set (no download
needed), and retrieval backend "none" -- no network, no Ollama server, no
corpus build required.
"""

from __future__ import annotations

import json

import pytest

import cra.rollout as rollout
from cra.config import AgentConfig, ExperimentConfig, ModelConfig, RetrievalConfig, load_experiment
from cra.data.trapset import load_trapset
from cra.rollout import (
    DEFAULT_CONTEXT_LENGTH,
    _ollama_num_ctx,
    build_retriever,
    load_questions,
    preflight_check,
    resolve_context_length,
    run_rollout,
)
from cra.tools.registry import default_registry
from cra.trace_io import read_trace_dir, write_traces
from cra.types import Trace


def _cfg(
    provider: str = "mock",
    base_url: str | None = None,
    tool_budget: int = 3,
    default_k: int = 3,
    max_tokens: int = 512,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(provider=provider, model_id="m", base_url=base_url, max_tokens=max_tokens, use_cache=False),
        agent=AgentConfig(tool_budget=tool_budget),
        retrieval=RetrievalConfig(backend="none", default_k=default_k),
        datasets=["trapset"],
        split="dev",
    )


# --------------------------------------------------------------------------
# resolve_context_length / _ollama_num_ctx
# --------------------------------------------------------------------------

def test_resolve_context_length_not_applicable_for_mock():
    assert resolve_context_length(_cfg(provider="mock")) == (0, "not_applicable")


def test_resolve_context_length_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "4096")
    cfg = _cfg(provider="openai_compat", base_url="http://localhost:11434/v1")
    assert resolve_context_length(cfg) == (4096, "OLLAMA_CONTEXT_LENGTH")


def test_resolve_context_length_queries_ollama(monkeypatch):
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)
    monkeypatch.setattr(rollout, "_ollama_num_ctx", lambda base_url, model_id: 16384)
    cfg = _cfg(provider="openai_compat", base_url="http://localhost:11434/v1")
    assert resolve_context_length(cfg) == (16384, "ollama_api_show")


def test_resolve_context_length_falls_back_to_default(monkeypatch, capsys):
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)
    monkeypatch.setattr(rollout, "_ollama_num_ctx", lambda base_url, model_id: None)
    cfg = _cfg(provider="openai_compat", base_url="http://localhost:11434/v1")
    ctx, source = resolve_context_length(cfg)
    assert ctx == DEFAULT_CONTEXT_LENGTH
    assert source == "default_assumed"
    assert "WARNING" in capsys.readouterr().out


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._payload


def test_ollama_num_ctx_parses_parameters_field(monkeypatch):
    monkeypatch.setattr(
        rollout.urllib.request,
        "urlopen",
        lambda req, timeout=10: _FakeResponse({"parameters": "num_ctx    8192\nstop    \"<eos>\""}),
    )
    assert _ollama_num_ctx("http://localhost:11434/v1", "qwen3:8b") == 8192


def test_ollama_num_ctx_parses_model_info_field(monkeypatch):
    monkeypatch.setattr(
        rollout.urllib.request,
        "urlopen",
        lambda req, timeout=10: _FakeResponse({"model_info": {"qwen3.context_length": 32768}}),
    )
    assert _ollama_num_ctx("http://localhost:11434/v1", "qwen3:8b") == 32768


def test_ollama_num_ctx_returns_none_on_connection_error(monkeypatch):
    def _raise(req, timeout=10):
        raise OSError("connection refused")

    monkeypatch.setattr(rollout.urllib.request, "urlopen", _raise)
    assert _ollama_num_ctx("http://localhost:11434/v1", "qwen3:8b") is None


# --------------------------------------------------------------------------
# preflight_check
# --------------------------------------------------------------------------

def test_preflight_check_always_passes_for_mock():
    registry = default_registry(retriever=None)
    result = preflight_check(_cfg(provider="mock"), registry, load_trapset()[:3])
    assert result.context_length == 0
    assert result.ok


def test_preflight_check_raises_on_overflow(monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "512")
    cfg = _cfg(provider="openai_compat", base_url="http://localhost:11434/v1", tool_budget=10, default_k=5)
    registry = default_registry(retriever=None)
    with pytest.raises(RuntimeError, match="Preflight check failed"):
        preflight_check(cfg, registry, load_trapset()[:3])


def test_preflight_check_passes_with_generous_context(monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "1000000")
    cfg = _cfg(provider="openai_compat", base_url="http://localhost:11434/v1")
    registry = default_registry(retriever=None)
    result = preflight_check(cfg, registry, load_trapset()[:3])
    assert result.ok
    assert not result.worst_case_risky


def test_preflight_check_warns_on_worst_case_risk(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "3000")
    cfg = _cfg(
        provider="openai_compat", base_url="http://localhost:11434/v1",
        tool_budget=3, default_k=1, max_tokens=100,
    )
    registry = default_registry(retriever=None)
    result = preflight_check(cfg, registry, load_trapset()[:3])
    assert result.ok  # the *expected* case (default_k) still fits
    assert result.worst_case_risky  # but a max-k model could overflow
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------
# load_questions / build_retriever
# --------------------------------------------------------------------------

def test_load_questions_applies_per_dataset_limit():
    cfg = ExperimentConfig(
        model=ModelConfig(provider="mock"),
        agent=AgentConfig(),
        retrieval=RetrievalConfig(backend="none"),
        datasets=["trapset"],
        split="dev",
        limit=5,
    )
    questions = load_questions(cfg)
    assert len(questions) == 5
    assert all(q.dataset == "trapset" for q in questions)


def test_build_retriever_none_backend_returns_none():
    cfg = ExperimentConfig(retrieval=RetrievalConfig(backend="none"))
    assert build_retriever(cfg) is None


# --------------------------------------------------------------------------
# run_rollout: end-to-end, offline
# --------------------------------------------------------------------------

_OFFLINE_EXPERIMENT = """
experiment_id: test_offline
model:
  provider: mock
  model_id: mock-heuristic
  use_cache: false
agent:
  mode: function_calling
  tool_budget: 1
  max_steps: 3
  seed: 1
retrieval:
  backend: none
datasets: [trapset]
split: dev
limit: 3
"""


def test_run_rollout_writes_traces_and_is_resumable(tmp_path, capsys):
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(_OFFLINE_EXPERIMENT, encoding="utf-8")
    out_dir = tmp_path / "traces"

    output_path = run_rollout(config=str(exp_path), out_dir=str(out_dir))
    assert output_path.exists()

    traces = read_trace_dir(output_path)
    assert len(traces) == 3
    assert all(t.question.dataset == "trapset" for t in traces)
    assert all(t.context_length == 0 for t in traces)  # mock: not applicable
    assert all(t.experiment_id == "test_offline" for t in traces)

    run_rollout(config=str(exp_path), out_dir=str(out_dir))
    assert "already traced" in capsys.readouterr().out
    assert len(read_trace_dir(output_path)) == 3


def test_run_rollout_retries_error_traces_but_keeps_genuine_outcomes(tmp_path, capsys):
    """An 'error' trace is an infrastructure crash (no usable answer) and
    should be re-attempted; an 'unparseable' trace is a real agent outcome
    and must survive a rerun untouched, or a resumed run would silently
    overwrite real data about how the agent behaves."""
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(_OFFLINE_EXPERIMENT, encoding="utf-8")
    out_dir = tmp_path / "traces"
    output_path = out_dir / "traces.jsonl.gz"

    cfg = load_experiment(str(exp_path))
    questions = load_questions(cfg)
    assert len(questions) >= 2

    crashed = Trace(
        question=questions[0], experiment_id=cfg.experiment_id,
        terminated_by="error", error="boom",
    )
    genuine_fail = Trace(
        question=questions[1], experiment_id=cfg.experiment_id,
        terminated_by="unparseable", error="bad json",
    )
    write_traces([crashed, genuine_fail], output_path)

    run_rollout(config=str(exp_path), out_dir=str(out_dir))
    assert "error-trace retries" in capsys.readouterr().out

    traces = {(t.question.dataset, t.question.qid): t for t in read_trace_dir(output_path)}
    assert len(traces) == len(questions)
    assert traces[(questions[0].dataset, questions[0].qid)].terminated_by != "error"
    kept = traces[(questions[1].dataset, questions[1].qid)]
    assert kept.terminated_by == "unparseable"
    assert kept.error == "bad json"


def test_run_rollout_dry_run_writes_nothing(tmp_path):
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(_OFFLINE_EXPERIMENT, encoding="utf-8")
    out_dir = tmp_path / "traces"

    output_path = run_rollout(config=str(exp_path), out_dir=str(out_dir), dry_run=True)
    assert not output_path.exists()
