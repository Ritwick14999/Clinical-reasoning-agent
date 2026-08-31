"""The rollout runner: turns an experiment config into committed traces.

``python tasks.py rollout -- --config headline_qwen3`` is the only place
that:

* runs the context-overflow preflight check before any model call (see
  ``docs/HANDOFF.md``, "The context-overflow hazard" -- Ollama truncates a
  too-long prompt from the *oldest* end, silently dropping the system prompt
  and the question, which would look like a reasoning failure downstream),
* is resumable, by skipping questions already present in the output file and
  relying on the LLM response cache for partial episodes, and
* records ``context_length`` on every trace, so the setting is part of
  provenance and cannot drift between the two model runs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from cra.agent.loop import build_registry, run_episode
from cra.agent.prompts import SYSTEM, render_question
from cra.config import ExperimentConfig, load_experiment
from cra.data.splits import get_split, sample_stratified
from cra.llm.factory import build_llm
from cra.trace_io import read_trace_dir, write_traces
from cra.types import Question, Trace

# Matches the 8 GB VRAM target machine (docs/HANDOFF.md decision 4). Used only
# when the actual configured value cannot be verified against Ollama.
DEFAULT_CONTEXT_LENGTH = 8192
# Crude estimate (no tokenizer dependency), consistent with the usage
# accounting HeuristicMockClient already uses elsewhere in this codebase.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class PreflightResult:
    context_length: int
    context_length_source: str
    expected_prompt_tokens: int
    absolute_worst_case_tokens: int
    max_tokens: int

    @property
    def ok(self) -> bool:
        if not self.context_length:
            return True
        return self.expected_prompt_tokens + self.max_tokens <= self.context_length

    @property
    def worst_case_risky(self) -> bool:
        if not self.context_length:
            return False
        return self.absolute_worst_case_tokens + self.max_tokens > self.context_length


def _ollama_num_ctx(base_url: str, model_id: str) -> int | None:
    """Query Ollama's ``/api/show`` for the model's configured context length.

    Ollama's OpenAI-compatible endpoint does not accept ``num_ctx`` directly,
    so this is the only way to learn the *actual* configured value rather
    than assume one (docs/HANDOFF.md: "Verify which against the installed
    Ollama version rather than assuming").
    """
    host = base_url.rsplit("/v1", 1)[0].rstrip("/")
    req = urllib.request.Request(
        f"{host}/api/show",
        data=json.dumps({"name": model_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return None

    for line in (payload.get("parameters") or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "num_ctx" and parts[1].isdigit():
            return int(parts[1])

    model_info = payload.get("model_info") or {}
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def resolve_context_length(cfg: ExperimentConfig) -> tuple[int, str]:
    """``(context_length, source)`` -- source is recorded so it can be audited.

    0 means "not applicable" (mock/anthropic providers have no fixed local
    context window to overflow).
    """
    if cfg.model.provider != "openai_compat":
        return 0, "not_applicable"

    env_ctx = os.environ.get("OLLAMA_CONTEXT_LENGTH")
    if env_ctx and env_ctx.isdigit():
        return int(env_ctx), "OLLAMA_CONTEXT_LENGTH"

    if cfg.model.base_url:
        queried = _ollama_num_ctx(cfg.model.base_url, cfg.model.model_id)
        if queried:
            return queried, "ollama_api_show"

    print(
        f"WARNING: could not verify num_ctx from Ollama; assuming {DEFAULT_CONTEXT_LENGTH}. "
        "Set OLLAMA_CONTEXT_LENGTH, or a Modelfile 'PARAMETER num_ctx', to be certain."
    )
    return DEFAULT_CONTEXT_LENGTH, "default_assumed"


def preflight_check(cfg: ExperimentConfig, registry, questions: list[Question]) -> PreflightResult:
    """Estimate worst-case prompt size and abort before any model call if it
    could overflow the model's context window.

    Two figures are computed. ``expected_prompt_tokens`` uses the *configured*
    ``retrieval.default_k`` -- what the tool actually returns when the model
    doesn't specify ``k`` -- and is what gates the run: it is the number that
    would actually occur under normal operation, and matches how
    ``docs/HANDOFF.md`` decision 4 was measured. ``absolute_worst_case_tokens``
    additionally assumes every search requests the maximum allowed ``k``; if
    that alone would overflow, a non-fatal warning is printed so the run isn't
    silently vulnerable to an unusually greedy model.
    """
    from cra.tools.retrieval import MAX_K, SNIPPET_CHARS

    context_length, source = resolve_context_length(cfg)

    schema_tokens = sum(
        estimate_tokens(
            json.dumps({"name": s.name, "description": s.description, "input_schema": s.input_schema})
        )
        for s in registry.schemas()
    )
    longest_question_tokens = max((estimate_tokens(render_question(q)) for q in questions), default=0)
    system_tokens = estimate_tokens(SYSTEM)
    base_tokens = system_tokens + longest_question_tokens + schema_tokens

    per_passage_tokens = estimate_tokens("x" * SNIPPET_CHARS)
    expected_tool_tokens = per_passage_tokens * cfg.retrieval.default_k * cfg.agent.tool_budget
    worst_case_tool_tokens = per_passage_tokens * MAX_K * cfg.agent.tool_budget

    result = PreflightResult(
        context_length=context_length,
        context_length_source=source,
        expected_prompt_tokens=base_tokens + expected_tool_tokens,
        absolute_worst_case_tokens=base_tokens + worst_case_tool_tokens,
        max_tokens=cfg.model.max_tokens,
    )
    if not result.ok:
        raise RuntimeError(
            f"Preflight check failed: an estimated prompt of ~{result.expected_prompt_tokens} tokens "
            f"plus a {cfg.model.max_tokens}-token response could exceed the model's context window "
            f"(~{context_length} tokens, source={source}). Ollama truncates silently from the oldest "
            "end -- the system prompt and question would be dropped, which would look like a "
            "reasoning failure downstream, not a truncation. Lower agent.tool_budget or "
            "retrieval.default_k, or raise num_ctx."
        )
    if result.worst_case_risky:
        print(
            f"WARNING: a model requesting the maximum k={MAX_K} on every search could still "
            f"overflow context (~{result.absolute_worst_case_tokens} worst-case tokens vs "
            f"~{context_length}). Monitor traces for terminated_by='error' or truncated-looking "
            "reasoning failures."
        )
    return result


def _existing_traces(output_path: Path) -> dict[tuple[str, str], Trace]:
    if not output_path.exists():
        return {}
    return {(t.question.dataset, t.question.qid): t for t in read_trace_dir(output_path)}


def _needs_retry(trace: Trace) -> bool:
    """``terminated_by == "error"`` means the episode crashed (a provider
    outage, a local bug) before producing a usable trace -- it carries no
    answer and is re-attempted on the next run. This is unlike
    ``unparseable`` or ``step_limit``, which are genuine agent outcomes
    (the model ran to completion and simply didn't produce a valid final
    answer) and are kept as-is: retrying those would silently discard real
    data about how the agent behaves."""
    return trace.terminated_by == "error"


def load_questions(cfg: ExperimentConfig) -> list[Question]:
    """300 per dataset (docs/HANDOFF.md decision 3), not 300 total."""
    questions: list[Question] = []
    for dataset in cfg.datasets:
        split_questions = get_split(dataset, cfg.split)
        questions.extend(sample_stratified(split_questions, cfg.limit, seed=cfg.agent.seed or 12345))
    return questions


def build_retriever(cfg: ExperimentConfig):
    if cfg.retrieval.backend == "none":
        return None
    if cfg.retrieval.backend == "bm25":
        from cra.retrieval.bm25 import BM25Retriever

        return BM25Retriever.from_corpus_file(cfg.retrieval.corpus)
    if cfg.retrieval.backend == "dense":
        from cra.retrieval.dense import DenseRetriever

        return DenseRetriever.from_corpus_file(cfg.retrieval.corpus, model_name=cfg.retrieval.embedding_model)
    raise ValueError(f"unknown retrieval backend: {cfg.retrieval.backend!r}")


def run_rollout(
    config: str,
    limit: int | None = None,
    split: str | None = None,
    out_dir: str | None = None,
    dry_run: bool = False,
) -> Path:
    overrides: dict = {}
    if limit is not None:
        overrides["limit"] = limit
    if split is not None:
        overrides["split"] = split
    cfg = load_experiment(config, overrides=overrides)

    questions = load_questions(cfg)
    if not questions:
        raise RuntimeError(f"no questions loaded for datasets={cfg.datasets} split={cfg.split!r}")

    retriever = build_retriever(cfg)
    registry = build_registry(cfg, retriever=retriever)
    preflight = preflight_check(cfg, registry, questions)

    output_dir = Path(out_dir) if out_dir else Path("results/traces") / cfg.experiment_id
    output_path = output_dir / "traces.jsonl.gz"

    print(
        f"[{cfg.experiment_id}] {len(questions)} question(s) across {cfg.datasets}, "
        f"model={cfg.model.model_id}, context_length={preflight.context_length} "
        f"({preflight.context_length_source}), expected prompt ~{preflight.expected_prompt_tokens} "
        f"tokens -> {output_path}"
    )
    if dry_run:
        print("--dry-run: preflight passed, nothing was run.")
        return output_path

    existing = _existing_traces(output_path)
    todo = [
        q for q in questions
        if (q.dataset, q.qid) not in existing or _needs_retry(existing[(q.dataset, q.qid)])
    ]
    n_retries = sum(
        1 for q in questions
        if (q.dataset, q.qid) in existing and _needs_retry(existing[(q.dataset, q.qid)])
    )
    if len(todo) < len(questions) or n_retries:
        done = len(questions) - len(todo)
        msg = f"Resuming: {done} already traced, {len(todo)} remaining."
        if n_retries:
            msg += f" ({n_retries} of those are error-trace retries.)"
        print(msg)
    if not todo:
        print("Nothing to do -- every question already has a usable trace.")
        return output_path

    llm = build_llm(cfg.model)
    traces = dict(existing)

    n_correct = n_answered = total_tokens = 0
    pbar = tqdm(todo, desc=cfg.experiment_id)
    for question in pbar:
        trace = run_episode(
            question,
            llm,
            registry,
            cfg=cfg.agent,
            experiment_id=cfg.experiment_id,
            config_hash=cfg.config_hash,
            max_tokens=cfg.model.max_tokens,
            context_length=preflight.context_length,
        )
        traces[(question.dataset, question.qid)] = trace
        total_tokens += trace.usage.input_tokens + trace.usage.output_tokens
        if trace.is_correct is not None:
            n_answered += 1
            n_correct += int(trace.is_correct)
        pbar.set_postfix(
            acc=f"{n_correct}/{n_answered}" if n_answered else "-",
            tokens=total_tokens,
            last_ms=f"{trace.wall_time_ms:.0f}",
        )
        # Rewritten after every episode (not batched) so a killed run loses at
        # most the episode in flight, never previously completed ones.
        write_traces(traces.values(), output_path)

    print(f"Wrote {len(traces)} trace(s) to {output_path}")
    return output_path
