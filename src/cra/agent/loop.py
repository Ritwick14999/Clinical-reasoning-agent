"""The agent control loop.

One controller serves both the function-calling and ReAct modes, and produces
an identical :class:`Trace` either way, so the evaluator never needs to know
which was used.

Design commitments, each of which exists to keep a failure mode measurable:

* The tool budget is enforced by **withholding tool schemas**, not by asking
  the model to restrain itself. Prompt-level budgets get violated; a schema
  the model was never shown cannot be called.
* Tool failures are observations. Nothing a model can emit raises out of here.
* Evidence IDs are assigned once, centrally, and are what the model is shown
  and what the evaluator resolves citations against.
* Every departure from the happy path -- a repair round, a refused call,
  forced finalisation -- is recorded as a field rather than inferred later
  from the shape of the trace.
"""

from __future__ import annotations

import time

from cra.agent import prompts
from cra.agent.parsing import parse_final
from cra.config import AgentConfig, ExperimentConfig, git_sha
from cra.llm.base import LLMClient, LLMResponse, assistant, system, tool_result, user
from cra.tools.base import EvidenceStore
from cra.tools.registry import ToolRegistry
from cra.types import Question, Step, ToolCallRecord, Trace


def run_episode(
    question: Question,
    llm: LLMClient,
    registry: ToolRegistry,
    cfg: AgentConfig | None = None,
    experiment_id: str = "adhoc",
    config_hash: str = "",
    max_tokens: int = 1024,
) -> Trace:
    """Run one question end to end and return its trace."""
    cfg = cfg or AgentConfig()
    started = time.perf_counter()

    trace = Trace(
        question=question,
        experiment_id=experiment_id,
        model_id=getattr(llm, "model_id", "unknown"),
        provider=getattr(llm, "provider", "unknown"),
        config_hash=config_hash,
        git_sha=git_sha(),
        seed=cfg.seed,
        temperature=cfg.temperature,
        tool_budget=0 if cfg.closed_book else cfg.tool_budget,
        max_steps=cfg.max_steps,
        agent_mode=cfg.mode,
    )

    store = EvidenceStore()
    allowed = prompts.allowed_answers(question)
    messages = [system(prompts.SYSTEM), user(prompts.render_question(question))]

    try:
        _drive(trace, question, llm, registry, cfg, store, messages, allowed, max_tokens)
    except Exception as exc:  # noqa: BLE001 - a provider outage must not lose the trace
        trace.error = f"{type(exc).__name__}: {exc}"
        trace.terminated_by = "error"

    trace.evidence = store.items
    trace.wall_time_ms = (time.perf_counter() - started) * 1000
    return trace


def _drive(trace, question, llm, registry, cfg, store, messages, allowed, max_tokens) -> None:
    for step_index in range(cfg.max_steps):
        remaining = 0 if cfg.closed_book else cfg.tool_budget - trace.n_tool_calls
        # Withholding the schemas is the budget enforcement mechanism.
        exposed = registry.schemas() if remaining > 0 else []
        if remaining <= 0 and trace.budget_exhausted_at_step is None and not cfg.closed_book:
            trace.budget_exhausted_at_step = step_index

        resp: LLMResponse = llm.chat(
            messages,
            tools=exposed,
            temperature=cfg.temperature,
            max_tokens=max_tokens,
            seed=cfg.seed,
        )
        trace.steps.append(
            Step(
                index=step_index,
                kind="model",
                text=resp.text or None,
                thinking=resp.thinking,
                n_tool_calls=len(resp.tool_calls),
                usage=resp.usage,
                latency_ms=resp.latency_ms,
            )
        )
        trace.usage = trace.usage + resp.usage
        messages.append(assistant(resp))

        if not resp.tool_calls:
            final, error = parse_final(resp.text, allowed=allowed)
            if final is not None:
                trace.final = final
                trace.terminated_by = "model"
                return
            if cfg.allow_repair and not trace.repair_used:
                trace.repair_used = True
                messages.append(user(prompts.REPAIR.format(error=error)))
                continue
            trace.terminated_by = "unparseable"
            trace.error = f"unparseable final answer: {error}"
            return

        _dispatch_calls(trace, resp, registry, store, messages, step_index, remaining)

    # Steps exhausted without a final answer: one last no-tools turn.
    _force_finalize(trace, llm, messages, cfg, allowed, max_tokens)


def _dispatch_calls(trace, resp, registry, store, messages, step_index, remaining) -> None:
    for call in resp.tool_calls:
        if remaining <= 0:
            # Refused, not run. It still needs a result: tool-use APIs reject a
            # turn whose tool_use blocks have no matching tool_result.
            note = (
                "ERROR: tool-call budget exhausted; this call was not executed. "
                "Answer from the evidence you already have."
            )
            trace.tool_calls.append(
                ToolCallRecord(
                    index=len(trace.tool_calls), step=step_index, name=call.name,
                    args=call.args, ok=False, output=note,
                    error="budget_exhausted: call refused", executed=False,
                )
            )
            messages.append(tool_result(call.id, note))
            continue

        record, observation = registry.dispatch(
            call, store, index=len(trace.tool_calls), step=step_index
        )
        trace.tool_calls.append(record)
        messages.append(tool_result(call.id, observation))
        remaining -= 1


def _force_finalize(trace, llm, messages, cfg, allowed, max_tokens) -> None:
    messages.append(user(prompts.FORCE_FINAL.format(contract=prompts.FINAL_ANSWER_CONTRACT)))
    resp = llm.chat(
        messages, tools=[], temperature=cfg.temperature, max_tokens=max_tokens, seed=cfg.seed
    )
    trace.steps.append(
        Step(
            index=len(trace.steps),
            kind="forced_final",
            text=resp.text or None,
            thinking=resp.thinking,
            usage=resp.usage,
            latency_ms=resp.latency_ms,
        )
    )
    trace.usage = trace.usage + resp.usage
    final, error = parse_final(resp.text, allowed=allowed)
    trace.final = final
    trace.terminated_by = "step_limit" if final is not None else "unparseable"
    if final is None:
        trace.error = f"forced finalisation produced no valid answer: {error}"


def build_registry(cfg: ExperimentConfig, retriever=None) -> ToolRegistry:
    """Assemble the tool set an experiment config asks for."""
    from cra.tools.registry import default_registry

    if cfg.agent.closed_book:
        return ToolRegistry([])
    return default_registry(
        retriever=retriever if cfg.agent.enable_retrieval else None,
        include_calculators=cfg.agent.enable_calculators,
        include_drugs=cfg.agent.enable_drugs,
        include_units=cfg.agent.enable_units,
    )
