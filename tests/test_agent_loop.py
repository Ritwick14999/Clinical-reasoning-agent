"""Control-loop behaviour.

These tests pin the invariants the evaluation framework depends on. If budget
enforcement or trace bookkeeping drifts, downstream failure-mode labels become
quietly wrong rather than obviously broken, so each invariant is asserted
directly.
"""

from __future__ import annotations

import pytest
from conftest import final_response, multi_tool_response, tool_response

from cra.agent.loop import run_episode
from cra.config import AgentConfig
from cra.llm.base import LLMResponse
from cra.llm.mock import HeuristicMockClient, ScriptedClient
from cra.tools.registry import ToolRegistry
from cra.types import Usage

SEARCH = ("search_literature", {"query": "atrial fibrillation anticoagulation"})


def test_happy_path(mcq, registry):
    llm = ScriptedClient([tool_response(*SEARCH), final_response()])
    trace = run_episode(mcq, llm, registry, AgentConfig())

    assert trace.terminated_by == "model"
    assert trace.final is not None
    assert trace.final.answer == "B"
    assert trace.is_correct is True
    assert trace.n_tool_calls == 1
    assert trace.tool_names == ["search_literature"]
    # Only the AF passage matches this query; the NSAID passage shares no terms.
    assert [e.evidence_id for e in trace.evidence] == ["E1"]
    assert not trace.repair_used


def test_evidence_ids_reach_the_model(mcq, registry):
    llm = ScriptedClient([tool_response(*SEARCH), final_response()])
    run_episode(mcq, llm, registry, AgentConfig())

    observation = llm.calls[1][-1].content
    assert "[E1]" in observation
    assert "Cite these as: E1" in observation
    assert "oral anticoagulation reduces" in observation, "passage text reaches the model"


def test_budget_enforced_by_withholding_schemas(mcq, registry):
    """Once the budget is spent the model must not be shown any tools."""
    llm = ScriptedClient(
        [tool_response(*SEARCH, call_id="a"), tool_response(*SEARCH, call_id="b"),
         final_response()]
    )
    trace = run_episode(mcq, llm, registry, AgentConfig(tool_budget=2))

    assert trace.n_tool_calls == 2
    assert llm.exposed_tools[0] and llm.exposed_tools[1], "tools available while budget remains"
    assert llm.exposed_tools[2] == [], "no tools offered once the budget is spent"
    assert trace.budget_exhausted_at_step == 2


def test_parallel_calls_beyond_budget_are_refused_not_executed(mcq, registry):
    llm = ScriptedClient(
        [
            multi_tool_response([SEARCH, SEARCH, SEARCH]),
            final_response(),
        ]
    )
    trace = run_episode(mcq, llm, registry, AgentConfig(tool_budget=2))

    assert trace.n_tool_calls == 2, "only two calls consume budget"
    assert len(trace.tool_calls) == 3, "the refused call is still recorded"
    refused = [tc for tc in trace.tool_calls if not tc.executed]
    assert len(refused) == 1
    assert "budget_exhausted" in refused[0].error


def test_every_tool_call_gets_a_result_message(mcq, registry):
    """Tool-use APIs reject a turn whose tool_use blocks have no tool_result."""
    llm = ScriptedClient([multi_tool_response([SEARCH, SEARCH, SEARCH]), final_response()])
    run_episode(mcq, llm, registry, AgentConfig(tool_budget=1))

    final_messages = llm.calls[-1]
    n_emitted = sum(len(m.tool_calls) for m in final_messages if m.role == "assistant")
    n_results = sum(1 for m in final_messages if m.role == "tool")
    assert n_emitted == n_results == 3


def test_closed_book_never_exposes_tools(mcq, registry):
    llm = ScriptedClient([final_response(citations=[])])
    trace = run_episode(mcq, llm, registry, AgentConfig(closed_book=True))

    assert llm.exposed_tools == [[]]
    assert trace.n_tool_calls == 0
    assert trace.tool_budget == 0
    assert trace.evidence == []


def test_repair_round_recovers_a_malformed_answer(mcq, registry):
    llm = ScriptedClient([LLMResponse(text="I think the answer is B."), final_response()])
    trace = run_episode(mcq, llm, registry, AgentConfig())

    assert trace.repair_used is True
    assert trace.final is not None
    assert trace.terminated_by == "model"
    assert "not a valid final answer" in llm.calls[1][-1].content


def test_only_one_repair_round(mcq, registry):
    llm = ScriptedClient([LLMResponse(text="prose"), LLMResponse(text="still prose")])
    trace = run_episode(mcq, llm, registry, AgentConfig())

    assert trace.repair_used is True
    assert trace.final is None
    assert trace.terminated_by == "unparseable"
    assert "unparseable" in trace.error


def test_repair_disabled(mcq, registry):
    llm = ScriptedClient([LLMResponse(text="prose")])
    trace = run_episode(mcq, llm, registry, AgentConfig(allow_repair=False))

    assert trace.repair_used is False
    assert trace.terminated_by == "unparseable"


def test_out_of_space_answer_is_rejected(mcq, registry):
    llm = ScriptedClient([final_response(answer="Z"), final_response(answer="C")])
    trace = run_episode(mcq, llm, registry, AgentConfig())

    assert trace.repair_used is True
    assert trace.final.answer == "C"
    assert trace.is_correct is False


def test_step_limit_forces_finalisation(mcq, registry):
    llm = ScriptedClient([tool_response(*SEARCH, call_id=f"c{i}") for i in range(3)]
                         + [final_response()])
    trace = run_episode(mcq, llm, registry, AgentConfig(max_steps=3, tool_budget=10))

    assert trace.terminated_by == "step_limit"
    assert trace.steps[-1].kind == "forced_final"
    assert trace.final is not None
    assert llm.exposed_tools[-1] == [], "the forced turn offers no tools"


def test_step_limit_without_a_parseable_answer(mcq, registry):
    llm = ScriptedClient([tool_response(*SEARCH, call_id="a"), LLMResponse(text="no json")])
    trace = run_episode(mcq, llm, registry, AgentConfig(max_steps=1, tool_budget=10))

    assert trace.terminated_by == "unparseable"
    assert trace.final is None
    assert trace.is_correct is None, "no answer is distinct from a wrong answer"


def test_tool_errors_do_not_end_the_episode(mcq, registry):
    llm = ScriptedClient(
        [tool_response("calc_meld", {"bilirubin_mg_dl": -1, "inr": 1.2, "creatinine_mg_dl": 1.0}),
         final_response(citations=[])]
    )
    trace = run_episode(mcq, llm, registry, AgentConfig())

    assert trace.terminated_by == "model"
    assert trace.tool_calls[0].ok is False
    assert trace.n_tool_calls == 1, "a failed call still consumes budget"


def test_unknown_tool_is_recorded_and_recoverable(mcq, registry):
    llm = ScriptedClient([tool_response("nonexistent_tool", {}), final_response(citations=[])])
    trace = run_episode(mcq, llm, registry, AgentConfig())

    assert "unknown_tool" in trace.tool_calls[0].error
    assert trace.final is not None


def test_provider_failure_is_captured_not_raised(mcq, registry):
    class Exploding:
        model_id, provider = "boom", "test"

        def chat(self, *a, **k):
            raise RuntimeError("upstream 503")

    trace = run_episode(mcq, Exploding(), registry, AgentConfig())

    assert trace.terminated_by == "error"
    assert "upstream 503" in trace.error
    assert trace.final is None


def test_usage_accumulates_across_steps(mcq, registry):
    llm = ScriptedClient(
        [
            LLMResponse(text="", tool_calls=tool_response(*SEARCH).tool_calls,
                        usage=Usage(input_tokens=100, output_tokens=10)),
            LLMResponse(text=final_response().text, usage=Usage(input_tokens=300, output_tokens=40)),
        ]
    )
    trace = run_episode(mcq, llm, registry, AgentConfig())

    assert trace.usage.input_tokens == 400
    assert trace.usage.output_tokens == 50


def test_provenance_is_recorded(mcq, registry):
    llm = ScriptedClient([final_response(citations=[])])
    trace = run_episode(
        mcq, llm, registry, AgentConfig(seed=7, temperature=0.3),
        experiment_id="exp-x", config_hash="deadbeef",
    )

    assert trace.experiment_id == "exp-x"
    assert trace.config_hash == "deadbeef"
    assert trace.seed == 7
    assert trace.temperature == 0.3
    assert trace.model_id == "mock-scripted"
    assert trace.schema_version == "1.0"


def test_duplicate_evidence_keeps_one_id(mcq, registry):
    """The same passage retrieved twice must not get two IDs."""
    llm = ScriptedClient(
        [tool_response(*SEARCH, call_id="a"), tool_response(*SEARCH, call_id="b"),
         final_response()]
    )
    trace = run_episode(mcq, llm, registry, AgentConfig())

    assert len(trace.evidence) == 1
    assert trace.tool_calls[0].evidence_ids == trace.tool_calls[1].evidence_ids == ["E1"]


def test_trace_round_trips_through_json(mcq, registry):
    """The trace is the hand-off to the evaluator; it must survive serialisation."""
    from cra.types import Trace

    llm = ScriptedClient([tool_response(*SEARCH), final_response()])
    trace = run_episode(mcq, llm, registry, AgentConfig())
    restored = Trace.model_validate_json(trace.model_dump_json())

    assert restored.final.answer == trace.final.answer
    assert [e.evidence_id for e in restored.evidence] == ["E1"]
    assert restored.evidence[0].text == trace.evidence[0].text, "evidence text is inlined"
    assert restored.is_correct == trace.is_correct


def test_heuristic_mock_runs_end_to_end(mcq, registry):
    trace = run_episode(mcq, HeuristicMockClient(), registry, AgentConfig())

    assert trace.final is not None
    assert trace.n_tool_calls == 1
    assert trace.terminated_by == "model"


def test_heuristic_mock_is_deterministic(mcq, registry):
    a = run_episode(mcq, HeuristicMockClient(), registry, AgentConfig(seed=1))
    b = run_episode(mcq, HeuristicMockClient(), registry, AgentConfig(seed=1))
    assert a.final.answer == b.final.answer

    c = run_episode(mcq, HeuristicMockClient(), registry, AgentConfig(seed=999))
    assert c.final is not None  # a different seed is still a valid episode


def test_empty_registry_is_survivable(mcq):
    llm = ScriptedClient([final_response(citations=[])])
    trace = run_episode(mcq, llm, ToolRegistry(), AgentConfig())
    assert trace.final is not None


@pytest.mark.parametrize("budget", [0, 1, 5])
def test_budget_is_never_exceeded(mcq, registry, budget):
    llm = ScriptedClient([tool_response(*SEARCH, call_id=f"c{i}") for i in range(10)]
                         + [final_response()])
    trace = run_episode(mcq, llm, registry, AgentConfig(tool_budget=budget, max_steps=12))
    assert trace.n_tool_calls <= budget
