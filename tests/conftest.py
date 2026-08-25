from __future__ import annotations

import pytest

from cra.llm.base import LLMResponse, ToolCall
from cra.tools.registry import default_registry
from cra.tools.retrieval import InMemoryRetriever, RetrievedPassage
from cra.types import Question

CORPUS = [
    RetrievedPassage(
        doc_id="10001",
        title="Anticoagulation in atrial fibrillation",
        text=(
            "In patients with non-valvular atrial fibrillation, oral anticoagulation reduces "
            "ischaemic stroke risk. Direct oral anticoagulants are preferred over warfarin for "
            "most patients."
        ),
    ),
    RetrievedPassage(
        doc_id="10002",
        title="NSAIDs and bleeding risk on warfarin",
        text=(
            "Concurrent NSAID use in patients taking warfarin substantially increases the risk "
            "of gastrointestinal haemorrhage."
        ),
    ),
]


@pytest.fixture
def retriever() -> InMemoryRetriever:
    return InMemoryRetriever(CORPUS)


@pytest.fixture
def registry(retriever):
    return default_registry(retriever=retriever)


@pytest.fixture
def mcq() -> Question:
    return Question(
        qid="t1",
        dataset="medqa",
        split="dev",
        question="A 78-year-old woman with atrial fibrillation and hypertension. Next step?",
        options={"A": "Aspirin", "B": "Oral anticoagulation", "C": "No therapy", "D": "Ablation"},
        gold_answer="B",
        expected_tools=["search_literature", "calc_cha2ds2_vasc"],
    )


def final_response(answer: str = "B", justification: str = "Because E1 says so.",
                   citations: list[str] | None = None) -> LLMResponse:
    import json

    return LLMResponse(
        text=json.dumps(
            {
                "answer": answer,
                "justification": justification,
                "citations": citations if citations is not None else ["E1"],
            }
        ),
        stop_reason="end_turn",
    )


def tool_response(name: str, args: dict, call_id: str = "c") -> LLMResponse:
    return LLMResponse(
        text="", tool_calls=[ToolCall(id=call_id, name=name, args=args)], stop_reason="tool_use"
    )


def multi_tool_response(calls: list[tuple[str, dict]]) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[
            ToolCall(id=f"c{i}", name=n, args=a) for i, (n, a) in enumerate(calls)
        ],
        stop_reason="tool_use",
    )
