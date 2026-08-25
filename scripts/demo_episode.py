#!/usr/bin/env python
"""Run one episode end to end with no credentials and print the trace.

    python scripts/demo_episode.py

Uses the deterministic mock client and a tiny in-memory corpus, so it works
offline. It exists to make the agent loop inspectable before the real datasets
and retrieval index are wired in.
"""

from __future__ import annotations

import argparse

from cra.agent.loop import run_episode
from cra.config import AgentConfig
from cra.llm.base import LLMResponse, ToolCall
from cra.llm.mock import HeuristicMockClient, ScriptedClient
from cra.tools.registry import default_registry
from cra.tools.retrieval import InMemoryRetriever, RetrievedPassage
from cra.trace_io import render_trace
from cra.types import Question

CORPUS = [
    RetrievedPassage(
        doc_id="21873455",
        title="Refining clinical risk stratification in atrial fibrillation",
        text=(
            "Among patients with non-valvular atrial fibrillation, the CHA2DS2-VASc score "
            "identifies those at truly low risk of stroke. Patients with a score of 2 or more "
            "derive net clinical benefit from oral anticoagulation, whereas aspirin provides "
            "little protection against cardioembolic stroke and carries a comparable bleeding "
            "risk in older patients."
        ),
    ),
    RetrievedPassage(
        doc_id="27567465",
        title="Antiplatelet therapy is not an adequate substitute for anticoagulation",
        text=(
            "Randomised comparisons show that aspirin monotherapy is substantially less "
            "effective than warfarin or a direct oral anticoagulant for stroke prevention in "
            "atrial fibrillation, without a meaningful safety advantage in elderly patients."
        ),
    ),
    RetrievedPassage(
        doc_id="30193606",
        title="Bleeding risk with concurrent NSAID and anticoagulant use",
        text=(
            "Concurrent non-steroidal anti-inflammatory drug use in anticoagulated patients "
            "substantially increases the risk of major gastrointestinal haemorrhage."
        ),
    ),
]

CASE = Question(
    qid="demo-1",
    dataset="medqa",
    split="dev",
    question=(
        "A 78-year-old woman with a 4-year history of non-valvular atrial fibrillation "
        "presents for review. She has hypertension and type 2 diabetes, and takes naproxen "
        "regularly for osteoarthritis. She has never had a stroke. Which is the most "
        "appropriate long-term antithrombotic strategy?"
    ),
    options={
        "A": "Aspirin 81 mg daily",
        "B": "Oral anticoagulation, with review of her NSAID use",
        "C": "No antithrombotic therapy",
        "D": "Clopidogrel 75 mg daily",
    },
    gold_answer="B",
    expected_tools=["search_literature", "calc_cha2ds2_vasc", "check_drug_interactions"],
)


def scripted_agent() -> ScriptedClient:
    """A hand-written 'good' episode, so the demo shows a realistic trajectory."""
    def call(cid: str, name: str, args: dict) -> LLMResponse:
        return LLMResponse(
            text="",
            thinking=f"Calling {name}.",
            tool_calls=[ToolCall(id=cid, name=name, args=args)],
            stop_reason="tool_use",
        )

    import json

    return ScriptedClient(
        [
            call("c1", "calc_cha2ds2_vasc",
                 {"age": 78, "sex": "female", "hypertension": True, "diabetes": True}),
            call("c2", "search_literature",
                 {"query": "atrial fibrillation anticoagulation aspirin stroke prevention", "k": 3}),
            call("c3", "check_drug_interactions", {"drugs": ["warfarin", "naproxen"]}),
            LLMResponse(
                text=json.dumps(
                    {
                        "answer": "B",
                        "justification": (
                            "Her CHA2DS2-VASc score is 5, which T1 classifies as high risk "
                            "warranting oral anticoagulation. E1 states that patients scoring 2 "
                            "or more derive net clinical benefit from oral anticoagulation. E2 "
                            "reports that aspirin is substantially less effective than "
                            "anticoagulation without a meaningful safety advantage in elderly "
                            "patients. T2 flags a major interaction between anticoagulants and "
                            "NSAIDs, so her naproxen use needs review."
                        ),
                        "citations": ["T1", "E1", "E2", "T2"],
                    }
                ),
                thinking="Synthesising the score, the literature and the interaction check.",
                stop_reason="end_turn",
            ),
        ],
        model_id="mock-scripted",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client", choices=["scripted", "heuristic"], default="scripted",
        help="scripted: a realistic hand-written trajectory. heuristic: the generic mock.",
    )
    parser.add_argument("--budget", type=int, default=5, help="tool-call budget")
    parser.add_argument("--closed-book", action="store_true", help="run with no tools at all")
    args = parser.parse_args()

    registry = default_registry(retriever=InMemoryRetriever(CORPUS))
    llm = scripted_agent() if args.client == "scripted" else HeuristicMockClient()
    cfg = AgentConfig(tool_budget=args.budget, closed_book=args.closed_book)

    trace = run_episode(CASE, llm, registry, cfg, experiment_id="demo")
    print(render_trace(trace))


if __name__ == "__main__":
    main()
