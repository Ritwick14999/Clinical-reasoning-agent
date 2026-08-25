"""Prompts and the final-answer contract.

The contract does real work downstream. One claim per sentence is what makes
rule-based claim extraction viable; explicit evidence IDs are what let the
evaluator separate "cited something that does not support this" from "cited
nothing at all". How often models honour the contract is itself measured
rather than assumed.
"""

from __future__ import annotations

from cra.types import Question

FINAL_ANSWER_CONTRACT = """When you are ready to answer, reply with a single JSON object and nothing else:

{"answer": "<the answer>", "justification": "<your reasoning>", "citations": ["E1", "T2"]}

Rules for the JSON object:
- "answer" must be exactly one of the allowed answers listed with the question.
- "justification" must state one claim per sentence, so each sentence can be
  checked independently against the evidence.
- "citations" must list the evidence IDs (E1, E2, ... for retrieved passages,
  T1, T2, ... for tool outputs) that your justification actually relies on.
  Cite only what you used, and use an empty list if you used no evidence.
- Do not assert anything the evidence does not support. If the evidence is
  insufficient, say so in the justification and answer with your best judgement."""

SYSTEM = f"""You are a careful clinical reasoning assistant working on benchmark questions.

You have tools available. Use them when they would change or confirm your answer:
- Search the literature before making a factual claim you cannot support from the question itself.
- Use a clinical calculator whenever the case supplies the inputs for one. Never compute a
  published risk score in your head.
- Check drug interactions whenever a case lists two or more medications.

You have a limited number of tool calls. Spend them on what you cannot determine otherwise.
When the tools are no longer offered to you, your budget is exhausted and you must answer
from what you already have.

Ground your answer in what the tools actually returned. A tool that returns nothing, or that
reports a drug it did not recognise, is not evidence of absence.

{FINAL_ANSWER_CONTRACT}"""

REPAIR = """Your last message was not a valid final answer: {error}

Reply now with only the JSON object described earlier. No preamble, no code fence, no commentary."""

FORCE_FINAL = """You have no tool calls remaining. Answer now from the evidence you already have.

{contract}"""


def allowed_answers(question: Question) -> list[str]:
    if question.options:
        return list(question.options)
    return ["yes", "no", "maybe"]


def render_question(question: Question) -> str:
    parts = [question.question.strip()]
    if question.options:
        parts.append("")
        parts += [f"{key}) {text}" for key, text in question.options.items()]
    parts.append("")
    parts.append(f"Allowed answers: {', '.join(allowed_answers(question))}")
    return "\n".join(parts)
