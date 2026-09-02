"""Gradio demo: run one episode and show the grounding check on its output.

The point of the demo is the second panel. Watching an agent call tools is
mildly interesting; watching each sentence of its justification get checked
against the passages it actually retrieved is the thing this project is about,
and it is hard to convey in a table.

Defaults to the deterministic mock model so the demo runs with no credentials
and no Ollama. Numbers produced from the mock are not results and the interface
says so.
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass

from cra.agent.loop import run_episode
from cra.config import AgentConfig, ModelConfig
from cra.eval.claims import extract_claims, strip_citation_framing
from cra.llm.factory import build_llm
from cra.tools.registry import default_registry
from cra.tools.retrieval import InMemoryRetriever, RetrievedPassage
from cra.types import Question, Trace

EXAMPLE_CASE = (
    "A 78-year-old woman with non-valvular atrial fibrillation, hypertension and type 2 "
    "diabetes takes naproxen regularly for osteoarthritis. She has never had a stroke. "
    "Which long-term antithrombotic strategy is most appropriate?"
)
EXAMPLE_OPTIONS = "\n".join(
    [
        "A) Aspirin 81 mg daily",
        "B) Oral anticoagulation, with review of her NSAID use",
        "C) No antithrombotic therapy",
        "D) Clopidogrel 75 mg daily",
    ]
)

# A small local corpus so the demo retrieves something real without an index build.
DEMO_CORPUS = [
    RetrievedPassage(
        doc_id="21873455",
        title="Risk stratification in atrial fibrillation",
        text=(
            "Among patients with non-valvular atrial fibrillation the CHA2DS2-VASc score "
            "identifies those at truly low risk of stroke. Patients scoring 2 or more derive "
            "net clinical benefit from oral anticoagulation, whereas aspirin provides little "
            "protection against cardioembolic stroke and carries a comparable bleeding risk "
            "in older patients."
        ),
    ),
    RetrievedPassage(
        doc_id="27567465",
        title="Antiplatelet therapy is not a substitute for anticoagulation",
        text=(
            "Randomised comparisons show aspirin monotherapy is substantially less effective "
            "than warfarin or a direct oral anticoagulant for stroke prevention in atrial "
            "fibrillation, without a meaningful safety advantage in elderly patients."
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

LABEL_STYLE = {
    "entailed": ("#1a7f37", "supported by the evidence shown"),
    "contradicted": ("#b42318", "contradicted by the evidence shown"),
    "not_addressed": ("#b54708", "not addressed by any retrieved passage"),
}


@dataclass
class Backend:
    """Whichever entailment checker is available, named for the interface."""

    name: str
    check: object
    note: str


def _load_checker(force_keyword: bool = False) -> Backend:
    if not force_keyword:
        try:
            from cra.eval.entailment.nli import NLIEntailmentChecker

            checker = NLIEntailmentChecker()
            return Backend("NLI (DeBERTa-v3 MNLI)", checker.check, "")
        except Exception:  # noqa: BLE001 - the extra is optional by design
            pass

    def keyword_check(claim: str, evidence_texts: list[str]) -> str:
        """Crude fallback so the demo runs without the dense extra.

        Overlap of distinctive words is not entailment. It is labelled as a
        fallback in the interface precisely so nobody mistakes it for the
        checker the results were produced with.
        """
        words = {w.strip(".,;:").lower() for w in claim.split() if len(w) > 5}
        if not words:
            return "not_addressed"
        for text in evidence_texts:
            other = {w.strip(".,;:").lower() for w in text.split() if len(w) > 5}
            if len(words & other) / len(words) >= 0.34:
                return "entailed"
        return "not_addressed"

    return Backend(
        "keyword overlap (fallback)",
        keyword_check,
        "The 'dense' extra is not installed, so this is a crude word-overlap stand-in, "
        "not the NLI checker the reported results use. Install it with: "
        "python tasks.py setup --extras dev,dense",
    )


def _parse_options(raw: str) -> dict[str, str] | None:
    options: dict[str, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if len(line) > 2 and line[0].isalpha() and line[1] in ").:":
            options[line[0].upper()] = line[2:].strip()
    return options or None


def _render_trace(trace: Trace) -> str:
    rows = ["<h3>What the agent did</h3>"]
    if not trace.tool_calls:
        rows.append("<p><i>No tools were called.</i></p>")
    for tc in trace.tool_calls:
        status = "ok" if tc.ok else ("refused" if not tc.executed else "failed")
        colour = "#1a7f37" if tc.ok else "#b42318"
        rows.append(
            f"<div style='margin:6px 0;padding:8px;border-left:3px solid {colour};"
            f"background:#00000008'><code>{html.escape(tc.name)}</code> "
            f"<span style='color:{colour}'>[{status}]</span><br>"
            f"<small>{html.escape(str(tc.args)[:220])}</small>"
            + (f"<br><small style='color:#b42318'>{html.escape(tc.error[:200])}</small>"
               if tc.error else "")
            + (f"<br><small>evidence: {', '.join(tc.evidence_ids)}</small>"
               if tc.evidence_ids else "")
            + "</div>"
        )

    if trace.evidence:
        rows.append("<h3>Evidence the agent saw</h3>")
        for e in trace.evidence:
            source = f" &lt;{html.escape(e.source_id)}&gt;" if e.source_id else ""
            rows.append(
                f"<div style='margin:6px 0;padding:8px;background:#00000008'>"
                f"<b>[{e.evidence_id}]</b> {html.escape(e.title or '(untitled)')}{source}<br>"
                f"<small>{html.escape(e.text[:400])}...</small></div>"
            )
    return "".join(rows)


def _render_grounding(trace: Trace, backend: Backend) -> str:
    if trace.final is None:
        return "<p><b>The agent produced no parseable answer.</b></p>"

    premises = [e.text for e in trace.evidence]
    claims = extract_claims(trace.final.justification)
    parts = [
        f"<h3>Grounding check <small style='font-weight:normal'>via {html.escape(backend.name)}"
        "</small></h3>"
    ]
    if backend.note:
        parts.append(
            f"<p style='padding:8px;background:#fff4e5;border-left:3px solid #b54708'>"
            f"<small>{html.escape(backend.note)}</small></p>"
        )
    if not premises:
        parts.append(
            "<p><i>Nothing was retrieved, so every claim is unsupported by construction. "
            "That is a property of this episode, not a measurement of the model.</i></p>"
        )

    unsupported = 0
    for claim in claims:
        hypothesis = strip_citation_framing(claim)
        label = backend.check(hypothesis, premises)
        colour, meaning = LABEL_STYLE[label]
        if label != "entailed":
            unsupported += 1
        stripped = (
            f"<br><small style='color:#666'>tested as: {html.escape(hypothesis)}</small>"
            if hypothesis != claim else ""
        )
        parts.append(
            f"<div style='margin:6px 0;padding:8px;border-left:3px solid {colour}'>"
            f"{html.escape(claim)}{stripped}<br>"
            f"<small style='color:{colour}'><b>{label}</b> — {meaning}</small></div>"
        )

    verdict = trace.is_correct
    verdict_text = {True: "correct", False: "incorrect", None: "no answer"}[verdict]
    summary = (
        f"<div style='margin-top:12px;padding:10px;background:#00000008'>"
        f"<b>Answer:</b> {html.escape(trace.final.answer)} ({verdict_text})<br>"
        f"<b>Cited:</b> {', '.join(trace.final.citations) or 'nothing'}<br>"
        f"<b>Unsupported claims:</b> {unsupported} of {len(claims)}</div>"
    )
    if verdict and unsupported:
        summary += (
            "<p style='padding:10px;border-left:3px solid #b54708;background:#fff4e5'>"
            "<b>Right answer, ungrounded justification.</b> This is the case standard "
            "accuracy benchmarks score as a clean win. Across the test split it accounts "
            "for 74–90% of correct answers.</p>"
        )
    return "".join(parts) + summary


def build_interface(model_config: ModelConfig, budget: int, force_keyword: bool = False):
    import gradio as gr

    backend = _load_checker(force_keyword=force_keyword)
    retriever = InMemoryRetriever(DEMO_CORPUS)

    def run(question: str, options_raw: str, gold: str, tool_budget: int, closed_book: bool):
        if not (question or "").strip():
            return "<p>Enter a clinical question.</p>", ""
        q = Question(
            qid="demo",
            dataset="medqa",
            split="dev",
            question=question.strip(),
            options=_parse_options(options_raw),
            gold_answer=(gold or "").strip() or "?",
        )
        registry = default_registry(retriever=retriever)
        cfg = AgentConfig(tool_budget=int(tool_budget), closed_book=closed_book)
        trace = run_episode(q, build_llm(model_config), registry, cfg, experiment_id="demo")
        return _render_trace(trace), _render_grounding(trace, backend)

    with gr.Blocks(title="Clinical reasoning agent — grounding check") as demo:
        gr.Markdown(
            "# Clinical reasoning agent\n"
            "Ask a clinical question, watch the agent use its tools, then see each sentence "
            "of its justification checked against the evidence it actually retrieved.\n\n"
            f"Model: `{model_config.model_id}`"
            + (
                "  \n*The mock model is deterministic and credential-free. It is a pipeline "
                "demonstration, not a result.*"
                if model_config.provider == "mock"
                else ""
            )
        )
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(label="Clinical question", lines=5, value=EXAMPLE_CASE)
                options = gr.Textbox(
                    label="Options (one per line, 'A) text')", lines=4, value=EXAMPLE_OPTIONS
                )
            with gr.Column(scale=1):
                gold = gr.Textbox(label="Correct answer (optional)", value="B")
                budget_slider = gr.Slider(0, 10, value=budget, step=1, label="Tool-call budget")
                closed_book = gr.Checkbox(
                    label="Closed-book (no tools)",
                    info="Removing tools costs 16-18 accuracy points on PubMedQA and "
                    "roughly halves the share of entailed claims.",
                )
                go = gr.Button("Run episode", variant="primary")
        with gr.Row():
            left = gr.HTML(label="Trace")
            right = gr.HTML(label="Grounding")
        go.click(run, [question, options, gold, budget_slider, closed_book], [left, right])
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="mock",
        help="mock (default, no credentials) or an Ollama model id such as qwen3-8k:8b",
    )
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--keyword-entailment", action="store_true",
        help="force the crude fallback checker even when the dense extra is installed",
    )
    args = parser.parse_args(argv)

    if args.model == "mock":
        cfg = ModelConfig(provider="mock", model_id="mock-heuristic", use_cache=False)
    else:
        cfg = ModelConfig(
            provider="openai_compat", model_id=args.model, base_url=args.base_url,
            use_cache=False,
        )
    build_interface(cfg, args.budget, force_keyword=args.keyword_entailment).launch(
        share=args.share
    )


if __name__ == "__main__":
    main()
