# Conventions and invariants

Read `docs/DESIGN.md` for the architecture. This file is the short version: the
things that will silently corrupt a result if they get undone.

## The one architectural invariant

Two stages, separated by a self-contained trace file:

```
ROLLOUT  (needs Ollama; stochastic; slow)  ->  results/traces/<exp>/*.jsonl.gz
EVAL     (pure function of traces; no network, no model)  ->  tables + figures
```

Evaluation code must never call a model to re-derive something the trace should
have recorded. That separation is what lets the instrument be improved and
re-run over existing traces at no rollout cost, and what lets a fresh clone
reproduce the headline numbers with no GPU and no credentials. It was used
repeatedly: eight defects found during development and annotation were each
corrected and every affected number re-scored without re-running a model.

A consequence: a `Trace` inlines the full text of every retrieved passage and
tool output. It has to stay interpretable after the retrieval index is rebuilt.

## Design commitments

1. **The tool budget is enforced by withholding tool schemas**, not by
   prompting. A schema the model was never shown cannot be called. See
   `agent/loop.py`.
2. **Nothing a model emits may raise.** Unknown tool, schema violation,
   unparseable arguments, tool exception, provider outage — all become recorded
   observations. A crash destroys exactly the traces `tool_misuse` studies.
3. **Evidence IDs are assigned once, centrally**, in `tools/base.EvidenceStore`
   (`E1..` passages, `T1..` tool outputs). The IDs the model sees are the ones
   the evaluator resolves citations against.
4. **Calculators are pure total functions** with published formulae and explicit
   validation errors, so a wrong score is unambiguously the agent's argument
   error rather than a tool defect. That is what makes `tool_misuse` meaningful.
5. **Refused calls are recorded with `executed=False`.** Tool-use APIs reject a
   turn whose `tool_use` blocks have no matching `tool_result`, so a
   budget-refused call still needs a result — but must not consume budget.
6. **A missing answer is distinct from a wrong answer** (`is_correct` returns
   `None`). Never collapse them.
7. **No test-set peeking.** All development and threshold calibration happens on
   `dev`. `split: test` requires `CRA_ALLOW_TEST=1` and was touched once.

## Failure-mode taxonomy

Ordered rules, exactly one primary label per trace:
`retrieval_failure` -> `tool_misuse` -> `reasoning_failure` -> `unsupported_claim`
(correct answer, ungrounded justification) -> `correct_grounded`, with
`no_answer` kept separate.

Hallucination is **also** an independent boolean flag on every trace, and the
hallucination rate is computed from the flags, never from the primary label —
otherwise hallucinations inside wrong answers vanish from the metric.

Two refinements came out of human annotation and are load-bearing. A malformed
tool call the agent retried successfully is `malformed_call_recovered` and does
not capture the label: 151 of 159 `tool_misuse` labels were exactly that, and
fixing it overturned a reported cross-model finding. And `retrieval_failure`
fires when a wrong answer was produced with no retrieved passage at all, not
only on a missed gold PMID, since MedQA has no gold provenance.

## Commands

Windows has no `make` and no `.venv/bin/python`, so `tasks.py` is the portable
entry point; the `Makefile` mirrors it for POSIX shells.

```
python tasks.py setup [--force] [--extras dev,dense,demo]
python tasks.py doctor      # Python 3.11, venv, Ollama models
python tasks.py test
python tasks.py demo        # one episode, mock model, no credentials
python tasks.py demo-app    # Gradio UI
python tasks.py lint
python tasks.py data        # download PubMedQA + MIRAGE into data/raw/
python tasks.py index       # build the BM25 corpus
python tasks.py rollout -- --config final_qwen3 --dry-run
```

## Environment constraints

- **Python 3.11 exactly** (`>=3.11,<3.12`). Pinned deps (torch, the dense stack)
  have no wheels for newer versions. `uv` can fetch 3.11.
- **Models via Ollama**, two open-weight models run *sequentially*
  (`qwen3:8b`, `llama3.1:8b`) on 8 GB of VRAM. The Anthropic adapter stays
  tested so a frontier run can be added without pipeline changes.
- **Native function calling only.** A ReAct mode was declared and never built;
  it has been removed rather than left as a setting that does nothing.
- **Ollama silently truncates over-long prompts** from the oldest tokens, which
  are the system prompt and the question. The longest episode in the final run
  reached 7,755 tokens against a 4,096 default. The window must be raised
  server-side (Modelfile `PARAMETER num_ctx`); the OpenAI-compatible endpoint
  ignores `options.num_ctx`. The preflight verifies this by sending a prompt of
  the expected size and checking the server evaluated all of it.

## Code conventions

- `ruff check src tests scripts tasks.py` must pass; line length 100.
- Tests: `pytest`, no network, no credentials, fast. Calculator expectations are
  computed by hand from published formulae — never captured from a previous run.
- Incremental commits with explanatory messages; do not squash the history.
- Do not commit raw datasets. `data/` is gitignored except `manifests/`,
  `trapset/` and `manual_labels/`.
- Report accuracy deltas with bootstrap CIs; paired comparisons use McNemar.
