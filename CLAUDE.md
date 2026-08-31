# Clinical Reasoning Agent + Evidence-Grounding Benchmark

A tool-using clinical reasoning agent, and an evaluation framework that measures
*why* it fails -- including whether its final claims are actually entailed by the
evidence it retrieved. **The evaluation framework is the contribution**; the
agent exists to generate traces for it.

Full design: `docs/DESIGN.md`. Read it before changing architecture.

## Working branch

`claude/clinical-reasoning-agent-sxi0mw` -- all work goes here. Never push to `main`.

## The one architectural invariant

Two stages, separated by a self-contained trace file:

```
ROLLOUT  (needs Ollama; stochastic; slow) -> results/traces/<exp>/*.jsonl.gz
EVAL     (pure function of traces; no network, no model) -> tables + figures
```

**Never let evaluation code call a model to re-derive something the trace should
have recorded.** The separation is what lets the eval be improved and re-run
over existing traces at zero rollout cost, and what lets `results` reproduce
headline numbers from a fresh clone with no GPU and no credentials.

Consequence: a `Trace` inlines the **full text** of every retrieved passage and
tool output. It must stay interpretable after the retrieval index is rebuilt.

## Layout

```
src/cra/
  types.py          Trace / Question / Evidence / ToolCallRecord  <- the contract
  config.py         dataclass configs + config_hash provenance
  llm/              LLMClient protocol; anthropic, openai_compat (Ollama/vLLM), mocks, cache
  agent/            loop.py (controller), prompts.py, parsing.py
  tools/            registry.py, retrieval.py, drugs.py, units.py, calculators/
  retrieval/        BM25 (default) + dense (optional, `dense` extra) over the PubMedQA corpus
  data/             download.py, pubmedqa.py, medqa.py, splits.py, expected_tools.py, trapset.py
  rollout.py, cli.py  the rollout runner: `python tasks.py rollout -- --config <name>`
  eval/             the evaluation framework      [Phase 3 - stub]
configs/            model/ agent/ retrieval/ experiments/  (composable YAML)
results/traces/     committed traces -- the reproducibility anchor
```

## Design commitments (do not quietly undo these)

1. **Tool budget is enforced by withholding tool schemas**, not by prompting.
   A schema the model was never shown cannot be called. See `agent/loop.py`.
2. **Nothing a model emits may raise.** Unknown tool, schema violation,
   unparseable arguments, tool exception, provider outage -- all become recorded
   observations. A crash destroys exactly the traces `tool_misuse` studies.
3. **Evidence IDs are assigned once, centrally**, in `tools/base.EvidenceStore`
   (`E1..` passages, `T1..` tool outputs). The IDs the model sees are the ones
   the evaluator resolves citations against.
4. **Calculators are pure total functions** with published formulae and explicit
   validation errors, so a wrong score is unambiguously the agent's argument
   error rather than a tool defect. That is what makes `tool_misuse` meaningful.
5. **Refused calls are recorded with `executed=False`.** Tool-use APIs reject a
   turn whose `tool_use` blocks have no matching `tool_result`, so a
   budget-refused call still needs a result -- but must not consume budget.
6. **A missing answer is distinct from a wrong answer** (`is_correct` returns
   `None`). Never collapse them.
7. **No test-set peeking.** All development and threshold calibration happens on
   `dev`. `split: test` is for final numbers only.

## Failure-mode taxonomy (Phase 3)

Ordered rules, exactly one primary label per trace:
`retrieval_failure` -> `tool_misuse` -> `reasoning_failure` -> `unsupported_claim`
(correct answer, ungrounded justification) -> `correct_grounded`.

**Hallucination is also an independent boolean flag on every trace, and the
hallucination rate is computed from the flags, never from the primary label** --
otherwise hallucinations inside wrong answers vanish from the metric.

## Commands

Windows: use `tasks.py` (no `make`, and `.venv/bin/python` does not exist there).

```
python tasks.py setup [--force] [--extras dev,dense]
python tasks.py doctor      # checks Python 3.11, venv, Ollama models
python tasks.py test
python tasks.py demo        # one episode, mock model, no credentials
python tasks.py lint
python tasks.py data        # download PubMedQA + MIRAGE into data/raw/
python tasks.py index       # build the BM25 corpus from downloaded data
python tasks.py rollout -- --config smoke_mock       # credential-free pipeline check
python tasks.py rollout -- --config headline_qwen3 --dry-run   # preflight only
```

POSIX: the `Makefile` has the same targets.

## Environment constraints

- **Python 3.11 exactly** (`>=3.11,<3.12`). Pinned deps (torch, the dense stack)
  have no wheels for newer versions. `uv` can fetch 3.11 for you.
- **Models: Ollama only**, two open-weight models run *sequentially*
  (`qwen3:8b`, `llama3.1:8b`). No paid API. The Anthropic adapter stays tested
  so a frontier run can be added later without pipeline changes.
- **Native function calling only.** `AgentConfig.mode == "react"` is declared but
  NOT implemented -- `cfg.mode` is only recorded on the trace. A model without
  tool-calling support will silently produce closed-book traces.
- **Target GPU has 8 GB VRAM.** Ollama silently truncates over-long prompts,
  which would drop the system prompt and the question and look like a reasoning
  failure. Keep episode prompts under the configured `num_ctx`, and record
  context length in the trace.

## Conventions

- `ruff check src tests scripts tasks.py` must pass; line length 100.
- Tests: `pytest`, no network, no credentials, fast. Calculator expectations are
  computed by hand from published formulae -- never captured from a previous run.
- Incremental commits with explanatory messages; do not squash the history.
- Do not commit raw datasets. `data/` is gitignored except `manifests/`,
  `trapset/` and `manual_labels/`.
- Report accuracy deltas with bootstrap CIs; paired comparisons use McNemar.
