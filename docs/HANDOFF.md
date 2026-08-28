# Handoff: current state and what to build next

This file exists so a Claude Code session opened on a fresh clone can continue
without re-deriving decisions already made. Read `CLAUDE.md` first (invariants
and conventions), then `docs/DESIGN.md` (full architecture), then this.

Last updated after Phase 1.

---

## Status

**Phase 1 is complete**: the agent, its tools, and the trace format. ~4 000 lines,
117 tests, lint clean. `python tasks.py test` and `python tasks.py demo` both run
with no network and no credentials.

**Phases 2-4 are unbuilt.** `src/cra/data/`, `src/cra/retrieval/` and
`src/cra/eval/` contain only `__init__.py`. **There are no results yet**, and
there cannot be until Phase 2 exists. Anything in `results/` is an empty
placeholder.

What already works:

| Module | State |
|---|---|
| `types.py` | Trace / Question / Evidence / ToolCallRecord. Stable; treat as the contract. |
| `config.py` | Composable YAML configs + `config_hash` provenance. |
| `llm/` | LLMClient protocol, Anthropic, OpenAI-compat (Ollama), 2 mocks, response cache. |
| `agent/loop.py` | Controller, budget enforcement, repair round, forced finalisation. |
| `tools/` | 4 calculators, drug interactions, unit conversion, retrieval tool + registry. |
| `trace_io.py` | Read/write `.jsonl.gz`, plus `render_trace` for humans. |

---

## Decisions already taken -- do not relitigate

1. **Ollama only, two open-weight models**, run *sequentially*: `qwen3:8b` then
   `llama3.1:8b`. No paid API. Both are already pulled on the target machine.
2. **Target machine**: Windows, RTX 5050, **8 GB VRAM**. Use `tasks.py`, not `make`.
3. **300 questions per dataset** for headline runs, not the full 1 773. Seeded
   and stratified; report bootstrap CIs.
4. **`default_k` 5 -> 3 and `SNIPPET_CHARS` 1400 -> 1000.** Measured: at k=5 a
   five-search episode reaches ~10 700 tokens, which overflows a typical Ollama
   context. This is the single most important correctness issue in Phase 2 --
   see the warning below.
5. **Validation of the failure-mode classifier uses two annotators**: an
   automated blind pass plus the repo owner as a second human annotator, so
   inter-annotator Cohen's kappa can be reported, not just classifier agreement.
6. **Entailment checking runs locally** (NLI model), not via a paid judge.

### The context-overflow hazard (read before writing the rollout runner)

Ollama **silently truncates** prompts longer than `num_ctx`, dropping the
*oldest* tokens. In this agent the oldest tokens are the system prompt and the
clinical question. A truncated episode therefore looks like a reasoning failure
in the evaluation, when in fact the model never saw the question. That would
silently corrupt the headline result.

Required mitigations in Phase 2:

- Set `num_ctx` explicitly (8192 for 8 GB VRAM). Ollama's OpenAI-compatible
  endpoint does **not** accept `num_ctx`; it needs either a custom Modelfile
  `PARAMETER num_ctx 8192` or the `OLLAMA_CONTEXT_LENGTH` environment variable.
  Verify which against the installed Ollama version rather than assuming.
- Add a **preflight token check** that fails the run *before* it starts if any
  prompt could exceed the configured context. Do not rely on knowing defaults.
- Record `context_length` on every trace, so the setting is part of provenance
  and cannot drift between the two model runs.

---

## Phase 2 -- datasets, retrieval, rollout runner

### 2a. `src/cra/data/`

Both sources were verified reachable and parsed during Phase 1:

- **PubMedQA** (expert-annotated PQA-L, 1 000 items, includes the abstracts that
  serve as gold evidence):
  `https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json`
  and `.../data/test_ground_truth.json` (the official 500-item test split IDs).
- **MedQA (USMLE, 4-option) test split, 1 273 items**, and the PubMedQA test
  split with PMIDs, both from the MIRAGE benchmark file:
  `https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json`
  Structure: `{"medqa": {"0000": {"question", "options", "answer"}}, "pubmedqa":
  {"<PMID>": {"question", "options", "answer", "PMID"}}, ...}`

Build:
- `download.py` -- fetch to `data/raw/`, record URL + sha256 in
  `data/manifests/`, so an upstream change is caught rather than absorbed.
- `pubmedqa.py`, `medqa.py` -- parse into `Question` objects. Populate
  `gold_source_ids` from the PubMedQA PMID (this is what makes
  `retrieval_failure` decidable). MedQA has no gold source: leave it empty.
- `splits.py` -- `dev` and `test`, with a module-level guard that raises unless
  `CRA_ALLOW_TEST=1`. All development and threshold calibration on `dev`.
- `expected_tools` oracle -- rule-based detection of when a case supplies
  calculator inputs (e.g. AF + age/CHF/HTN/diabetes/stroke => CHA2DS2-VASc).
  **Hand-audit it on a sample and report its precision as a stated limitation**
  of the tool-misuse metric. It is the weakest link in the taxonomy.
- `trapset.py` + `data/trapset/trapset_v1.jsonl` -- 30-50 hand-written items,
  each with a plausible distractor the corpus does *not* support, and a recorded
  rationale per item.

### 2b. `src/cra/retrieval/`

- Corpus from PubMedQA abstracts (~1 000 from PQA-L; PQA-A can extend it for the
  corpus-size ablation). Gold PMID provenance is what makes Hit@k measurable.
- `bm25.py` using `rank-bm25` (already a dependency) -- the offline default.
- `dense.py` using FAISS + sentence-transformers -- behind the `dense` extra,
  optional because it pulls torch.
- Wire into the existing `SearchLiteratureTool`, which already takes any object
  satisfying the `Retriever` protocol.
- Honest limitation to state: this corpus matches PubMedQA topically but only
  partly matches MedQA, so MedQA retrieval will be weak. Report that as a
  finding; it is what the corpus-quality ablation is for.

### 2c. Rollout runner

`python tasks.py rollout --config headline_qwen3` should:
- load the experiment config, build the LLM client and tool registry,
- run the preflight context check (above) and abort on overflow,
- loop over questions, calling the existing `run_episode`,
- write `results/traces/<experiment_id>/*.jsonl.gz` via `trace_io.write_traces`,
- be resumable: skip questions already traced, and rely on the response cache,
- show progress and a running token/latency summary.

---

## Phase 3 -- the evaluation framework (the actual contribution)

Consumes traces only. **No model calls to re-derive anything the trace should
have recorded.**

- `claims.py` -- claim extraction: v1 rule-based sentence split, v2 LLM-based.
  Report agreement between them on `dev`.
- `entailment/` -- `EntailmentChecker` protocol; `nli.py` (DeBERTa-MNLI class,
  each claim scored against each passage *independently* to avoid premise
  truncation) and `judge.py` (local model against the committed rubric).
  Thresholds calibrated on `dev` only. Report cross-method agreement.
- `failure_modes.py` -- the ordered rules R1-R5 from `CLAUDE.md`. Plus
  independent boolean flags per trace, and a co-occurrence matrix.
  **Hallucination rate comes from the flags, never from the primary label.**
- `retrieval_metrics.py` (Hit@k, MRR vs gold PMID), `tool_metrics.py`
  (precision/recall vs the `expected_tools` oracle), bootstrap CIs, McNemar.
- `agreement.py` -- Cohen's kappa and confusion matrices, for classifier-vs-human
  *and* human-vs-human (see decision 5).
- `report.py` + `viz/plots.py` -- tables and the stacked failure-mode bar chart.

## What the pieces are for (the goal, stated plainly)

Two artefacts get produced, and they serve different purposes. Confusing them
is the most common way a project like this ends up unfalsifiable.

**The ~40 hand labels are not a result. They are a licence.** Their only job is
to establish whether the automatic failure-mode classifier can be trusted:

```
  40 traces labelled by hand  ─┐
                                ├─► Cohen's kappa ─► classifier trustworthy?
  same 40 labelled by the code ─┘        │
                                         ├─ high: apply the classifier to all traces
                                         └─ low : the classifier is wrong. Fix it and
                                                  re-run over the SAME traces -- free,
                                                  no GPU, no re-rollout. This is exactly
                                                  what the rollout/eval split buys.
```

**The model traces are the results.** Once the classifier is validated, run it
over all of them (300 questions x 2 datasets x 2 models = 1 200 traces) to produce:

- accuracy per model and dataset;
- the failure-mode breakdown -- the headline stacked bar chart;
- hallucination rate and grounding rate (from the per-trace flags, not the label);
- qwen3 vs llama3.1, to show whether a profile is one model's quirk;
- tools vs closed-book, to test whether grounding measurably reduces hallucination.

**The claim the project exists to support** has this shape, with the real numbers
deciding the wording: accuracy is X%, but N% of errors are retrieval failures
rather than reasoning failures -- the model reasons adequately over evidence it
never received -- and M% of *correct* answers contain at least one claim
unsupported by any retrieved evidence. Accuracy benchmarks score those as wins.
An agent can be right for the wrong reasons, and accuracy cannot see it.

**Report honestly.** Kappa may come back low; that is a finding about the
instrument, and fixing it is cheap. The cross-model comparison may be null. At
300 questions per dataset the confidence intervals are wide -- do not claim a
difference the CIs do not support.

## Phase 4 -- ablations, demo, write-up

- Ablations: tools vs closed-book (`configs/agent/closed_book.yaml` exists),
  budget 2 / 5 / unlimited (configs exist), corpus size/quality.
- Gradio demo (`src/cra/demo/app.py`), runnable on the mock client without
  credentials.
- `README.md` with setup, headline table, and the failure-mode chart inline.
  Write it PowerShell-first: the target machine is Windows.
- `writeup/paper.md`, 1 500-2 500 words. State plainly: two 8B open-weight
  models is a weaker generalisation claim than open-weight vs frontier; it shows
  whether a failure profile is one model's quirk, not how it shifts with
  capability.
- Verify a fresh clone reproduces the headline numbers from committed traces
  with no network and no GPU.

---

## Known gaps

- **`AgentConfig.mode == "react"` is declared but not implemented.** `cfg.mode`
  is only recorded on the trace; the loop always uses native function calling.
  A model without tool-calling support will silently produce closed-book traces.
  Either implement the ReAct fallback or keep to tool-capable models.
- The `expected_tools` oracle (Phase 2a) is rule-based and fallible by
  construction. Audit it and report its precision rather than presenting the
  tool-misuse label as ground truth.
