# Handoff: current state and what to build next

This file exists so a Claude Code session opened on a fresh clone can continue
without re-deriving decisions already made. Read `CLAUDE.md` first (invariants
and conventions), then `docs/DESIGN.md` (full architecture), then this.

Last updated after Phase 2.

---

## Status

**Phases 1 and 2 are complete**: the agent, its tools, the trace format, the
datasets, the retrieval corpus, and the rollout runner. ~4 050 lines of `src`,
153 tests, lint clean. `python tasks.py test` and `python tasks.py demo` both
run with no network and no credentials. `python tasks.py data` and
`python tasks.py index` were run against the live sources during this phase;
`data/manifests/` reflects a real download (sha256-checksummed).

**Phase 3 (the evaluation framework) is unbuilt.** `src/cra/eval/` contains
only `__init__.py` and `entailment/__init__.py`. **There are still no
headline results** -- that needs a real Ollama rollout on the target machine
plus the eval framework to turn traces into numbers. Anything in
`results/traces/` right now is whatever you've run locally; nothing is
committed as a reportable result yet.

What already works, beyond the Phase 1 table (`types.py`, `config.py`, `llm/`,
`agent/loop.py`, `tools/`, `trace_io.py` -- still stable, still the contract):

| Module | State |
|---|---|
| `data/download.py` | Fetches PubMedQA + MIRAGE to `data/raw/`, checksums into `data/manifests/`. Verified against the live sources. |
| `data/pubmedqa.py` | 1000 PQA-L items -> `Question`s. Split by official `test_ground_truth.json` membership (500/500). Abstract is *not* inlined -- it's a retrieval target. |
| `data/medqa.py` | 1273-item MIRAGE test split -> `Question`s. **No independent MedQA dev split exists in the allowed sources**; `dev` is a seeded, answer-stratified re-partition of the official test pool. Stated as a limitation in the module docstring -- don't over-read dev/test deltas for this dataset. |
| `data/splits.py` | `get_split()` (CRA_ALLOW_TEST-gated) + `sample_stratified()` (seeded, gold-answer-stratified, used for the "N per dataset" headline cap). |
| `data/expected_tools.py` | The keyword-rule tool-misuse oracle. Deliberately imprecise -- audit before trusting (see below). |
| `data/trapset.py` + `data/trapset/trapset_v1.jsonl` | 30 hand-written items, each a vignette with a plausible-but-wrong distractor and a recorded rationale. All pharmacology/calculator traps were checked against `cra.tools.drugs.INTERACTIONS` and the calculator formulae directly, not just written from memory. |
| `retrieval/corpus.py`, `retrieval/bm25.py` | Corpus built from PubMedQA abstracts (1000 docs, keyed by PMID); BM25 is the offline default, rebuilt in-process at experiment start (cheap at this scale, no persisted index format). Verified: the gold PMID ranks first for a sampled query. |
| `retrieval/dense.py` | FAISS + sentence-transformers, behind the `dense` extra. **Not exercised in this session** -- the extra wasn't installed. Written to the same `Retriever` protocol as BM25; sanity-check it before reporting any dense-retrieval numbers. |
| `rollout.py`, `cli.py` | `python tasks.py rollout -- --config <name>` (or `cra rollout --config <name>` once installed). Preflight context check, resumable, writes `results/traces/<experiment_id>/traces.jsonl.gz`. Exercised end-to-end with the mock client against the trap set and against `smoke_mock` (real PubMedQA + MedQA, mock model). **Not yet run against a live Ollama server** -- that needs the target Windows + RTX 5050 machine. |

---

## Decisions already taken -- do not relitigate

Phase 1 decisions (Ollama-only sequential models, Windows target machine, 300
questions/dataset, `default_k` 5->3 and `SNIPPET_CHARS` 1400->1000, two-annotator
validation, local NLI) still hold -- see the previous version of this file in
git history if you need the original reasoning. New in Phase 2:

7. **`expected_tools` oracle is keyword-regex, not NLP.** It fires
   `calc_cha2ds2_vasc` on any "atrial fibrillation" mention (no check that the
   question is actually asking about stroke risk), and `check_drug_interactions`
   on any 2+ recognized drug names (no check that an interaction is actually
   at issue). Both false positives are visible in the trap set itself
   (`trap-015`, `trap-022`) -- left uncorrected on purpose, as real examples for
   the precision/recall audit `docs/DESIGN.md` Sec 6 asks for. **Do this audit
   before Phase 3 reports a `tool_misuse` number.**
2. **MedQA's `dev` split is a re-partition, not an independent set** (see the
   table above). If Phase 3 threshold calibration behaves oddly on MedQA `dev`,
   check whether this is why before assuming it's a real signal.
3. **The preflight context check gates on `retrieval.default_k`, not the tool's
   `MAX_K` (10).** A model could in principle request `k=10` on every search
   and still overflow context even when the preflight passes; that case prints
   a `WARNING` (see `rollout.preflight_check`'s `worst_case_risky`) rather than
   blocking the run, because gating on `MAX_K` would make `agent/budget_unlimited`
   -- and most real configs -- fail preflight unconditionally. If Ollama runs
   start showing `terminated_by='error'` or reasoning failures that look like
   truncation, check for that warning first.
4. **BM25 needs no persisted index.** `RetrievalConfig.index_dir` exists in the
   config schema (and is inert) for forward compatibility with a future
   persisted/dense index; `BM25Retriever` tokenizes `corpus.jsonl` in-process
   at experiment start, which is well under a second at ~1000 documents.
5. **A latent bug in `HeuristicMockClient` was fixed in this phase**: it
   defaulted to answering `"A"` for any question with no lettered options,
   which is invalid for PubMedQA's yes/no/maybe answer space and made
   `smoke_mock` silently fail on every PubMedQA item. It now reads the
   "Allowed answers: ..." line `prompts.render_question` always appends. If
   you see old traces with `terminated_by='unparseable'` and an "answer 'A' is
   not one of the allowed answers" error on PubMedQA questions, they predate
   this fix and should be regenerated, not treated as a model failure.

---

## Phase 3 -- the evaluation framework (the actual contribution)

Unchanged from the original plan in `docs/DESIGN.md` Sec 6-7 and the prior
version of this file:

- `claims.py` -- claim extraction: v1 rule-based sentence split, v2 LLM-based.
  Report agreement between them on `dev`.
- `entailment/` -- `EntailmentChecker` protocol; `nli.py` (DeBERTa-MNLI class,
  each claim scored against each passage *independently* to avoid premise
  truncation) and `judge.py` (local model against the committed rubric).
  Thresholds calibrated on `dev` only. Report cross-method agreement.
- `failure_modes.py` -- the ordered rules R1-R5 from `CLAUDE.md`. Plus
  independent boolean flags per trace, and a co-occurrence matrix.
  **Hallucination rate comes from the flags, never from the primary label.**
  R2 (`tool_misuse`) should be reported alongside the oracle's audited
  precision/recall (see decision 1 above), not as ground truth.
- `retrieval_metrics.py` (Hit@k, MRR vs gold PMID -- only meaningful for
  PubMedQA; MedQA and the trap set have no gold source), `tool_metrics.py`
  (precision/recall vs the `expected_tools` oracle), bootstrap CIs, McNemar.
- `agreement.py` -- Cohen's kappa and confusion matrices, for classifier-vs-human
  *and* human-vs-human.
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
`src/cra/rollout.py::run_rollout` is what feeds this stage: point it at traces
via `results/traces/<experiment_id>/traces.jsonl.gz` and `trace_io.read_trace_dir`.

## Before the first real rollout, on the target machine

1. `python tasks.py doctor` -- confirms Ollama is serving `qwen3:8b` and
   `llama3.1:8b`.
2. `python tasks.py data` then `python tasks.py index` if `data/raw/` and
   `data/processed/corpus_pqal.jsonl` aren't already populated (they're
   gitignored; this session's copies don't travel with the repo).
3. `python tasks.py rollout -- --config smoke_mock` -- credential-free
   pipeline check, no reportable numbers.
4. `python tasks.py rollout -- --config headline_qwen3 --dry-run` -- confirms
   the preflight check passes (or explains why not) before spending real
   inference time. Watch for the `worst_case_risky` warning (decision 3
   above) even on a pass.
5. Only then drop `--dry-run` for the real run, and repeat for
   `headline_llama31`.

## Phase 4 -- ablations, demo, write-up

Unchanged from the original plan:

- Ablations: tools vs closed-book (`configs/agent/closed_book.yaml` exists),
  budget 2 / 5 / unlimited (configs exist -- note `budget_unlimited` may fail
  the preflight check on an 8192 num_ctx; that's a real, correctly-caught
  hazard, not a bug to route around), corpus size/quality.
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

- **`AgentConfig.mode == "react"` is declared but not implemented** (Phase 1
  gap, still open). `cfg.mode` is only recorded on the trace; the loop always
  uses native function calling. A model without tool-calling support will
  silently produce closed-book traces.
- **The `expected_tools` oracle is rule-based and fallible by construction**
  (decision 1 above). Audit it on a sample and report its precision/recall
  rather than presenting the tool-misuse label as ground truth.
- **`retrieval/dense.py` is untested in this environment** (the `dense` extra
  was not installed). It follows the same protocol as `BM25Retriever` and
  should be a drop-in swap via `retrieval.backend: dense`, but verify with a
  small corpus before trusting it for a reported ablation.
- **The rollout runner has not been exercised against a live Ollama server.**
  Everything upstream of the HTTP call (`preflight_check`, `resolve_context_length`'s
  `/api/show` query path, resumability, trace writing) is unit-tested with a
  faked response; the real integration point -- Ollama's actual `/api/show`
  payload shape and actual tool-calling behaviour for `qwen3:8b` /
  `llama3.1:8b` -- should be spot-checked on the target machine before a
  headline run.
