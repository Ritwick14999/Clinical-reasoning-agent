# Clinical Reasoning Agent + Evidence-Grounding Benchmark

A tool-using clinical reasoning agent, and an evaluation framework that measures
**why** it fails — including whether its final claims are actually entailed by the
evidence it retrieved.

The evaluation framework is the contribution. The agent exists to generate traces for it.

## Headline result

> Across two open-weight models on PubMedQA, **74–90% of *correct* answers contain at
> least one claim not entailed by the evidence retrieved to justify it.**

Accuracy cannot see this. An agent that answers correctly while inventing its
justification scores as a clean win on every standard benchmark.

The failure-mode classifier that produces this number is validated against blind human
annotation at **Cohen's kappa = 0.737** (82.5% agreement, n=40, held out).

The range is a stated uncertainty, not hedging. 74.4% is the figure at the default
entailment cut of 0.5; a cut calibrated on one annotation pass and reported on a
disjoint one (kappa 0.737 → 0.771) gives 90.1%. The calibrated cut is **not** adopted,
because only 1 of 24 threshold-sensitive held-out traces changes label between the two
— the direction of the finding is robust to the threshold, its magnitude is not pinned
down by 80 annotations. See `results/tables/calibration.md`.

Removing tools entirely makes it worse on both counts, which rules out the obvious
objection that retrieval is decorative: accuracy falls 16–18 points on PubMedQA
(p < 10⁻⁷, both models) and the share of entailed claims roughly halves. The evidence is
demonstrably used — and the justification still is not a faithful account of it.

![Failure-mode breakdown](results/figures/failure_modes.png)

## Results (test split, 300 questions per dataset per model)

| Model | Dataset | Accuracy (95% CI) | Hit@k | Tool P/R |
|---|---|---|---|---|
| qwen3:8b | MedQA | **76.5%** [71.5, 81.2] | n/a | n/a (n=21, no tools used) |
| qwen3:8b | PubMedQA | 45.7% [40.3, 51.3] | 79.3% | 100.0% / 81.0% (n=300) |
| llama3.1:8b | MedQA | 58.1% [52.2, 63.6] | n/a | 95.0% / 90.5% (n=21) |
| llama3.1:8b | PubMedQA | **58.7%** [53.0, 64.0] | 95.3% | 98.3% / 97.7% (n=300) |

**The ranking crosses over by dataset**, with non-overlapping intervals in both
directions. Either model alone would have supported the opposite conclusion — which is
why the protocol requires two.

### Failure modes

| Model | Dataset | no answer | retrieval | tool misuse | reasoning | unsupported | grounded |
|---|---|---|---|---|---|---|---|
| qwen3:8b | MedQA | 8% | **22%** | 0% | 0% | 71% | 0% |
| qwen3:8b | PubMedQA | 0% | 17% | 0% | 37% | 33% | 13% |
| llama3.1:8b | MedQA | 3% | 14% | 4% | 23% | 55% | 1% |
| llama3.1:8b | PubMedQA | 0% | 3% | 1% | 38% | 45% | 14% |

The two models fail differently at comparable accuracy. **qwen3 fails by not looking
things up** (22% retrieval failure, 0% reasoning failure on MedQA). **llama3.1 looks
things up and misreads them** (14% vs 23%). A single-model study would have reported
one of these as "the" failure profile.

## Setup

Requires **Python 3.11 exactly** (pinned dependencies have no wheels for newer
versions) and [uv](https://docs.astral.sh/uv/), which can fetch 3.11 for you.

Windows (PowerShell):

```powershell
git clone https://github.com/Ritwick14999/Project_new.git
cd Project_new
python tasks.py setup --extras dev,dense
python tasks.py doctor
python tasks.py test
```

macOS / Linux: the `Makefile` has the same targets, or use `tasks.py` identically.

`python tasks.py doctor` checks the Python version, the virtualenv, and whether Ollama
is serving the models a rollout needs.

## Reproducing the numbers

Traces are committed, and evaluation is a pure function of traces — no GPU, no model,
no network:

```powershell
python -m cra.cli eval --experiments final_qwen3 final_llama31 --nli
```

That regenerates every table and the figure above from `results/traces/`.

To re-run the agent itself you need Ollama and the two models:

```powershell
ollama pull qwen3:8b
ollama pull llama3.1:8b
# Ollama serves a 4096-token window by default and truncates silently above it.
# The longest episodes here reach 7755 tokens, so the window must be raised
# server-side; the OpenAI-compatible endpoint ignores options.num_ctx.
ollama create qwen3-8k:8b -f Modelfile     # FROM qwen3:8b + PARAMETER num_ctx 12288
python tasks.py data
python tasks.py index
python tasks.py rollout -- --config final_qwen3 --dry-run   # preflight only
```

The preflight sends a prompt of the expected size and verifies the server evaluated all
of it. It refuses to start a run whose prompts would be truncated.

## How the evaluation works

Two stages, separated by a self-contained trace file:

```
ROLLOUT  (needs Ollama; stochastic; slow)  ->  results/traces/<exp>/*.jsonl.gz
EVAL     (pure function of traces)         ->  tables + figures
```

A trace inlines the full text of every retrieved passage and tool output, so it stays
interpretable after the retrieval index is rebuilt, and so the instrument can be
improved and re-run over existing traces at zero rollout cost. That property was used
repeatedly: three classifier defects found during human annotation were each corrected
and re-scored without re-running a single model.

Each trace gets exactly one primary label by ordered rules — `retrieval_failure` →
`tool_misuse` → `reasoning_failure` → `unsupported_claim` (correct answer, ungrounded
justification) → `correct_grounded`. Hallucination is **also** an independent boolean on
every trace, and the hallucination rate is computed from those flags rather than from
the primary label; otherwise hallucinations inside wrong answers would vanish from the
metric.

## Validation

The classifier was validated in two passes, kept deliberately separate:

- A **development pass** (40 traces) that found three real defects: recovered tool
  errors capturing the `tool_misuse` label, retrieval failure being undetectable without
  gold provenance, and citation framing inflating the hallucination rate.
- A **held-out pass** (40 unseen traces, balanced across both models) scored *after*
  those corrections: **kappa = 0.737, 82.5% agreement**.

Agreement is strongest where the contribution lies — 19/20 on `unsupported_claim`, 8/9
on `reasoning_failure` — and weakest on `retrieval_failure` (3/6), where the classifier
cannot see evidence a reader judges retrieved-but-unhelpful.

Held-out kappa (0.737) exceeds in-sample kappa (0.673), so the corrections generalise
rather than fitting the labels that prompted them.

## Limitations

- **Two 8B open-weight models** is a weaker generalisation claim than open-weight vs
  frontier. It shows a failure profile is not one model's quirk; it says nothing about
  how the profile changes with capability.
- **MedQA grounding figures (98–100%) measure corpus coverage, not fabrication.** The
  retrieval corpus is built from PubMedQA abstracts and does not cover USMLE vignettes,
  and qwen3 answers most MedQA questions without searching at all. PubMedQA is where the
  grounding measure means what it says.
- **The `expected_tools` oracle is a keyword rule**, not ground truth. Tool
  precision/recall are reported against the oracle, and its own error rate is not
  characterised.
- **The entailment threshold is not pinned down.** It is calibrated non-circularly
  (fitted on the development annotation pass, reported on the disjoint held-out pass) and
  the fitted cut of 0.95 improves held-out kappa from 0.737 to 0.771. It is nonetheless
  left at the 0.5 default, because only 1 of 24 threshold-sensitive held-out traces
  changes label between the two cuts while the headline moves 16 points. Pinning the
  magnitude would need roughly 150–200 labelled traces concentrated on correct answers,
  not 80 spread across all outcomes.
- **The second annotation pass was not blind to feedback** from the first, and used a
  single annotator who is not a clinician.
- **`AgentConfig.mode == "react"` is declared but not implemented.** The loop uses native
  function calling only.

## Layout

```
src/cra/
  types.py          Trace / Question / Evidence / ToolCallRecord — the contract
  llm/              provider-agnostic client; Ollama, Anthropic, mocks, response cache
  agent/            control loop, prompts, final-answer parsing
  tools/            retrieval, drug interactions, 4 clinical calculators, unit conversion
  retrieval/        BM25 (default) and dense indexes over the PubMedQA corpus
  data/             dataset download, splits, trap set, expected-tools oracle
  eval/             claims, entailment, failure modes, metrics, annotation harness
configs/            composable YAML: model / agent / retrieval / experiments
results/traces/     committed traces — the reproducibility anchor
docs/DESIGN.md      full design rationale
```

265 tests, no network and no credentials required: `python tasks.py test`.

## Data and licensing

No patient data. PubMedQA and MedQA are public QA benchmarks, downloaded by
`python tasks.py data` and not redistributed here — `data/` is gitignored except for
manifests (URLs and checksums), the hand-curated trap set, and human annotation labels.

The drug-interaction table is a curated teaching table assembled for benchmarking. It is
deliberately incomplete and is **not clinical decision support**.
