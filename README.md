# Clinical Reasoning Agent + Evidence-Grounding Benchmark

A tool-using clinical QA agent, plus an evaluation framework that measures *why* it
fails instead of only how often. The framework is the actual contribution here; the
agent mostly exists to produce traces to run it on.

The question I care about isn't "how often is the agent right". It's "when it is right,
is it right for the reasons it gives".

## What I found

Across two open-weight 8B models on the PubMedQA test split, somewhere between 74% and
90% of *correct* answers contain at least one claim that isn't entailed by the evidence
the agent retrieved to justify it. At the claim level, only 20-24% of extracted claims
are entailed by any retrieved passage.

That range is real uncertainty, not a hedge. 74.4% is the figure at the default
entailment cutoff of 0.5. Calibrating that cutoff on one annotation pass and reporting
it on a disjoint one gives 90.1%, and improves agreement with human labels (kappa 0.737
to 0.771). I don't adopt the calibrated cutoff, because only 1 of 24
threshold-sensitive held-out traces flips label between the two settings while the
headline number moves sixteen points. The direction survives every threshold in the
grid; the magnitude just isn't identified by 80 annotations. Details in
`results/tables/calibration.md`.

Accuracy can't see any of this. An agent that answers correctly and invents its
justification looks like a clean win on every standard benchmark.

![Failure-mode breakdown](results/figures/failure_modes.png)

### Retrieval isn't just decoration

The obvious objection is that the agent might be ignoring what it retrieves, in which
case of course its justifications aren't grounded in it. A closed-book ablation rules
that out. Same questions, tools removed entirely:

| | accuracy on PubMedQA | McNemar p | claims entailed |
|---|---|---|---|
| qwen3, tools to closed-book | 49.5% to 31.7% | 2.9e-07 | 22.0% to 11.4% |
| llama3.1, tools to closed-book | 63.3% to 47.3% | 4.5e-09 | 24.0% to 13.6% |

Retrieval is worth sixteen to eighteen accuracy points and roughly doubles the share of
entailed claims, and that replicates independently on both models. So the evidence
demonstrably does get used. The finding isn't that retrieval fails. Retrieval works, and
the stated reasoning still isn't a faithful account of it.

On MedQA the same ablation shows nothing significant (+4.0pp and -1.5pp), which is what
a corpus that doesn't cover USMLE vignettes should produce. That dataset ends up serving
as a negative control for retrieval quality.

One scoring detail: closed-book claims are checked against the passages the tool-using
run retrieved for the same question. Scoring them against their own empty evidence would
return 100% unsupported by construction and would be measuring the ablation rather than
the agent.

## Test-split results

300 questions per dataset per model, official test splits, run once.

| Model | Dataset | Accuracy (95% CI) | Hit@k | Tool P/R |
|---|---|---|---|---|
| qwen3:8b | MedQA | 76.5% [71.5, 81.2] | n/a | n/a (n=21, no tools used) |
| qwen3:8b | PubMedQA | 45.7% [40.3, 51.3] | 79.3% | 100.0% / 81.0% (n=300) |
| llama3.1:8b | MedQA | 58.1% [52.2, 63.6] | n/a | 95.0% / 90.5% (n=21) |
| llama3.1:8b | PubMedQA | 58.7% [53.0, 64.0] | 95.3% | 98.3% / 97.7% (n=300) |

The ranking crosses over depending on the dataset, with non-overlapping intervals in
both directions. Either model on its own would let you argue the opposite conclusion
about which one is better. That's the clearest reason running two models was substantive
rather than procedural.

Failure modes, as a percentage of 300:

| Model | Dataset | no answer | retrieval | tool misuse | reasoning | unsupported | grounded |
|---|---|---|---|---|---|---|---|
| qwen3:8b | MedQA | 8% | 22% | 0% | 2% | 71% | 0% |
| qwen3:8b | PubMedQA | 0% | 17% | 0% | 37% | 33% | 13% |
| llama3.1:8b | MedQA | 3% | 14% | 4% | 23% | 55% | 1% |
| llama3.1:8b | PubMedQA | 0% | 3% | 1% | 38% | 45% | 14% |

The two models fail differently at comparable accuracy. qwen3 on MedQA fails almost
entirely by never retrieving (22% retrieval failure against 2% reasoning failure), and
it only searched on 8% of MedQA questions. llama3.1 does retrieve, then misreads what it
gets. An intervention that helps one of them would do very little for the other.

## Validation

The classifier is validated against blind human annotation in two disjoint passes of
forty traces each. The development pass turned up three defects in the classifier and
drove two taxonomy corrections. The held-out pass, drawn from traces the annotator
hadn't seen and balanced across both models, then scored **Cohen's kappa 0.737, 82.5%
agreement**. Held-out kappa comes out above the in-sample figure of 0.673, so the
corrections generalise rather than just fitting the labels that prompted them.

Per-label agreement is strongest where the contribution is, 19 of 20 on
`unsupported_claim` and 8 of 9 on `reasoning_failure`, and weakest on
`retrieval_failure` at 3 of 6, where the classifier can catch a missed gold PMID or an
empty retrieval but can't see evidence that a reader judges retrieved-but-unhelpful.

An LLM-as-judge pass on a stratified sample gives 75.0% and 77.5% trace-level
hallucination against the NLI checker's 82-86%. Two instruments from different model
families landing within a few points is weak corroboration, but it's real.

## Other results

**Tool budget isn't a useful lever here.** At a budget of five the agent averages 2.19
calls; halving it to two gives 1.77 and doubling it to ten gives 2.30, with only 23 of
555 episodes going past five calls when allowed ten. The agent stops on its own well
before it hits any ceiling. The caveat is that the effect is small because *this* agent
under-uses its budget, not because budget can't matter in principle. I tried a fifty-call
arm and the preflight refused it: its worst-case prompt is roughly 40k tokens against a
12,288 window, and 8 GB of VRAM can't serve that.

**The adversarial trap set doesn't trap.** Thirty hand-written items, each pairing a
clinical vignette with a planted distractor. Both models resisted almost completely
(qwen3 30 of 30, llama3.1 28 of 30, against 25% chance). The items are badly designed:
each one pairs a recklessly wrong distractor with a cautious, hedged gold answer, so a
model with ordinary medical priors picks the cautious option without needing evidence at
all. They test a preference for careful phrasing, not resistance to unsupported claims.

The grounding measurement on those same traces is still the sharpest version of the main
result. qwen3 answered all thirty correctly while searching on three of them, and 96% of
its traces carry at least one unsupported claim (21% of claims entailed). Getting the
right answer and justifying it from evidence are separable, and these items separate them
completely.

## Setup

Python 3.11 exactly, since several pinned dependencies have no wheels for newer
versions, plus [uv](https://docs.astral.sh/uv/), which can fetch 3.11 for you.

```powershell
git clone https://github.com/Ritwick14999/Project_new.git
cd Project_new
python tasks.py setup --extras dev,dense
python tasks.py doctor
python tasks.py test
```

`tasks.py` works the same on Windows, macOS and Linux; the `Makefile` has the same
targets if you're on a POSIX shell. `doctor` checks the Python version, the virtualenv,
and whether Ollama is serving the models a rollout needs.

## Reproducing

Traces are committed and evaluation is a pure function of them, so every number and
figure regenerates with no GPU, no model and no network:

```powershell
python -m cra.cli eval --experiments final_qwen3 final_llama31 --nli
```

Re-running the agent itself needs Ollama. One detail matters here and is easy to get
wrong: Ollama serves a 4,096-token context by default and truncates longer prompts
silently, dropping the oldest tokens, which are the system prompt and the question. The
longest episode in the final run reached 7,755 tokens. The window has to be raised
server-side, since the OpenAI-compatible endpoint ignores `options.num_ctx`, so the runs
use a Modelfile variant:

```powershell
ollama pull qwen3:8b
ollama create qwen3-8k:8b -f Modelfile     # FROM qwen3:8b + PARAMETER num_ctx 12288
python tasks.py data
python tasks.py index
python tasks.py rollout -- --config final_qwen3 --dry-run
```

The preflight sends a prompt of the expected size and checks that the server actually
evaluated all of it, then refuses to start a run whose prompts would be truncated.
Reading the config isn't evidence: the value can be unset, overridden per-model, or
changed on the server.

## How the evaluation works

Two stages, separated by a self-contained trace file.

```
ROLLOUT  (needs Ollama, stochastic, slow)  ->  results/traces/<exp>/*.jsonl.gz
EVAL     (pure function of traces)         ->  tables + figures
```

A trace inlines the full text of every retrieved passage and every tool output, so it
stays interpretable after the retrieval index is rebuilt, and the instrument can be
improved and re-run over existing traces at no rollout cost. I leaned on that property
repeatedly: eight defects found during development and annotation were each corrected
and every affected number re-scored without re-running a single model.

Each trace gets exactly one primary label by ordered rules: retrieval failure, then tool
misuse, then reasoning failure, then unsupported claim (a correct answer with an
ungrounded justification), then correct and grounded. A missing answer is kept distinct
from a wrong one. Hallucination is also an independent boolean on every trace, and the
hallucination rate comes from those flags rather than from the primary label. Deriving it
from the label would make hallucinations inside wrong answers invisible.

## Demo

```powershell
python tasks.py setup --extras dev,demo
python tasks.py demo-app
```

Type in a clinical question and watch the agent's tool calls, the evidence it retrieves,
and the per-claim grounding check. It runs on a deterministic mock model by default, so
it needs no credentials; point it at Ollama to drive a real one.

## Limitations

Two 8B open-weight models is a weaker generalisation claim than open-weight against
frontier would be. It shows a failure profile isn't one model's quirk, but says nothing
about how that profile changes with capability.

The MedQA grounding figures of 98-100% are measuring corpus coverage, not fabrication.
The retrieval corpus is built from PubMedQA abstracts and doesn't cover USMLE vignettes,
and qwen3 answers most MedQA questions without searching at all. PubMedQA is where the
grounding measure means what it says.

The `expected_tools` oracle is a keyword rule whose own error rate is uncharacterised, so
tool precision and recall are reported against the oracle rather than against ground
truth. It fires on only 21 of 300 MedQA questions, so the blast radius is small, but it
still isn't ground truth.

The entailment threshold's magnitude is unidentified, as described above. Pinning it down
would need roughly 150-200 labelled traces concentrated on correct answers, rather than
80 spread across all outcomes.

The second annotation pass wasn't blind to feedback from the first, and used a single
annotator who isn't a clinician, so inter-annotator agreement is unmeasured.

The trap set doesn't discriminate, for the reasons given above. A useful version needs
distractors that are plausible and cautiously phrased, and that differ from the gold
answer in a factual claim the corpus can actually settle.

## Layout

```
src/cra/
  types.py          Trace / Question / Evidence / ToolCallRecord, the contract
  llm/              provider-agnostic client; Ollama, Anthropic, mocks, response cache
  agent/            control loop, prompts, final-answer parsing
  tools/            retrieval, drug interactions, four clinical calculators, unit conversion
  retrieval/        BM25 (default) and dense indexes over the PubMedQA corpus
  data/             dataset download, splits, trap set, expected-tools oracle
  eval/             claims, entailment, failure modes, calibration, ablations, annotation
configs/            composable YAML: model / agent / retrieval / experiments
results/METRICS.md  every reported number with the caveat that constrains it
results/traces/     committed traces, the reproducibility anchor
docs/DESIGN.md      design rationale
writeup/paper.md    the write-up
```

301 tests, no network or credentials required: `python tasks.py test`.

## Data

No patient data. PubMedQA and MedQA are public benchmarks fetched by `python tasks.py
data` from `pubmedqa/pubmedqa` and the MIRAGE benchmark, recorded with URL and SHA-256 in
`data/manifests/` and not redistributed here.

The drug-interaction table is a curated teaching table I assembled for benchmarking. It's
deliberately incomplete and is not clinical decision support.
