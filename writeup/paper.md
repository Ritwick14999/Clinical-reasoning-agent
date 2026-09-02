# Right for the Wrong Reasons: Measuring Evidence Grounding in Tool-Using Clinical QA Agents

## Abstract

Retrieval-augmented agents are evaluated almost entirely on answer accuracy. Accuracy
cannot distinguish an agent that reasons correctly from evidence it retrieved from one
that retrieves evidence, ignores it, and happens to be right. We build a tool-using
clinical reasoning agent and an evaluation framework that classifies *why* each episode
fails, including whether the agent's own justification is entailed by the passages it
retrieved. On the PubMedQA test split, across two open-weight 8B models, **74.4% of
correct answers contain at least one claim not entailed by the evidence retrieved to
justify it.** The failure-mode classifier is validated against blind human annotation at
Cohen's kappa = 0.737 on a held-out sample. We also find that the two models' accuracy
ranking inverts between datasets, and that their failure profiles differ sharply at
comparable accuracy — one fails by not retrieving, the other by misreading what it
retrieves.

## 1. Motivation

A clinical QA agent that answers 76% of USMLE-style questions correctly sounds useful.
The number says nothing about whether the agent's reasoning is connected to the evidence
it cited. In a deployed setting the justification is what a clinician actually reads, and
a fluent, confident, unsupported justification is worse than an obviously uncertain one.

Standard benchmarks are blind to this by construction. They compare a predicted label to
a gold label. Two agents with identical accuracy — one grounded, one confabulating — are
indistinguishable. The gap matters most exactly where these systems are most attractive:
domains where the user cannot easily verify the answer themselves.

This project asks a different question. Not *how often is the agent right*, but *when it
is right, is it right for the reasons it gives*; and when it is wrong, *at which stage
did it break* — retrieval, tool use, or reasoning.

## 2. Method

### 2.1 The agent

A native function-calling loop with a hard tool-call budget (default 5). Tools: literature
retrieval over a BM25 index, a drug-interaction checker, four clinical calculators
(CHA2DS2-VASc, Wells' PE, MELD/MELD-Na, anion gap) implemented as pure total functions
with published formulae, and a lab unit converter. The agent must emit a final answer as
JSON with an explicit list of the evidence IDs it relied on.

Three design commitments exist to keep failure modes measurable rather than to make the
agent better:

**The tool budget is enforced by withholding tool schemas**, not by prompting. Prompt-level
budgets are routinely violated; a schema the model was never shown cannot be called.

**Nothing a model emits may raise.** Unknown tool, schema violation, unparseable arguments,
tool exception, provider outage — all become recorded observations. A crash would destroy
exactly the traces the tool-misuse analysis needs.

**Evidence IDs are assigned once, centrally.** The IDs the model is shown are the IDs the
evaluator resolves citations against.

### 2.2 The two-stage architecture

Rollout and evaluation are separated by a self-contained trace file. A trace inlines the
full text of every retrieved passage and tool output, so it remains interpretable after
the retrieval index is rebuilt, and so evaluation is a pure function of committed traces —
no network, no model, no GPU.

This is not a packaging convenience. It is what made the project's central finding
trustworthy. Human annotation exposed three defects in the classifier; each was corrected
and every result re-scored **without re-running a single model**. Under a coupled design,
each correction would have cost eight hours of local inference, and the realistic outcome
is that fewer corrections get made.

### 2.3 Failure-mode taxonomy

Ordered rules assign exactly one primary label per trace: `retrieval_failure` →
`tool_misuse` → `reasoning_failure` → `unsupported_claim` (correct answer, ungrounded
justification) → `correct_grounded`, with `no_answer` kept distinct from a wrong answer.

Hallucination is **also** an independent boolean on every trace, and the hallucination
rate is computed from those flags rather than from the primary label. Deriving it from the
label would make hallucinations inside wrong answers invisible to the metric, which is
precisely the population one should worry about most.

### 2.4 Grounding check

Justifications are split into sentences, filtered to assertions, and each claim is scored
against **each retrieved passage independently** by an NLI model (DeBERTa-v3
MNLI-FEVER-ANLI), taking the maximum entailment score. Independent scoring avoids the
premise truncation that concatenation would introduce.

Two normalisations proved necessary, both discovered through human annotation. The agent
is *instructed* to cite its sources, so justifications read "E1 states that X". Asking an
NLI model whether an abstract entails a statement about what E1 says is structurally
unanswerable regardless of the abstract's content: measured over the traces, claims
mentioning an evidence ID were scored entailed 14% of the time against 36% for claims
without one. Citation framing is now stripped before the claim becomes a hypothesis.
Separately, sentences like "E2 is unrelated to resuscitation devices" are the agent
correctly noticing bad retrieval — the opposite of the behaviour the metric should
penalise — and are filtered as non-assertions.

## 3. Experimental setup

Two open-weight models run sequentially through Ollama: `qwen3:8b` and `llama3.1:8b`, both
with native tool calling. 300 questions per dataset per model on the official test splits
of MedQA (USMLE, 4-option) and PubMedQA, giving 1,200 episodes. Retrieval corpus: PubMedQA
abstracts, which carry gold PMID provenance and therefore make Hit@k — and hence
`retrieval_failure` — decidable.

One implementation detail is load-bearing. Ollama serves a 4,096-token context by default
and **truncates over-long prompts silently, from the oldest tokens** — which are the system
prompt and the question. A truncated episode looks exactly like a reasoning failure. The
longest llama3.1 episode in the final run reached 7,755 tokens. Running against the default
would have corrupted the headline result with nothing in the traces to reveal it. The
preflight therefore sends a prompt of the expected size and verifies the server evaluated
all of it; a configuration read is not evidence, because the value can be unset, overridden
per-model, or changed server-side.

## 4. Results

### 4.1 Accuracy inverts between datasets

| Model | MedQA | PubMedQA |
|---|---|---|
| qwen3:8b | **76.5%** [71.5, 81.2] | 45.7% [40.3, 51.3] |
| llama3.1:8b | 58.1% [52.2, 63.6] | **58.7%** [53.0, 64.0] |

The ranking reverses, with non-overlapping 95% intervals in both directions. A
single-model study would have supported either conclusion depending on which model was
chosen. This is the clearest available argument for the multi-model requirement being
substantive rather than procedural.

### 4.2 The grounding result

Of answers that were **correct**, the share containing at least one claim not entailed by
the retrieved evidence:

| Model | PubMedQA | MedQA |
|---|---|---|
| qwen3:8b | 71.5% (98/137) | 100% (212/212) |
| llama3.1:8b | 76.7% (135/176) | 98.2% (166/169) |

**74.4% combined on PubMedQA.** At the claim level, only 20–24% of extracted claims are
entailed by any retrieved passage.

We report PubMedQA as the headline because it is where the measure means what it says:
Hit@k is 79–95%, so relevant evidence is genuinely in hand. The MedQA figures should not
be read as a fabrication rate. The corpus is built from PubMedQA abstracts and does not
cover USMLE vignettes, and qwen3 answers most MedQA questions without searching at all —
those numbers measure corpus coverage and retrieval behaviour.

An independent LLM-as-judge pass on a stratified dev sample gives 75.0% and 77.5% for the
two models, against the NLI checker's 82–86% of traces carrying at least one unsupported
claim. Two instruments of different families agreeing to within a few points is weak but
real corroboration.

### 4.3 Failure profiles differ at comparable accuracy

On MedQA, qwen3 shows 22% `retrieval_failure` and **0%** `reasoning_failure`; llama3.1
shows 14% and 23%. qwen3 searched on only 8% of MedQA questions. The two models are not
better and worse versions of each other — they fail at different stages, and an
intervention that helps one (better retrieval prompting) would do little for the other
(better evidence integration).

### 4.4 Classifier validation

Blind human annotation, two passes kept separate. A development pass of 40 traces found
three defects. A held-out pass of 40 unseen traces, balanced across both models, scored
after the corrections: **kappa = 0.737, 82.5% agreement**.

Agreement is strongest where the contribution lies — 19/20 on `unsupported_claim`, 8/9 on
`reasoning_failure` — and weakest on `retrieval_failure` (3/6), where the classifier
detects a missed gold PMID or an empty retrieval but cannot see evidence a human judges
retrieved-but-unhelpful.

Held-out kappa (0.737) exceeds in-sample kappa (0.673), indicating the corrections
generalise rather than fitting the labels that prompted them.

## 5. Discussion

### What human annotation actually bought

The annotation did not merely certify the classifier; it corrected it, three times.

The annotator used `tool_misuse` zero times across 40 traces while the classifier assigned
it to 27–30% of llama3.1's episodes. Investigation showed **151 of 159 such labels were
traces where the agent emitted a malformed tool call, immediately retried, and succeeded**.
A recovered formatting slip explains nothing, yet it outranked `reasoning_failure` in the
precedence and captured the label. Before this correction, the reported cross-model finding
was "llama3.1 fails by tool misuse, qwen3 by reasoning" — an artifact that does not survive.

The annotator also labelled MedQA traces as retrieval failures where the classifier could
not, because `retrieval_failure` fired only on a missed gold PMID and MedQA has no gold
source. The taxonomy's own wording — correct evidence existed but was not retrieved — is
satisfied just as plainly by retrieving nothing at all.

The lesson generalises beyond this project: a validation step that can only confirm or deny
is worth much less than one that can locate the defect. Reporting kappa without reporting
what the disagreements *were* discards most of the value.

### On disclosure

Nine of the first pass's forty labels contradicted the objective gold-answer comparison,
because the label scheme required the annotator to encode correctness — a string comparison,
not a judgement. The annotation template was rebuilt to state the answer status and offer
only labels consistent with it. This is a design flaw in the task, not annotator error, and
the fact that it survived into a first pass is itself a finding about how easily annotation
protocols encode avoidable ambiguity.

## 6. Limitations

Two 8B open-weight models is a weaker generalisation claim than open-weight versus frontier:
it shows a failure profile is not one model's quirk, but says nothing about how the profile
changes with capability. The `expected_tools` oracle is a keyword rule whose own error rate
is uncharacterised, so tool precision/recall are reported against the oracle rather than
against ground truth. NLI thresholds sit at their 0.5 default; calibrating them on the
annotation labels would have fitted the classifier to its own validation set, so it was not
done, and the residual `correct_grounded`/`unsupported_claim` disagreements are partly
attributable to that. The second annotation pass was not blind to feedback from the first,
and used a single annotator who is not a clinician — inter-annotator agreement is therefore
unmeasured. Finally, the retrieval corpus is small and topically mismatched to MedQA, which
is why MedQA grounding figures are reported as a coverage artifact rather than a headline.

## 7. Conclusion

Accuracy is a poor proxy for reliability in retrieval-augmented clinical QA. Measuring
grounding directly is feasible, cheap once traces exist, and produces a number that
accuracy cannot: roughly three quarters of correct answers rest on justifications the
retrieved evidence does not support. The framework that produces it agrees with human
judgement at kappa = 0.737 out of sample, and — more usefully — was corrected three times
by that human judgement at zero re-inference cost, because evaluation was kept a pure
function of committed traces.
