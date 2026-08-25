# Design Proposal — Clinical Reasoning Agent + Evidence-Grounding Benchmark

Status: **proposed, awaiting sign-off**. No tool code written yet.

This document is the contract for Phases 1–4. The evaluation framework is the
contribution; the agent exists to generate traces that the framework analyses.

---

## 0. Environment constraints discovered (these shape the design)

Probed in the build environment on 2026-08-25:

| Capability | Status | Consequence |
|---|---|---|
| `pypi.org`, `files.pythonhosted.org` | reachable | dependency install works |
| `raw.githubusercontent.com` | reachable | datasets can be fetched (see §4) |
| `eutils.ncbi.nlm.nih.gov` (PubMed/Entrez) | **403 — blocked by egress policy** | live PubMed retrieval unusable here |
| `api.fda.gov` (openFDA) | **403 — blocked** | live drug-interaction API unusable here |
| `huggingface.co` | **403 — blocked** | no HF datasets, no NLI model download here |
| `ollama.com` / `registry.ollama.ai` | unreachable | no local open-weight model here |
| `ANTHROPIC_API_KEY` / any LLM key | **absent** | no real LLM rollouts can be run here |
| Hardware | 4 vCPU / 15 GB RAM / no GPU | no local 7B+ inference here anyway |

**Design response — offline-first, provider-agnostic, two-stage pipeline.**
Every network-dependent component sits behind an interface with a local
implementation that works with zero egress. The expensive, key-requiring stage
(rollouts) is fully separated from the analysis stage (eval), and traces are
committed, so headline numbers are reproducible from a fresh clone with **no
network and no API key**. Live-API adapters are still written and tested against
recorded fixtures, so they work on a machine with open egress.

This is not a workaround bolted on — it is how a benchmark should be built
anyway: rollouts are expensive and stochastic, analysis should be re-runnable.

---

## 1. Two-stage pipeline (the central architectural decision)

```
                   stage 1: ROLLOUT (needs LLM key; stochastic; expensive)
  dataset ──▶ agent loop ──▶ tools ──▶ Trace ──▶ results/traces/<exp_id>/*.jsonl.gz
                                                        │  (committed to git)
                   stage 2: EVAL (pure function of traces; no network)
                                                        ▼
   traces ──▶ claim extraction ──▶ entailment ──▶ failure-mode labels ──▶ tables + figures
```

Consequences, all of them deliberate:

- The eval can be improved (better claim extractor, recalibrated thresholds)
  and **re-run over existing traces at zero rollout cost** — essential when
  iterating on a measurement instrument.
- `make results` regenerates every number and figure in the README from
  committed traces. That is what satisfies "fresh clone reproduces headline
  numbers with no manual intervention", honestly, despite LLM nondeterminism.
- `make rollouts` re-runs stage 1 (needs keys) and is verified to reproduce
  committed traces bit-for-bit under the response cache, and statistically
  (within CI) without it.

**The Trace is the interface between the stages, so it must be self-contained**:
full text of every retrieved passage and tool output is inlined, not referenced
by ID into an index that may later change.

---

## 2. Repo structure

```
.
├── README.md                    # setup, headline table, failure-mode chart (inline)
├── pyproject.toml               # py3.11, all deps pinned (uv.lock committed)
├── Makefile                     # setup | data | index | rollouts | eval | results | demo | test
├── docs/DESIGN.md               # this file
├── writeup/paper.md             # 1500–2500 word arXiv-style write-up
├── configs/
│   ├── model/                   # claude_sonnet.yaml, claude_haiku.yaml, qwen3_8b_ollama.yaml, mock.yaml
│   ├── agent/                   # fc.yaml (function calling), react.yaml, closed_book.yaml
│   ├── retrieval/               # bm25.yaml, dense_small.yaml, dense_full.yaml
│   ├── eval/                    # judge.yaml, nli.yaml
│   └── experiments/             # headline.yaml, ablation_{tools,budget,corpus}.yaml
├── src/cra/
│   ├── types.py                 # Trace / Step / ToolCall / Evidence / FinalAnswer (pydantic v2)
│   ├── config.py                # yaml -> frozen dataclasses + config hash for provenance
│   ├── llm/
│   │   ├── base.py              # LLMClient protocol, Message, ToolSpec, LLMResponse
│   │   ├── anthropic_client.py  # native tool-use
│   │   ├── openai_compat.py     # vLLM / Ollama / OpenAI — one adapter, three backends
│   │   ├── mock.py              # deterministic scripted client: CI + demo without keys
│   │   └── cache.py             # content-addressed on-disk response cache
│   ├── agent/
│   │   ├── loop.py              # the controller (§3)
│   │   ├── prompts.py           # system prompt + final-answer contract
│   │   ├── budget.py            # tool-call budget enforcement
│   │   └── parsing.py           # final-answer parse + one repair round
│   ├── tools/
│   │   ├── base.py              # Tool protocol: name / json_schema / run() -> ToolResult
│   │   ├── registry.py          # dispatch, arg validation, error capture
│   │   ├── retrieval.py         # search_literature
│   │   ├── drugs.py             # check_drug_interactions (local table | openFDA adapter)
│   │   ├── calculators/         # chads_vasc.py wells_pe.py meld.py anion_gap.py
│   │   └── units.py             # convert_lab_units (optional)
│   ├── retrieval/               # corpus.py bm25.py dense.py index_build.py
│   ├── data/                    # download.py medqa.py pubmedqa.py trapset.py splits.py
│   ├── eval/
│   │   ├── run.py               # traces -> per-trace records -> metrics
│   │   ├── accuracy.py  retrieval_metrics.py  tool_metrics.py
│   │   ├── claims.py            # claim extraction (rule-based v1, LLM v2)
│   │   ├── entailment/          # base.py nli.py judge.py rubric.md
│   │   ├── failure_modes.py     # the ordered decision procedure (§6)
│   │   ├── agreement.py         # Cohen's kappa, confusion matrix vs manual labels
│   │   └── report.py            # markdown/LaTeX tables
│   ├── viz/plots.py             # failure-mode stacked bar, ablation plots
│   └── demo/app.py              # Gradio: live tool calls + grounding check
├── scripts/                     # thin CLI entrypoints only
├── data/                        # gitignored EXCEPT:
│   ├── manifests/*.json         # URLs + sha256 of every raw file
│   ├── trapset/trapset_v1.jsonl # hand-curated adversarial set (committed)
│   └── manual_labels/*.jsonl    # human failure-mode labels (committed)
├── results/
│   ├── traces/<exp_id>/*.jsonl.gz   # committed — the reproducibility anchor
│   ├── tables/  figures/
└── tests/                       # unit (calculators, parsing, budget) + golden-trace + e2e-with-mock
```

Rationale for `src/cra/` over the requested flat `/agent /tools /eval`: one
installable package avoids `sys.path` hacks and makes `pytest` + the Gradio app
import cleanly. The requested top-level concepts map 1:1 onto subpackages, and
`/data`, `/results`, `/configs` stay at top level where they are expected.
**Say the word and I will flatten it instead** — this is cosmetic, not structural.

---

## 3. Agent core loop

Primary mode is **native function calling**; a **ReAct text mode** is a fallback
for models without tool-calling APIs. Both produce the identical `Trace`, so the
eval never knows which was used.

```python
def run_episode(question, registry, llm, cfg) -> Trace:
    trace = Trace.new(question, cfg)                    # records config_hash, seed, model
    messages = [system(prompts.SYSTEM), user(render(question))]

    for step in range(cfg.max_steps):
        remaining = cfg.tool_budget - trace.n_tool_calls
        # Budget is enforced by WITHHOLDING TOOL SCHEMAS, never by asking nicely.
        exposed = registry.schemas() if remaining > 0 else []

        resp = llm.chat(messages, tools=exposed, temperature=cfg.temperature, seed=cfg.seed)
        trace.steps.append(Step(kind="model", text=resp.text, thinking=resp.thinking,
                                usage=resp.usage, latency_ms=resp.latency_ms))
        messages.append(assistant(resp))

        if not resp.tool_calls:
            final, err = parse_final(resp.text)          # strict JSON contract
            if final is None and not trace.repair_used:  # exactly one repair round
                trace.repair_used = True
                messages.append(user(prompts.REPAIR.format(error=err)))
                continue
            trace.final = final
            trace.terminated_by = "model" if final else "unparseable"
            break

        for tc in resp.tool_calls[:remaining]:
            result = registry.dispatch(tc)               # NEVER raises; failures are observations
            trace.tool_calls.append(ToolCall(name=tc.name, args=tc.args, ok=result.ok,
                                             error=result.error, output=result.render(),
                                             step=step, latency_ms=result.latency_ms))
            trace.evidence.extend(result.evidence)       # append-only, deduped, stable IDs E1..En
            messages.append(tool_result(tc.id, result.render()))
    else:
        trace.final = force_finalize(llm, messages)      # one final no-tools call
        trace.terminated_by = "step_limit"

    return trace
```

Five decisions worth sanity-checking:

1. **Budget by schema withholding.** Prompt-level budget instructions are
   routinely violated; withholding the schemas makes over-budget calls
   structurally impossible. `budget_exhausted_at_step` is recorded so
   "ran out of budget" is measurable rather than inferred.
2. **Tool errors are observations, not exceptions.** A crash on a bad argument
   would destroy exactly the traces the `tool_misuse` analysis needs.
3. **Append-only evidence store with stable IDs.** Every passage/tool output the
   agent has *seen* gets `E1, E2, …`; the model is instructed to cite these IDs.
   Grounding is then checked against **all** seen evidence (per the spec's
   definition of hallucination), while the citation list yields a *second*,
   independent metric: citation precision (did it cite what it actually used?).
4. **One repair round, recorded.** Unlimited retries would silently paper over a
   real failure mode (models that cannot follow an output contract).
5. **Forced finalization is labelled.** "No answer because out of steps" and
   "wrong answer" are different failures and must not be merged.

**Final-answer contract** (enforced, repaired once, then failed loudly):

```json
{"answer": "B",
 "justification": "One claim per sentence. Cite evidence inline.",
 "citations": ["E3", "E7", "T2"]}
```

`answer` ∈ {A,B,C,D} for MedQA, {yes,no,maybe} for PubMedQA. The one-claim-per-sentence
instruction makes the v1 rule-based claim extractor viable, and we measure how
often models honour it.

---

## 4. Tools

| Tool | Local (offline, default here) | Live adapter (open-egress machines) |
|---|---|---|
| `search_literature(query, k)` | BM25 + FAISS dense index over a local corpus (§5) | Entrez E-utilities client (blocked here; tested against recorded fixtures) |
| `check_drug_interactions(drugs[])` | curated table (~120 clinically-significant pairs, sourced + cited in-repo) | openFDA / RxNorm adapter, same interface |
| `calc_chads_vasc` / `calc_wells_pe` / `calc_meld` / `calc_anion_gap` | pure functions, exhaustively unit-tested against published worked examples | n/a |
| `convert_lab_units` (optional) | pure function, curated conversion table | n/a |

Calculators are **pure, total functions with explicit validation errors** — no
LLM in the loop — so a wrong score is unambiguously the agent's argument error,
which is precisely what `tool_misuse` needs to be able to claim.

Every tool returns a uniform `ToolResult(ok, output, evidence[], error)`. Only
retrieval and drug-interaction results carry `Evidence`; calculator outputs are
also citable evidence (`T1, T2…`) since a computed score can ground a claim.

---

## 5. Data and retrieval corpus

Sources, all fetchable from the allowed egress (verified today):

- **PubMedQA** — `raw.githubusercontent.com/pubmedqa/pubmedqa` (`ori_pqal.json`,
  1 000 expert-annotated QA + abstracts; official `test_ground_truth.json`).
  Verified reachable, 2.5 MB.
- **MedQA (USMLE, 4-option)** and the **PubMedQA official 500-question test
  split** — from the MIRAGE benchmark file (`Teddy-XiongGZ/MIRAGE/benchmark.json`),
  verified reachable, contains the official MedQA test split (1 273 items) and
  PubMedQA test split (500 items) with linked PMIDs.
- **Trap set** — 30–50 hand-curated items I will write, each with a
  plausible-sounding distractor that the corpus does **not** support, plus a
  recorded rationale per item. Committed to the repo.

Every download is checksummed into `data/manifests/` so a re-download that
changes upstream is caught, not silently absorbed.

**Retrieval corpus.** PubMed's live API is blocked, so the corpus is built from
PubMedQA abstracts (~1 000 from PQA-L, expandable with PQA-A for the corpus-size
ablation). This has a real methodological advantage: **gold provenance is known**
(each PubMedQA question links to its source abstract by PMID), which is what
makes `Hit@k` and therefore the `retrieval_failure` label measurable at all.

Honest limitation to state in the write-up: this corpus is topically matched to
PubMedQA but only partially to MedQA, so MedQA retrieval will be weak. That is
reported as a finding (and is exactly what the corpus-size/quality ablation is
for) — not hidden.

**Split discipline.** `splits.py` exposes `dev` and `test`; a module-level guard
raises unless `CRA_ALLOW_TEST=1` is set, and every trace records which split it
came from. All development, threshold calibration, and prompt iteration happens
on `dev`. Test is touched once, at the end.

---

## 6. Failure-mode classification (the core of the contribution)

Two tiers, because collapsing them loses information.

**Tier 1 — outcome:** `correct` / `incorrect`, from the gold answer.

**Tier 2 — primary attribution label, exactly one per trace**, by ordered rules
(precedence = most upstream cause wins):

| # | Label | Rule |
|---|---|---|
| R1 | `retrieval_failure` | gold source known ∧ gold ∉ retrieved evidence ∧ answer incorrect |
| R2 | `tool_misuse` | ¬R1 ∧ answer incorrect ∧ any of: required tool never called, wrong tool, unnecessary call, arguments contradict the case, malformed call, budget exhausted with no answer |
| R3 | `reasoning_failure` | ¬R1 ∧ ¬R2 ∧ answer incorrect (correct evidence in hand, wrong conclusion) |
| R4 | `unsupported_claim` | answer correct ∧ ≥1 justification claim not-addressed or contradicted |
| R5 | `correct_grounded` | answer correct ∧ all claims entailed |

The headline stacked bar chart has exactly these five segments.

**Critically:** hallucination is *also* recorded as an independent boolean flag on
every trace, including incorrect ones. **Hallucination rate is computed from the
flags over all traces, never from the primary label** — otherwise hallucinations
inside wrong answers would vanish from the metric. Alongside the exclusive
labels I report a **multi-label co-occurrence matrix**, which shows how often
failures compound (e.g. retrieval failure → hallucinated justification).

**Where labels are not assignable, say so.** MedQA has no gold source document,
so R1 cannot fire there; those traces carry `retrieval_gold_available=false` and
the MedQA chart omits (rather than zeroes) that segment.

**The `expected_tools` oracle** that R2 depends on is the weakest link, so it is
built explicitly and audited: a deterministic rule detects when a case supplies
the inputs for a calculator (e.g. AF + age/CHF/HTN/diabetes/stroke ⇒
CHA₂DS₂-VASc). I hand-verify the detector on a sample and **report its precision
and recall as a stated limitation of the tool-misuse metric**, rather than
presenting R2 as ground truth.

---

## 7. Grounding / hallucination check

1. **Claim extraction.** v1: sentence-split + filter non-assertions (questions,
   hedges-only, pure restatement of the option text). v2: LLM extractor with a
   fixed prompt. I report agreement between v1 and v2 on the dev set so the
   choice of extractor is visible rather than assumed.
2. **Entailment.** `EntailmentChecker` protocol, two implementations:
   - `nli.py` — small NLI model (DeBERTa-v3 MNLI class), each claim scored
     against **each evidence passage independently**, label = `entailed` if
     max P(entail) > τ_e; `contradicted` if max P(contradict) > τ_c and no
     entailment; else `not_addressed`. Independent scoring avoids the premise
     truncation that wrecks concatenated-context NLI.
   - `judge.py` — LLM-as-judge against a **written rubric committed at
     `eval/entailment/rubric.md`**, strict three-way output, temperature 0.
   - τ thresholds calibrated **on dev only**, reported in the config.
3. **Cross-method agreement** (NLI vs judge) is reported. Two instruments
   disagreeing is itself a result about the measurability of grounding.
4. `hallucination_rate = (#not_addressed + #contradicted) / #claims`, reported
   per-claim and per-trace (fraction of traces with ≥1 unsupported claim).

**Judge-circularity caveat, stated up front:** if the judge belongs to the same
model family as an evaluated agent, that pairing is reported separately and
never used as the sole basis for a headline claim. The NLI checker is the
family-independent control.

---

## 8. Validation of the automatic classifier

Without this the framework is unfalsifiable, so it is not optional.

- 40 traces stratified across dataset × model × predicted label.
- **Blind protocol:** labels are written against the rubric *before* the
  automatic labels are revealed; the annotation file records timestamps and
  order of operations.
- Report **Cohen's κ** and a full confusion matrix — not bare agreement, which
  is inflated under skewed classes.
- Separate κ for the failure-mode label and for per-claim entailment.
- **Annotator caveat, stated plainly:** I am not a clinician, and I am
  the same class of system being evaluated. I will produce the first
  annotation pass under the blind protocol; **a second independent pass by you
  on the same 40 traces would let us report inter-annotator κ**, which is a
  materially stronger claim. Flagged for your decision, not assumed.

---

## 9. Models

Provider-agnostic by construction (`LLMClient` protocol).

**Decision (2026-08-25): two open-weight models via Ollama, no paid API.**

- **`qwen3:8b`** and **`llama3.1:8b`**, both through the OpenAI-compatible
  adapter. Both support native tool calling, which is required: the loop
  implements function calling only, so a model without tool support would
  produce silently closed-book traces rather than an obvious error.
- **`MockClient`** — deterministic scripted responses; powers CI, the
  golden-trace tests, and a credential-free demo. Not a result, never reported
  as one.
- The Anthropic adapter stays in the codebase and tested, so a frontier-model
  run can be added later without touching the pipeline. Because eval is a pure
  function of traces, adding a third model costs one rollout and no re-analysis.

**Stated limitation.** Comparing two 8B open-weight models is a weaker
generalisation claim than open-weight versus frontier. It still answers the
question the comparison exists to answer — whether a failure-mode profile is
one model's quirk — but it cannot speak to how the profile changes with
capability. The write-up says so explicitly rather than implying broader
coverage.

**Sampling.** Headline runs use a stratified 300 questions per dataset rather
than the full 1 773-question test set: local 8B inference over the full set
takes many hours per model, and 300 supports the failure-mode breakdown with
bootstrap CIs. The sample size is a config field, the sampling is seeded, and
the reported numbers carry CIs that reflect it.

**Neither model can be run in the build environment** (blocked model
registries, no GPU); rollouts run on the user's Windows + GPU machine. See §11.

---

## 10. Reproducibility mechanics

- `uv` + committed lockfile; Python 3.11 pinned.
- Every trace embeds: `schema_version`, `config_hash`, model ID, decoding params,
  seed, dataset split, corpus/index hash, code git SHA.
- Content-addressed LLM response cache → a rerun with the cache is byte-identical;
  a rerun without it is compared statistically (bootstrap CI overlap), and both
  checks are part of `make test`.
- `make results` = zero network, zero API cost, regenerates every README number
  and figure from committed traces.
- Bootstrap 95% CIs on every headline number; McNemar's test for paired
  ablation comparisons. Accuracy deltas without CIs are not claims.

---

## 11. Phase plan and what "done" means here

| Phase | Content | Runnable in this environment? |
|---|---|---|
| 1 | Scaffold, types/Trace schema, LLM abstraction + mock, agent loop, tools, unit tests | **Yes, fully** — end-to-end on the mock client |
| 2 | Data download + prep + splits + corpus index (BM25 offline; dense if the embedder can be sourced from PyPI) | **Yes** (verified reachable) |
| 3 | Eval framework: claims, entailment (judge + NLI), failure modes, metrics, plots, validation harness | **Code yes; NLI weights and judge calls need egress/keys** |
| 4 | Ablations, README, write-up, Gradio demo | Code yes; real numbers need a rollout run |

**The gap is real and I will not paper over it:** this environment can produce a
complete, tested, reproducible *pipeline* with mock-generated traces, but it
cannot produce real headline numbers. Any `results/` table built from mock
traces is structurally correct and empirically meaningless, and is labelled as
such until a real run replaces it.

**Agreed resolution (2026-08-25):** rollouts run on the user's Windows + GPU
machine against local Ollama. This repo therefore ships a cross-platform task
runner (`tasks.py`) alongside the Makefile, since `make` and the POSIX
`.venv/bin/python` path do not exist on Windows. `python tasks.py doctor`
checks the Python version, the virtualenv and whether Ollama is serving the
required models.
