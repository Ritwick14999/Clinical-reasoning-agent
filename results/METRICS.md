# Consolidated metrics

Every headline number in one place, with what produced it and what qualifies it.
Regenerate any of these from committed traces: no GPU, model or network needed.

```
python -m cra.cli eval      --experiments final_qwen3 final_llama31 --nli
python -m cra.cli ablation  --baseline headline_qwen3 --variants ablation_closedbook_qwen3
python -m cra.cli calibrate --experiments headline_qwen3 headline_llama31 \
                            --dev results/annotations/dev_pass_labels.csv \
                            --heldout results/annotations/heldout.csv
```

---

## 1. Headline result

**74–90% of *correct* answers on PubMedQA contain at least one claim not entailed by
the evidence retrieved to justify it.**

| | qwen3-8k | llama31-8k | combined |
|---|---|---|---|
| at default cut (entail > 0.5) | 71.5% (98/137) | 76.7% (135/176) | **74.4%** (233/313) |
| at calibrated cut (entail > 0.95) | 89.1% (122/137) | 90.9% (160/176) | **90.1%** (282/313) |

Reported as a range. The calibrated cut is **not adopted** — see §4.

Claim level: only **20–24%** of extracted claims are entailed by any retrieved passage
(qwen3 415/2067, llama3.1 453/1828 on the test split).

MedQA reads 98–100% but **must not be reported as a fabrication rate**: the corpus is
built from PubMedQA abstracts and does not cover USMLE vignettes, and qwen3 answers
most MedQA questions without searching at all. It measures corpus coverage.

## 2. Test-split accuracy (300 questions per dataset per model, run once)

| Model | Dataset | Accuracy (95% CI) | Answered | Hit@k | Tool P/R |
|---|---|---|---|---|---|
| qwen3-8k | MedQA | **76.5%** [71.5, 81.2] | 277/300 | n/a | n/a (n=21, no tools used) |
| qwen3-8k | PubMedQA | 45.7% [40.3, 51.3] | 300/300 | 79.3% | 100.0% / 81.0% (n=300) |
| llama31-8k | MedQA | 58.1% [52.2, 63.6] | 291/300 | n/a | 95.0% / 90.5% (n=21) |
| llama31-8k | PubMedQA | **58.7%** [53.0, 64.0] | 300/300 | 95.3% | 98.3% / 97.7% (n=300) |

**The ranking crosses over by dataset with non-overlapping intervals in both
directions.** Either model alone supports the opposite conclusion — the strongest
argument that the two-model requirement is substantive.

## 3. Failure modes (test split, % of N=300)

| Model | Dataset | no answer | retrieval | tool misuse | reasoning | unsupported | grounded |
|---|---|---|---|---|---|---|---|
| qwen3-8k | MedQA | 8% | **22%** | 0% | **2%** | 71% | 0% |
| qwen3-8k | PubMedQA | 0% | 17% | 0% | 37% | 33% | 13% |
| llama31-8k | MedQA | 3% | 14% | 4% | 23% | 55% | 1% |
| llama31-8k | PubMedQA | 0% | 3% | 1% | 38% | 45% | 14% |

**The two models fail differently at comparable accuracy.** qwen3 on MedQA fails almost
entirely by not retrieving (22% vs 2% reasoning); llama3.1 retrieves and misreads
(14% vs 23%). qwen3 searched on only **8% of MedQA questions** (20/255 on dev).

## 4. Classifier validation

Two disjoint annotation passes, 40 traces each, 0 overlap.

| | dev (fitted) | held-out (reported) |
|---|---|---|
| default cut, entail > 0.5 | 0.673 | **0.737** |
| calibrated cut, entail > 0.95 | 0.740 | 0.771 |

Held-out κ = **0.737**, 82.5% agreement, balanced 20/20 across both models.
Held-out exceeds in-sample (0.673), so the taxonomy corrections generalise.

Per-label agreement at the default cut (held-out): `unsupported_claim` **19/20**,
`reasoning_failure` **8/9**, `correct_grounded` 3/4, `retrieval_failure` **3/6**.
Agreement is strongest exactly where the contribution lies; weakest on retrieval, where
the classifier cannot see evidence a human judges retrieved-but-unhelpful.

**Why the calibrated cut is not adopted.** Of 24 threshold-sensitive traces, **2 change
label on dev and 1 on held-out** between the two cuts, while the headline moves 16
points. All 3 flips moved toward the human label, so the direction is consistent, but 80
annotations do not identify the magnitude. Pinning it needs ~150–200 labels concentrated
on correct answers.

Cross-method check: an LLM-as-judge pass on a stratified dev sample gives 75.0%
(llama3.1) and 77.5% (qwen3) trace-level hallucination, against the NLI checker's 82–86%.
Two instruments of different families within a few points.

## 5. Closed-book ablation (dev, paired on the same questions, McNemar)

| | accuracy | Δ | p | claims entailed | traces w/ unsupported claim |
|---|---|---|---|---|---|
| qwen3 tools → closed-book | 61.3% → 52.9% | −8.4pp | 4.4e-05 | 22.0% → 11.4% | 79.8% → 93.2% |
| llama3.1 tools → closed-book | 62.3% → 52.9% | −9.4pp | 1.1e-08 | 24.0% → 13.6% | 85.2% → 93.7% |

Per dataset:

| | PubMedQA | p | MedQA | p |
|---|---|---|---|---|
| qwen3 | 49.5% → 31.7% (**−17.8pp**) | 2.9e-07 | 76.3% → 80.3% (+4.0pp) | 0.14 n.s. |
| llama3.1 | 63.3% → 47.3% (**−16.0pp**) | 4.5e-09 | 61.1% → 59.6% (−1.5pp) | 0.18 n.s. |

**This rules out the sceptical reading of the headline.** If retrieval were decorative,
removing it would change nothing. It costs 16–18 points where the corpus matches and
roughly halves the entailed-claim share, on both models independently. The evidence is
demonstrably used — and the justification still is not a faithful account of it.

MedQA shows no significant effect, functioning as a negative control for corpus quality.

Closed-book claims are scored against the passages the tool-using run retrieved for the
same question; scoring them against their own (empty) evidence would return 100%
unsupported by construction.

## 6. Tool-budget ablation (llama3.1, dev)

| Budget | max calls | mean calls |
|---|---|---|
| 2 | 2 | 1.77 |
| 5 (baseline) | 5 | 2.19 |
| 10 | 10 | 2.30 |

**Tool budget is not a meaningful lever for this agent.** It averages 2.19 of 5 and only
23/555 episodes exceed 5 calls when allowed 10. Caveat: the effect is small *because*
the agent under-uses its budget, not because budget cannot matter in principle.

A budget-50 arm was attempted and **refused by the preflight**: worst-case prompt ~40k
tokens against a 12288 window. Not runnable on 8 GB VRAM.

## 7. Defects found and corrected (each changed a reported number)

| Defect | Impact | How found |
|---|---|---|
| Recovered tool errors captured `tool_misuse` | **151 of 159** labels were transient errors the agent retried successfully. Invalidated the original "llama3.1 fails by tool misuse" claim. | human annotation |
| Citation framing broke entailment | Claims citing an evidence ID scored entailed **14%** vs **36%** for uncited ones — the grader punished the agent for following its prompt | human annotation |
| `retrieval_failure` undetectable without gold provenance | R1 could not fire on MedQA at all; wrong answers with zero retrieval were labelled reasoning failures | human annotation |
| Annotation transcripts truncated evidence to 220 chars | Annotator judged entailment on ~22% of the premise the checker read | flagged by annotator |
| Annotation transcripts showed the **wrong model's** trace | All 40 transcripts were llama3.1 while rows claimed both models | flagged by annotator |
| Preflight read the architectural context max, not the served one | llama3.1's longest test episode was **7,755 tokens** against Ollama's default 4,096 — would have truncated silently | pre-run verification |
| Cache held an exclusive lock across inference | Concurrent eval runs failed with `database is locked` | hit during calibration |
| Trap set had gold=B and trap=A on all 30 items | A B-biased model would score 100% | pre-run inspection |

## 8. Scale

1,200 test-split episodes (2 models × 600), 2,220 dev episodes across baselines and
ablations, 301 tests, ~9k lines of Python. Test split touched exactly once.

## 9. Standing limitations

Two 8B open-weight models (weaker than open-weight vs frontier). The `expected_tools`
oracle is an uncharacterised keyword rule firing on only 21/300 MedQA questions. The
entailment threshold's magnitude is unidentified (§4). The second annotation pass was not
blind to feedback from the first, single non-clinician annotator, so inter-annotator
agreement is unmeasured. The retrieval corpus is topically mismatched to MedQA.
