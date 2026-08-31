# Entailment rubric

You are checking whether a single claim from a clinical justification is
supported by the evidence passages the model had access to when it wrote
that justification.

You will be given:

- **CLAIM**: one sentence from a model's justification for its final answer.
- **EVIDENCE**: one or more passages -- retrieved literature, or a computed
  tool output (e.g. a risk score) -- the model was allowed to cite.

Decide exactly one label:

- **entailed** -- at least one evidence passage states, or directly implies,
  the claim. The claim does not need to be a verbatim quote; a careful reader
  of the evidence would agree the claim follows from it.
- **contradicted** -- at least one evidence passage states something that
  directly conflicts with the claim.
- **not_addressed** -- the evidence, read carefully, neither supports nor
  contradicts the claim. It simply isn't covered.

Rules:

- Judge only against the EVIDENCE given. Do not use outside medical knowledge
  to decide whether the claim is *true* -- only whether the evidence in front
  of you supports it. A claim can be entailed by the evidence and still be
  medically wrong; that is not your concern here.
- A generic restatement of the final answer with no independent factual
  content (e.g. "Therefore the answer is B") is `not_addressed` -- it isn't a
  claim the evidence could support or contradict.
- If no evidence was provided at all, the label is always `not_addressed`.
- Output exactly one word: `entailed`, `contradicted`, or `not_addressed`.
  No punctuation, no explanation, no other text.
