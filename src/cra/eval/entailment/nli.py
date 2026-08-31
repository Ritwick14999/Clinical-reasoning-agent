"""NLI-based entailment checker (a DeBERTa-v3 MNLI-class model), behind the
``dense`` extra.

This is the checker that actually satisfies CLAUDE.md's two-stage invariant
for eval: deterministic, local, no network call, no per-run API cost. See
``judge.py``'s module docstring for the contrast -- that checker calls a live
model by design and is deliberately kept out of the default ``cra eval`` path.

Each claim is scored against **each evidence passage independently**, never
against all passages concatenated. Concatenating would let NLI's max-length
truncation (which drops from the end) silently discard exactly the passages
needed to judge a claim that cites evidence late in a long trace -- taking the
max entailment score across independent per-passage runs avoids that failure
mode entirely (docs/DESIGN.md Sec 7).
"""

from __future__ import annotations

from cra.eval.entailment.base import EntailmentLabel

_MISSING_EXTRA = (
    "NLI entailment requires the 'dense' extra. Install with: python tasks.py setup --extras dev,dense (not a bare pip -- it may belong to a different Python, and a uv-created venv has no pip) "
    "(pulls transformers and torch)."
)

# A widely-used, permissively-licensed MNLI+FEVER+ANLI checkpoint with a
# small (~184M param) footprint -- feasible on CPU for a few thousand claim x
# passage pairs, which is the scale a 300-question headline run produces.
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class NLIEntailmentChecker:
    name = "nli"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        entail_threshold: float = 0.5,
        contradict_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(_MISSING_EXTRA) from exc

        self._torch = torch
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    def _label_scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        inputs = self._tokenizer(
            premise, hypothesis, truncation=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = self._torch.softmax(logits, dim=-1)
        # Read the label order from the model config rather than assuming a
        # fixed index -> meaning mapping: it varies by checkpoint, and a wrong
        # assumption here would silently swap entailment and contradiction.
        return {self._model.config.id2label[i].lower(): float(probs[i]) for i in range(len(probs))}

    def check(self, claim: str, evidence_texts: list[str]) -> EntailmentLabel:
        if not evidence_texts:
            return "not_addressed"

        best_entail = best_contradict = 0.0
        for passage in evidence_texts:
            scores = self._label_scores(passage, claim)
            entail = next((v for k, v in scores.items() if "entail" in k), 0.0)
            contradict = next((v for k, v in scores.items() if "contra" in k), 0.0)
            best_entail = max(best_entail, entail)
            best_contradict = max(best_contradict, contradict)

        if best_entail > self.entail_threshold:
            return "entailed"
        if best_contradict > self.contradict_threshold:
            return "contradicted"
        return "not_addressed"
