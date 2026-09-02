"""NLI-based entailment checker (a DeBERTa-v3 MNLI-class model), behind the
``dense`` extra.

This is the checker that actually satisfies docs/CONVENTIONS.md's two-stage invariant
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
from cra.eval.entailment.cache import DEFAULT_CACHE_PATH, EntailmentCache, score_key

_MISSING_EXTRA = (
    "NLI entailment requires the 'dense' extra. Install with: python tasks.py setup --extras dev,dense (not a bare pip -- it may belong to a different Python, and a uv-created venv has no pip) "
    "(pulls transformers and torch)."
)

# A widely-used, permissively-licensed MNLI+FEVER+ANLI checkpoint with a
# small (~184M param) footprint -- feasible on CPU for a few thousand claim x
# passage pairs, which is the scale a 300-question headline run produces.
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# The checkpoint declares no maximum length, so transformers disables truncation
# and warns. Left alone, one long passage would build an outsized tensor and
# stall the run. 512 is this architecture's trained context, and a retrieval
# snippet (capped at 1000 characters, roughly 250 tokens) plus a one-sentence
# hypothesis fits inside it with room to spare.
MAX_LENGTH = 512


class NLIEntailmentChecker:
    name = "nli"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        entail_threshold: float = 0.5,
        contradict_threshold: float = 0.5,
        device: str = "cpu",
        cache_path: str | None = DEFAULT_CACHE_PATH,
        use_cache: bool = True,
        batch_size: int = 16,
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
        self.model_name = model_name
        self.batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.cache = EntailmentCache(cache_path or DEFAULT_CACHE_PATH, enabled=use_cache)

    def _label_scores_batch(self, premises: list[str], hypothesis: str) -> list[dict[str, float]]:
        """Score one hypothesis against many premises in a single forward pass.

        Previously this ran one pass per premise. The model is small enough
        that per-call overhead, not compute, dominated: batching keeps the same
        arithmetic and the same verdicts while giving the hardware something
        worth doing.
        """
        # id2label ordering varies by checkpoint; read it rather than assume a
        # fixed index -> meaning map, which would silently swap entailment and
        # contradiction.
        id2label = self._model.config.id2label
        out: list[dict[str, float]] = []
        for start in range(0, len(premises), self.batch_size):
            chunk = premises[start : start + self.batch_size]
            inputs = self._tokenizer(
                chunk,
                [hypothesis] * len(chunk),
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                logits = self._model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)
            for row in range(len(chunk)):
                out.append(
                    {id2label[i].lower(): float(probs[row][i]) for i in range(probs.shape[1])}
                )
        return out

    def score(self, claim: str, evidence_texts: list[str]) -> tuple[float, float]:
        """``(max_entail, max_contradict)`` over the premises, cached.

        Scores rather than labels are the cached unit: they do not depend on the
        thresholds, so a threshold sweep reads them instead of re-running the
        model. That is what makes calibration cheap enough to do at all.
        """
        if not evidence_texts:
            return 0.0, 0.0

        key = score_key(self.name, self.model_name, claim, evidence_texts)
        cached = self.cache.get_scores(key)
        if cached is not None:
            return cached

        best_entail = best_contradict = 0.0
        for scores in self._label_scores_batch(evidence_texts, claim):
            entail = next((v for k, v in scores.items() if "entail" in k), 0.0)
            contradict = next((v for k, v in scores.items() if "contra" in k), 0.0)
            best_entail = max(best_entail, entail)
            best_contradict = max(best_contradict, contradict)

        self.cache.put_scores(key, best_entail, best_contradict)
        return best_entail, best_contradict

    @staticmethod
    def label_from_scores(
        entail: float, contradict: float, entail_threshold: float, contradict_threshold: float
    ) -> EntailmentLabel:
        """The decision rule, isolated so a sweep applies exactly what check() does."""
        if entail > entail_threshold:
            return "entailed"
        if contradict > contradict_threshold:
            return "contradicted"
        return "not_addressed"

    def check(self, claim: str, evidence_texts: list[str]) -> EntailmentLabel:
        if not evidence_texts:
            return "not_addressed"
        entail, contradict = self.score(claim, evidence_texts)
        return self.label_from_scores(
            entail, contradict, self.entail_threshold, self.contradict_threshold
        )

    def flush(self) -> None:
        """Commit cached verdicts. Safe to call repeatedly."""
        self.cache.commit()
