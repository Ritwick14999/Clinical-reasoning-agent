"""Inter-annotator agreement: Cohen's kappa and a confusion matrix.

Used two ways (docs/DESIGN.md Sec 8): classifier-vs-human (does the automatic
failure-mode/entailment label match a blind human read?) and human-vs-human
(does a second annotator agree with the first?) -- the latter is the
materially stronger claim, if a second annotator is available. Report kappa,
not bare agreement, which is inflated under skewed classes (most traces will
be ``correct_grounded`` or ``reasoning_failure``, so agreeing "by default"
on the majority class must not look like a validated classifier).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class AgreementResult:
    n: int
    observed_agreement: float
    kappa: float
    confusion: dict[tuple[str, str], int]  # (label_a, label_b) -> count
    labels: list[str]


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> AgreementResult:
    if len(labels_a) != len(labels_b):
        raise ValueError("paired label sequences must be the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("cannot compute agreement over zero items")

    all_labels = sorted(set(labels_a) | set(labels_b))
    confusion = Counter(zip(labels_a, labels_b, strict=True))
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n

    count_a, count_b = Counter(labels_a), Counter(labels_b)
    expected = sum((count_a[label] / n) * (count_b[label] / n) for label in all_labels)

    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)

    return AgreementResult(
        n=n, observed_agreement=observed, kappa=kappa,
        confusion=dict(confusion), labels=all_labels,
    )
