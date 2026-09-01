"""Blind human-annotation workflow for validating the automatic classifier.

``sample_for_annotation`` picks a stratified subset (docs/DESIGN.md Sec 8: 40
traces, stratified across dataset x model x predicted label), and
``write_annotation_template`` writes it to a CSV with the automatic label
withheld, plus a companion transcript file so the annotator has everything
needed without running any code. "Blind" means the automatic label stays
withheld until the human has already written a judgement -- ``read_annotations``
only reads the human column back; joining it against the predicted label to
compute kappa is the caller's job (via ``cra.eval.agreement.cohens_kappa``),
done only after annotation is complete.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from cra.eval.agreement import AgreementResult, cohens_kappa
from cra.eval.records import TraceEvalRecord
from cra.trace_io import read_trace_dir, render_trace

ANNOTATION_FIELDS = [
    "trace_id", "dataset", "qid", "model_id", "answer_status", "allowed_labels", "human_label",
]

# Whether the answer was right is a string comparison against gold, not a
# judgement -- yet the taxonomy encodes it in the label, so an annotator can
# pick a label that contradicts a known fact. In the first annotation pass nine
# of forty labels did exactly that. The template therefore states the answer
# status and offers only the labels consistent with it, leaving the annotator
# the part that genuinely needs a human: for a wrong answer, which upstream
# cause; for a right one, whether the justification is grounded.
ALLOWED_BY_STATUS: dict[str, tuple[str, ...]] = {
    "incorrect": ("retrieval_failure", "tool_misuse", "reasoning_failure"),
    "correct": ("unsupported_claim", "correct_grounded"),
    "no_answer": ("no_answer",),
}


def answer_status(record: TraceEvalRecord) -> str:
    if record.is_correct is None:
        return "no_answer"
    return "correct" if record.is_correct else "incorrect"


def allowed_labels(record: TraceEvalRecord) -> tuple[str, ...]:
    return ALLOWED_BY_STATUS[answer_status(record)]

# Valid values for the human_label column -- the same taxonomy the
# classifier uses, so kappa is computed on directly comparable labels.
VALID_HUMAN_LABELS = (
    "no_answer", "retrieval_failure", "tool_misuse",
    "reasoning_failure", "unsupported_claim", "correct_grounded",
)


def sample_for_annotation(
    records: list[TraceEvalRecord], n: int = 40, seed: int = 12345
) -> list[TraceEvalRecord]:
    """Stratified by (dataset, model_id, failure_mode) -- see module docstring.

    Traces with no parseable answer are excluded. Their label is forced --
    ``no_answer`` is the only option consistent with the answer status, and the
    classifier derives it from the same objective fact -- so including them
    would contribute guaranteed agreement that measures nothing about the
    classifier while inflating kappa.
    """
    records = [r for r in records if answer_status(r) != "no_answer"]
    if not records:
        return []

    groups: dict[tuple, list[TraceEvalRecord]] = {}
    for r in records:
        groups.setdefault((r.dataset, r.model_id, r.failure_mode), []).append(r)

    rng = random.Random(seed)
    keys = sorted(groups, key=str)
    for key in keys:
        groups[key] = sorted(groups[key], key=lambda r: r.trace_id)
        rng.shuffle(groups[key])

    quota = n / len(records)
    picked: list[TraceEvalRecord] = []
    for key in keys:
        take = max(1, round(len(groups[key]) * quota))
        picked.extend(groups[key][:take])

    picked.sort(key=lambda r: r.trace_id)
    rng.shuffle(picked)
    return picked[:n]


def write_annotation_template(
    records: list[TraceEvalRecord], trace_dirs: list[str], out_path: str | Path
) -> tuple[Path, Path]:
    """Writes the blank-label CSV plus a ``.transcripts.txt`` sibling with the
    full rendered trace for every sampled item. Returns (csv_path, transcript_path)."""
    traces_by_key = {}
    for d in trace_dirs:
        for t in read_trace_dir(d):
            traces_by_key[(t.question.dataset, t.question.qid)] = t

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path = out_path.with_name(out_path.stem + ".transcripts.txt")

    with transcript_path.open("w", encoding="utf-8") as fh:
        for r in records:
            trace = traces_by_key.get((r.dataset, r.qid))
            fh.write(f"### trace_id={r.trace_id}\n")
            # full=True: the annotator must read the same evidence text the
            # entailment checker does, or the kappa measures the missing text.
            fh.write(
                render_trace(trace, full=True)
                if trace is not None
                else "(trace not found)\n"
            )
            fh.write("\n\n")

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "trace_id": r.trace_id, "dataset": r.dataset, "qid": r.qid,
                    "model_id": r.model_id,
                    "answer_status": answer_status(r),
                    "allowed_labels": " | ".join(allowed_labels(r)),
                    "human_label": "",
                }
            )

    return out_path, transcript_path


def validate_annotations(path: str | Path) -> list[str]:
    """Problems in a filled-in annotation CSV, as human-readable strings.

    Catches a label that contradicts the row's own answer status. Silently
    scoring such a row would mix an annotation slip into the kappa and read as
    classifier disagreement.
    """
    problems: list[str] = []
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("human_label") or "").strip()
            if not label:
                continue
            if label not in VALID_HUMAN_LABELS:
                problems.append(f"{row['trace_id']}: {label!r} is not a valid label")
                continue
            allowed = [a.strip() for a in (row.get("allowed_labels") or "").split("|") if a.strip()]
            if allowed and label not in allowed:
                problems.append(
                    f"{row['trace_id']}: {label!r} contradicts answer_status="
                    f"{row.get('answer_status')!r}; allowed here: {', '.join(allowed)}"
                )
    return problems


def read_annotations(path: str | Path) -> dict[str, str]:
    """trace_id -> human_label, for rows where a label has actually been filled in."""
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["trace_id"]: row["human_label"] for row in reader if row.get("human_label")}


@dataclass
class AnnotationScore:
    agreement: AgreementResult | None  # None if fewer than 2 gradable pairs
    n_scored: int
    n_ungraded_skipped: int  # human labeled it, but the classifier had no prediction yet
    n_missing: int  # sampled but never labeled by the human


def score_annotations(
    records: list[TraceEvalRecord], annotations: dict[str, str]
) -> AnnotationScore:
    """Pairs each completed human label against the classifier's prediction
    for the *same* trace_id, and computes kappa over the pairs where the
    classifier actually made a prediction.

    A record with ``failure_mode is None`` ("correct, ungraded" -- see
    ``failure_modes.py``) has no automatic label to compare against yet; its
    human label is not discarded, just excluded from *this* kappa
    computation. Re-run this after grading those records with NLI or
    ``judge-check`` and the same human labels become comparable -- the point
    of writing them once, blind, is that they don't need to be re-collected.
    """
    predicted: list[str] = []
    human: list[str] = []
    n_ungraded_skipped = 0
    n_missing = 0

    for record in records:
        human_label = annotations.get(record.trace_id)
        if human_label is None:
            n_missing += 1
            continue
        if record.failure_mode is None:
            n_ungraded_skipped += 1
            continue
        predicted.append(record.failure_mode)
        human.append(human_label)

    agreement = cohens_kappa(predicted, human) if len(predicted) >= 2 else None
    return AnnotationScore(
        agreement=agreement, n_scored=len(predicted),
        n_ungraded_skipped=n_ungraded_skipped, n_missing=n_missing,
    )
