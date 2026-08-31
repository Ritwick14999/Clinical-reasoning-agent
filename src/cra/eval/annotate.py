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

ANNOTATION_FIELDS = ["trace_id", "dataset", "qid", "model_id", "human_label"]

# Valid values for the human_label column -- the same taxonomy the
# classifier uses, so kappa is computed on directly comparable labels.
VALID_HUMAN_LABELS = (
    "no_answer", "retrieval_failure", "tool_misuse",
    "reasoning_failure", "unsupported_claim", "correct_grounded",
)


def sample_for_annotation(
    records: list[TraceEvalRecord], n: int = 40, seed: int = 12345
) -> list[TraceEvalRecord]:
    """Stratified by (dataset, model_id, failure_mode) -- see module docstring."""
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
                {"trace_id": r.trace_id, "dataset": r.dataset, "qid": r.qid,
                 "model_id": r.model_id, "human_label": ""}
            )

    return out_path, transcript_path


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
