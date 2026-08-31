"""Reading and writing traces, and rendering one for a human.

Traces are stored as gzipped JSON Lines: one self-contained trace per line,
streamable, appendable, and small enough to commit. Committed traces are what
make ``make results`` reproduce the headline numbers with no network and no
credentials.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from cra.types import Trace


def write_traces(traces: Iterable[Trace], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    # Written to a temp file and swapped in atomically: the rollout runner
    # rewrites this whole file after every episode, and a kill mid-write must
    # leave the previous complete file in place, not a truncated one.
    tmp = path.with_name(path.name + ".tmp")
    with opener(tmp, "wt", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(trace.model_dump_json() + "\n")
    tmp.replace(path)
    return path


def read_traces(path: str | Path) -> Iterator[Trace]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Trace.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{path}:{line_no}: could not parse trace: {exc}") from exc


def read_trace_dir(path: str | Path, pattern: str = "*.jsonl*") -> list[Trace]:
    root = Path(path)
    if root.is_dir():
        # write_traces's atomic-write temp file ("traces.jsonl.gz.tmp")
        # matches "*.jsonl*" too -- excluded explicitly rather than relying
        # on timing, since a read landing mid-write would otherwise try to
        # gzip-decode a filename whose suffix is ".tmp", not ".gz", and
        # crash on the raw gzip magic bytes as if they were UTF-8 text.
        files = sorted(f for f in root.glob(pattern) if not f.name.endswith(".tmp"))
    else:
        files = [root]
    return [t for f in files for t in read_traces(f)]


def render_trace(trace: Trace, width: int = 88, show_evidence: bool = True) -> str:
    """A readable transcript. Used by the CLI and the demo."""
    rule = "-" * width
    out: list[str] = [
        rule,
        f"{trace.question.dataset}/{trace.question.qid}  [{trace.question.split}]  "
        f"model={trace.model_id}  budget={trace.tool_budget}",
        rule,
        trace.question.question.strip(),
    ]
    if trace.question.options:
        out += [f"  {k}) {v}" for k, v in trace.question.options.items()]
    out.append("")

    for step in trace.steps:
        label = {"model": "MODEL", "repair": "REPAIR", "forced_final": "MODEL (forced)"}[step.kind]
        out.append(f"[step {step.index}] {label}")
        if step.thinking:
            out.append(f"  thinking: {step.thinking.strip()[:400]}")
        calls = [tc for tc in trace.tool_calls if tc.step == step.index]
        for tc in calls:
            status = "ok" if tc.ok else ("refused" if not tc.executed else "FAILED")
            out.append(f"  -> {tc.name}({json.dumps(tc.args, ensure_ascii=False)[:160]}) [{status}]")
            if tc.error:
                out.append(f"     error: {tc.error[:200]}")
            elif tc.evidence_ids:
                out.append(f"     evidence: {', '.join(tc.evidence_ids)}")
        if step.text and not calls:
            out.append(f"  text: {step.text.strip()[:400]}")
        out.append("")

    if show_evidence and trace.evidence:
        out.append("EVIDENCE SEEN")
        for e in trace.evidence:
            head = f"  [{e.evidence_id}] {e.title or '(untitled)'}"
            if e.source_id:
                head += f"  <{e.source_id}>"
            out.append(head)
            out.append(f"      {e.text.strip()[:220].replace(chr(10), ' ')}...")
        out.append("")

    out.append("FINAL")
    if trace.final is None:
        out.append(f"  (no parseable answer; terminated_by={trace.terminated_by})")
        if trace.error:
            out.append(f"  error: {trace.error}")
    else:
        correct = trace.is_correct
        verdict = "correct" if correct else "incorrect"
        out.append(f"  answer: {trace.final.answer}  (gold: {trace.question.gold_answer}) -> {verdict}")
        out.append(f"  citations: {trace.final.citations or '(none)'}")
        out.append(f"  justification: {trace.final.justification}")
    out.append(
        f"  terminated_by={trace.terminated_by}  tool_calls={trace.n_tool_calls}"
        f"  repair={trace.repair_used}  tokens={trace.usage.input_tokens}/{trace.usage.output_tokens}"
    )
    out.append(rule)
    return "\n".join(out)
