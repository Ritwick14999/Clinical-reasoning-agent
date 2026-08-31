"""Trace I/O: round-tripping, atomic writes, and directory reads.

Regression coverage for a real bug: ``write_traces``'s atomic-write temp file
(``traces.jsonl.gz.tmp``) matches ``read_trace_dir``'s default
``*.jsonl*`` glob pattern. A read landing while a background rollout was
mid-write picked up the ``.tmp`` file, whose ``.tmp`` suffix isn't recognized
as gzip, and crashed trying to decode raw gzip bytes as UTF-8 text.
"""

from __future__ import annotations

from cra.trace_io import read_trace_dir, read_traces, render_trace, write_traces
from cra.types import FinalAnswer, Question, Trace


def _trace(qid="q1") -> Trace:
    q = Question(qid=qid, dataset="pubmedqa", split="dev", question="Q?", gold_answer="yes")
    return Trace(question=q, final=FinalAnswer(answer="yes", justification="Because E1."))


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "traces.jsonl.gz"
    write_traces([_trace("a"), _trace("b")], path)
    traces = list(read_traces(path))
    assert [t.question.qid for t in traces] == ["a", "b"]


def test_write_traces_cleans_up_temp_file(tmp_path):
    path = tmp_path / "traces.jsonl.gz"
    write_traces([_trace()], path)
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_read_trace_dir_ignores_leftover_tmp_file(tmp_path):
    """A stray .tmp file (e.g. left behind by a killed process, or present
    momentarily during a concurrent write) must never be read as a trace file."""
    path = tmp_path / "traces.jsonl.gz"
    write_traces([_trace("real")], path)

    # Simulate a temp file caught mid-write: real gzip bytes, wrong suffix.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(b"\x1f\x8bnot a complete gzip stream")

    traces = read_trace_dir(tmp_path)
    assert [t.question.qid for t in traces] == ["real"]


def test_write_traces_overwrites_cleanly(tmp_path):
    path = tmp_path / "traces.jsonl.gz"
    write_traces([_trace("a")], path)
    write_traces([_trace("a"), _trace("b")], path)  # simulates a resumed run
    traces = list(read_traces(path))
    assert [t.question.qid for t in traces] == ["a", "b"]


def test_render_trace_smoke():
    text = render_trace(_trace())
    assert "pubmedqa/q1" in text
    assert "FINAL" in text
