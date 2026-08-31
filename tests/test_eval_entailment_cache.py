"""Entailment verdict cache.

Grading a headline run is the most expensive step in the eval stage, and the
eval is re-run by design. These tests pin the two properties that make caching
safe: a key that changes whenever anything affecting the verdict changes, and
durability across an interrupted run.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from cra.eval.entailment.cache import EntailmentCache, verdict_key

BASE = dict(
    checker="nli",
    model_name="model-x",
    entail_threshold=0.5,
    contradict_threshold=0.5,
    hypothesis="Aspirin is less effective.",
    premises=["passage one", "passage two"],
)


def test_roundtrip(tmp_path):
    cache = EntailmentCache(tmp_path / "v.sqlite")
    key = verdict_key(**BASE)
    assert cache.get(key) is None
    cache.put(key, "entailed")
    cache.commit()
    assert cache.get(key) == "entailed"
    assert (cache.hits, cache.misses) == (1, 1)
    cache.close()


def test_persists_across_instances(tmp_path):
    path = tmp_path / "v.sqlite"
    first = EntailmentCache(path)
    first.put(verdict_key(**BASE), "contradicted")
    first.close()
    assert EntailmentCache(path).get(verdict_key(**BASE)) == "contradicted"


@pytest.mark.parametrize(
    "field,value",
    [
        ("checker", "judge"),
        ("model_name", "model-y"),
        ("entail_threshold", 0.6),
        ("contradict_threshold", 0.7),
        ("hypothesis", "Aspirin is more effective."),
        ("premises", ["passage one"]),
        ("premises", ["passage one", "passage three"]),
    ],
)
def test_key_changes_when_anything_affecting_the_verdict_changes(field, value):
    """A stale verdict surviving a threshold change would be silently wrong.

    Thresholds are calibrated on dev and *will* move, so this is not
    hypothetical.
    """
    assert verdict_key(**{**BASE, field: value}) != verdict_key(**BASE)


def test_premise_order_is_significant():
    """The verdict maxes over premises, so order cannot change it -- but a
    reordering means different evidence was retrieved, which must not silently
    reuse an old verdict."""
    swapped = {**BASE, "premises": list(reversed(BASE["premises"]))}
    assert verdict_key(**swapped) != verdict_key(**BASE)


def test_disabled_cache_is_a_working_no_op(tmp_path):
    cache = EntailmentCache(tmp_path / "v.sqlite", enabled=False)
    key = verdict_key(**BASE)
    cache.put(key, "entailed")
    cache.commit()
    assert cache.get(key) is None


def test_survives_an_interrupted_run(tmp_path):
    """Verdicts already paid for must not be lost when a long run is killed."""
    path = tmp_path / "v.sqlite"
    code = (
        "from cra.eval.entailment.cache import EntailmentCache;"
        f"c=EntailmentCache(r'{path}');"
        "[c.put(f'k{i}','entailed') for i in range(300)];"
        "import os; os._exit(1)"
    )
    subprocess.run([sys.executable, "-c", code], check=False)
    reopened = EntailmentCache(path)
    try:
        assert len(reopened) >= EntailmentCache.COMMIT_EVERY
    finally:
        reopened.close()
