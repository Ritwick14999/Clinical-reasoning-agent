"""On-disk cache for entailment verdicts.

Grading a headline run means tens of thousands of claim-by-passage forward
passes, and the evaluation is re-run often by design -- after a claim-extraction
fix, a threshold change, or a taxonomy refinement. Recomputing identical
verdicts each time is the single largest cost in the eval stage, and it is
entirely avoidable: the checker is deterministic, so the same inputs always
yield the same label.

The key covers everything that could change a verdict -- checker, model,
thresholds, the hypothesis, and every premise. Miss one and a stale label
would silently survive a change that should have invalidated it. Thresholds in
particular are calibrated on dev and *will* move.

SQLite rather than one file per entry: verdicts are three-word strings and
there are tens of thousands of them, so a directory tree would waste both
inodes and lookup time. It is also in the standard library, and its
transactions make a killed run leave a valid cache rather than a torn one.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

DEFAULT_CACHE_PATH = ".cache/entailment/verdicts.sqlite"


def verdict_key(
    checker: str,
    model_name: str,
    entail_threshold: float,
    contradict_threshold: float,
    hypothesis: str,
    premises: list[str],
) -> str:
    parts = [
        checker,
        model_name,
        f"{entail_threshold:.6f}",
        f"{contradict_threshold:.6f}",
        hypothesis,
        # Order matters: the verdict takes a max over premises, so a reordering
        # is equivalent -- but sorting here would hide a genuine change in which
        # passages were retrieved. Keep the sequence as given.
        *premises,
    ]
    blob = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class EntailmentCache:
    """Key -> verdict, persisted. A disabled cache is a working no-op."""

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH, enabled: bool = True) -> None:
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._uncommitted = 0
        self._conn: sqlite3.Connection | None = None
        if not enabled:
            return
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, label TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        if self._conn is None:
            return None
        row = self._conn.execute("SELECT label FROM verdicts WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return row[0]

    # Commit in batches rather than per row: one fsync per verdict would cost
    # more than the inference it saves, but never committing would lose the whole
    # cache if a long grading run is interrupted -- which is exactly when it is
    # most worth keeping.
    COMMIT_EVERY = 256

    def put(self, key: str, label: str) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO verdicts (key, label) VALUES (?, ?)", (key, label)
        )
        self._uncommitted += 1
        if self._uncommitted >= self.COMMIT_EVERY:
            self.commit()

    def __len__(self) -> int:
        """Number of stored verdicts."""
        if self._conn is None:
            return 0
        return int(self._conn.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0])

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._uncommitted = 0

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def __enter__(self) -> EntailmentCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        # A grading run that ends without an explicit close must not discard
        # verdicts that are already computed and paid for.
        try:
            self.close()
        except Exception:
            pass
