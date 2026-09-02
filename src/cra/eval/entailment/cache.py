"""On-disk cache for entailment results.

Grading a headline run means tens of thousands of claim-by-passage forward
passes, and the evaluation is re-run often by design -- after a claim-extraction
fix, a threshold change, or a taxonomy refinement. Recomputing identical results
each time is the single largest cost in the eval stage, and it is entirely
avoidable: the checker is deterministic, so the same inputs always yield the
same output.

Two tables, because they are invalidated by different things.

``scores`` holds the (entailment, contradiction) probability pair, keyed
*without* thresholds. Probabilities are a property of the model and the text;
thresholds only decide where to cut them. This is what makes a threshold sweep
cheap -- it reads cached scores instead of re-running the model -- and therefore
what makes calibration feasible at all.

``verdicts`` holds a final label, and its key must include both thresholds: a
stale label surviving a threshold change would be silently wrong, and
thresholds are calibrated on dev and *will* move.

SQLite rather than one file per entry: results are tiny and there are tens of
thousands of them, so a directory tree would waste both inodes and lookup time.
It is also in the standard library, and autocommit means a killed run leaves a
valid cache holding everything computed up to that point.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

DEFAULT_CACHE_PATH = ".cache/entailment/verdicts.sqlite"


def score_key(
    checker: str,
    model_name: str,
    hypothesis: str,
    premises: list[str],
) -> str:
    """Key for a cached (entailment, contradiction) probability pair.

    Deliberately excludes thresholds. Probabilities are a property of the model
    and the text; thresholds only decide where to cut them. Keying scores by
    threshold -- as the verdict cache must -- would force a full recompute for
    every candidate cut, which is what made calibration expensive enough to skip.
    """
    parts = [checker, model_name, hypothesis, *premises]
    blob = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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
        # Eval runs are routinely started in parallel -- a scoring pass alongside
        # a rollout, two ablation arms at once -- so the cache must tolerate
        # concurrent access. Three settings together achieve that:
        #
        # isolation_level=None gives autocommit, so each write is its own short
        # transaction. Batching 256 writes previously held an exclusive lock
        # across the model inference happening between them, blocking any other
        # process for minutes. Writes are negligible next to inference, so
        # committing each one costs nothing measurable.
        #
        # WAL lets readers proceed during a write, and the busy timeout makes a
        # writer wait its turn rather than fail outright.
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, timeout=60.0, isolation_level=None
        )
        self._conn.execute("PRAGMA busy_timeout=60000")
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            # Switching journal mode needs a brief exclusive lock, which another
            # process mid-write will not yield. The cache still works on the
            # default mode -- writers just serialise -- so this is a degradation,
            # not a failure, and must not abort an evaluation run.
            pass
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, label TEXT NOT NULL)"
        )
        # Scores live in their own table, keyed without thresholds, so a
        # threshold sweep reads rather than recomputes.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS scores ("
            "key TEXT PRIMARY KEY, entail REAL NOT NULL, contradict REAL NOT NULL)"
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

    # Retained for the durability test and for callers that still reference it;
    # under autocommit every write is already durable the moment it returns.
    COMMIT_EVERY = 1

    def put(self, key: str, label: str) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO verdicts (key, label) VALUES (?, ?)", (key, label)
        )
        self._uncommitted += 1
        if self._uncommitted >= self.COMMIT_EVERY:
            self.commit()

    def get_scores(self, key: str) -> tuple[float, float] | None:
        """Cached ``(max_entail, max_contradict)`` for a claim, or None."""
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT entail, contradict FROM scores WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return float(row[0]), float(row[1])

    def put_scores(self, key: str, entail: float, contradict: float) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO scores (key, entail, contradict) VALUES (?, ?, ?)",
            (key, entail, contradict),
        )
        self._uncommitted += 1
        if self._uncommitted >= self.COMMIT_EVERY:
            self.commit()

    def __len__(self) -> int:
        """Number of stored verdicts."""
        if self._conn is None:
            return 0
        return int(self._conn.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0])

    def n_scores(self) -> int:
        if self._conn is None:
            return 0
        return int(self._conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0])

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
