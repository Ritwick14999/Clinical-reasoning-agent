"""Configuration objects and provenance hashing.

Every trace carries a ``config_hash`` covering everything that could change
the agent's behaviour. Two traces with the same hash were produced under the
same conditions; two with different hashes were not, whatever the filenames
say. That is the difference between a reproducible experiment and a plausible
claim about one.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

CONFIG_ROOT = Path(os.environ.get("CRA_CONFIG_ROOT", "configs"))


@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["anthropic", "openai_compat", "mock", "mock_scripted"] = "mock"
    model_id: str = "mock-heuristic"
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 1024
    use_cache: bool = True
    cache_dir: str = ".cache/llm"
    # Mock-only knobs, ignored by real providers.
    mock: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    # Only native function calling is implemented. A "react" option was declared
    # here and never built: the loop never branched on it, so setting it would
    # have silently produced function-calling traces labelled as ReAct. Removing
    # the option is more honest than leaving a setting that does nothing, and a
    # config naming it now fails loudly instead.
    mode: Literal["function_calling"] = "function_calling"
    tool_budget: int = 5
    max_steps: int = 8
    temperature: float = 0.0
    seed: int | None = 12345
    allow_repair: bool = True
    # Closed-book: no tools at all, for the grounding ablation.
    closed_book: bool = False
    enable_retrieval: bool = True
    enable_calculators: bool = True
    enable_drugs: bool = True
    enable_units: bool = True

    def __post_init__(self) -> None:
        # A Literal annotation is a type hint, not a runtime check: without this,
        # mode="react" would be accepted, recorded on every trace, and silently
        # produce function-calling episodes labelled as something else.
        if self.mode != "function_calling":
            raise ValueError(
                f"unsupported agent mode {self.mode!r}. Only 'function_calling' is "
                "implemented; a 'react' mode was declared but never built, and the loop "
                "never branched on it."
            )
        if self.tool_budget < 0:
            raise ValueError(f"tool_budget must be >= 0, got {self.tool_budget}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")


@dataclass(frozen=True)
class RetrievalConfig:
    backend: Literal["bm25", "dense", "none"] = "bm25"
    index_dir: str = "data/index/bm25_pqal"
    corpus: str = "data/processed/corpus_pqal.jsonl"
    # 5 -> 3: at k=5 with the old 1400-char snippet a five-search episode reached
    # ~10700 tokens, which overflows an 8192 num_ctx. 
    default_k: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str = "adhoc"
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    datasets: list[str] = field(default_factory=lambda: ["pubmedqa", "medqa"])
    split: Literal["dev", "test"] = "dev"
    limit: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        """Covers behaviour-affecting fields only.

        ``experiment_id``, ``notes`` and ``limit`` are excluded on purpose: a
        renamed run or a shorter run of the same configuration should hash
        alike, so traces from a pilot and a full run remain comparable.
        """
        payload = self.to_dict()
        payload.pop("experiment_id", None)
        payload.pop("notes", None)
        payload.pop("limit", None)
        payload["model"].pop("use_cache", None)
        payload["model"].pop("cache_dir", None)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve(section: str, value: Any) -> dict[str, Any]:
    """A section may be an inline mapping or the name of a file in configs/<section>/."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return _load_yaml(CONFIG_ROOT / section / f"{value}.yaml")


def _filter(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"unknown keys for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return data


def load_experiment(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load an experiment config, resolving named model/agent/retrieval sections."""
    path = Path(path)
    if not path.exists() and not path.suffix:
        path = CONFIG_ROOT / "experiments" / f"{path.name}.yaml"
    raw = _load_yaml(path)
    raw.update(overrides or {})

    model = ModelConfig(**_filter(ModelConfig, _resolve("model", raw.pop("model", None))))
    agent = AgentConfig(**_filter(AgentConfig, _resolve("agent", raw.pop("agent", None))))
    retrieval = RetrievalConfig(
        **_filter(RetrievalConfig, _resolve("retrieval", raw.pop("retrieval", None)))
    )
    rest = _filter(ExperimentConfig, raw)
    return ExperimentConfig(model=model, agent=agent, retrieval=retrieval, **rest)


def with_overrides(cfg: ExperimentConfig, **kwargs: Any) -> ExperimentConfig:
    """Return a copy with top-level fields replaced (used by the ablation sweeps)."""
    return replace(cfg, **kwargs)
