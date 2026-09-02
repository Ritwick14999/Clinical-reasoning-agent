"""``EntailmentChecker`` protocol: does a set of evidence passages support a claim?

Two implementations, deliberately asymmetric in status:

* :mod:`cra.eval.entailment.nli` -- local, deterministic, no network. This is
  the checker the default ``cra eval`` pipeline uses, because it is the only
  one that actually satisfies docs/CONVENTIONS.md's "eval is a pure function of traces;
  no network, no model" invariant.
* :mod:`cra.eval.entailment.judge` -- calls a real LLM. Useful as a
  cross-validation signal (docs/DESIGN.md Sec 7: "cross-method agreement...
  is itself a result"), but it is *not* part of the default eval path -- see
  its module docstring for why, and how this project avoids judge-circularity.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

EntailmentLabel = Literal["entailed", "contradicted", "not_addressed"]


@runtime_checkable
class EntailmentChecker(Protocol):
    name: str

    def check(self, claim: str, evidence_texts: list[str]) -> EntailmentLabel: ...
