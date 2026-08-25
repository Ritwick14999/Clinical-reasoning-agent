"""Tool registry: validation, dispatch, and observation composition.

This is the only place that turns a model-emitted tool call into an
observation. It is also the only place that assigns evidence IDs, so the IDs
the model sees in an observation are exactly the ones the evaluator later
resolves citations against.

Nothing here raises on bad model output. Every failure mode -- unknown tool,
schema violation, unparseable arguments, an exception inside a tool -- is
recorded as a failed :class:`ToolCallRecord` and fed back to the model as text.
"""

from __future__ import annotations

import time
from typing import Any

import jsonschema

from cra.llm.base import ToolCall, ToolSpec
from cra.tools.base import EvidenceStore, Tool, ToolResult, spec_of
from cra.types import Evidence, ToolCallRecord


def compose_observation(result: ToolResult, evidence: list[Evidence]) -> str:
    """The text the model sees back, with evidence IDs resolved."""
    parts = [result.output] if result.output else []
    parts += [e.render() for e in evidence]
    if evidence:
        ids = ", ".join(e.evidence_id for e in evidence)
        parts.append(f"(Cite these as: {ids})")
    return "\n\n".join(parts)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[ToolSpec]:
        return [spec_of(t) for t in self._tools.values()]

    def dispatch(
        self,
        call: ToolCall,
        store: EvidenceStore,
        index: int,
        step: int,
    ) -> tuple[ToolCallRecord, str]:
        """Run one tool call. Returns the record and the observation text."""
        started = time.perf_counter()

        def fail(error: str, output: str | None = None) -> tuple[ToolCallRecord, str]:
            text = output or f"ERROR: {error}"
            record = ToolCallRecord(
                index=index, step=step, name=call.name, args=call.args, ok=False,
                output=text, error=error, latency_ms=(time.perf_counter() - started) * 1000,
            )
            return record, text

        # The adapter could not parse the arguments the model emitted.
        if call.parse_error:
            return fail(f"malformed_call: {call.parse_error}")

        tool = self._tools.get(call.name)
        if tool is None:
            return fail(
                f"unknown_tool: no tool named {call.name!r}. "
                f"Available tools: {', '.join(self.names)}"
            )

        try:
            jsonschema.validate(call.args, tool.input_schema)
        except jsonschema.ValidationError as exc:
            path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
            return fail(f"schema_violation: at {path}: {exc.message}")

        try:
            result = tool.run(**call.args)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not end the episode
            return fail(
                f"tool_exception: {type(exc).__name__}: {exc}",
                output=f"ERROR: the tool raised {type(exc).__name__}: {exc}",
            )

        evidence = store.add(result.evidence, source_tool=tool.name)
        observation = compose_observation(result, evidence)
        record = ToolCallRecord(
            index=index,
            step=step,
            name=call.name,
            args=call.args,
            ok=result.ok,
            output=observation,
            error=result.error,
            evidence_ids=[e.evidence_id for e in evidence],
            latency_ms=result.latency_ms or (time.perf_counter() - started) * 1000,
        )
        return record, observation


def default_registry(
    retriever: Any | None = None,
    include_calculators: bool = True,
    include_drugs: bool = True,
    include_units: bool = True,
) -> ToolRegistry:
    """The standard tool set. Omitting the retriever gives a closed-book agent."""
    from cra.tools.calculators.tool import ALL_CALCULATORS
    from cra.tools.drugs import DrugInteractionTool
    from cra.tools.retrieval import SearchLiteratureTool
    from cra.tools.units import UnitConversionTool

    tools: list[Tool] = []
    if retriever is not None:
        tools.append(SearchLiteratureTool(retriever))
    if include_drugs:
        tools.append(DrugInteractionTool())
    if include_calculators:
        tools.extend(ALL_CALCULATORS)
    if include_units:
        tools.append(UnitConversionTool())
    return ToolRegistry(tools)
