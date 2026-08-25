from __future__ import annotations

import pytest

from cra.llm.base import ToolCall
from cra.tools.base import EvidenceDraft, EvidenceStore
from cra.tools.drugs import DrugInteractionTool, find_interactions, normalize_drug
from cra.tools.registry import ToolRegistry, default_registry
from cra.tools.retrieval import InMemoryRetriever, RetrievedPassage, SearchLiteratureTool
from cra.tools.units import UnitConversionTool


class TestDrugInteractions:
    def test_brand_names_resolve_to_generics(self):
        assert normalize_drug("Coumadin") == "warfarin"
        assert normalize_drug("  ADVIL tablets ") == "ibuprofen"
        assert normalize_drug("Bactrim") == "trimethoprim-sulfamethoxazole"

    def test_class_rule_matches_any_member(self):
        for nsaid in ("ibuprofen", "naproxen", "ketorolac"):
            found, _, _ = find_interactions(["warfarin", nsaid])
            assert len(found) == 1
            assert found[0]["severity"] == "major"

    def test_contraindicated_pairs_are_flagged(self):
        found, _, _ = find_interactions(["phenelzine", "sertraline"])
        assert found[0]["severity"] == "contraindicated"

    def test_results_sorted_most_severe_first(self):
        found, _, _ = find_interactions(["nitroglycerin", "sildenafil", "warfarin", "sertraline"])
        assert found[0]["severity"] == "contraindicated"

    def test_unrecognised_drugs_are_named_not_silently_ignored(self):
        result = DrugInteractionTool().run(drugs=["warfarin", "fictionalmab"])
        assert "NOT CHECKED" in result.output
        assert "fictionalmab" in result.output

    def test_no_interaction_case_is_explicit(self):
        result = DrugInteractionTool().run(drugs=["acetaminophen", "levothyroxine"])
        assert "No interactions found" in result.output

    def test_requires_at_least_two_drugs(self):
        result = DrugInteractionTool().run(drugs=["warfarin"])
        assert result.ok is False
        assert "at least two" in result.error

    def test_duplicates_do_not_self_interact(self):
        found, _, _ = find_interactions(["warfarin", "warfarin"])
        assert found == []

    def test_qt_class_interacts_with_itself(self):
        found, _, _ = find_interactions(["amiodarone", "haloperidol"])
        assert len(found) == 1
        assert "QT" in found[0]["mechanism"]

    def test_disclaimer_always_present(self):
        assert "not clinical decision support" in DrugInteractionTool().run(
            drugs=["warfarin", "ibuprofen"]
        ).output


class TestUnitConversion:
    @pytest.mark.parametrize(
        "analyte,value,src,dst,expected",
        [
            ("glucose", 100, "mg/dL", "mmol/L", 5.55),
            ("creatinine", 1.0, "mg/dL", "umol/L", 88.4),
            ("creatinine", 88.4, "umol/L", "mg/dL", 1.0),
            ("albumin", 3.5, "g/dL", "g/L", 35.0),
            ("bilirubin", 1.0, "mg/dL", "umol/L", 17.1),
        ],
    )
    def test_round_trip_values(self, analyte, value, src, dst, expected):
        result = UnitConversionTool().run(analyte=analyte, value=value, from_unit=src, to_unit=dst)
        assert result.ok
        assert f"{expected:g}" in result.output

    def test_micro_sign_is_accepted(self):
        result = UnitConversionTool().run(
            analyte="creatinine", value=88.4, from_unit="µmol/L", to_unit="mg/dL"
        )
        assert result.ok

    def test_unknown_analyte_lists_options(self):
        result = UnitConversionTool().run(
            analyte="unobtainium", value=1, from_unit="mg/dL", to_unit="mmol/L"
        )
        assert result.ok is False
        assert "glucose" in result.error

    def test_unsupported_unit_pair_rejected(self):
        result = UnitConversionTool().run(
            analyte="glucose", value=1, from_unit="mg/dL", to_unit="furlongs"
        )
        assert result.ok is False


class TestEvidenceStore:
    def test_ids_are_prefixed_by_kind(self):
        store = EvidenceStore()
        items = store.add(
            [
                EvidenceDraft(kind="passage", text="a", source_id="1"),
                EvidenceDraft(kind="tool_output", text="b"),
                EvidenceDraft(kind="passage", text="c", source_id="2"),
            ],
            source_tool="t",
        )
        assert [i.evidence_id for i in items] == ["E1", "T1", "E2"]

    def test_duplicates_return_the_original(self):
        store = EvidenceStore()
        first = store.add([EvidenceDraft(kind="passage", text="a", source_id="1")], "t")
        second = store.add([EvidenceDraft(kind="passage", text="a", source_id="1")], "t")
        assert first[0].evidence_id == second[0].evidence_id
        assert len(store.items) == 1


class TestRegistry:
    def test_rejects_duplicate_names(self):
        registry = ToolRegistry([UnitConversionTool()])
        with pytest.raises(ValueError, match="duplicate"):
            registry.register(UnitConversionTool())

    def test_schema_violation_is_reported_with_a_path(self, registry):
        record, _ = ToolRegistry([UnitConversionTool()]).dispatch(
            ToolCall("c", "convert_lab_units", {"analyte": "glucose", "value": "lots",
                                                "from_unit": "mg/dL", "to_unit": "mmol/L"}),
            EvidenceStore(), 0, 0,
        )
        assert record.ok is False
        assert "schema_violation" in record.error
        assert "value" in record.error

    def test_additional_properties_rejected(self):
        record, _ = ToolRegistry([UnitConversionTool()]).dispatch(
            ToolCall("c", "convert_lab_units", {"analyte": "glucose", "value": 1,
                                                "from_unit": "mg/dL", "to_unit": "mmol/L",
                                                "extra": 1}),
            EvidenceStore(), 0, 0,
        )
        assert record.ok is False

    def test_parse_error_from_adapter_surfaces_as_malformed_call(self):
        call = ToolCall("c", "convert_lab_units", {}, parse_error="unterminated string")
        record, _ = ToolRegistry([UnitConversionTool()]).dispatch(call, EvidenceStore(), 0, 0)
        assert "malformed_call" in record.error

    def test_tool_exception_is_contained(self):
        class Exploding:
            name, description = "boom", "explodes"
            input_schema = {"type": "object", "properties": {}, "required": []}

            def run(self, **kwargs):
                raise ZeroDivisionError("nope")

        record, observation = ToolRegistry([Exploding()]).dispatch(
            ToolCall("c", "boom", {}), EvidenceStore(), 0, 0
        )
        assert record.ok is False
        assert "tool_exception" in record.error
        assert "ZeroDivisionError" in observation

    def test_closed_book_registry_is_empty(self):
        assert len(default_registry(retriever=None, include_calculators=False,
                                    include_drugs=False, include_units=False)) == 0

    def test_default_registry_omits_search_without_a_retriever(self):
        assert "search_literature" not in default_registry(retriever=None)


class TestSearchTool:
    def test_empty_result_warns_against_assuming_evidence(self):
        tool = SearchLiteratureTool(InMemoryRetriever([]))
        result = tool.run(query="anything at all")
        assert result.ok
        assert result.evidence == []
        assert "do not assume evidence exists" in result.output

    def test_k_is_bounded(self):
        tool = SearchLiteratureTool(InMemoryRetriever([]))
        assert tool.run(query="x", k=99).ok is False
        assert tool.run(query="x", k=0).ok is False

    def test_empty_query_rejected(self):
        tool = SearchLiteratureTool(InMemoryRetriever([]))
        assert tool.run(query="   ").ok is False

    def test_long_passages_are_truncated(self):
        long_text = "warfarin " * 500
        tool = SearchLiteratureTool(InMemoryRetriever([RetrievedPassage("1", long_text, "t")]))
        result = tool.run(query="warfarin interaction")
        assert len(result.evidence[0].text) < len(long_text)
        assert result.evidence[0].text.endswith("...")

    def test_source_id_is_preserved_for_hit_at_k(self):
        tool = SearchLiteratureTool(
            InMemoryRetriever([RetrievedPassage("PMID123", "warfarin bleeding risk", "t")])
        )
        result = tool.run(query="warfarin bleeding")
        assert result.evidence[0].source_id == "PMID123"
