"""Dataset parsing, splits and the expected_tools oracle.

No network: PubMedQA/MedQA parsers are exercised against tiny local fixtures
shaped exactly like the real ori_pqal.json / test_ground_truth.json /
MIRAGE benchmark.json files, not the downloaded data.
"""

from __future__ import annotations

import json

import pytest

from cra.data.expected_tools import expected_tools_for
from cra.data.medqa import load_medqa
from cra.data.pubmedqa import load_pubmedqa
from cra.data.splits import get_split, sample_stratified
from cra.data.trapset import load_trapset
from cra.types import Question


def test_expected_tools_af_triggers_cha2ds2_vasc():
    text = "A 78-year-old woman with atrial fibrillation and hypertension."
    assert "calc_cha2ds2_vasc" in expected_tools_for(text)


def test_expected_tools_pe_triggers_wells():
    text = "Suspected pulmonary embolism after recent surgery with tachycardia."
    assert "calc_wells_pe" in expected_tools_for(text)


def test_expected_tools_two_drugs_triggers_interaction_check():
    text = "A patient on warfarin is started on ibuprofen for pain."
    assert "check_drug_interactions" in expected_tools_for(text)


def test_expected_tools_one_drug_does_not_trigger():
    text = "A patient is started on warfarin for a new diagnosis of atrial fibrillation."
    tools = expected_tools_for(text)
    assert "check_drug_interactions" not in tools
    assert "calc_cha2ds2_vasc" in tools  # AF is still present


def test_expected_tools_no_signal_is_empty():
    assert expected_tools_for("What is the mechanism of action of a beta blocker?") == []


# --------------------------------------------------------------------------
# pubmedqa
# --------------------------------------------------------------------------

def _write_pubmedqa_fixture(tmp_path):
    ori = {
        "111": {
            "QUESTION": "Does drug X help condition Y?",
            "CONTEXTS": ["Background sentence.", "Results sentence supporting yes."],
            "LABELS": ["BACKGROUND", "RESULTS"],
            "MESHES": ["Drug X"],
            "YEAR": "2020",
            "final_decision": "yes",
        },
        "222": {
            "QUESTION": "Does drug Z worsen condition W?",
            "CONTEXTS": ["Background.", "Results showing no effect."],
            "LABELS": ["BACKGROUND", "RESULTS"],
            "MESHES": [],
            "YEAR": "2019",
            "final_decision": "no",
        },
        "333": {
            "QUESTION": "Is this inconclusive?",
            "CONTEXTS": ["Some context."],
            "LABELS": ["BACKGROUND"],
            "MESHES": [],
            "YEAR": "2018",
            "final_decision": "maybe",
        },
    }
    test_gt = {"111": "yes"}  # only 111 is in the official test split
    ori_path = tmp_path / "ori_pqal.json"
    gt_path = tmp_path / "test_ground_truth.json"
    ori_path.write_text(json.dumps(ori), encoding="utf-8")
    gt_path.write_text(json.dumps(test_gt), encoding="utf-8")
    return ori_path, gt_path


def test_load_pubmedqa_split_membership(tmp_path):
    ori_path, gt_path = _write_pubmedqa_fixture(tmp_path)
    questions = load_pubmedqa(ori_path, gt_path)
    by_qid = {q.qid: q for q in questions}

    assert by_qid["111"].split == "test"
    assert by_qid["222"].split == "dev"
    assert by_qid["333"].split == "dev"
    assert by_qid["111"].gold_answer == "yes"
    assert by_qid["111"].gold_source_ids == ["111"]
    # The abstract is not inlined into the question -- it becomes a retrieval
    # corpus document instead, so the agent must retrieve it.
    assert "Background sentence" not in by_qid["111"].question
    assert by_qid["111"].options is None


def test_load_pubmedqa_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pubmedqa(tmp_path / "missing.json", tmp_path / "also_missing.json")


# --------------------------------------------------------------------------
# medqa
# --------------------------------------------------------------------------

def _write_medqa_fixture(tmp_path, n_per_answer=10):
    medqa = {}
    i = 0
    for answer in "ABCD":
        for _ in range(n_per_answer):
            medqa[f"{i:04d}"] = {
                "question": f"Question number {i} with answer {answer}.",
                "options": {"A": "opt A", "B": "opt B", "C": "opt C", "D": "opt D"},
                "answer": answer,
            }
            i += 1
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"medqa": medqa}), encoding="utf-8")
    return path


def test_load_medqa_stratified_dev_split(tmp_path):
    path = _write_medqa_fixture(tmp_path, n_per_answer=10)
    questions = load_medqa(path, dev_fraction=0.2, seed=1)

    assert len(questions) == 40
    assert all(q.gold_source_ids == [] for q in questions)
    dev = [q for q in questions if q.split == "dev"]
    test = [q for q in questions if q.split == "test"]
    assert len(dev) == 8  # 20% of 40, stratified
    assert len(test) == 32
    # Stratification: each answer letter contributes proportionally to dev.
    for answer in "ABCD":
        n_dev_for_answer = sum(1 for q in dev if q.gold_answer == answer)
        assert n_dev_for_answer == 2  # 20% of 10


def test_load_medqa_seed_is_deterministic(tmp_path):
    path = _write_medqa_fixture(tmp_path)
    a = load_medqa(path, seed=42)
    b = load_medqa(path, seed=42)
    assert [q.split for q in a] == [q.split for q in b]


# --------------------------------------------------------------------------
# trapset (real committed fixture, not synthetic)
# --------------------------------------------------------------------------

def test_load_trapset_real_file():
    questions = load_trapset()
    assert len(questions) >= 30
    for q in questions:
        assert q.dataset == "trapset"
        assert q.gold_answer in (q.options or {})
        assert q.gold_source_ids == []
        assert q.metadata.get("rationale")


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def test_get_split_test_requires_env(monkeypatch):
    monkeypatch.delenv("CRA_ALLOW_TEST", raising=False)
    with pytest.raises(RuntimeError, match="CRA_ALLOW_TEST"):
        get_split("trapset", "test")


def test_get_split_test_allowed_with_env(monkeypatch):
    monkeypatch.setenv("CRA_ALLOW_TEST", "1")
    # Should not raise, even though the trap set currently has no test items.
    result = get_split("trapset", "test")
    assert isinstance(result, list)


def _fake_questions(n: int, answers: list[str]) -> list[Question]:
    return [
        Question(
            qid=f"q{i}",
            dataset="trapset",
            split="dev",
            question=f"question {i}",
            gold_answer=answers[i % len(answers)],
        )
        for i in range(n)
    ]


def test_sample_stratified_respects_cap_and_is_deterministic():
    questions = _fake_questions(100, ["A", "B", "C", "D"])
    a = sample_stratified(questions, 20, seed=7)
    b = sample_stratified(questions, 20, seed=7)
    assert len(a) == 20
    assert {q.qid for q in a} == {q.qid for q in b}


def test_sample_stratified_preserves_balance():
    questions = _fake_questions(100, ["A", "B", "C", "D"])  # 25 each
    sample = sample_stratified(questions, 40, seed=1)
    counts = {a: sum(1 for q in sample if q.gold_answer == a) for a in "ABCD"}
    assert all(c == 10 for c in counts.values())


def test_sample_stratified_noop_when_n_exceeds_pool():
    questions = _fake_questions(5, ["A", "B"])
    assert sample_stratified(questions, 100, seed=1) == questions


def test_sample_stratified_none_returns_all():
    questions = _fake_questions(5, ["A"])
    assert sample_stratified(questions, None, seed=1) == questions
