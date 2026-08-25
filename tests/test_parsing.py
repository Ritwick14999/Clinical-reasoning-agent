from __future__ import annotations

import pytest

from cra.agent.parsing import parse_final

MCQ = ["A", "B", "C", "D"]
YN = ["yes", "no", "maybe"]


def test_plain_json():
    final, err = parse_final('{"answer":"B","justification":"j","citations":["E1"]}', MCQ)
    assert err == ""
    assert (final.answer, final.justification, final.citations) == ("B", "j", ["E1"])


def test_code_fenced_json_with_prose():
    text = 'Sure.\n```json\n{"answer":"yes","justification":"j","citations":[]}\n```\nDone.'
    final, _ = parse_final(text, YN)
    assert final.answer == "yes"


def test_json_embedded_in_prose():
    text = 'My conclusion: {"answer": "C", "justification": "j"} -- that is my answer.'
    final, _ = parse_final(text, MCQ)
    assert final.answer == "C"


def test_answer_is_normalised_to_the_allowed_form():
    final, _ = parse_final('{"answer":"(b)","justification":"j"}', MCQ)
    assert final.answer == "B"


def test_citations_extracted_from_a_string():
    final, _ = parse_final('{"answer":"A","justification":"j","citations":"E3, T1 and E3"}', MCQ)
    assert final.citations == ["E3", "T1"]


def test_citations_deduplicated_and_cleaned():
    final, _ = parse_final('{"answer":"A","justification":"j","citations":["[E1]","E1","nonsense"]}', MCQ)
    assert final.citations == ["E1"]


def test_malformed_citation_container_is_tolerated():
    final, _ = parse_final('{"answer":"A","justification":"j","citations":{"a":1}}', MCQ)
    assert final.citations == []


def test_missing_justification_is_empty_not_fatal():
    final, err = parse_final('{"answer":"A"}', MCQ)
    assert err == ""
    assert final.justification == ""


@pytest.mark.parametrize(
    "text,fragment",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("The answer is B.", "no JSON object"),
        ('{"justification":"j"}', "no JSON object"),
        ('{"answer":"","justification":"j"}', "empty"),
        ('{"answer":"Z","justification":"j"}', "not one of the allowed"),
    ],
)
def test_rejections_explain_themselves(text, fragment):
    final, err = parse_final(text, MCQ)
    assert final is None
    assert fragment in err


def test_raw_text_is_retained():
    text = '{"answer":"A","justification":"j"}'
    final, _ = parse_final(text, MCQ)
    assert final.raw == text


def test_first_valid_object_wins_over_later_ones():
    text = '{"answer":"A","justification":"first"} then {"answer":"D","justification":"second"}'
    final, _ = parse_final(text, MCQ)
    assert final.answer == "A"


def test_no_allowed_list_accepts_free_text():
    final, _ = parse_final('{"answer":"anticoagulation","justification":"j"}', None)
    assert final.answer == "anticoagulation"
