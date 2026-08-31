"""Corpus construction and the BM25 retriever backend.

No network, no downloaded data: everything runs against tiny in-memory or
tmp_path fixtures.
"""

from __future__ import annotations

import json

from cra.retrieval.bm25 import BM25Retriever
from cra.retrieval.corpus import (
    CorpusDoc,
    build_corpus_from_pubmedqa,
    read_corpus_jsonl,
    write_corpus_jsonl,
)


def test_corpus_jsonl_roundtrip(tmp_path):
    docs = [
        CorpusDoc(doc_id="1", text="alpha beta", title="T1", metadata={"year": "2020"}),
        CorpusDoc(doc_id="2", text="gamma delta", title=None, metadata={}),
    ]
    path = write_corpus_jsonl(docs, tmp_path / "corpus.jsonl")
    loaded = read_corpus_jsonl(path)
    assert loaded == docs


def test_build_corpus_from_pubmedqa_joins_contexts(tmp_path):
    ori = {
        "111": {"CONTEXTS": ["First sentence.", "Second sentence."], "LABELS": ["A", "B"], "YEAR": "2021"},
        "222": {"CONTEXTS": [], "LABELS": [], "YEAR": "2021"},  # no text -> skipped
    }
    path = tmp_path / "ori_pqal.json"
    path.write_text(json.dumps(ori), encoding="utf-8")

    docs = build_corpus_from_pubmedqa(path)
    assert len(docs) == 1
    assert docs[0].doc_id == "111"
    assert "First sentence." in docs[0].text
    assert "Second sentence." in docs[0].text


def test_bm25_retriever_ranks_relevant_doc_first():
    docs = [
        CorpusDoc(doc_id="A", text="Atrial fibrillation increases stroke risk in elderly patients."),
        CorpusDoc(doc_id="B", text="Peptic ulcer disease is caused by H. pylori infection."),
        CorpusDoc(doc_id="C", text="Anticoagulation reduces stroke risk in atrial fibrillation."),
    ]
    retriever = BM25Retriever(docs)
    hits = retriever.search("atrial fibrillation stroke risk", k=2)

    assert len(hits) == 2
    assert {h.doc_id for h in hits} == {"A", "C"}
    assert hits[0].score >= hits[1].score


def test_bm25_retriever_empty_query_returns_nothing():
    retriever = BM25Retriever([CorpusDoc(doc_id="A", text="some text")])
    assert retriever.search("!!!", k=5) == []


def test_bm25_from_corpus_file(tmp_path):
    # rank_bm25's IDF is <=0 when a term appears in half or more of the corpus
    # (log((N-freq+0.5)/(freq+0.5))), so a query term needs to be a minority
    # across enough filler documents to score positive.
    docs = [
        CorpusDoc(doc_id="1", text="hypertension management guidelines"),
        CorpusDoc(doc_id="2", text="unrelated orthopedic fracture care"),
        CorpusDoc(doc_id="3", text="dermatology skin biopsy technique"),
        CorpusDoc(doc_id="4", text="pediatric vaccination schedule"),
    ]
    path = write_corpus_jsonl(docs, tmp_path / "corpus.jsonl")
    retriever = BM25Retriever.from_corpus_file(path)
    hits = retriever.search("hypertension guidelines", k=1)
    assert hits and hits[0].doc_id == "1"
