"""Build the retrieval corpus from downloaded PubMedQA data.

    python tasks.py index
    python -m cra.retrieval.index_build --raw data/raw/pubmedqa/ori_pqal.json \\
        --out data/processed/corpus_pqal.jsonl

Writes ``data/processed/corpus_pqal.jsonl``. BM25 itself needs no persisted
index -- it tokenizes the corpus at experiment start, which is cheap at this
scale (~1000 documents); this step only needs to run once per corpus change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cra.retrieval.corpus import build_corpus_from_pubmedqa, write_corpus_jsonl

DEFAULT_RAW = Path("data/raw/pubmedqa/ori_pqal.json")
DEFAULT_OUT = Path("data/processed/corpus_pqal.jsonl")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    docs = build_corpus_from_pubmedqa(args.raw)
    out = write_corpus_jsonl(docs, args.out)
    print(f"Wrote {len(docs)} documents to {out}")


if __name__ == "__main__":
    main()
