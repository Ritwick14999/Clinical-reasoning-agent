"""Dataset download with checksummed provenance.

Fetches source files to ``data/raw/`` and records URL + sha256 + size in
``data/manifests/`` (which *is* committed, unlike ``data/raw/``), so an
upstream change to a source file is caught on the next download rather than
silently absorbed into a rebuilt corpus or a re-parsed split.

    python tasks.py data              # everything
    python -m cra.data.download --only pubmedqa_ori
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw")
MANIFEST_DIR = Path("data/manifests")

SOURCES: dict[str, dict[str, str]] = {
    "pubmedqa_ori": {
        "url": "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
        "dest": "pubmedqa/ori_pqal.json",
    },
    "pubmedqa_test_gt": {
        "url": "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json",
        "dest": "pubmedqa/test_ground_truth.json",
    },
    "mirage_benchmark": {
        "url": "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json",
        "dest": "mirage/benchmark.json",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(name: str, url: str, dest: Path, manifest_dir: Path = MANIFEST_DIR) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cra-benchmark/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())

    sha256 = _sha256(dest)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{name}.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if previous and previous.get("sha256") != sha256:
        print(
            f"WARNING: {name} sha256 changed since the last download "
            f"({previous['sha256'][:12]}... -> {sha256[:12]}...). Upstream data has moved; "
            "re-check parsing assumptions before trusting a rebuilt corpus or split."
        )
    manifest_path.write_text(
        json.dumps(
            {
                "name": name,
                "url": url,
                "dest": str(dest),
                "sha256": sha256,
                "bytes": dest.stat().st_size,
                "downloaded_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dest


def download_all(
    raw_dir: Path = RAW_DIR, manifest_dir: Path = MANIFEST_DIR, only: list[str] | None = None
) -> None:
    for name, spec in SOURCES.items():
        if only and name not in only:
            continue
        dest = raw_dir / spec["dest"]
        print(f"Downloading {name} -> {dest}")
        download_one(name, spec["url"], dest, manifest_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only", nargs="*", default=None, choices=sorted(SOURCES), help="download a subset of sources"
    )
    args = parser.parse_args(argv)
    download_all(only=args.only)


if __name__ == "__main__":
    main()
