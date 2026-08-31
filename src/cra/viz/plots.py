"""The stacked failure-mode bar chart -- docs/DESIGN.md Sec 6's headline
figure. One bar per (model, dataset); segments are the taxonomy's labels, in
causal order (R1 -> R5) so the bar reads bottom-to-top as "how far the agent
got before things went wrong."

MedQA (and the trap set) have no gold source document, so ``retrieval_failure``
is structurally impossible there. A bar for a group with no gold-available
traces gets an explicit annotation rather than a silent zero -- a real
``retrieval_failure`` count of zero and "this label cannot fire here" are
different claims, and docs/DESIGN.md Sec 6 says the omission must be visible:
"the MedQA chart omits (rather than zeroes) that segment."
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to render and save
import matplotlib.pyplot as plt

from cra.eval.records import TraceEvalRecord

_LABEL_ORDER = [
    "retrieval_failure", "tool_misuse", "reasoning_failure",
    "unsupported_claim", "correct_grounded", None, "no_answer",
]
_LABEL_DISPLAY = {
    "retrieval_failure": "Retrieval failure",
    "tool_misuse": "Tool misuse",
    "reasoning_failure": "Reasoning failure",
    "unsupported_claim": "Unsupported claim",
    "correct_grounded": "Correct, grounded",
    None: "Correct, ungraded",
    "no_answer": "No answer",
}
_LABEL_COLOR = {
    "retrieval_failure": "#c44e52",
    "tool_misuse": "#dd8452",
    "reasoning_failure": "#ccb974",
    "unsupported_claim": "#8172b3",
    "correct_grounded": "#55a868",
    None: "#b0b0b0",
    "no_answer": "#4c4c4c",
}


def plot_failure_modes(records: list[TraceEvalRecord], out_path: str | Path) -> Path:
    groups: dict[tuple[str, str], Counter] = {}
    gold_ever_available: dict[tuple[str, str], bool] = {}
    for r in records:
        key = (r.model_id, r.dataset)
        groups.setdefault(key, Counter())[r.failure_mode] += 1
        gold_ever_available[key] = gold_ever_available.get(key, False) or r.retrieval_gold_available

    keys = sorted(groups)
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.8), 5.5))

    bottoms = [0.0] * len(keys)
    for label in _LABEL_ORDER:
        heights = []
        for key in keys:
            total = sum(groups[key].values())
            heights.append((groups[key][label] / total) if total else 0.0)
        ax.bar(
            [f"{model}\n{dataset}" for model, dataset in keys],
            heights, bottom=bottoms,
            label=_LABEL_DISPLAY[label], color=_LABEL_COLOR[label],
        )
        bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]

    for i, key in enumerate(keys):
        if not gold_ever_available[key]:
            ax.text(i, 1.02, "no gold source\n(retrieval_failure\ncannot fire)",
                     ha="center", va="bottom", fontsize=7, color="#666666")

    ax.set_ylabel("Share of traces")
    ax.set_ylim(0, 1.15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Failure-mode breakdown")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
