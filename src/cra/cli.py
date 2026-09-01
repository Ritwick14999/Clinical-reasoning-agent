"""Command-line entry point (the ``cra`` console script, or ``python -m cra.cli``).

Subcommands: ``rollout`` (Phase 2, drive the agent and write traces),
``show-trace`` (render one for a human), ``eval`` (Phase 3, the pure
no-network path: traces -> tables + figure), ``annotate`` (the blind
human-validation workflow), and ``judge-check`` (the explicitly live-model
cross-judge entailment pass -- see ``cra.eval.entailment.judge``'s module
docstring for why it is kept separate from ``eval``).
"""

from __future__ import annotations

import argparse
import sys


def _use_utf8_stdio() -> None:
    """Windows terminals default stdout/stderr to the system codepage (often
    cp1252), which cannot encode most of what a real model actually says --
    en dashes, curly quotes, non-Latin text. Reconfiguring here (rather than
    hoping every ``print`` call is ASCII-only) is what lets ``show-trace``
    render real Ollama output instead of crashing on it.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> None:
    _use_utf8_stdio()
    parser = argparse.ArgumentParser(prog="cra")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rollout = sub.add_parser("rollout", help="run an experiment and write traces")
    p_rollout.add_argument("--config", required=True, help="experiment config name or path")
    p_rollout.add_argument("--limit", type=int, default=None, help="override the per-dataset question cap")
    p_rollout.add_argument("--split", choices=["dev", "test"], default=None, help="override the split")
    p_rollout.add_argument("--out-dir", default=None, help="override results/traces/<experiment_id>")
    p_rollout.add_argument(
        "--dry-run", action="store_true", help="run the preflight check and print the plan; run nothing"
    )

    p_show = sub.add_parser("show-trace", help="render one or more traces as text")
    p_show.add_argument("path", help="a trace file, or a results/traces/<experiment_id> directory")
    p_show.add_argument("--limit", type=int, default=None, help="show at most N traces")

    p_eval = sub.add_parser(
        "eval", help="score committed traces into tables and a figure (pure function of traces)"
    )
    p_eval.add_argument("--experiments", required=True, nargs="+", help="experiment_id(s) under results/traces/")
    p_eval.add_argument("--out-dir", default="results")
    p_eval.add_argument(
        "--nli", action="store_true",
        help="grade correct-answer traces (R4 vs R5) with the local NLI checker; "
        "requires the 'dense' extra. Without this flag those traces are left ungraded.",
    )

    p_annotate = sub.add_parser("annotate", help="write a blind human-annotation template")
    p_annotate.add_argument("--experiments", required=True, nargs="+")
    p_annotate.add_argument("--sample", type=int, default=40)
    p_annotate.add_argument("--out", default="results/annotations/sample.csv")
    p_annotate.add_argument(
        "--nli", action="store_true", help="grade with NLI first, so the sample is drawn across real R4/R5 labels"
    )

    p_judge = sub.add_parser(
        "judge-check",
        help="live cross-model LLM-judge entailment pass on a stratified sample "
        "(calls real models -- not part of `eval`)",
    )
    p_judge.add_argument("--experiments", required=True, nargs="+")
    p_judge.add_argument("--sample", type=int, default=40)
    p_judge.add_argument("--out", default="results/tables/judge_check.md")

    p_score = sub.add_parser(
        "annotate-score", help="compute Cohen's kappa between a completed annotation CSV and the classifier"
    )
    p_score.add_argument("--experiments", required=True, nargs="+")
    p_score.add_argument("--annotations", required=True, help="the filled-in CSV from `cra annotate`")
    p_score.add_argument(
        "--nli", action="store_true", help="grade with NLI first, so R4/R5 predictions are available to score against"
    )

    args = parser.parse_args(argv)

    if args.command == "rollout":
        from cra.rollout import run_rollout

        run_rollout(
            config=args.config,
            limit=args.limit,
            split=args.split,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
        )
    elif args.command == "show-trace":
        from cra.trace_io import read_trace_dir, render_trace

        traces = read_trace_dir(args.path)
        for trace in traces[: args.limit]:
            print(render_trace(trace))
    elif args.command == "eval":
        _run_eval_command(args)
    elif args.command == "annotate":
        _run_annotate_command(args)
    elif args.command == "judge-check":
        _run_judge_check_command(args)
    elif args.command == "annotate-score":
        _run_annotate_score_command(args)


def _build_nli_checker():
    from cra.eval.entailment.nli import NLIEntailmentChecker

    print("Loading the local NLI model (first run downloads the checkpoint)...")
    return NLIEntailmentChecker()


def _run_eval_command(args: argparse.Namespace) -> None:
    from cra.eval.report import render_failure_mode_table, render_markdown_table
    from cra.eval.run import run_eval

    entailment = _build_nli_checker() if args.nli else None
    trace_dirs = [f"results/traces/{e}" for e in args.experiments]
    _traces, records, summaries = run_eval(trace_dirs, out_dir=args.out_dir, entailment=entailment)

    print(render_markdown_table(summaries))
    print()
    print(render_failure_mode_table(summaries))
    n_ungraded = sum(1 for r in records if r.is_correct and r.hallucinated is None)
    if n_ungraded:
        print(
            f"\n{n_ungraded} correct-answer trace(s) are ungraded (failure_mode=None): "
            "no entailment checker was run. Pass --nli to grade them, or use "
            "`cra judge-check` for a live cross-model pass."
        )
    print(f"\nWrote {args.out_dir}/tables/*.md and {args.out_dir}/figures/failure_modes.png")


def _run_annotate_command(args: argparse.Namespace) -> None:
    from cra.eval.annotate import sample_for_annotation, write_annotation_template
    from cra.eval.run import build_records
    from cra.trace_io import read_trace_dir

    entailment = _build_nli_checker() if args.nli else None
    trace_dirs = [f"results/traces/{e}" for e in args.experiments]
    traces = [t for d in trace_dirs for t in read_trace_dir(d)]
    records = build_records(traces, entailment=entailment)

    sample = sample_for_annotation(records, n=args.sample)
    csv_path, transcript_path = write_annotation_template(sample, trace_dirs, args.out)
    print(f"Wrote {len(sample)} trace(s) to annotate:")
    print(f"  {csv_path}  (fill in the human_label column)")
    print(f"  {transcript_path}  (the full transcript for each trace_id)")
    print(
        "\nValid labels: no_answer, retrieval_failure, tool_misuse, reasoning_failure, "
        "unsupported_claim, correct_grounded. Judge each trace against the rubric in "
        "src/cra/eval/entailment/rubric.md and the taxonomy in CLAUDE.md before looking at "
        "anything the classifier said -- that's what makes 'blind' mean something.\n"
        "When done: cra annotate-score --experiments ... --annotations " + str(csv_path)
    )


def _run_annotate_score_command(args: argparse.Namespace) -> None:
    import csv as csv_module

    from cra.eval.annotate import read_annotations, score_annotations, validate_annotations
    from cra.eval.run import build_records
    from cra.trace_io import read_trace_dir

    # Catch a label contradicting its own row before spending a grading pass on it.
    problems = validate_annotations(args.annotations)
    if problems:
        print(f"{len(problems)} annotation problem(s) -- fix these before scoring:")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)

    entailment = _build_nli_checker() if args.nli else None
    trace_dirs = [f"results/traces/{e}" for e in args.experiments]
    traces = [t for d in trace_dirs for t in read_trace_dir(d)]
    records = build_records(traces, entailment=entailment)

    with open(args.annotations, newline="", encoding="utf-8") as fh:
        sampled_trace_ids = {row["trace_id"] for row in csv_module.DictReader(fh)}
    sampled_records = [r for r in records if r.trace_id in sampled_trace_ids]

    annotations = read_annotations(args.annotations)
    score = score_annotations(sampled_records, annotations)

    print(
        f"Sampled: {len(sampled_trace_ids)}, human-labeled: {len(annotations)}, "
        f"still blank: {score.n_missing}, no automatic prediction yet: {score.n_ungraded_skipped}"
    )
    if score.agreement is None:
        print("Not enough gradable pairs yet to compute kappa (need >=2 with a real automatic prediction).")
    else:
        a = score.agreement
        print(
            f"Cohen's kappa = {a.kappa:.3f} over n={a.n} scored item(s) "
            f"(observed agreement {a.observed_agreement:.1%})"
        )
        print(f"Labels seen: {a.labels}")
        if score.n_ungraded_skipped:
            print(
                f"({score.n_ungraded_skipped} human label(s) saved but not scored -- their "
                "classifier prediction doesn't exist yet. Re-run with --nli, or after "
                "`cra judge-check`, to include them.)"
            )


def _run_judge_check_command(args: argparse.Namespace) -> None:
    from cra.config import load_experiment
    from cra.eval.run import cross_judge_traces
    from cra.llm.factory import build_llm
    from cra.trace_io import read_trace_dir

    traces_by_model: dict[str, list] = {}
    llms_by_model: dict[str, object] = {}
    for exp in args.experiments:
        cfg = load_experiment(exp)
        traces = read_trace_dir(f"results/traces/{exp}")
        traces_by_model.setdefault(cfg.model.model_id, []).extend(traces)
        llms_by_model.setdefault(cfg.model.model_id, build_llm(cfg.model))

    records = cross_judge_traces(traces_by_model, llms_by_model, sample_n=args.sample)

    from collections import Counter

    print(f"Judged {len(records)} trace(s) (each by a *different* model than the one that produced it):")
    for model_id in sorted(traces_by_model):
        subset = [r for r in records if r.model_id == model_id]
        if not subset:
            continue
        n_hallucinated = sum(1 for r in subset if r.hallucinated)
        counts = Counter(r.failure_mode for r in subset)
        print(f"  {model_id}: n={len(subset)}, hallucination_rate={n_hallucinated / len(subset):.1%}, "
              f"labels={dict(counts)}")

    out_path = args.out
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["| Model | N | Hallucination rate |", "|---|---|---|"]
    for model_id in sorted(traces_by_model):
        subset = [r for r in records if r.model_id == model_id]
        if not subset:
            continue
        rate = sum(1 for r in subset if r.hallucinated) / len(subset)
        lines.append(f"| {model_id} | {len(subset)} | {rate:.1%} |")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
