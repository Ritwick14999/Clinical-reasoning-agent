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
    p_annotate.add_argument(
        "--exclude", default=None,
        help="path to an existing annotation CSV whose trace_ids to leave out, so a later "
        "pass is genuinely held out from the labels that informed the taxonomy",
    )
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

    p_cal = sub.add_parser(
        "calibrate",
        help="sensitivity of the failure-mode labels to the entailment threshold, "
        "fitted on one annotation pass and reported on a disjoint one",
    )
    p_cal.add_argument("--experiments", required=True, nargs="+")
    p_cal.add_argument("--dev", required=True, help="annotation CSV to fit on")
    p_cal.add_argument("--heldout", required=True, help="disjoint annotation CSV to report on")
    p_cal.add_argument("--out", default="results/tables/calibration.md")

    p_abl = sub.add_parser(
        "ablation", help="paired comparison of an ablation arm against its baseline"
    )
    p_abl.add_argument("--baseline", required=True, help="baseline experiment_id")
    p_abl.add_argument(
        "--variants", required=True, nargs="+", help="one or more variant experiment_id(s)"
    )
    p_abl.add_argument("--out", default="results/tables/ablations.md")
    p_abl.add_argument(
        "--no-borrow",
        action="store_true",
        help="do not score an evidence-less arm against the baseline's evidence. Without "
        "borrowing, a closed-book arm is 100%% unsupported by construction, which measures "
        "the ablation rather than the agent.",
    )

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
    elif args.command == "calibrate":
        _run_calibrate_command(args)
    elif args.command == "ablation":
        _run_ablation_command(args)
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

    excluded: set[str] = set()
    if args.exclude:
        import csv as _csv

        with open(args.exclude, newline="", encoding="utf-8") as fh:
            excluded = {row["trace_id"] for row in _csv.DictReader(fh)}
        print(f"Excluding {len(excluded)} already-annotated trace(s) from {args.exclude}")
    sample = sample_for_annotation(records, n=args.sample, exclude_trace_ids=excluded)
    csv_path, transcript_path = write_annotation_template(sample, trace_dirs, args.out)
    print(f"Wrote {len(sample)} trace(s) to annotate:")
    print(f"  {csv_path}  (fill in the human_label column)")
    print(f"  {transcript_path}  (the full transcript for each trace_id)")
    print(
        "\nValid labels: no_answer, retrieval_failure, tool_misuse, reasoning_failure, "
        "unsupported_claim, correct_grounded. Judge each trace against the rubric in "
        "src/cra/eval/entailment/rubric.md and the taxonomy in docs/CONVENTIONS.md before looking at "
        "anything the classifier said -- that's what makes 'blind' mean something.\n"
        "When done: cra annotate-score --experiments ... --annotations " + str(csv_path)
    )


def _run_calibrate_command(args: argparse.Namespace) -> None:
    import csv as _csv
    from pathlib import Path

    from cra.eval.calibrate import calibrate, render_report, score_traces
    from cra.trace_io import read_trace_dir

    def _labels(path: str) -> dict[str, str]:
        with open(path, newline="", encoding="utf-8") as fh:
            return {
                row["trace_id"]: row["human_label"].strip()
                for row in _csv.DictReader(fh)
                if (row.get("human_label") or "").strip()
            }

    dev_human = _labels(args.dev)
    heldout_human = _labels(args.heldout)
    overlap = set(dev_human) & set(heldout_human)
    if overlap:
        # Fitting and reporting on the same traces would tune the classifier to
        # its own validation set, which is the whole reason this is split.
        raise SystemExit(
            f"{len(overlap)} trace(s) appear in both annotation sets; they must be disjoint "
            "for the reported figure to be out of sample."
        )

    checker = _build_nli_checker()
    traces = [t for e in args.experiments for t in read_trace_dir(f"results/traces/{e}")]
    dev_scored = score_traces(traces, checker, trace_ids=set(dev_human))
    heldout_scored = score_traces(traces, checker, trace_ids=set(heldout_human))
    checker.flush()

    result = calibrate(dev_scored, dev_human, heldout_scored, heldout_human)

    # How many labels the threshold can actually move decides whether the fitted
    # cut is identified at all, so it is reported rather than left implicit.
    flips = {
        name: sum(
            1 for s in scored if s.label_at(result.default_entail) != s.label_at(result.best_entail)
        )
        for name, scored in (("dev", dev_scored), ("held-out", heldout_scored))
    }
    sensitive = {
        name: sum(1 for s in scored if s.fixed_label is None)
        for name, scored in (("dev", dev_scored), ("held-out", heldout_scored))
    }

    report = render_report(result)
    identification = (
        f"\nIdentification: of {sensitive['dev']} threshold-sensitive dev traces "
        f"{flips['dev']} change label between the two cuts, and of "
        f"{sensitive['held-out']} held-out traces {flips['held-out']} do. The kappa "
        "difference therefore rests on a handful of traces, so the fitted cut is "
        "reported as a sensitivity analysis and NOT adopted as the default: the "
        "direction of the finding is robust to the threshold, its magnitude is not "
        "pinned down by this many annotations."
    )
    print(report)
    print(identification)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n" + identification + "\n", encoding="utf-8")
    print("")
    print(f"Wrote {out}")


def _run_ablation_command(args: argparse.Namespace) -> None:
    from pathlib import Path

    from cra.eval.ablation import compare, render_table
    from cra.trace_io import read_trace_dir

    entailment = _build_nli_checker()
    baseline = read_trace_dir(f"results/traces/{args.baseline}")
    if not baseline:
        raise SystemExit(f"no traces found for baseline {args.baseline!r}")

    comparisons = []
    for variant_id in args.variants:
        variant = read_trace_dir(f"results/traces/{variant_id}")
        if not variant:
            print(f"skipping {variant_id!r}: no traces found")
            continue
        result = compare(baseline, variant, entailment, borrow_evidence=not args.no_borrow)
        comparisons.append((variant_id, result))

        p = result.p_value
        # Never round a p-value into "0.0": a genuinely tiny value must stay
        # legible as tiny rather than be reported as exactly zero.
        p_str = "n/a" if p is None else (f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}")
        print("")
        print(f"{variant_id} vs {args.baseline}")
        print(f"  paired on {result.n_paired} question(s)")
        print(
            f"  accuracy {result.baseline.accuracy:.1%} -> {result.variant.accuracy:.1%} "
            f"({result.accuracy_delta:+.1%}), McNemar b={result.b} c={result.c} p={p_str}"
        )
        print(
            f"  traces with an unsupported claim "
            f"{result.baseline.hallucination_rate_traces:.1%} -> "
            f"{result.variant.hallucination_rate_traces:.1%} "
            f"({result.hallucination_delta:+.1%})"
        )
        print(
            f"  claims entailed {result.baseline.grounding_rate_claims:.1%} -> "
            f"{result.variant.grounding_rate_claims:.1%}"
        )
        for note in result.notes:
            print(f"  note: {note}")

    if comparisons:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_table(comparisons) + "\n", encoding="utf-8")
        print("")
        print(f"Wrote {out}")


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
