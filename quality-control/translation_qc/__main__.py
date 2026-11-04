"""Laptop-only translation QC; all commands default to offline operation."""

import argparse
import json
from pathlib import Path

from .common import require, write_json
from .inputs import prepare, validate
from .main_experiment import prepare_main
from .human import evaluate_labels
from .jev import MODEL, current_decisions, judge
from .explanations import export_prompts, import_annotations
from .mock_run import run_mock
from .reports import export_review, report


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Audit four pilot runs and export windows, sample and review files")
    preparation.add_argument("--runs-dir", type=Path, default=Path("pi-runs"))
    preparation.add_argument("--reference-dir", type=Path, default=Path("data/mcif-pilot-JhbtCwcsWY"))
    preparation.add_argument("--out", type=Path, required=True, help="New directory for QC artifacts")
    preparation.add_argument("--target-s", type=float, default=20)
    preparation.add_argument("--per-config", type=int, default=10)
    preparation.add_argument("--seed", type=int, default=20261018)
    preparation.add_argument("--context-s", type=int, choices=[0, 8], default=0)
    main_preparation = commands.add_parser("prepare-main", help="Audit 96 main runs; retain 48 identical-repeat text outputs; pair 24-second windows")
    main_preparation.add_argument("--runs-dir", type=Path, required=True)
    main_preparation.add_argument("--evaluation", type=Path, required=True, help="Completed evaluation-main-fixed directory")
    main_preparation.add_argument("--out", type=Path, required=True, help="New QC directory outside raw inputs")
    main_preparation.add_argument("--per-talk", type=int, default=2, help="Same number of seeded passages per talk; each has eight configurations")
    main_preparation.add_argument("--seed", type=int, default=20261022)
    validation = commands.add_parser("validate", help="Verify prepared artifact and original input fingerprints, offline")
    validation.add_argument("--input", type=Path, required=True)
    asr = commands.add_parser("asr-errors", help="Reproduce main WER as substitutions/deletions/insertions; requires jiwer==4.0.0")
    asr.add_argument("--input", type=Path, required=True)
    judging = commands.add_parser("judge", help="Prepare Jev requests offline; network requires --live-jev")
    judging.add_argument("--input", type=Path, required=True, help="Prepared windows.jsonl")
    judging.add_argument("--out", type=Path, required=True, help="Same prepared QC directory")
    judging.add_argument("--model", default=MODEL, choices=[MODEL])
    mode = judging.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: save requests/estimate, no network")
    mode.add_argument("--live-jev", action="store_true", help="Authorize external transfer of audited public MCIF text and paid Decisions calls")
    judging.add_argument("--limit", type=int, default=3, help="First N candidate windows (default 3; includes cached windows)")
    judging.add_argument("--full", action="store_true", help="Select from all windows instead of the seeded review subset")
    judging.add_argument("--selection", choices=["sample", "diagnostic", "all"], default="sample")
    judging.add_argument("--max-requests", type=int, default=3, help="Hard HTTP attempt cap INCLUDING retries, maximum 100")
    judging.add_argument("--max-cost-usd", type=float, default=0.01, help="Local planning/observed-cost stop; not a provider billing guarantee")
    judging.add_argument("--retries", type=int, default=2, help="0–2 retries on 429/5xx; timeouts are not retried")
    judging.add_argument("--timeout", type=float, default=30, help="Per HTTP operation timeout, at most 60 seconds")
    reporting = commands.add_parser("report", help="Regenerate review and threshold summaries from saved results, offline")
    reporting.add_argument("--input", type=Path, required=True)
    reporting.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    human = commands.add_parser("human-eval", help="Evaluate against manually completed labels, offline")
    human.add_argument("--input", type=Path, required=True)
    human.add_argument("--labels", type=Path, help="Default: INPUT/human_labels.csv")
    human.add_argument("--threshold", type=float, default=0.5)
    human.add_argument("--reviewer", help="Select one reviewer when labels contain multiple reviewers")
    explanations = commands.add_parser("explain", help="Export a small generative MQM prompt set; no network implementation")
    explanations.add_argument("--input", type=Path, required=True)
    explanations.add_argument("--model", required=True, help="Generative model identifier for exported payloads")
    explanations.add_argument("--provider", default="openrouter")
    explanations.add_argument("--max-flagged", type=int, default=5)
    explanations.add_argument("--unflagged", type=int, default=2, help="Seeded random controls; 1–5")
    explanations.add_argument("--seed", type=int, default=20261018)
    explanations.add_argument("--threshold", type=float, default=0.5)
    explanations.add_argument("--include-reference", action="store_true", help="Include clearly labelled unaligned whole-talk English evidence")
    importing = commands.add_parser("import-explanations", help="Validate offline LLM JSON/spans; preserve separately from human labels")
    importing.add_argument("--input", type=Path, required=True)
    importing.add_argument("--responses", type=Path, required=True)
    mock = commands.add_parser("mock", help="Offline injected-error demo with scripted answers, not a model evaluation")
    mock.add_argument("--out", type=Path, required=True)
    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "prepare":
            destination = arguments.out.resolve()
            for protected in (arguments.runs_dir, arguments.reference_dir, Path("models")):
                require(not destination.is_relative_to(protected.resolve()), "QC output must be separate from inputs/models")
            windows, checks = prepare(arguments.runs_dir, arguments.reference_dir, arguments.out,
                                      arguments.target_s, arguments.per_config, arguments.seed, arguments.context_s)
            export_review(arguments.out, windows, checks)
            print(json.dumps({"windows": len(windows), "selected": sum(window["review_selected"] for window in windows),
                              "out": str(arguments.out), "preview": [
                                  {key: window[key] for key in ("window_id", "chunk_ids", "source_en", "target_de")}
                                  for window in windows[:3]]}, ensure_ascii=False, indent=2))
        elif arguments.command == "prepare-main":
            windows, checks = prepare_main(arguments.runs_dir, arguments.evaluation, arguments.out,
                                          arguments.per_talk, arguments.seed)
            export_review(arguments.out, windows, checks)
            print(json.dumps({"audited_runs": 96, "retained_runs": 48, "windows": len(windows),
                              "selected": sum(w["review_selected"] for w in windows),
                              "diagnostic": sum(w["diagnostic_selected"] for w in windows),
                              "network_requests": 0, "out": str(arguments.out)}, indent=2))
        elif arguments.command == "validate":
            windows, checks, manifest = validate(arguments.input)
            decisions, stale = current_decisions(arguments.input, windows)
            result = {"status": "passed", "windows": len(windows), "selected": manifest["selected_count"],
                      "inputs_unchanged": True, "current_decisions": len(decisions), "ignored_stale_records": stale}
            write_json(arguments.input / "validation.json", result)
            print(json.dumps(result, indent=2))
        elif arguments.command == "asr-errors":
            from .asr_errors import analyze_asr
            print(json.dumps(analyze_asr(arguments.input), indent=2))
        elif arguments.command == "judge":
            result = judge(arguments.out, arguments.input, live=arguments.live_jev, full=arguments.full,
                           limit=arguments.limit, max_requests=arguments.max_requests, max_cost_usd=arguments.max_cost_usd,
                           model=arguments.model, retries=arguments.retries, timeout=arguments.timeout,
                           selection=arguments.selection)
            print(json.dumps(result, indent=2))
        elif arguments.command == "report":
            result = report(arguments.input, arguments.thresholds)
            print(json.dumps(result["counts"], indent=2))
        elif arguments.command == "human-eval":
            result = evaluate_labels(arguments.input, arguments.labels or arguments.input / "human_labels.csv",
                                     arguments.threshold, arguments.reviewer)
            print(json.dumps(result, indent=2))
        elif arguments.command == "explain":
            result = export_prompts(arguments.input, arguments.model, arguments.provider, arguments.max_flagged,
                                    arguments.unflagged, arguments.seed, arguments.threshold, arguments.include_reference)
            print(json.dumps(result, indent=2))
        elif arguments.command == "import-explanations":
            print(json.dumps(import_annotations(arguments.input, arguments.responses), indent=2))
        elif arguments.command == "mock":
            print(json.dumps(run_mock(arguments.out), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
