"""Exercise requests/cache/reports offline using explicitly scripted answers."""

import json

from .common import append_jsonl, fingerprint, require, write_json, write_jsonl
from .fixtures import CASES
from .inputs import build_windows, select_review
from .jev import Budget, MODEL, QUESTION_TEXT, build_request, evaluate_request
from .reports import configuration_summary, export_review


def run_mock(out):
    require(not out.exists(), "Use a new directory for the synthetic mock run")
    windows, checks = [], []
    for case in CASES:
        chunk = {"chunk_index": 0, "source_start_s": 0, "source_end_s": 20, "chunk_duration_s": 20,
                 "processing_start_s": 20, "emitted_s": 20.1, "english": case["source_en"],
                 "german": case["target_de"], "words": case["target_de"].split()}
        group, findings = build_windows([chunk], "synthetic-" + case["name"], "synthetic", 20, {"public_pilot": False})
        group[0]["talk_id"] = "synthetic-injected-fixture"
        group[0]["window_id"] = "synthetic/" + case["name"]
        group[0]["content_fingerprint"] = fingerprint({key: value for key, value in group[0].items() if key != "content_fingerprint"})
        findings[0]["window_id"] = group[0]["window_id"]
        windows.extend(group)
        checks.extend(findings)
    select_review(windows)
    out.mkdir(parents=True)
    decisions = {}
    for case, window in zip(CASES, windows):
        raw = {"model": MODEL + "-20260917", "provider": "SYNTHETIC MOCK",
               "answers": {name: {"noul": 0.9 if name == case["injected_category"] else 0.1} for name in QUESTION_TEXT}}

        def transport(payload, api_key, timeout):
            return 200, json.dumps(raw), {}

        result = evaluate_request(build_request(window), out / "cache", Budget(1), "unused-mock-key", transport=transport)
        result["mode"] = "synthetic_mock"
        result["note"] = "Scripted answer, not Jev inference and not evidence of detection accuracy"
        decisions[window["window_id"]] = result
        append_jsonl(out / "mock_decisions.jsonl", result)
    write_jsonl(out / "synthetic_windows.jsonl", windows)
    export_review(out, windows, checks, decisions)
    review_path = out / "review.md"
    review_path.write_text("# SYNTHETIC MOCK — SCRIPTED ANSWERS, NO MODEL EVALUATION\n\n" + review_path.read_text(encoding="utf-8"), encoding="utf-8")
    summary = {"mode": "synthetic_mock", "network_requests": 0,
               "note": "All scores are scripted fixture values; no model accuracy has been measured.",
               "counts": configuration_summary(windows, decisions, (0.5, 0.95))}
    write_json(out / "summary.json", summary)
    return summary
