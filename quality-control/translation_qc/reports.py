"""Readable review exports; statuses never imply proven adequacy."""

import html
from collections import Counter

from .common import canonical, finite_number, now, read_jsonl, require, write_csv, write_json
from .inputs import validate
from .jev import QUESTION_TEXT, current_decisions


def export_review(directory, windows, deterministic, decisions=None, threshold=0.5, annotations=None):
    decisions = decisions or {}
    annotations = annotations or {}
    checks = {row["window_id"]: row for row in deterministic}
    rows = []
    for window in windows:
        check = checks[window["window_id"]]
        decision = decisions.get(window["window_id"], {})
        probabilities = decision.get("probabilities", {}) if decision.get("status") == "ok" else {}
        flags = [name for name, score in probabilities.items() if score >= threshold]
        status = decision.get("status", "not_judged")
        semantic = ("candidate" if flags else "unflagged") if status == "ok" else status
        rows.append({
            "window_id": window["window_id"], "config": window["config"],
            "talk_id": window["talk_id"], "paired_passage_id": window.get("paired_passage_id", ""),
            "diagnostic_selected": window.get("diagnostic_selected", False),
            "selected": window["review_selected"], "source_start_s": window["source_start_s"],
            "source_end_s": window["source_end_s"], "chunk_ids": canonical(window["chunk_ids"]),
            "chunk_boundaries_s": canonical(window["chunk_boundaries_s"]),
            "source_en": window["source_en"], "target_de": window["target_de"],
            "context_before": canonical([{k: c[k] for k in ("chunk_index", "english", "german")}
                                          for c in window.get("context_before", [])]),
            "context_after": canonical([{k: c[k] for k in ("chunk_index", "english", "german")}
                                         for c in window.get("context_after", [])]),
            "structural_status": check["structural_status"], "semantic_status": semantic,
            "judgment_source": decision.get("mode", "none"),
            "review_status": "candidate" if flags or check["warnings"] else "unreviewed",
            "scores": canonical(probabilities), "suggested_categories": ";".join(flags),
            "deterministic_warnings": canonical(check["warnings"]),
            "boundary_ambiguity": check["boundary_ambiguity"], "origin": "uncertain",
            "human_issue": "", "human_notes": "",
            "llm_suggestions": canonical(annotations.get(window["window_id"], {}).get("annotation", {}).get("issues", [])),
        })
    rows.sort(key=lambda row: (row["review_status"] != "candidate", not row["selected"],
                               row["config"], row["source_start_s"]))
    if rows:
        write_csv(directory / "review.csv", tuple(rows[0]), rows)
    lines = ["# Translation fidelity review", "",
             "Exploratory MT fidelity relative to logged ASR. Candidate suggestions are not verified errors.",
             "Unflagged does not establish adequacy. Unjudged and failed requests have no semantic judgment.",
             "Main windows share source-time intervals; pilot pairing depends on the preparation settings. Origin remains uncertain.",
             "Whole-talk references are in reference_evidence.json; no sentence/window alignment is asserted.",
             "German judgments are model-based. Supported A2 author review is not expert German validation.",
             "Review.csv is regenerated. Any author annotations and API model labels must remain separate.", ""]
    for row in rows:
        lines.extend([f"## {row['window_id']}", "",
                      f"{row['config']} | {row['source_start_s']:.3f}–{row['source_end_s']:.3f} s | selected={row['selected']}",
                      f"Chunks: {row['chunk_ids']}; internal boundaries: {row['chunk_boundaries_s']}",
                      f"Structural: {row['structural_status']}; semantic: {row['semantic_status']}; review: {row['review_status']}; source: {row['judgment_source']}", "",
                      "English ASR / actual MT input:", "", f"<pre>{html.escape(row['source_en'])}</pre>", "",
                      "German emitted output:", "", f"<pre>{html.escape(row['target_de'])}</pre>", "",
                      f"Context before: <pre>{html.escape(row['context_before'])}</pre>",
                      f"Context after: <pre>{html.escape(row['context_after'])}</pre>", "",
                      f"Scores: `{row['scores']}`", f"Candidate categories: {row['suggested_categories'] or 'none'}",
                      f"Deterministic warnings: `{row['deterministic_warnings']}`", row["boundary_ambiguity"], "",
                      f"Separate LLM suggestions (not human labels): <pre>{html.escape(row['llm_suggestions'])}</pre>", "",
                      "Human issue (yes/no/uncertain): ___; category: ___; severity: ___; origin: ___; notes: ___", ""])
    (directory / "review.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def configuration_summary(windows, decisions, thresholds):
    success = [decisions[window["window_id"]] for window in windows
               if decisions.get(window["window_id"], {}).get("status") == "ok"]
    statuses = Counter(decisions.get(window["window_id"], {}).get("status", "not_judged") for window in windows)
    failed = sum(count for status, count in statuses.items() if status not in ("ok", "not_judged", "skipped_budget"))
    result = {"windows": len(windows), "review_selected": sum(window["review_selected"] for window in windows),
              "successful": len(success), "failed": failed, "not_judged": statuses["not_judged"],
              "skipped_budget": statuses["skipped_budget"], "statuses": dict(statuses),
              "processed_rate": ratio(len(success) + failed, len(windows)),
              "success_rate": ratio(len(success), len(windows)), "failure_rate": ratio(failed, len(windows)),
              "not_judged_rate": ratio(statuses["not_judged"], len(windows)),
              "skipped_budget_rate": ratio(statuses["skipped_budget"], len(windows)), "thresholds": {}}
    for threshold in thresholds:
        flags = sum(any(score >= threshold for score in row["probabilities"].values()) for row in success)
        result["thresholds"][str(threshold)] = {
            "candidate_review_rate": ratio(flags, len(success)),
            "suggested_categories": {name: ratio(sum(row["probabilities"][name] >= threshold for row in success), len(success))
                                     for name in QUESTION_TEXT},
        }
    return result


def usage_summary(directory):
    path = directory / "jev_attempts.jsonl"
    attempts = read_jsonl(path) if path.exists() else []
    require(len({attempt["attempt_id"] for attempt in attempts}) == len(attempts), "Duplicate HTTP attempt IDs")
    result = {"http_attempts": len(attempts), "fields": {}, "providers": {}, "resolved_models": {},
              "scope": "All recorded HTTP attempts, including retries, failures and previous prompt versions; cache hits are not billed again."}
    result["providers"] = dict(Counter(attempt.get("provider", "not_returned") for attempt in attempts))
    result["resolved_models"] = dict(Counter(attempt.get("model", "not_returned") for attempt in attempts))
    for field in ("input_tokens", "output_tokens", "cost"):
        values = []
        for attempt in attempts:
            usage = attempt.get("usage")
            value = usage.get(field) if isinstance(usage, dict) else None
            if finite_number(value) and value >= 0:
                values.append(value)
        result["fields"][field] = {"known_sum": sum(values) if values else None,
                                   "reported_attempts": len(values), "unknown_attempts": len(attempts) - len(values)}
    return result


def report(directory, thresholds=(0.5, 0.7, 0.9)):
    require(bool(thresholds) and all(finite_number(value) and 0 <= value <= 1 for value in thresholds),
            "Thresholds must be finite probabilities in [0, 1]")
    windows, deterministic, manifest = validate(directory)
    decisions, stale = current_decisions(directory, windows)
    annotations_path = directory / "llm_annotations.jsonl"
    annotations = {row["window_id"]: row for row in read_jsonl(annotations_path)} if annotations_path.exists() else {}
    export_review(directory, windows, deterministic, decisions, thresholds[0], annotations)
    groups = {}
    for window in windows:
        groups.setdefault(window["config"], []).append(window)
    summary = {
        "created_utc": now(), "scope": manifest.get("scope", "Development pilot; exploratory MT-fidelity diagnostics"),
        "label_source": "API model judgments; no expert German validation",
        "time_paired": manifest.get("time_paired", False),
        "selection": {"method": windows[0]["selection"] if windows else None,
                      "inventory": len(windows), "review_subset": manifest["selected_count"],
                      "judging": "Default judge processes first three selected windows; --limit and --full explicitly change this."},
        "counts": configuration_summary(windows, decisions, thresholds),
        "by_configuration": {config: configuration_summary(group, decisions, thresholds) for config, group in sorted(groups.items())},
        "review_subset": configuration_summary([window for window in windows if window["review_selected"]], decisions, thresholds),
        "sample_by_configuration": {config: configuration_summary([w for w in group if w["review_selected"]], decisions, thresholds)
                                    for config, group in sorted(groups.items())},
        "sample_by_talk": {talk: configuration_summary([w for w in windows if w["talk_id"] == talk and w["review_selected"]], decisions, thresholds)
                           for talk in sorted({w["talk_id"] for w in windows})},
        "diagnostic_subset": configuration_summary([w for w in windows if w.get("diagnostic_selected")], decisions, thresholds),
        "deterministic_candidate_windows": sum(bool(row["warnings"]) for row in deterministic),
        "ignored_stale_or_nonlive_records": stale, "provider_usage": usage_summary(directory),
        "llm_annotated_windows": len(annotations), "llm_label_source": "llm, never human ground truth",
        "limitations": ["Review rates are model suggestions, not measured translation error rates or calibrated probabilities.",
                        "ASR omissions absent from both texts can escape Jev; source is the actual MT input.",
                        "Reference evidence is whole-talk and unaligned; attribution remains uncertain.",
                        "Main windows are paired in source time, not guaranteed semantic alignment. No causal boundary effect is established.",
                        "API failures and unjudged windows are excluded from semantic-rate denominators.",
                        "Human labels and optional imported LLM explanations remain separate from Jev probabilities.",
                        "No significance claims; chrF, COMET and LongYAAL remain unchanged. QC API time is not inference latency."],
    }
    write_json(directory / "summary.json", summary)
    lines = ["# Translation QC summary", "", summary["scope"], "",
             f"Inventory: {len(windows)} windows; seeded review subset: {manifest['selected_count']}.",
             f"Jev successes: {summary['counts']['successful']}; failed: {summary['counts']['failed']}; "
             f"unjudged: {summary['counts']['not_judged']}; budget-skipped: {summary['counts']['skipped_budget']}.", "",
             "Model-based German judgments; no expert German fluency or adequacy validation.", "",
             "| Configuration | Inventory | Selected | Jev successful | Failed | Unjudged | Budget-skipped |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for config, result in summary["by_configuration"].items():
        lines.append(f"| {config} | {result['windows']} | {result['review_selected']} | {result['successful']} | "
                     f"{result['failed']} | {result['not_judged']} | {result['skipped_budget']} |")
    lines.extend(["", "Fixed-sample model flags (successful judgments only; not validated error rates):", "",
                  "| Configuration | Sample inventory | Judged | Omission flags | Any category flags |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for config, result in summary["sample_by_configuration"].items():
        t = result["thresholds"][str(thresholds[0])]
        lines.append(f"| {config} | {result['windows']} | {result['successful']} | "
                     f"{t['suggested_categories']['meaningful_omission']['numerator']} | {t['candidate_review_rate']['numerator']} |")
    for label, subset in (("Fixed sample", summary["review_subset"]), ("Targeted diagnostic subset", summary["diagnostic_subset"])):
        lines.extend(["", label + " threshold sensitivity:", ""])
        for threshold, result in subset["thresholds"].items():
            rate = result["candidate_review_rate"]
            lines.append(f"- Threshold {threshold}: {rate['numerator']}/{rate['denominator']}"
                         + (f" ({rate['value']:.1%})" if rate["value"] is not None else " (not available)"))
    lines.append("A passage can belong to both selections; these subsets must not be added together.")
    lines.extend(["", "Provider usage/cost (null means not returned, never zero-filled):", "", "```json",
                  canonical(summary["provider_usage"]), "```", "", "Human labels: human_labels.csv. Review: review.md / review.csv.",
                  "Optional LLM explanations: llm_annotations.jsonl, separate from human labels.", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
