"""Evaluate automated screening only against explicit human labels."""

import csv
import json

from .common import finite_number, now, require, write_json
from .inputs import validate
from .jev import current_decisions

CATEGORIES = {"omission", "mistranslation", "addition", "entity_or_terminology", "fluency", "incomplete_output"}
SEVERITIES = {"minor", "major", "critical"}
ORIGINS = {"asr", "mt", "streaming_boundary", "uncertain"}


def confusion(labels, decisions, truth_field, threshold):
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    excluded = {"unlabelled": 0, "uncertain": 0, "no_successful_judgment": 0}
    for row in labels:
        truth = row[truth_field]
        if not truth:
            excluded["unlabelled"] += 1
            continue
        if truth == "uncertain":
            excluded["uncertain"] += 1
            continue
        decision = decisions.get(row["window_id"], {})
        if decision.get("status") != "ok":
            excluded["no_successful_judgment"] += 1
            continue
        scores = decision["probabilities"]
        predicted = (scores["meaningful_omission"] >= threshold if truth_field == "omission"
                     else any(score >= threshold for score in scores.values()))
        counts[("t" if predicted == (truth == "yes") else "f") + ("p" if predicted else "n")] += 1
    true_positive, false_positive, false_negative = counts["tp"], counts["fp"], counts["fn"]
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    denominator = 2 * true_positive + false_positive + false_negative
    return {"confusion_matrix": counts, "evaluated": sum(counts.values()), "excluded": excluded,
            "precision": precision, "recall": recall,
            "f1": 2 * true_positive / denominator if denominator else None}


def evaluate_labels(directory, labels_path, threshold=0.5, reviewer=None):
    require(finite_number(threshold) and 0 <= threshold <= 1, "Threshold must be in [0, 1]")
    windows, checks, manifest = validate(directory)
    known_ids = {window["window_id"] for window in windows}
    with labels_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"window_id", "reviewer", "issue", "error_category", "severity", "possible_origin", "omission"}
        require(required <= set(reader.fieldnames or []), "Human label CSV is missing required columns")
        labels = list(reader)
    for row in labels:
        require(None not in row and all(value is not None for value in row.values()), "Malformed CSV row")
        require(row["window_id"] in known_ids, "Unknown human-label window_id")
        require(row["issue"] in ("", "yes", "no", "uncertain") and row["omission"] in ("", "yes", "no", "uncertain"),
                "Labels must be blank, yes, no or uncertain")
        categories = set(filter(None, row["error_category"].split(";")))
        require(categories <= CATEGORIES, "Unknown error category")
        require(row["severity"] in SEVERITIES | {""}, "Unknown severity")
        require(row["possible_origin"] in ORIGINS | {""}, "Unknown possible origin")
        require(not (row["issue"] == "no" and (categories or row["omission"] == "yes")), "Contradictory no-error label")
        require(not (row["omission"] == "no" and "omission" in categories), "Contradictory omission label")
    reviewers = {row["reviewer"] for row in labels if row["issue"] or row["omission"]}
    require(reviewer is not None or len(reviewers) <= 1, "Multiple reviewers: select --reviewer or supply adjudicated labels")
    if reviewer is not None:
        labels = [row for row in labels if row["reviewer"] == reviewer]
    require(len({row["window_id"] for row in labels}) == len(labels), "Duplicate labels for a window; adjudicate before scoring")
    decisions, stale = current_decisions(directory, windows)
    result = {"created_utc": now(), "label_source": "human", "labels_path": str(labels_path.resolve()),
              "reviewer": reviewer, "threshold": threshold, "csv_rows": len(labels),
              "annotated_instances": sum(bool(row["issue"] or row["omission"]) for row in labels),
              "scope": "Manually labelled development-pilot instances only; uncertain excluded, not treated as negative",
              "issue_detection": confusion(labels, decisions, "issue", threshold),
              "omission_detection": confusion(labels, decisions, "omission", threshold),
              "note": "Omission metrics use the explicit omission column. Blank omission labels are excluded, even if issue is labelled."}
    write_json(directory / "human_evaluation.json", result)
    (directory / "human_evaluation.md").write_text(
        "# Human-label evaluation\n\n" + result["scope"] + "\n\n```json\n"
        + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    return result
