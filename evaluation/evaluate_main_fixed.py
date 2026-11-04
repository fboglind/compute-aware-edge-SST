#!/usr/bin/env python3
"""Offline scoring fix for the frozen TinyML kit dated 2026-10-18.

Save this file OUTSIDE the experiment directory (for example in Downloads).
Run it using .venv-evaluation/bin/python from the original experiment directory,
with the same --data/--runs arguments and a NEW --out directory.

No original source, run, plan, model, reference, or timestamp is overwritten.
The original integrity checks remain enabled. This adapter adds empty timestamp
arrays only when a resegmented prediction contains zero words. Missing times for
nonempty predictions remain errors. Empty translations remain in quality scores
and have unavailable, not zero, latency. Both this file's hash and the normalized
segment indices are recorded in the new scoring output.
"""
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

EXPECTED_EVALUATOR_SHA256 = "175c5c2ad2eb49de1c2e06d423e64a3e78431b1c8c5de78ad10d2a17545517b1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_empty_timestamps(rows):
    normalized, changes = [], []
    for i, original in enumerate(rows):
        row = dict(original)
        prediction = row.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"Resegmented row {i}: missing or invalid prediction")
        words = len(prediction.split())
        added = []
        for key in ("emission_cu", "emission_ca"):
            values = row.get(key)
            if values is None:
                if words:
                    raise ValueError(f"Resegmented row {i}: missing {key} for {words} output words; refusing to invent timestamps")
                row[key] = []
                added.append(key)
            elif not isinstance(values, list) or len(values) != words:
                raise ValueError(f"Resegmented row {i}: {key} count does not match output words")
            elif any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError(f"Resegmented row {i}: invalid numeric value in {key}")
        if added:
            changes.append({"row_0_based": i, "fields_set_to_empty_arrays": added})
        normalized.append(row)
    return normalized, changes


def load_evaluator(experiment_root):
    root = Path(experiment_root).resolve()
    source = root / "evaluate_main.py"
    if not source.is_file():
        raise ValueError("Run this command from the original TinyML-main-experiment-2026-10-18 directory")
    if Path(__file__).resolve().parent == root:
        raise ValueError("Move evaluate_main_fixed.py outside the experiment directory (e.g. to Downloads), then rerun it by its full path. The frozen plan checks all top-level Python files.")
    if digest(source) != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("evaluate_main.py differs from the original delivered kit. This adapter requires the unchanged original; do not change the recorded plan hashes.")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("tinyml_original_evaluator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.ROOT.resolve() != root:
        raise ValueError("Imported common.py belongs to a different experiment directory")

    read_original = module.read_jsonl
    write_original = module.write_json
    modifications = {}
    provenance = {"file": Path(__file__).name, "sha256": digest(__file__),
                  "original_evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
                  "change": "Missing/null CU/CA arrays become [] only for zero-word resegmented predictions.",
                  "raw_runs_modified": False, "frozen_plan_checks": "unchanged"}

    def read_fixed(path):
        rows = read_original(path)
        if Path(path).name == "instances.resegmented.jsonl":
            rows, changed = normalize_empty_timestamps(rows)
            modifications[str(Path(path).parent.resolve())] = changed
        return rows

    def write_fixed(path, value):
        path = Path(path)
        if path.name in ("validation.json", "scores.json", "scores.partial.json"):
            value = dict(value)
            value["evaluation_fix"] = provenance
            if path.name == "validation.json":
                value["empty_timestamp_normalizations"] = modifications.get(str(path.parent.resolve()), [])
        write_original(path, value)
        if path.name in ("scores.json", "scores.partial.json"):
            # Retain the exact adapter with the evaluation for reproduction.
            saved_source = path.parent / "evaluate_main_fixed.py"
            if saved_source.exists() and digest(saved_source) != provenance["sha256"]:
                raise ValueError("A different scoring adapter already exists in the output directory")
            if not saved_source.exists():
                saved_source.write_bytes(Path(__file__).read_bytes())

    module.read_jsonl = read_fixed
    module.write_json = write_fixed
    return module


def main():
    evaluator = load_evaluator(Path.cwd())
    print("Using the empty-segment scoring fix; all frozen-plan and raw-run checks remain enabled.", flush=True)
    evaluator.main()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}")
