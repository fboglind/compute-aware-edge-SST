"""The inspected run_pilot chunk contract and whole-chunk windows."""

import math
import random
from pathlib import Path

from .common import (digest, file_record, finite_number, fingerprint, normalized,
                     now, read_json, read_jsonl, require, write_csv, write_json, write_jsonl)

TALK = "JhbtCwcsWY"
AUDIO_SHA256 = "eddaed93ed3d04531a18019c6ea262d7619d9b0549d9994bdb9cbc9612d1ddc9"
RUNNER_SHA256 = "86618d8f56f659ac63b9ad203956d3d403391a2f2f6e1705b2c44f0728cb65fa"
SEGMENTATION_VERSION = "whole-chunks-at-least-target-v1"
FILES = ("chunks.jsonl", "metadata.json", "summary.json", "transcript.en.txt",
         "translation.de.txt", "words.jsonl", "hypothesis.jsonl")
LABEL_FIELDS = ("window_id", "reviewer", "issue", "error_category", "severity",
                "source_evidence_span", "target_evidence_span", "possible_origin",
                "omission", "notes")


def close(left, right):
    return finite_number(left) and finite_number(right) and abs(left - right) <= 1e-5


def audit_chunks(chunks, chunk_s, duration):
    require(finite_number(chunk_s) and chunk_s > 0, "Invalid chunk schedule")
    require(finite_number(duration) and duration > 0, "Invalid audio duration")
    require(len(chunks) == math.ceil(duration / chunk_s), "Missing chunks or incorrect chunk count")
    previous_emission = 0
    for index, chunk in enumerate(chunks):
        required = ("chunk_index", "source_start_s", "source_end_s", "chunk_duration_s",
                    "emitted_s", "processing_start_s", "english", "german", "words")
        require(all(key in chunk for key in required), f"Chunk {index}: missing required fields")
        require(type(chunk["chunk_index"]) is int and chunk["chunk_index"] == index,
                f"Chunk {index}: missing, duplicate or out-of-order chunk ID")
        require(all(isinstance(chunk[key], str) for key in ("english", "german")),
                f"Chunk {index}: text must be a string")
        start, end = chunk["source_start_s"], chunk["source_end_s"]
        require(close(start, index * chunk_s) and close(end, min((index + 1) * chunk_s, duration)),
                f"Chunk {index}: interval gap, overlap or schedule mismatch")
        require(close(chunk["chunk_duration_s"], end - start), f"Chunk {index}: duration mismatch")
        processing, emission = chunk["processing_start_s"], chunk["emitted_s"]
        require(finite_number(processing) and finite_number(emission)
                and processing + 1e-5 >= max(end, previous_emission) and emission >= processing,
                f"Chunk {index}: noncausal or nonmonotonic emission")
        require(chunk["words"] == chunk["german"].split(), f"Chunk {index}: word/text mismatch")
        previous_emission = emission


def load_run(run, asr, chunk_s, reference_manifest, *, talk_id=TALK, audio_sha256=AUDIO_SHA256):
    try:
        metadata = read_json(run / "metadata.json")
        summary = read_json(run / "summary.json")
        chunks = read_jsonl(run / "chunks.jsonl")
        duration = reference_manifest["audio_duration_s"]
        require(summary["complete"] is True, "Incomplete run")
        require(metadata["args"]["asr"] == summary["asr_model"] == asr, "ASR config mismatch")
        require(metadata["args"]["chunk_s"] == summary["chunk_s"] == chunk_s, "Chunk config mismatch")
        require(metadata["args"]["threads"] == 4 and metadata["decoding"]["asr_beam"] == 1
                and metadata["decoding"]["mt_beam"] == 4, "Unexpected pilot decoding settings")
        require(metadata["audio_sha256"] == audio_sha256
                == reference_manifest["prepared_files"][f"{talk_id}.wav"]["sha256"], "Audio hash mismatch")
        require(close(summary["audio_duration_s"], duration)
                and close(metadata["audio_duration_s"], duration), "Audio duration mismatch")
        require(summary["chunks"] == len(chunks), "Summary chunk count mismatch")
        audit_chunks(chunks, chunk_s, duration)
        for key, filename in (("english", "transcript.en.txt"), ("german", "translation.de.txt")):
            require(normalized(" ".join(chunk[key] for chunk in chunks))
                    == normalized((run / filename).read_text(encoding="utf-8")), f"{filename}: concatenation mismatch")
        words = read_jsonl(run / "words.jsonl")
        hypotheses = read_jsonl(run / "hypothesis.jsonl")
        require(len(hypotheses) == 1, "Expected one full-talk hypothesis (not a resegmented file)")
        hypothesis = hypotheses[0]
        expected = [(word, chunk) for chunk in chunks for word in chunk["words"]]
        require(len(expected) == len(words) == summary["output_words"]
                == len(hypothesis["delays"]) == len(hypothesis["elapsed"]), "Word/export count mismatch")
        require(hypothesis["prediction"].split() == [word for word, chunk in expected], "Hypothesis text mismatch")
        require(hypothesis["source"] == f"{talk_id}.wav"
                and close(hypothesis["source_length"], duration * 1000), "Hypothesis source mismatch")
        for index, (event, (word, chunk)) in enumerate(zip(words, expected)):
            require(event["word_index"] == index and event["word"] == word
                    and event["chunk_index"] == chunk["chunk_index"], "Word ordering mismatch")
            require(close(event["source_consumed_s"], chunk["source_end_s"])
                    and close(event["emitted_s"], chunk["emitted_s"])
                    and close(hypothesis["delays"][index], chunk["source_end_s"] * 1000)
                    and close(hypothesis["elapsed"][index], chunk["emitted_s"] * 1000), "Word/export clock mismatch")
        require(metadata["script_sha256"]["run_pilot.py"] == RUNNER_SHA256,
                "Uninspected runner version: verify MT input handling before preparing these logs")
        provenance = {
            "inputs": [file_record(run / filename) for filename in FILES],
            "audio_sha256": metadata["audio_sha256"], "script_sha256": metadata["script_sha256"],
            "mt_input_contract": "Logged english passed unchanged to tokenizer.encode; no truncation; empty input skips MT.",
            "public_pilot": talk_id == TALK and audio_sha256 == AUDIO_SHA256,
        }
        return chunks, provenance
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{run}: {error}") from error


def chunk_checks(chunks):
    findings = {}
    seen = {"english": {}, "german": {}}
    for chunk in chunks:
        warnings = []
        for field, language in (("english", "en"), ("german", "de")):
            text = chunk[field]
            if not text.strip():
                warnings.append({"code": f"empty_{language}", "kind": "structural"})
            elif text in seen[field]:
                warnings.append({"code": f"exact_repeated_{language}", "kind": "candidate",
                                 "previous_chunk_id": seen[field][text]})
            else:
                seen[field][text] = chunk["chunk_index"]
        if chunk["english"].strip() and not chunk["german"].strip():
            warnings.append({"code": "nonempty_en_zero_de", "kind": "candidate"})
        if chunk.get("mt_limit_reached"):
            warnings.append({"code": "mt_output_limit", "kind": "candidate"})
        findings[chunk["chunk_index"]] = warnings
    return findings


def build_windows(chunks, run_id, asr, chunk_s, provenance, target_s=20, *,
                  talk_id=TALK, device=None, context_s=0):
    """Close each window at the first complete chunk reaching target_s; keep the tail."""
    require(finite_number(target_s) and 16 <= target_s <= 24, "Target must be between 16 and 24 seconds")
    require(context_s in (0, 8), "Context must be 0 or 8 seconds")
    groups, pending = [], []
    for chunk in chunks:
        pending.append(chunk)
        if chunk["source_end_s"] - pending[0]["source_start_s"] >= target_s:
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    findings = chunk_checks(chunks)
    windows, deterministic = [], []
    for ordinal, group in enumerate(groups):
        start, end = group[0]["source_start_s"], group[-1]["source_end_s"]
        window_id = f"{talk_id}/{run_id}/w{ordinal:03d}/{start:.6f}-{end:.6f}"
        window = {
            "window_id": window_id, "talk_id": talk_id, "run_id": run_id,
            "config": (f"{device}/" if device else "") + f"{asr}/{chunk_s:g}s", "ordinal": ordinal,
            "source_start_s": start, "source_end_s": end, "audio_span_s": end - start,
            "chunk_ids": [chunk["chunk_index"] for chunk in group],
            "chunk_boundaries_s": [chunk["source_end_s"] for chunk in group[:-1]],
            "source_en": " ".join(chunk["english"] for chunk in group),
            "target_de": " ".join(chunk["german"] for chunk in group),
            "chunks": group, "provenance": provenance,
            "segmentation": {"version": SEGMENTATION_VERSION, "target_s": target_s,
                             "context_s": context_s},
            "reference_evidence": {"file": "reference_evidence.json", "alignment": "whole_talk_unaligned",
                                   "note": "Additional human evidence only; no reference sentence assigned to any chunk/window."},
        }
        if context_s:
            window["context_before"] = [c for c in chunks if c["source_end_s"] <= start
                                        and c["source_end_s"] > start - context_s]
            window["context_after"] = [c for c in chunks if c["source_start_s"] >= end
                                       and c["source_start_s"] < end + context_s]
        window["content_fingerprint"] = fingerprint(window)
        warnings = [{"chunk_id": chunk["chunk_index"], **warning}
                    for chunk in group for warning in findings[chunk["chunk_index"]]]
        for field, language in (("source_en", "en"), ("target_de", "de")):
            if not window[field].strip():
                warnings.append({"code": f"empty_window_{language}", "kind": "structural"})
        deterministic.append({
            "window_id": window_id, "warnings": warnings,
            "chunk_results": [{"chunk_id": chunk["chunk_index"], "warnings": findings[chunk["chunk_index"]]}
                              for chunk in group],
            "structural_status": "warning" if warnings else "ok",
            "semantic_status": "not_judged", "review_status": "candidate" if warnings else "unreviewed",
            "boundary_ambiguity": "Window edges can split ideas; internal chunks must be read together. No causal attribution.",
            "origin": "uncertain",
        })
        windows.append(window)
    return windows, deterministic


def select_review(windows, per_config=10, seed=20261018):
    require(type(per_config) is int and per_config > 0, "Sample size must be positive")
    groups = {}
    for window in windows:
        groups.setdefault(window["run_id"], []).append(window)
    for run_id, group in sorted(groups.items()):
        generator = random.Random(f"{seed}:{run_id}")
        count = min(per_config, len(group))
        selected = {generator.randrange(index * len(group) // count, (index + 1) * len(group) // count)
                    for index in range(count)}
        for index, window in enumerate(group):
            window["review_selected"] = index in selected
            window["selection"] = {"method": "one seeded random window per equal ordinal stratum",
                                   "seed": seed, "requested_per_config": per_config}


def prepare(runs_dir, reference_dir, out, target_s=20, per_config=10, seed=20261018, context_s=0):
    require(not out.exists(), f"Output already exists: {out}; use a new QC directory")
    manifest_path = reference_dir / "manifest.json"
    reference_manifest = read_json(manifest_path)
    require(reference_manifest["talk_id"] == TALK, "Only the public MCIF pilot is supported")
    reference = {"talk_id": TALK, "alignment": "whole_talk_unaligned", "origin": "official MCIF pinned reference",
                 "inputs": [file_record(manifest_path)], "use": "Human adjudication only; never the primary Jev source"}
    for language in ("en", "de"):
        filename = reference_manifest["references"][language]["file"]
        path = reference_dir / filename
        require(digest(path) == reference_manifest["prepared_files"][filename]["sha256"], "Reference hash mismatch")
        reference[language] = path.read_text(encoding="utf-8")
        reference["inputs"].append(file_record(path))
    windows, deterministic, runs = [], [], []
    for label in ("tiny", "base"):
        for chunk_s in (4, 8):
            run = runs_dir / f"mcif-{TALK}-{label}-{chunk_s}s-mtbeam4-r1"
            chunks, provenance = load_run(run, label + ".en", chunk_s, reference_manifest)
            run_windows, checks = build_windows(chunks, run.name, label + ".en", chunk_s, provenance,
                                               target_s, context_s=context_s)
            windows.extend(run_windows)
            deterministic.extend(checks)
            runs.append({"run_id": run.name, "chunks": len(chunks), "windows": len(run_windows), **provenance})
    select_review(windows, per_config, seed)
    out.mkdir(parents=True)
    write_jsonl(out / "windows.jsonl", windows)
    write_jsonl(out / "deterministic.jsonl", deterministic)
    write_jsonl(out / "jev_decisions.jsonl", [])
    write_json(out / "reference_evidence.json", reference)
    write_csv(out / "human_labels.csv", LABEL_FIELDS,
              [{"window_id": window["window_id"]} for window in windows if window["review_selected"]])
    write_json(out / "manifest.json", {
        "schema_version": 1, "created_utc": now(), "talk_id": TALK, "runs": runs,
        "reference_inputs": reference["inputs"], "window_count": len(windows),
        "selected_count": sum(window["review_selected"] for window in windows),
        "segmentation_version": SEGMENTATION_VERSION,
        "scope": "Development pilot; MT fidelity relative to ASR, not original-audio fidelity",
        "time_paired": target_s == 24,
        "artifacts": {name: digest(out / name) for name in ("windows.jsonl", "deterministic.jsonl", "reference_evidence.json")},
        "qc_code_sha256": {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
    })
    return windows, deterministic


def validate(directory, check_inputs=True):
    manifest = read_json(directory / "manifest.json")
    require(manifest["schema_version"] == 1, "Unsupported QC manifest version")
    for name, expected in manifest["artifacts"].items():
        require(digest(directory / name) == expected, f"QC artifact changed: {name}")
    windows = read_jsonl(directory / "windows.jsonl")
    deterministic = read_jsonl(directory / "deterministic.jsonl")
    identifiers = [window["window_id"] for window in windows]
    require(len(set(identifiers)) == len(identifiers) == manifest["window_count"], "Window count/ID mismatch")
    require([row["window_id"] for row in deterministic] == identifiers, "Deterministic result IDs mismatch")
    require(sum(window["review_selected"] for window in windows) == manifest["selected_count"], "Selection mismatch")
    if check_inputs:
        records = manifest["reference_inputs"] + [record for run in manifest["runs"] for record in run["inputs"]]
        for record in records:
            require(digest(record["path"]) == record["sha256"], f"Input changed: {record['path']}")
    return windows, deterministic, manifest
