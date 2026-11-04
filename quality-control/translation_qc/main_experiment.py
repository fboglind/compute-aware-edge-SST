"""Read-only adapter for the frozen, six-talk main experiment.

This module prepares MT-fidelity requests. It performs no semantic assessment.
"""

from pathlib import Path

from .common import (digest, file_record, fingerprint, now, read_json, require,
                     write_csv, write_json, write_jsonl)
from .inputs import LABEL_FIELDS, build_windows, load_run

MAIN_AUDIO = {
    "ERmKpJPPDc": "9e38e8df0c16f430fa8c35b9274e2de86cf2f49bf369709f69e7f6a99d155a2e",
    "EqmWoxNDIr": "584f600cb98ed52df02f609eda50e49317dbbaa93ba610adaa4fc0179b9734fb",
    "UOlPKyCVgg": "52793303ef8e090ecc23bda7d03f57c1958206d56e614bc48f7339e998e29709",
    "krJSAnVcGR": "850ef7caf4b56b8f902508e0a17539ea43d62bd89edb501d2cf16597c602c290",
    "rxrToXvRyM": "9ccc27917580959598c023fe5da4f5f23157696005c685445b15324e3cb3010b",
    "xhjBHGVOyQ": "f0ee15e29c39840aae75eb6e02b7b7eb801cfe9587e9ac4a0f733ddfc13eeab0",
}
MAIN_SEGMENTATION = "common-24s-core-plus-8s-context-v1"


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Input path escapes its root")
    return path


def select_paired(windows, per_talk=2, seed=20261022):
    """Hash-rank passage IDs once per talk; the selection ignores output text."""
    require(type(per_talk) is int and per_talk > 0, "Passages per talk must be positive")
    talks = sorted({w["talk_id"] for w in windows})
    records = []
    for talk in talks:
        group = [w for w in windows if w["talk_id"] == talk]
        passages = sorted({w["paired_passage_id"] for w in group})
        counts = {p: sum(w["paired_passage_id"] == p for w in group) for p in passages}
        require(set(counts.values()) == {8}, "Each main passage must contain all eight configurations")
        require(per_talk <= len(passages), "Requested sample exceeds passage inventory")
        ranked = sorted(passages, key=lambda p: fingerprint({"seed": seed, "passage_id": p}))
        selected = set(ranked[:per_talk])
        records.append({"talk_id": talk, "population": len(passages), "selected_passage_ids": sorted(selected)})
        for window in group:
            window["review_selected"] = window["paired_passage_id"] in selected
            window["selection"] = {"method": "lowest SHA256 of canonical seed/passage_id; same passages for all configs",
                                   "seed": seed, "passages_per_talk": per_talk}
    return records


def prepare_main(runs_dir, evaluation_dir, out, per_talk=2, seed=20261022):
    require(not out.exists(), f"Output already exists: {out}; use a new QC directory")
    for root in (runs_dir, evaluation_dir):
        require(not out.resolve().is_relative_to(root.resolve()), "QC output must be separate from inputs")
    scores_path = evaluation_dir / "scores.json"
    scores = read_json(scores_path)
    require(scores["complete_grid"] is True and len(scores["runs"]) == 96, "Require the complete 96-run main evaluation")
    data = scores["manifest"]
    talks = {t["id"]: t for t in data["talks"]}
    require(len(data["talks"]) == 6 and set(talks) == set(MAIN_AUDIO), "Unexpected main talk set")
    require(data["reference_segments"] == 266, "Unexpected main reference count")
    expected = {(d, t, a, c, r) for d in ("pi", "laptop") for t in talks
                for a in ("tiny.en", "base.en") for c in (4, 8) for r in (1, 2)}
    indexed = {(r["device"], r["talk"], r["asr"], r["chunk_s"], r["repetition"]): r for r in scores["runs"]}
    require(len(indexed) == 96 and set(indexed) == expected, "Main job grid is missing/duplicated/mislabeled")
    references = {"alignment": "whole_talk_unaligned", "talks": {},
                  "use": "Additional evidence; never substituted for the actual MT input"}
    reference_inputs = [file_record(scores_path)]
    for talk_id, talk in sorted(talks.items()):
        require(talk["audio_sha256"] == MAIN_AUDIO[talk_id], "Unexpected main audio fingerprint")
        evidence = {"talk_id": talk_id, "origin": "official MCIF frozen main references",
                    "alignment": "whole_talk_unaligned"}
        for name in ("en.txt", "de.txt", "audio-segments.yaml"):
            path = evaluation_dir / "references" / talk_id / name
            expected_hash = data["files_sha256"][talk["reference_dir"] + "/" + name]
            require(digest(path) == expected_hash, f"Main reference hash mismatch: {talk_id}/{name}")
            reference_inputs.append(file_record(path))
            evidence[name.split(".")[0]] = path.read_text(encoding="utf-8")
        require(len(evidence["en"].splitlines()) == len(evidence["de"].splitlines())
                == talk["reference_segments"], "Reference line count mismatch")
        references["talks"][talk_id] = evidence

    loaded, run_records = {}, []
    for identity, row in sorted(indexed.items()):
        device, talk_id, asr, chunk_s, repetition = identity
        job_dir = inside(runs_dir, f"{device}/jobs/{row['id']}")
        complete_path = job_dir / "complete.json"
        complete = read_json(complete_path)
        require(all(row[k] == v for k, v in complete["job"].items()), "Completion/job metadata mismatch")
        attempt = inside(job_dir, complete["attempt"])
        input_records = [file_record(complete_path)]
        for name, expected_hash in complete["output_sha256"].items():
            path = inside(attempt, name)
            require(digest(path) == expected_hash, f"Completed input changed: {row['key']}/{name}")
            input_records.append(file_record(path))
        for name, expected_hash in row["input_sha256"].items():
            require(digest(inside(attempt, name)) == expected_hash, f"Scored input changed: {row['key']}/{name}")
        segment_path = inside(evaluation_dir, row["segments_file"])
        require(digest(segment_path) == row["segments_sha256"], "Scored segments changed")
        reference_inputs.append(file_record(segment_path))
        segments = read_json(segment_path)
        require([s["src"] for s in segments]
                == [s.strip() for s in references["talks"][talk_id]["en"].splitlines()],
                "Scored source/reference mismatch")
        talk = talks[talk_id]
        contract = {"audio_duration_s": talk["duration_s"],
                    "prepared_files": {f"{talk_id}.wav": {"sha256": MAIN_AUDIO[talk_id]}}}
        chunks, provenance = load_run(attempt, asr, chunk_s, contract,
                                      talk_id=talk_id, audio_sha256=MAIN_AUDIO[talk_id])
        provenance.update(public_main=True, dataset="MCIF-main-six-talks-2026-10-18",
                          run_key=row["key"], inputs=input_records)
        loaded[identity] = (chunks, provenance)
        run_records.append({"run_id": row["key"], "repetition": repetition, "chunks": len(chunks), **provenance})

    windows, checks, repeat_checks = [], [], []
    for device, talk_id, asr, chunk_s in sorted({i[:4] for i in indexed}):
        first, provenance = loaded[(device, talk_id, asr, chunk_s, 1)]
        second, _ = loaded[(device, talk_id, asr, chunk_s, 2)]
        signature = lambda chunks: [(c["source_start_s"], c["source_end_s"], c["english"], c["german"]) for c in chunks]
        require(signature(first) == signature(second), "Repetitions differ: do not collapse these quality outputs")
        repeat_checks.append({"device": device, "talk_id": talk_id, "asr": asr, "chunk_s": chunk_s,
                              "identical_chunk_text_and_schedule": True, "retained_repetition": 1})
        run_id = indexed[(device, talk_id, asr, chunk_s, 1)]["key"]
        group, group_checks = build_windows(first, run_id, asr, chunk_s, provenance, 24,
                                            talk_id=talk_id, device=device, context_s=8)
        for window, check in zip(group, group_checks):
            window["segmentation"]["version"] = MAIN_SEGMENTATION
            window["paired_passage_id"] = f"{talk_id}/w{window['ordinal']:03d}/{window['source_start_s']:.6f}-{window['source_end_s']:.6f}"
            window["device"] = device
            window["diagnostic_selected"] = any(w["code"] in ("mt_output_limit", "nonempty_en_zero_de")
                                                 for w in check["warnings"])
            # Content identity excludes machine-specific input locations.
            identity = {k: v for k, v in window.items() if k not in ("content_fingerprint", "provenance")}
            identity["input_hashes"] = sorted(r["sha256"] for r in provenance["inputs"])
            window["content_fingerprint"] = fingerprint(identity)
        windows.extend(group)
        checks.extend(group_checks)
    sample = select_paired(windows, per_talk, seed)
    by_passage = {}
    for window in windows:
        by_passage.setdefault(window["paired_passage_id"], []).append(window)
    for group in by_passage.values():
        require(len(group) == 8 and len({(w["source_start_s"], w["source_end_s"]) for w in group}) == 1,
                "Common-time pairing failed")
    # This is a text audit, not a new quality score or semantic label.
    out.mkdir(parents=True)
    write_jsonl(out / "windows.jsonl", windows)
    write_jsonl(out / "deterministic.jsonl", checks)
    write_jsonl(out / "jev_decisions.jsonl", [])
    write_json(out / "reference_evidence.json", references)
    write_json(out / "sample_manifest.json", {"seed": seed, "selection": sample,
               "per_talk": per_talk, "unit": "shared source-time passage", "configuration_outputs_per_passage": 8})
    write_json(out / "repeat_audit.json", repeat_checks)
    write_csv(out / "human_labels.csv", LABEL_FIELDS,
              [{"window_id": w["window_id"]} for w in windows if w["review_selected"]])
    artifacts = ("windows.jsonl", "deterministic.jsonl", "reference_evidence.json",
                 "sample_manifest.json", "repeat_audit.json")
    write_json(out / "manifest.json", {
        "schema_version": 1, "created_utc": now(), "dataset": "main", "talk_ids": sorted(talks),
        "scope": "Six-talk main experiment; one text-identical repetition retained per device/configuration; MT fidelity relative to ASR",
        "time_paired": True, "runs": run_records, "audited_runs": 96, "retained_runs": 48,
        "reference_inputs": reference_inputs, "window_count": len(windows),
        "selected_count": sum(w["review_selected"] for w in windows),
        "diagnostic_count": sum(w["diagnostic_selected"] for w in windows),
        "independent_talks": 6, "segmentation_version": MAIN_SEGMENTATION,
        "artifacts": {name: digest(out / name) for name in artifacts},
        "qc_code_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "human_validation": "No expert German labels available; supported author review is not gold-standard validation",
    })
    return windows, checks
