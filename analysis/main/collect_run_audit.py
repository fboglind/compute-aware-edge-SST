"""Extract numeric diagnostics from existing main-experiment logs. No inference.

Optional reconstruction of data/run-audit.json:
python collect_run_audit.py --scores /path/evaluation-main-fixed/scores.json \
    --runs /path/runs --out data/run-audit.json
The supplied snapshot already contains the result for all 96 completed runs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(scores_path, runs_root):
    scores = read(scores_path)
    result = {"scores_sha256": sha(scores_path), "runs": [], "sessions": {}}
    fields = ["chunk_index", "source_start_s", "source_end_s", "chunk_duration_s",
              "processing_start_s", "emitted_s", "process_s", "asr_s", "mt_s",
              "queue_wait_s", "lag_after_chunk_end_s", "mt_tokens", "mt_limit_reached"]
    for run in scores["runs"]:
        path = runs_root / run["device"] / "jobs" / run["id"] / "attempt-001"
        chunks = [json.loads(line) for line in (path / "chunks.jsonl").read_text().splitlines()]
        metadata = read(path / "metadata.json")
        validation_path = scores_path.parent / "results" / run["device"] / run["id"] / "validation.json"
        validation = read(validation_path)
        assert validation["timestamp_roundtrip"] == "passed"
        assert metadata["effective_compute_type"] == {"asr": "int8_float32", "mt": "int8_float32"}
        assert abs(sum(c["process_s"] for c in chunks) - run["total_processing_s"]) < 1e-6
        assert abs(max(c["queue_wait_s"] for c in chunks) - run["queue_wait_max_s"]) < 1e-6
        result["runs"].append({
            "key": run["key"], "metadata": {k: metadata[k] for k in
                ("platform", "machine", "python", "models", "decoding", "policy", "emission",
                 "warmup", "model_load_s", "warmup_s", "effective_compute_type", "packages")},
            "source_sha256": {name: sha(path / name) for name in
                ("chunks.jsonl", "metadata.json", "resources.jsonl", "summary.json")},
            "validation": {"timestamp_roundtrip": validation["timestamp_roundtrip"],
                           "empty_timestamp_normalizations": validation.get("empty_timestamp_normalizations", [])},
            "chunks": [{**{k: c[k] for k in fields}, "asr_words": len(c["english"].split()),
                        "mt_words": len(c["german"].split())} for c in chunks],
            "long_output_examples": [{"chunk_index": c["chunk_index"], "english": c["english"],
                                      "german": c["german"]} for c in chunks
                                     if len(c["english"].split()) >= 100 or c["mt_limit_reached"]],
        })
    for device in ("pi", "laptop"):
        result["sessions"][device] = [read(p) for p in sorted((runs_root / device / "sessions").glob("*.json"))]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(collect(args.scores, args.runs), ensure_ascii=False, indent=2) + "\n")
