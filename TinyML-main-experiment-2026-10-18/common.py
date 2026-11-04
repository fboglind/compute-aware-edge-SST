"""Shared integrity checks; no inference/evaluation dependencies."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRID = [("tiny.en", 4), ("tiny.en", 8), ("base.en", 4), ("base.en", 8)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def source_hashes():
    paths = list(ROOT.glob("*.py")) + list((ROOT / "runner").glob("*.py"))
    paths += list(ROOT.glob("requirements-*.txt"))
    paths += [ROOT / "selection.json", ROOT / "expected-models.json"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def dataset(data, with_audio=True):
    m = read_json(Path(data) / "manifest.json")
    require(m["manifest_id"] == fingerprint({k: v for k, v in m.items() if k != "manifest_id"}),
            "Manifest was modified after preparation")
    require(m["selection_sha256"] == sha(ROOT / "selection.json"), "Selection changed")
    require([t["id"] for t in m["talks"]] == [t["id"] for t in read_json(ROOT / "selection.json")["talks"]],
            "Manifest talk IDs differ from frozen selection")
    for rel, expected in m["files_sha256"].items():
        if with_audio or not rel.endswith(".wav"):
            require(sha(Path(data) / rel) == expected, f"Input hash differs: {rel}")
    return m


def jobs_for(m, device):
    require(device in ("pi", "laptop"), "Device must be pi or laptop")
    first = GRID if device == "pi" else GRID[2:] + GRID[:2]
    jobs = []
    for rep in (1, 2):
        order = first if rep == 1 else list(reversed(first))
        talks = m["talks"] if rep == 1 else list(reversed(m["talks"]))
        for asr, seconds in order:
            for talk in talks:
                jobs.append({"id": f"{talk['id']}-{asr[:-3]}-{seconds}s-r{rep}",
                             "talk": talk["id"], "asr": asr, "chunk_s": seconds,
                             "repetition": rep, "device": device, "order": len(jobs) + 1})
    return jobs


def plan_for(m, device):
    p = {"schema": 1, "device": device, "manifest_id": m["manifest_id"],
         "source_sha256": source_hashes(), "jobs": jobs_for(m, device),
         "settings": {"threads": 4, "asr_beam": 1, "mt_beam": 4,
                      "effective_compute_type": "int8_float32"}}
    p["plan_id"] = fingerprint(p)
    return p


def validate_plan(plan, m):
    require(plan == plan_for(m, plan["device"]),
            "Plan, manifest, or implementation changed. Preserve existing results; investigate before continuing.")


def check_models(models):
    expected = read_json(ROOT / "expected-models.json")
    for name, identity in expected.items():
        folder = Path(models) / name
        stored = read_json(folder / "pilot-model.json")
        require(all(stored[k] == identity[k] for k in identity), f"Model manifest changed: {name}")
        for name_in_model, digest in identity["files_sha256"].items():
            require(sha(folder / name_in_model) == digest, f"Model bytes differ: {folder / name_in_model}")
    return expected


def audit_run(run, talk, job):
    # Retain the pilot's complete chunk/word/export audit, adding main-set checks.
    from pilot_audit import audit
    manifest = {"audio_file": Path(talk["audio"]).name, "audio_duration_s": talk["duration_s"],
                "prepared_files": {Path(talk["audio"]).name: {"sha256": talk["audio_sha256"]}}}
    summary, meta, en, de, words = audit(Path(run), manifest, job["asr"], job["chunk_s"])
    expected_scripts = {n: sha(ROOT / "runner" / n) for n in ("run_pilot.py", "stream_timing.py")}
    require(meta["script_sha256"] == expected_scripts, "Runner differs from validated source")
    require(meta["effective_compute_type"] == {"asr": "int8_float32", "mt": "int8_float32"},
            "Effective precision differs from the protocol")
    expected = read_json(ROOT / "expected-models.json")
    for key, name in [("asr", "asr-" + job["asr"]), ("mt", "mt-en-de-int8")]:
        require(all(meta["models"][key][k] == v for k, v in expected[name].items()), "Model identity differs")
    require(meta["decoding"]["vad"] is False and meta["decoding"]["condition_on_previous_text"] is False
            and meta["decoding"]["temperature"] == 0 and meta["decoding"]["asr_task"] == "transcribe",
            "Decoding policy differs")
    chunks = read_jsonl(Path(run) / "chunks.jsonl")
    for field, total in [("asr_s", "total_asr_s"), ("mt_s", "total_mt_s")]:
        require(all(math.isfinite(c[field]) and c[field] >= 0 for c in chunks), "Invalid stage timing")
        require(abs(sum(c[field] for c in chunks) - summary[total]) < 1e-5, "Stage totals disagree")
    return summary, meta, en, de, words


def completed_attempt(jobdir, plan, job, talk):
    marker = Path(jobdir) / "complete.json"
    if not marker.exists():
        return None
    c = read_json(marker)
    require(c["plan_id"] == plan["plan_id"] and c["job"] == job, "Completion record belongs to a different plan/job")
    require(Path(c["attempt"]).name == c["attempt"], "Invalid attempt path")
    path = Path(jobdir) / c["attempt"]
    for name, digest in c["output_sha256"].items():
        require(sha(path / name) == digest, f"Completed output changed: {path / name}")
    audit_run(path, talk, job)
    return path
