"""Shared integrity checks; no inference/evaluation dependencies."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRID = [("tiny.en", 1), ("tiny.en", 2), ("base.en", 1), ("base.en", 2)]


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
    paths += [ROOT / "selection.json", ROOT / "expected-models.json", ROOT / "PROTOCOL.md", ROOT / "source-provenance.json"]
    paths += list((ROOT / "model-manifests").glob("*.json"))
    paths += [ROOT.parent / "TinyML-main-experiment-2026-10-18" / "runner" / n for n in ("run_pilot.py", "stream_timing.py")]
    import os
    return {os.path.relpath(p, ROOT): sha(p) for p in sorted(paths)}


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
                jobs.append({"id": f"{talk['id']}-{asr[:-3]}-la2-{seconds}s-r{rep}",
                             "talk": talk["id"], "asr": asr, "min_chunk_s": seconds,
                             "repetition": rep, "device": device, "order": len(jobs) + 1})
    return jobs


def plan_for(m, device):
    p = {"schema": 1, "device": device, "manifest_id": m["manifest_id"],
         "source_sha256": source_hashes(), "jobs": jobs_for(m, device),
         "settings": {"threads": 4, "asr_beam": 1, "mt_beam": 4,
                      "effective_compute_type": "int8_float32", "buffer_trim_s": 15,
                      "policy": "LocalAgreement-2 English; translate each commit; labelled EOF tail",
                      "clocks": "consumed audio and elapsed on one CA-adaptive trajectory"}}
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
    from audit_events import validate, close
    from dataclasses import asdict
    from local_agreement import LocalAgreement2, Word
    run = Path(run)
    meta, summary, _ = validate(run)
    require((run / "updates.jsonl").exists(), "Expected adaptive update logs")
    require(meta["audio_sha256"] == talk["audio_sha256"], "Audio identity differs")
    require(close(meta["audio_duration_s"], talk["duration_s"]) and
            close(meta["audio_total_duration_s"], talk["duration_s"]) and meta["source_offset_s"] == 0,
            "Truncated or shifted recording")
    args = meta["args"]
    require(args["asr"] == job["asr"] and args["min_chunk_s"] == job["min_chunk_s"]
            and args["buffer_trim_s"] == 15 and args["threads"] == 4 and args["max_audio_s"] is None,
            "Runner settings differ")
    shared = ROOT.parent / "TinyML-main-experiment-2026-10-18" / "runner"
    paths = [ROOT / "run_localagreement.py", ROOT / "local_agreement.py",
             shared / "run_pilot.py", shared / "stream_timing.py"]
    require(meta["script_sha256"] == {str(p.relative_to(ROOT.parent)): sha(p) for p in paths},
            "Inference source differs")
    require(meta["effective_compute_type"] == {"asr": "int8_float32", "mt": "int8_float32"},
            "Effective precision differs")
    expected = read_json(ROOT / "expected-models.json")
    for key, name in [("asr", "asr-" + job["asr"]), ("mt", "mt-en-de-int8")]:
        require(all(meta["models"][key][k] == v for k, v in expected[name].items()), "Model identity differs")
    decoding = {"asr_beam": 1, "mt_beam": 4, "temperature": 0,
                "vad": False, "condition_on_previous_text": False, "word_timestamps": True,
                "without_timestamps": False, "prompt_max_words": 200, "asr_language": "en",
                "asr_task": "transcribe", "mt_max_output_tokens": 256, "mt_max_input_tokens": 512}
    require(meta["decoding"] == decoding, "Decoding policy differs")
    events = read_jsonl(run / "updates.jsonl")
    agreement = LocalAgreement2()
    for event in events:
        if event["event"] == "update":
            confirmed = agreement.update(Word(**w) for w in event["asr_hypothesis"])
            require([asdict(w) for w in confirmed] == event["confirmed_words"] and
                    [asdict(w) for w in agreement.pending] == event["pending_words"], "Agreement replay differs")
            ready = min(event["source_start_s"] + job["min_chunk_s"], talk["duration_s"])
            require(close(ready, event["update_ready_s"]), "Update readiness differs")
            require(event["source_end_s"] + 1/16000 >= ready, "Update before minimum arrival")
            require(close(event["update_wait_s"], max(0, event["processing_start_s"] - ready)), "Update wait differs")
        else:
            require([asdict(w) for w in agreement.finish()] == event["flushed_words"], "EOF tail differs")
        require(math.isfinite(event["policy_s"]) and event["policy_s"] >= 0, "Invalid policy timing")
        require(event["asr_s"] + event["mt_s"] + event["policy_s"] <= event["process_s"] + 1e-6,
                "Stage time exceeds processing")
    for field, total in [("asr_s", "total_asr_s"), ("mt_s", "total_mt_s"), ("policy_s", "total_policy_s")]:
        require(close(sum(e[field] for e in events), summary[total]), "Stage totals disagree")
    require(close(summary["final_tail_s"], events[-1]["emitted_s"] - talk["duration_s"]), "Final tail differs")
    require(summary["agreement_words"] == sum(len(e["confirmed_words"]) for e in events[:-1]) and
            summary["eof_unconfirmed_words"] == len(events[-1]["flushed_words"]), "Commit counts differ")
    for field, value in {
        "updates": len(events)-1,
        "output_words": len(read_jsonl(run / "words.jsonl")),
        "agreement_commits": sum(bool(e["english"]) for e in events[:-1]),
        "buffer_trims": sum(e["trim_at_s"] is not None for e in events[:-1]),
        "mt_limit_updates": sum(e["mt_limit_reached"] for e in events),
        "update_wait_max_s": max(e["update_wait_s"] for e in events[:-1]),
        "max_decoded_buffer_s": max(e["decoded_audio_s"] for e in events[:-1]),
        "decoded_audio_total_s": sum(e["decoded_audio_s"] for e in events[:-1]),
    }.items():
        require(close(summary[field], value), "Summary differs: " + field)
    en = (run / "transcript.en.txt").read_text().strip()
    de = (run / "translation.de.txt").read_text().strip()
    return summary, meta, en, de, read_jsonl(run / "words.jsonl")


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
