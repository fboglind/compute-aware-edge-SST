"""Preparation utilities for 01_prepare.ipynb. No inference is performed here."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path

import yaml

PILOT_ID = "JhbtCwcsWY"
CONFIGURATIONS = [
    {"id": f"{model}-{seconds}s", "asr": f"{model}.en", "chunk_s": seconds}
    for model in ("tiny", "base") for seconds in (4, 8)
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def write_once(path, content):
    """Reuse identical files; never replace a different prepared input."""
    path = Path(path)
    content = content.encode("utf-8") if isinstance(content, str) else content
    if path.exists():
        require(path.read_bytes() == content,
                f"{path} already contains different data. Use a new experiment ID; existing files were kept.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
    return path


def copy_once(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        require(sha256(source) == sha256(destination), f"Existing copy differs: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids replacing an existing recording.
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    return destination


def resolve_mcif_root(path):
    path = Path(path).expanduser().resolve()
    for candidate in (path, path / "mcif-long-trans"):
        if all((candidate / p).exists() for p in ("audio", "ref/en.txt", "ref/de.txt", "audio-segments.yaml")):
            return candidate
    raise FileNotFoundError(
        f"MCIF_ROOT is not the extracted dataset folder: {path}. "
        "Select the folder containing audio/, ref/, and audio-segments.yaml in notebook section 1.")


def wav_info(path):
    with wave.open(str(path), "rb") as audio:
        rate, frames = audio.getframerate(), audio.getnframes()
        require(rate > 0 and frames > 0, f"Empty or invalid WAV: {path}")
        return {"sample_rate_hz": rate, "channels": audio.getnchannels(),
                "sample_width_bytes": audio.getsampwidth(), "encoding": audio.getcomptype(),
                "frames": frames, "duration_s": frames / rate}


def canonical_wav(info):
    return (info["sample_rate_hz"], info["channels"], info["sample_width_bytes"], info["encoding"]) == (16000, 1, 2, "NONE")


def load_dataset(root, lock):
    root = resolve_mcif_root(root)
    paths = {"audio-segments.yaml": root / "audio-segments.yaml",
             "en.txt": root / "ref/en.txt", "de.txt": root / "ref/de.txt"}
    for name, path in paths.items():
        require(sha256(path) == lock["reference_input_sha256"][name],
                f"{path} differs from the official reference files validated for the pilot. "
                "Check that you selected the same mcif-long-trans package.")
    segments = yaml.safe_load(paths["audio-segments.yaml"].read_text(encoding="utf-8"))
    texts = {lang: paths[lang + ".txt"].read_text(encoding="utf-8").splitlines() for lang in ("en", "de")}
    require(isinstance(segments, list) and len(segments) == len(texts["en"]) == len(texts["de"]),
            "The YAML and EN/DE files must have the same number of entries. Blank lines must not be removed.")
    groups = {}
    for index, segment in enumerate(segments):
        name = Path(segment["wav"]).name
        require(re.fullmatch(r"[A-Za-z0-9_-]+\.wav", name), f"Unexpected recording name: {name}")
        offset, duration = float(segment["offset"]), float(segment["duration"])
        require(math.isfinite(offset) and math.isfinite(duration) and offset >= 0 and duration > 0,
                f"Invalid timing in reference row {index + 1}")
        require(all(texts[lang][index].strip() for lang in ("en", "de")),
                f"Empty reference in row {index + 1}; inspect the original files.")
        groups.setdefault(Path(name).stem, []).append(index)
    for talk, indices in groups.items():
        offsets = [segments[i]["offset"] for i in indices]
        require(offsets == sorted(offsets), f"Reference rows are out of time order: {talk}")
    return {"root": root, "segments": segments, "texts": texts, "groups": groups,
            "reference_sha256": {name: sha256(path) for name, path in paths.items()}}


def inventory(dataset):
    rows = []
    for talk, indices in sorted(dataset["groups"].items()):
        path = dataset["root"] / "audio" / (talk + ".wav")
        last_end = max(float(dataset["segments"][i]["offset"]) + float(dataset["segments"][i]["duration"]) for i in indices)
        row = {"talk_id": talk, "role": "reserved pilot" if talk == PILOT_ID else "main candidate",
               "reference_segments": len(indices), "reference_end_s": round(last_end, 3),
               "audio_minutes": None, "format": "missing WAV"}
        if path.is_file():
            try:
                info = wav_info(path)
                row.update(audio_minutes=round(info["duration_s"] / 60, 2),
                           audio_duration_s=info["duration_s"],
                           format="mono 16 kHz PCM16" if canonical_wav(info) else "conversion needed")
                require(last_end <= info["duration_s"] + 1 / info["sample_rate_hz"],
                        f"A reference segment extends beyond {talk}.wav")
            except wave.Error:
                row["format"] = "unsupported WAV; inspect format"
        rows.append(row)
    return rows


def check_selection(ids, dataset):
    require(isinstance(ids, list), "MAIN_TALK_IDS must be a Python list of talk IDs.")
    require(len(ids) == len(set(ids)), "MAIN_TALK_IDS contains a duplicate.")
    require(PILOT_ID not in ids, f"{PILOT_ID} is reserved for the pilot and cannot enter the main evaluation.")
    durations = []
    for talk in ids:
        require(talk in dataset["groups"], f"Unknown talk ID: {talk}. Use an ID from the inventory table.")
        durations.append(wav_info(dataset["root"] / "audio" / (talk + ".wav"))["duration_s"])
    if ids:
        require(30 <= sum(durations) / 60 <= 40,
                f"Selected audio totals {sum(durations) / 60:.2f} minutes; select 30–40 minutes of whole talks.")
    return sum(durations)


def ensure_runner(project, support, lock):
    project, support = Path(project), Path(support)
    require(project.is_dir(), f"Project folder does not exist: {project}")
    rows = []
    for name in ("run_pilot.py", "stream_timing.py", "test_timing.py", "check_runtime.py", "requirements.txt"):
        source, destination = support / name, project / name
        expected = lock["runner_sha256"].get(name, sha256(source))
        require(sha256(source) == expected, f"Bundled source does not match the pilot: {name}")
        if not destination.exists():
            copy_once(source, destination)
        require(sha256(destination) == expected,
                f"{destination} differs from the supplied pilot source. It has not been replaced. "
                "Use a separate project folder for this reproducible preparation, or inspect the difference first.")
        rows.append({"file": name, "status": "matches supplied source", "sha256": expected[:16] + "…"})
    return rows


def verify_model(folder, expected):
    folder = Path(folder)
    marker = folder / "pilot-model.json"
    if not folder.exists():
        return {"folder": folder.name, "status": "missing", "revision": expected["revision"][:12]}
    require(marker.is_file(), f"{folder} has no pilot-model.json. Keep it intact and inspect it before proceeding.")
    actual = read_json(marker)
    require(actual.get("repo") == expected["repo"] and actual.get("revision") == expected["revision"],
            f"Model repository/revision differs from the pilot: {folder}")
    require(actual.get("kind") == expected["kind"], f"Model preparation type differs: {folder}")
    require(actual.get("files_sha256") == expected["files_sha256"], f"Model file manifest differs from the pilot: {folder}")
    for name, digest in expected["files_sha256"].items():
        path = folder / name
        require(path.is_file() and sha256(path) == digest, f"Model file is missing or changed: {path}")
    return {"folder": folder.name, "status": "verified", "revision": expected["revision"][:12],
            "model_mib": round((folder / "model.bin").stat().st_size / 2**20, 1)}


def audit_models(project, lock):
    return [verify_model(Path(project) / "models" / name, record) for name, record in lock["models"].items()]


def prepare_talk(project, experiment_id, dataset, talk, role, lock, support, allow_conversion=False):
    require(re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id), "Use letters, numbers, hyphens or underscores in EXPERIMENT_ID.")
    require(talk in dataset["groups"], f"Unknown talk: {talk}")
    require(role in ("pilot", "main") and ((role == "pilot") == (talk == PILOT_ID)), "Pilot/main split mismatch")
    project = Path(project).resolve()
    source = dataset["root"] / "audio" / (talk + ".wav")
    require(source.is_file(), f"Recording is missing: {source}")
    source_hash, source_info = sha256(source), wav_info(source)
    if talk == PILOT_ID:
        require(source_hash == lock["pilot"]["prepared_files"][talk + ".wav"]["sha256"],
                "Pilot WAV differs from the recording used for the completed pilot runs.")
        for lang in ("en", "de"):
            selected = " ".join(dataset["texts"][lang][i] for i in dataset["groups"][talk])
            original = (Path(support) / f"pilot-reference.{lang}.txt").read_text(encoding="utf-8")
            require(" ".join(selected.split()) == " ".join(original.split()),
                    f"Timed {lang} reference does not match the original pilot reference.")
    destination = project / "data" / "prepared" / experiment_id / role / talk
    audio = destination / source.name
    method = "byte-for-byte copy of the complete original WAV"
    if canonical_wav(source_info):
        copy_once(source, audio)
    else:
        require(allow_conversion, f"{source.name} requires conversion. Set ALLOW_AUDIO_CONVERSION=True in the notebook.")
        require(shutil.which("ffmpeg"), "Install ffmpeg on the laptop before converting audio.")
        previous = destination / "manifest.json"
        if audio.exists():
            require(previous.is_file(), f"Existing converted WAV has no preparation manifest: {audio}")
            old = read_json(previous)
            require(old["source_audio_sha256"] == source_hash and sha256(audio) == old["prepared_files"][audio.name]["sha256"],
                    f"Existing converted WAV or source changed: {audio}")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination) as tmp:
                temporary = Path(tmp) / source.name
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source),
                                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temporary)], check=True)
                copy_once(temporary, audio)
        method = "full-recording ffmpeg conversion to mono 16 kHz PCM16; no trimming or silence removal"
    info = wav_info(audio)
    require(canonical_wav(info), f"Prepared WAV is not in the runner's required format: {audio}")
    require(abs(info["duration_s"] - source_info["duration_s"]) <= 0.001,
            f"Conversion changed recording duration unexpectedly: {audio}")
    indices = dataset["groups"][talk]
    segments = [dict(dataset["segments"][i]) for i in indices]
    for segment in segments:
        require(float(segment["offset"]) + float(segment["duration"]) <= info["duration_s"] + 1 / 16000,
                f"Reference extends beyond the prepared WAV: {talk}")
    files = [audio]
    for lang in ("en", "de"):
        text = "\n".join(dataset["texts"][lang][i] for i in indices) + "\n"
        files.append(write_once(destination / f"reference.{lang}.txt", text))
    files.append(write_once(destination / "audio-segments.yaml", yaml.safe_dump(segments, sort_keys=False, allow_unicode=True)))
    manifest = {"schema_version": 1, "talk_id": talk, "role": role,
                "dataset": "FBK-MT/MCIF; official mcif-long-trans package",
                "license": "CC-BY-4.0", "attribution": "FBK-MT, MCIF authors, and original talk creators",
                "audio_file": audio.name, "audio_duration_s": info["duration_s"], "audio_frames": info["frames"],
                "audio_format": info, "source_audio_sha256": source_hash, "audio_processing": method,
                "source_interval_s": [0.0, info["duration_s"]],
                "reference_clock": "original full-recording clock; no cropping or offset reset",
                "reference_rows_1_based": [i + 1 for i in indices], "reference_segments": len(indices),
                "reference_input_sha256": dataset["reference_sha256"],
                "references": {lang: {"file": f"reference.{lang}.txt", "sentences_timed": True} for lang in ("en", "de")},
                "prepared_files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}}
    write_once(destination / "manifest.json", json_bytes(manifest))
    return {"talk_id": talk, "audio": audio.relative_to(project).as_posix(),
            "manifest": (destination / "manifest.json").relative_to(project).as_posix(),
            "audio_duration_s": info["duration_s"], "audio_sha256": sha256(audio),
            "manifest_sha256": sha256(destination / "manifest.json"), "reference_segments": len(indices)}


def freeze_experiment(project, experiment_id, split, talks, lock, repetitions):
    require(talks, "No prepared talks were supplied.")
    require(all(r["status"] == "verified" for r in audit_models(project, lock)),
            "Complete model preparation before freezing the experiment.")
    require(split in ("pilot", "main") and repetitions in (1, 2), "Invalid split or repetition count")
    require(split != "main" or (all(t["talk_id"] != PILOT_ID for t in talks) and repetitions == 2),
            "Main evaluation must exclude the pilot and use two repetitions.")
    project = Path(project).resolve()
    require(len(talks) == len({t["talk_id"] for t in talks}), "Duplicate prepared talk.")
    if split == "main":
        require(30 <= sum(t["audio_duration_s"] for t in talks) / 60 <= 40,
                "The main set must contain 30–40 minutes of whole talks.")
    for name, expected in lock["runner_sha256"].items():
        require(sha256(project / name) == expected, f"Runner changed since the initial check: {name}")
    for talk in talks:
        require(sha256(project / talk["audio"]) == talk["audio_sha256"], "Prepared audio changed.")
        require(sha256(project / talk["manifest"]) == talk["manifest_sha256"], "Talk manifest changed.")
    output = project / "experiments" / experiment_id / (split + ".json")
    existing = read_json(output) if output.exists() else {}
    settings = {"threads": 4, "asr_beam": 1, "mt_beam": 4, "compute_type_requested": "int8",
                "pilot_effective_compute_type": "int8_float32", "temperature": 0,
                "vad": False, "condition_on_previous_text": False, "overlap_s": 0,
                "revisions": False, "normal_speed_replay": True,
                "loading_and_warmup_outside_replay": True,
                "emission_policy": "entire chunk translation after MT completes"}
    jobs = []
    for device in ("pi", "laptop"):
        for repeat in range(1, repetitions + 1):
            for talk in talks:
                for config in CONFIGURATIONS:
                    run_name = f"mcif-{talk['talk_id']}-{config['id']}-mtbeam4-r{repeat}"
                    jobs.append({"device": device, "repetition": repeat, "talk_id": talk["talk_id"],
                                 "configuration": config["id"], "audio": talk["audio"],
                                 "out": f"runs/{experiment_id}/{split}/{device}/{run_name}"})
    document = {"schema_version": 1, "experiment_id": experiment_id, "split": split,
                "created_utc": existing.get("created_utc", datetime.now(timezone.utc).isoformat()),
                "job_status": "planned; this file does not claim any inference run has been executed",
                "reserved_pilot_talk": PILOT_ID, "configurations": CONFIGURATIONS,
                "settings": settings, "repetitions_per_device": repetitions, "devices": ["pi", "laptop"],
                "talks": talks, "total_audio_s": sum(t["audio_duration_s"] for t in talks),
                "models": {name: {key: item[key] for key in ("repo", "revision", "kind", "files_sha256")}
                           for name, item in lock["models"].items()},
                "runner_sha256": lock["runner_sha256"], "jobs": jobs,
                "evaluation_only": "Reference text and segment times are not inputs to the inference runner."}
    write_once(output, json_bytes(document))
    return output
