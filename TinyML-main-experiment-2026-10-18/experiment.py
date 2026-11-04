#!/usr/bin/env python3
"""Prepare, check and run the frozen MCIF experiment on the Pi/laptop."""
import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path

from common import (ROOT, audit_run, check_models, completed_attempt, dataset,
                    fingerprint, plan_for, read_json, require, sha, utc,
                    validate_plan, write_json)


def pcm_info(path):
    with wave.open(str(path), "rb") as w:
        return {"channels": w.getnchannels(), "rate": w.getframerate(),
                "width": w.getsampwidth(), "compression": w.getcomptype(),
                "frames": w.getnframes(), "duration_s": w.getnframes() / w.getframerate()}


def prepare(args):
    import yaml
    root, out = args.dataset.resolve(), args.out.resolve()
    require(not out.exists(), f"Destination exists: {out}. Use a new path; do not overwrite a frozen dataset.")
    selection = read_json(ROOT / "selection.json")
    source = {"audio-segments.yaml": root / "audio-segments.yaml",
              "en.txt": root / "ref/en.txt", "de.txt": root / "ref/de.txt"}
    for name, path in source.items():
        require(sha(path) == selection["reference_source_sha256"][name],
                f"Official reference file differs from the validated pilot copy: {path}")
    segments = yaml.safe_load(source["audio-segments.yaml"].read_text())
    texts = {lang: source[lang + ".txt"].read_text(encoding="utf-8").splitlines() for lang in ("en", "de")}
    require(len(segments) == len(texts["en"]) == len(texts["de"]) == 919, "Official line mapping differs")
    # Validate input presence before creating anything.
    for item in selection["talks"]:
        require((root / "audio" / (item["id"] + ".wav")).is_file(), f"Missing audio: {item['id']}")
    staging = out.with_name(out.name + ".preparing-" + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    talks, conversions = [], []
    for item in selection["talks"]:
        ident = item["id"]
        indices = [i for i, s in enumerate(segments) if Path(s["wav"]).name == ident + ".wav"]
        require([i + 1 for i in indices] == item["reference_lines_1_based"], "Reference mapping changed")
        segs = [dict(segments[i], wav=ident + ".wav") for i in indices]
        folder = staging / ident
        folder.mkdir()
        src, dst = root / "audio" / (ident + ".wav"), folder / (ident + ".wav")
        try:
            original = pcm_info(src)
        except (wave.Error, EOFError):
            original = None
        command = None
        if original and (original["channels"], original["rate"], original["width"], original["compression"]) == (1, 16000, 2, "NONE"):
            shutil.copyfile(src, dst)
        else:
            require(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Install ffmpeg for WAV conversion")
            if original is None:
                probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(src)], text=True)
                original = {"duration_s": float(json.loads(probe)["format"]["duration"])}
            command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(src), "-map", "0:a:0",
                       "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
            subprocess.run(command, check=True)
        info = pcm_info(dst)
        require(abs(info["duration_s"] - original["duration_s"]) < 0.05, "Conversion changed recording duration")
        duration = info["duration_s"]
        require(duration > 0 and all(s["duration"] > 0 and s["offset"] >= 0
                and s["offset"] + s["duration"] <= duration + 1e-6 for s in segs), "Reference outside WAV")
        require([s["offset"] for s in segs] == sorted(s["offset"] for s in segs), "References out of order")
        (folder / "audio-segments.yaml").write_text(yaml.safe_dump(segs, sort_keys=False), encoding="utf-8")
        for lang in ("en", "de"):
            lines = [texts[lang][i] for i in indices]
            require(all(s.strip() for s in lines), "Empty official reference")
            (folder / (lang + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        talks.append({**item, "duration_s": duration, "audio": f"{ident}/{ident}.wav",
                      "audio_sha256": sha(dst), "original_audio_sha256": sha(src),
                      "reference_dir": ident, "excerpt_offset_s": 0})
        conversions.append({"talk": ident, "source_info": original, "prepared_info": info,
                            "command": command, "operation": "full-recording conversion" if command else "byte copy"})
        print(f"Prepared {ident}: {duration / 60:.2f} min, {len(indices)} reference segments", flush=True)
    total = sum(t["duration_s"] for t in talks)
    require(1800 <= total <= 2400, f"Actual audio total {total / 60:.2f} min outside 30–40 min; investigate before running")
    m = {"schema": 1, "prepared_utc": utc(), "selection_sha256": sha(ROOT / "selection.json"),
         "excluded_pilot": selection["excluded_pilot"], "talks": talks, "duration_s": total,
         "reference_segments": sum(t["reference_segments"] for t in talks),
         "reference_source_sha256": selection["reference_source_sha256"], "conversions": conversions,
         "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
                           if any(c["command"] for c in conversions) else None,
         "files_sha256": {str(p.relative_to(staging)): sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}}
    m["manifest_id"] = fingerprint(m)
    write_json(staging / "manifest.json", m)
    staging.rename(out)
    print(f"Frozen: {out / 'manifest.json'}\n{total / 60:.2f} min; 48 jobs/device; replay floor {8 * total / 3600:.2f} h/device")


def environment():
    return {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_TORCH": "0", "USE_TF": "0",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}


def hardware():
    def text_file(path):
        try:
            return Path(path).read_text().strip().replace("\x00", "")
        except OSError:
            return None
    def command(argv):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=5)
            return p.stdout.strip() or p.stderr.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            return None
    paths = list(Path("/sys/class/thermal").glob("thermal_zone*/type"))
    paths += list(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"))
    paths += list(Path("/sys/class/power_supply").glob("*/online"))
    return {"utc": utc(), "host": platform.node(), "platform": platform.platform(),
            "machine": platform.machine(), "python": sys.version, "cpu_count": os.cpu_count(),
            "device_model": text_file("/proc/device-tree/model"), "lscpu": command(["lscpu"]),
            "memory": text_file("/proc/meminfo"), "power_profile": command(["powerprofilesctl", "get"]),
            "sysfs": {str(p): text_file(p) for p in paths},
            "resource_sensor_note": "Runner temperature is thermal_zone0/temp; interpret with its recorded type. Frequency is CPU0 scaling_cur_freq, not all-core average.",
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def check_runtime(models, device):
    require(platform.system() == "Linux", "This measurement kit targets Linux")
    require(platform.machine() in (("aarch64", "arm64") if device == "pi" else ("x86_64", "AMD64")), "Device label does not match architecture")
    pins = {}
    for line in (ROOT / "requirements-inference.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==")
            pins[name] = version
            require(importlib.metadata.version(name) == version, f"Install the pinned inference environment: {name}=={version}")
    identities = check_models(models)
    # This process is separate from every measured child process. No model warm-up
    # or previous state can leak into the actual replay's process lifetime RSS.
    script = """
import json,sys
from pathlib import Path
from run_pilot import Cascade
import numpy as np
models=Path(sys.argv[1]); out={}
for label in ('tiny.en','base.en'):
    c=Cascade(models/('asr-'+label),models/'mt-en-de-int8',4,mt_beam=4)
    types={'asr':c.asr.model.compute_type,'mt':c.mt.compute_type}
    assert types=={'asr':'int8_float32','mt':'int8_float32'}, types
    c.transcribe(np.zeros(16000,dtype=np.float32))
    translation,_,limit=c.translate('This is a test of English to German translation.')
    assert translation.strip() and not limit
    out[label]={'types':types,'translation':translation}
    del c
print(json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", script, str(models.resolve())], cwd=ROOT / "runner",
                       env=environment(), capture_output=True, text=True)
    require(p.returncode == 0, "Runtime check failed:\n" + p.stdout + p.stderr)
    return {"models": identities, "pins": pins, "runtime_stdout": p.stdout, "runtime_stderr": p.stderr}


def make_plan(args):
    m = dataset(args.data)
    plan = plan_for(m, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "plan.json"
    if path.exists():
        validate_plan(read_json(path), m)
        require(read_json(path)["device"] == args.device, "Device label changed")
    else:
        write_json(path, plan)
    print(f"{len(plan['jobs'])} sequential jobs in {path}; replay floor {8 * m['duration_s']/3600:.2f} h")
    return m, plan


def run_batch(args):
    # One advisory lock per device on this host, across output directories. It is
    # released by the OS if this process dies; never use a stale PID as a lock.
    lockdir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/tinyml-{os.getuid()}"))
    lockdir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (lockdir / "tinyml-main-inference.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another main-experiment batch is already running on this machine")
        m, plan = make_plan(args)
        if args.dry_run:
            for j in plan["jobs"]:
                print(f"{j['order']:02d} {j['id']}")
            return
        require(args.notes and args.notes.strip(), "Use --notes to record power, cooling and background workload")
        session_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
        session = {"plan_id": plan["plan_id"], "notes": args.notes, "hardware": hardware(),
                   "invocation": sys.argv, "runtime": check_runtime(args.models, args.device)}
        write_json(args.out / "sessions" / (session_id + ".json"), session)
        talks = {t["id"]: t for t in m["talks"]}
        if args.preflight_only:
            print("Preflight passed. No main replay started.")
            return
        count = 0
        for job in plan["jobs"]:
            folder = args.out / "jobs" / job["id"]
            talk = talks[job["talk"]]
            if completed_attempt(folder, plan, job, talk):
                print("Verified, skipping", job["id"], flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            attempts = sorted(folder.glob("attempt-*"))
            # A process may have finished writing valid output just before its
            # parent died. Adopt that first valid complete attempt, never rerun it.
            adopted = None
            for previous in attempts:
                info = read_json(previous / "attempt.json")
                require(info["plan_id"] == plan["plan_id"] and info["job"] == job,
                        "Existing attempt belongs to a different plan/job")
                raw_previous = previous / "raw"
                if (raw_previous / "summary.json").exists() and read_json(raw_previous / "summary.json").get("complete"):
                    audit_run(raw_previous, talk, job)
                    for path in raw_previous.iterdir():
                        path.rename(previous / path.name)
                    raw_previous.rmdir()
                if (previous / "summary.json").exists() and read_json(previous / "summary.json").get("complete"):
                    audit_run(previous, talk, job)
                    adopted = previous
                    break
            if adopted:
                finish_job(folder, adopted, plan, job, session_id, recovered=True)
                print("Recovered completed run", job["id"], flush=True)
                continue
            if attempts and not args.retry_incomplete:
                raise ValueError(f"Incomplete attempt retained in {folder}. Inspect its console/failure log, then resume with --retry-incomplete.")
            attempt = folder / f"attempt-{len(attempts)+1:03d}"
            attempt.mkdir()
            command = [sys.executable, str(ROOT / "runner/run_pilot.py"), str((args.data / talk["audio"]).resolve()),
                       "--asr", job["asr"], "--chunk-s", str(job["chunk_s"]), "--threads", "4", "--mt-beam", "4",
                       "--models-dir", str(args.models.resolve()), "--out", str((attempt / "raw").resolve())]
            # Launch into an empty child directory: the retained runner rejects
            # any pre-existing logs. Move completed raw files one level up after exit.
            write_json(attempt / "attempt.json", {"job": job, "plan_id": plan["plan_id"], "session_id": session_id,
                       "started_utc": utc(), "command": command, "hardware_before": hardware()})
            print(f"[{job['order']}/{len(plan['jobs'])}] {job['id']}", flush=True)
            with (attempt / "console.log").open("w") as log:
                child = subprocess.Popen(command, env=environment(), stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(lock.fileno(),))
                try:
                    rc = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            raw = attempt / "raw"
            if raw.exists():
                for p in raw.iterdir():
                    p.rename(attempt / p.name)
                raw.rmdir()
            write_json(attempt / "exit.json", {"returncode": rc, "ended_utc": utc()})
            require(rc == 0, f"Inference failed; inspect {attempt / 'console.log'}. Resume only after fixing the cause.")
            audit_run(attempt, talk, job)
            finish_job(folder, attempt, plan, job, session_id)
            count += 1
            if args.max_jobs and count >= args.max_jobs:
                print("Requested job limit reached; rerun the same command without --max-jobs to continue.")
                return
        print("All 48 planned jobs are complete and verified.")


def finish_job(folder, attempt, plan, job, session_id, recovered=False):
    write_json(folder / "complete.json", {"plan_id": plan["plan_id"], "job": job,
               "attempt": attempt.name, "session_id": session_id, "verified_utc": utc(), "recovered": recovered,
               "output_sha256": {p.name: sha(p) for p in sorted(attempt.iterdir()) if p.is_file()}})


def status(args):
    m = dataset(args.data, with_audio=False)
    plan = read_json(args.out / "plan.json")
    validate_plan(plan, m)
    require(plan["device"] == args.device, "Wrong device label")
    talks = {t["id"]: t for t in m["talks"]}
    completed, pending = [], []
    for job in plan["jobs"]:
        if completed_attempt(args.out / "jobs" / job["id"], plan, job, talks[job["talk"]]):
            completed.append(job)
        else:
            pending.append(job)
    print(f"{args.device}: {len(completed)}/{len(plan['jobs'])} completed and verified")
    for job in pending[:3]:
        print("Pending:", job["id"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze the six talks using local MCIF WAVs")
    prep.add_argument("--dataset", type=Path, required=True)
    prep.add_argument("--out", type=Path, default=Path("data/main"))
    for name in ("plan", "run", "status"):
        q = sub.add_parser(name)
        q.add_argument("--data", type=Path, default=Path("data/main"))
        q.add_argument("--device", choices=["pi", "laptop"], required=True)
        q.add_argument("--out", type=Path, required=True)
        if name == "run":
            q.add_argument("--models", type=Path, default=Path("../models"))
            q.add_argument("--notes")
            q.add_argument("--dry-run", action="store_true")
            q.add_argument("--preflight-only", action="store_true")
            q.add_argument("--retry-incomplete", action="store_true")
            q.add_argument("--max-jobs", type=int)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "plan":
        make_plan(args)
    elif args.command == "status":
        status(args)
    else:
        require(args.max_jobs is None or args.max_jobs > 0, "--max-jobs must be positive")
        run_batch(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"ERROR: {exc}")
