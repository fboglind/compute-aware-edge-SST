#!/usr/bin/env python3
"""Prepare, check and run the LocalAgreement MCIF experiment on the Pi/laptop."""
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
from run_localagreement import TimestampedCascade as Cascade
import numpy as np
models=Path(sys.argv[1]); out={}
for label in ('tiny.en','base.en'):
    c=Cascade(models/('asr-'+label),models/'mt-en-de-int8',4,mt_beam=4)
    types={'asr':c.asr.model.compute_type,'mt':c.mt.compute_type}
    assert types=={'asr':'int8_float32','mt':'int8_float32'}, types
    c.transcribe_words(np.zeros(16000,dtype=np.float32))
    translation,_,limit=c.translate('This is a test of English to German translation.')
    assert translation.strip() and not limit
    out[label]={'types':types,'translation':translation}
    del c
print(json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", script, str(models.resolve())], cwd=ROOT,
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
            raise ValueError("Another TinyML inference batch is already running on this machine")
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
            print("Preflight passed. No experiment replay started.")
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
            command = [sys.executable, str(ROOT / "run_localagreement.py"), str((args.data / talk["audio"]).resolve()),
                       "--asr", job["asr"], "--min-chunk-s", str(job["min_chunk_s"]), "--threads", "4", "--buffer-trim-s", "15",
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
    for name in ("plan", "run", "status"):
        q = sub.add_parser(name)
        q.add_argument("--data", type=Path, default=ROOT.parent / "work/data/main")
        q.add_argument("--device", choices=["pi", "laptop"], required=True)
        q.add_argument("--out", type=Path, required=True)
        if name == "run":
            q.add_argument("--models", type=Path, default=ROOT.parent / "models")
            q.add_argument("--notes")
            q.add_argument("--dry-run", action="store_true")
            q.add_argument("--preflight-only", action="store_true")
            q.add_argument("--retry-incomplete", action="store_true")
            q.add_argument("--max-jobs", type=int)
    args = p.parse_args()
    if args.command == "plan":
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
