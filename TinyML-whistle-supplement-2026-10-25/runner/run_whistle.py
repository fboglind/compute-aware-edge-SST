#!/usr/bin/env python3
"""Supplementary Whistle-to-Marian cascade; normal-speed fixed-chunk replay."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

from stream_timing import replay
from whistle_backend import Whistle, verified_assets


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified_manifest(folder):
    path = folder / "pilot-model.json"
    manifest = json.loads(path.read_text())
    for filename, expected in manifest["files_sha256"].items():
        if hash_file(folder / filename) != expected:
            raise ValueError(f"Model checksum mismatch: {folder / filename}")
    return manifest


def throttled():
    try:
        result = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                                text=True, timeout=3)
        return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def read_number(path, divisor=1):
    try:
        return float(Path(path).read_text().strip()) / divisor
    except (OSError, ValueError):
        return None


class ResourceSampler:
    """Sample every five seconds, independently of slow model calls."""
    def __init__(self, path):
        import psutil
        self.psutil = psutil
        self.proc = psutil.Process()
        self.path = path
        self.stop = threading.Event()
        self.rows = []
        self.error = None

    def sample(self):
        # /proc/self follows this process even in environments with remapped PIDs.
        # On ordinary Linux this agrees with psutil.Process().memory_info().rss.
        try:
            status = Path("/proc/self/status").read_text()
            rss_mib = next(float(line.split()[1]) / 1024
                           for line in status.splitlines() if line.startswith("VmRSS:"))
        except (OSError, ValueError, StopIteration):
            rss_mib = self.proc.memory_info().rss / 2**20
        row = {
            "elapsed_s": time.perf_counter() - self.origin,
            "rss_mib": rss_mib,
            "system_available_mib": self.psutil.virtual_memory().available / 2**20,
            "system_swap_used_mib": self.psutil.swap_memory().used / 2**20,
            "cpu_temp_c": read_number("/sys/class/thermal/thermal_zone0/temp", 1000),
            "process_threads": self.proc.num_threads(),
            "cpu_freq_mhz": read_number("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000),
        }
        self.rows.append(row)
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()

    def worker(self):
        try:
            while not self.stop.wait(5):
                self.sample()
        except Exception as e:
            self.error = repr(e)

    def __enter__(self):
        self.origin = time.perf_counter()
        self.file = self.path.open("w", encoding="utf-8")
        self.sample()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        try:
            self.sample()
        finally:
            self.file.close()


class Cascade:
    def __init__(self, asr_path, mt_path, threads, mt_beam=1):
        import ctranslate2
        from transformers import MarianTokenizer
        self.asr = Whistle(asr_path)
        self.tokenizer = MarianTokenizer.from_pretrained(str(mt_path / "tokenizer"),
                                                         local_files_only=True)
        self.mt = ctranslate2.Translator(str(mt_path), device="cpu", compute_type="int8",
                                         inter_threads=1, intra_threads=threads)
        self.mt_beam = mt_beam

    def transcribe(self, audio):
        self.last_asr_response = self.asr.transcribe(audio)
        return " ".join(self.last_asr_response["text"].split())

    def translate(self, english):
        if not english.strip():
            return "", 0, False
        ids = self.tokenizer.encode(english, add_special_tokens=True, truncation=False)
        if len(ids) > 512:
            raise ValueError("ASR text exceeds 512 MT input tokens; refusing silent truncation")
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        result = self.mt.translate_batch([tokens], beam_size=self.mt_beam, max_batch_size=1,
                                         max_input_length=0, max_decoding_length=256)[0]
        target = result.hypotheses[0]
        german = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target),
                                       skip_special_tokens=True)
        german = " ".join(german.split())
        return german, len(target), len(target) >= 256

    def __call__(self, pcm):
        import numpy as np
        start = time.perf_counter()
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        english = self.transcribe(audio)
        asr_end = time.perf_counter()
        german, token_count, hit_limit = self.translate(english)
        mt_end = time.perf_counter()
        return dict(english=english, german=german, asr_s=asr_end - start,
                    mt_s=mt_end - asr_end, mt_tokens=token_count, mt_limit_reached=hit_limit,
                    asr_response=self.last_asr_response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Mono 16 kHz, signed 16-bit PCM WAV")
    parser.add_argument("--asr", choices=["whistle"], default="whistle")
    parser.add_argument("--chunk-s", type=float, default=4)
    parser.add_argument("--threads", type=int, default=4, help="MT threads; Whistle uses its native defaults")
    parser.add_argument("--mt-beam", type=int, default=4, help="MT beam size; this protocol fixes beam 4")
    parser.add_argument("--models-dir", type=Path, default=Path("../models"))
    parser.add_argument("--whistle-assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New, empty run directory")
    args = parser.parse_args()
    if not (0 < args.chunk_s <= 30) or args.threads < 1 or args.mt_beam < 1:
        parser.error("Use 0 < --chunk-s <= 30, --threads >= 1 and --mt-beam >= 1")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must not contain an earlier run; use a new directory")
    # All model/tokenizer access must be local during inference.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with wave.open(str(args.audio), "rb") as f:
        if (f.getnchannels(), f.getframerate(), f.getsampwidth(), f.getcomptype()) != (1, 16000, 2, "NONE"):
            parser.error("Audio must be mono 16 kHz, 16-bit uncompressed PCM WAV; convert with ffmpeg")
        duration = f.getnframes() / 16000
        if duration <= 0:
            parser.error("Empty audio file")
    asr_path = args.whistle_assets
    mt_path = args.models_dir / "mt-en-de-int8"
    manifests = {"asr": verified_assets(asr_path), "mt": verified_manifest(mt_path)}
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
        "audio_sha256": hash_file(args.audio), "audio_duration_s": duration,
        "script_sha256": {p.name: hash_file(p) for p in [Path(__file__), Path(__file__).with_name("stream_timing.py"), Path(__file__).with_name("whistle_backend.py"), Path(__file__).with_name("needle.h")]},
        "models": manifests,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "decoding": {"asr_beam": "engine_default", "asr_threads": "engine_default_not_exposed",
                     "whistle_policy": manifests["asr"]["inference_policy"],
                     "mt_beam": args.mt_beam, "temperature": "engine_default",
                     "vad": False, "condition_on_previous_text": False,
                     "asr_language": "en", "asr_task": "transcribe", "mt_max_output_tokens": 256},
        "policy": "successive disjoint fixed chunks, no revision, no overlap, one worker",
        "timestamp_unit": "seconds in chunks/words; milliseconds in hypothesis.jsonl",
        "word_unit": "whitespace-separated surface tokens, punctuation retained",
        "emission": "whole translation committed atomically after MT; no interpolated word times",
        "warmup": "one ASR call on synthetic silence of chunk duration; one English MT sentence",
        "throttled_before": throttled(),
    }
    write_json(args.out / "metadata.json", metadata)
    print("Loading models (outside replay timing)...", flush=True)
    started = time.perf_counter()
    cascade = Cascade(asr_path, mt_path, args.threads, mt_beam=args.mt_beam)
    metadata["model_load_s"] = time.perf_counter() - started
    metadata["effective_compute_type"] = {"asr": "cactus-quants-shipped",
                                           "mt": cascade.mt.compute_type}
    import numpy as np
    print("Warming up (outside replay timing)...", flush=True)
    started = time.perf_counter()
    cascade.transcribe(np.zeros(round(args.chunk_s * 16000), dtype=np.float32))
    warmup_de, _, _ = cascade.translate("This is a test of English to German translation.")
    metadata["warmup_s"] = time.perf_counter() - started
    metadata["mt_warmup_output"] = warmup_de
    write_json(args.out / "metadata.json", metadata)
    rows, words = [], []
    print("Starting normal-speed replay...", flush=True)
    try:
        with ResourceSampler(args.out / "resources.jsonl") as resources, \
                wave.open(str(args.audio), "rb") as wav, \
                (args.out / "chunks.jsonl").open("w", encoding="utf-8") as chunks_file, \
                (args.out / "words.jsonl").open("w", encoding="utf-8") as words_file:
            def emit(record):
                rows.append(record)
                chunks_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunks_file.flush()
                for word in record["words"]:
                    event = dict(word_index=len(words), word=word, chunk_index=record["chunk_index"],
                                 source_consumed_s=record["source_end_s"], emitted_s=record["emitted_s"])
                    words.append(event)
                    words_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                words_file.flush()
            replay_wall = replay(wav, args.chunk_s, cascade, emit)
        processing = sum(x["process_s"] for x in rows)
        hypothesis = {
            "source": args.audio.name,
            "prediction": " ".join(w["word"] for w in words),
            "delays": [w["source_consumed_s"] * 1000 for w in words],
            "elapsed": [w["emitted_s"] * 1000 for w in words],
            "source_length": duration * 1000,
        }
        (args.out / "hypothesis.jsonl").write_text(json.dumps(hypothesis, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out / "translation.de.txt").write_text(hypothesis["prediction"] + "\n", encoding="utf-8")
        (args.out / "transcript.en.txt").write_text(" ".join(r["english"] for r in rows if r["english"]) + "\n", encoding="utf-8")
        summary = {
            "complete": True, "audio_duration_s": duration, "chunks": len(rows),
            "output_words": len(words), "asr_model": args.asr, "chunk_s": args.chunk_s,
            "total_processing_s": processing, "rtf_processing": processing / duration,
            "total_asr_s": sum(x["asr_s"] for x in rows),
            "total_mt_s": sum(x["mt_s"] for x in rows),
            "replay_wall_s": replay_wall,
            "queue_wait_first_s": rows[0]["queue_wait_s"],
            "queue_wait_last_s": rows[-1]["queue_wait_s"],
            "queue_wait_max_s": max(r["queue_wait_s"] for r in rows),
            "mean_lag_after_chunk_end_s": statistics.mean(r["lag_after_chunk_end_s"] for r in rows),
            "final_tail_s": rows[-1]["emitted_s"] - duration,
            "empty_asr_chunks": sum(not r["english"] for r in rows),
            "mt_limit_chunks": sum(r["mt_limit_reached"] for r in rows),
            "process_peak_rss_mib_including_load_warmup": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024),
            "sampled_peak_replay_rss_mib": max(r["rss_mib"] for r in resources.rows),
            "resource_sampling_error": resources.error,
            "throttled_before": metadata["throttled_before"], "throttled_after": throttled(),
            "latency_note": "Queue wait and chunk-end lag are diagnostics, not LongYAAL.",
        }
        write_json(args.out / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        print(f"Output: {args.out / 'translation.de.txt'}")
    except BaseException as e:
        write_json(args.out / "failure.json", {"complete": False, "error": repr(e), "completed_chunks": len(rows)})
        raise


if __name__ == "__main__":
    main()
