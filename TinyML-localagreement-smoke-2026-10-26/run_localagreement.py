#!/usr/bin/env python3
"""LocalAgreement-2 English ASR -> Marian German, with normal-speed WAV replay."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import wave

from local_agreement import LocalAgreement2, Word, confirmed_segment_cut, replay

# Reuse the frozen experiment's local model verification, MT, and resource logs.
HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
SHARED = PILOT / "TinyML-main-experiment-2026-10-18" / "runner"
sys.path.insert(0, str(SHARED))
from run_pilot import Cascade, ResourceSampler, hash_file, throttled, verified_manifest, write_json


class TimestampedCascade(Cascade):
    def transcribe_words(self, audio, prompt=""):
        segments, _ = self.asr.transcribe(
            audio, language="en", task="transcribe", beam_size=1, best_of=1,
            temperature=0.0, condition_on_previous_text=False,
            initial_prompt=prompt or None, vad_filter=False,
            word_timestamps=True, without_timestamps=False, no_speech_threshold=0.6,
        )
        # Materialize the lazy decoder AND timestamp alignment inside the ASR timer.
        words, ends = [], []
        for segment in segments:
            ends.append(segment.end)
            for word in segment.words or []:
                if word.word.strip():
                    words.append(Word(word.start, word.end, word.word.strip()))
        return words, ends


class OnlineCascade:
    def __init__(self, backend, trim_s=15):
        import numpy as np
        self.np = np
        self.backend = backend
        self.trim_s = trim_s
        self.buffer = np.empty(0, dtype=np.float32)
        self.offset_frames = 0
        self.agreement = LocalAgreement2()

    def update(self, pcm, source_end):
        started = time.perf_counter()
        new_audio = self.np.frombuffer(pcm, dtype="<i2").astype(self.np.float32) / 32768
        self.buffer = self.np.concatenate((self.buffer, new_audio))
        offset = self.offset_frames / 16000
        if abs(offset + len(self.buffer) / 16000 - source_end) > 1 / 16000:
            raise ValueError("Audio buffer lost or duplicated samples")
        # Paper: prompt from up to 200 confirmed words outside the current buffer.
        history = [w.text for w in self.agreement.committed if w.end <= offset]
        prompt = " ".join(history[-200:])
        relative_words, relative_ends = self.backend.transcribe_words(self.buffer, prompt)
        words = [Word(w.start + offset, w.end + offset, w.text) for w in relative_words]
        asr_end = time.perf_counter()
        confirmed = self.agreement.update(words)
        cut = confirmed_segment_cut([e + offset for e in relative_ends],
                                    self.agreement.last_end, offset, source_end, self.trim_s)
        if cut is not None:
            cut_frame = math.floor(cut * 16000)
            self.buffer = self.buffer[cut_frame - self.offset_frames:]
            self.offset_frames = cut_frame
        policy_end = time.perf_counter()
        english = " ".join(w.text for w in confirmed)
        german, token_count, hit_limit = self.backend.translate(english)
        mt_end = time.perf_counter()
        return {
            "english": english, "german": german, "commit_reason": "agreement",
            "asr_s": asr_end - started, "policy_s": policy_end - asr_end,
            "mt_s": mt_end - policy_end, "mt_tokens": token_count,
            "mt_limit_reached": hit_limit, "asr_hypothesis": [asdict(w) for w in words],
            "confirmed_words": [asdict(w) for w in confirmed],
            "pending_words": [asdict(w) for w in self.agreement.pending],
            "buffer_start_s": offset, "decoded_audio_s": source_end - offset,
            "buffer_start_after_s": self.offset_frames / 16000,
            "trim_at_s": cut, "prompt": prompt,
        }

    def finish(self):
        started = time.perf_counter()
        tail = self.agreement.finish()
        english = " ".join(w.text for w in tail)
        policy_end = time.perf_counter()
        german, token_count, hit_limit = self.backend.translate(english)
        ended = time.perf_counter()
        return {
            "english": english, "german": german, "commit_reason": "eof_unconfirmed",
            "asr_s": 0, "policy_s": policy_end - started,
            "mt_s": ended - policy_end, "mt_tokens": token_count,
            "mt_limit_reached": hit_limit, "confirmed_words": [],
            "flushed_words": [asdict(w) for w in tail], "pending_words": [],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--asr", choices=["tiny.en", "base.en"], default="tiny.en")
    parser.add_argument("--min-chunk-s", type=float, default=1)
    parser.add_argument("--buffer-trim-s", type=float, default=15)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--models-dir", type=Path, default=PILOT / "models")
    parser.add_argument("--max-audio-s", type=float, help="Optional prefix for an unscored diagnostic")
    parser.add_argument("--out", type=Path, required=True, help="New run directory; never overwritten")
    args = parser.parse_args()
    if not math.isfinite(args.min_chunk_s) or not 0 < args.min_chunk_s <= 30:
        parser.error("--min-chunk-s must be finite and in (0, 30]")
    if not math.isfinite(args.buffer_trim_s) or not 0 < args.buffer_trim_s <= 30:
        parser.error("--buffer-trim-s must be finite and in (0, 30]")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.max_audio_s is not None and (not math.isfinite(args.max_audio_s) or args.max_audio_s <= 0):
        parser.error("--max-audio-s must be finite and positive")
    if args.out.exists():
        parser.error("--out already exists; use a new directory")
    with wave.open(str(args.audio), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
            parser.error("Use mono 16 kHz signed 16-bit PCM WAV")
        total_frames = wav.getnframes()
        frames = min(total_frames, round(args.max_audio_s * 16000)) if args.max_audio_s else total_frames
        if frames <= 0:
            parser.error("Empty audio selection")
    duration = frames / 16000
    for key, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_TORCH": "0",
        "USE_TF": "0", "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ[key] = value
    asr_path = args.models_dir / f"asr-{args.asr}"
    mt_path = args.models_dir / "mt-en-de-int8"
    manifests = {"asr": verified_manifest(asr_path), "mt": verified_manifest(mt_path)}
    args.out.mkdir(parents=True)
    source_files = [Path(__file__), HERE / "local_agreement.py",
                    SHARED / "run_pilot.py", SHARED / "stream_timing.py"]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
        "audio_sha256": hash_file(args.audio), "audio_duration_s": duration,
        "audio_total_duration_s": total_frames / 16000, "source_offset_s": 0,
        "script_sha256": {str(p.relative_to(PILOT)): hash_file(p) for p in source_files},
        "models": manifests,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "decoding": {"asr_beam": 1, "mt_beam": 4, "temperature": 0,
                     "vad": False, "condition_on_previous_text": False,
                     "word_timestamps": True, "without_timestamps": False,
                     "prompt_max_words": 200, "asr_language": "en", "asr_task": "transcribe",
                     "mt_max_output_tokens": 256, "mt_max_input_tokens": 512},
        "policy": "LocalAgreement-2 on English words; each new confirmed span translated once",
        "translation_policy": "independent Marian translation per ASR commit; no German agreement or revision",
        "buffer_trimming": "confirmed Whisper segment, retain final segment; threshold is not a hard cap",
        "scheduler": "adaptive real-time: consume all arrived audio after minimum new-audio wait",
        "eof_policy": "flush latest unconfirmed ASR tail once, translate separately, label eof_unconfirmed",
        "timestamp_unit": "seconds in updates/words; milliseconds in hypothesis.jsonl",
        "cu_note": "delays is consumed audio on this CA-adaptive trajectory, not a separate CU simulation",
        "emission": "all German words in each translation emitted atomically after MT",
        "warmup": "timestamped ASR on synthetic silence, then one English MT sentence; excluded from replay",
        "throttled_before": throttled(),
        "power_supply_online": {str(p): p.read_text().strip() for p in Path('/sys/class/power_supply').glob('*/online')},
    }
    write_json(args.out / "metadata.json", metadata)
    rows, words = [], []
    try:
        print("Loading local models (outside replay timing)...", flush=True)
        started = time.perf_counter()
        backend = TimestampedCascade(asr_path, mt_path, args.threads, mt_beam=4)
        metadata["model_load_s"] = time.perf_counter() - started
        metadata["effective_compute_type"] = {"asr": backend.asr.model.compute_type, "mt": backend.mt.compute_type}
        import numpy as np
        started = time.perf_counter()
        backend.transcribe_words(np.zeros(round(args.min_chunk_s * 16000), dtype=np.float32))
        warmup, _, _ = backend.translate("This is a test of English to German translation.")
        metadata.update(warmup_s=time.perf_counter() - started, mt_warmup_output=warmup)
        write_json(args.out / "metadata.json", metadata)
        online = OnlineCascade(backend, args.buffer_trim_s)
        print(f"Starting {duration:.2f}s normal-speed replay; see updates.jsonl for progress...", flush=True)
        with ResourceSampler(args.out / "resources.jsonl") as resources, \
                wave.open(str(args.audio), "rb") as wav, \
                (args.out / "updates.jsonl").open("w", encoding="utf-8") as update_file, \
                (args.out / "words.jsonl").open("w", encoding="utf-8") as word_file:
            def emit(row):
                rows.append(row)
                update_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                update_file.flush()
                for word in row["german"].split():
                    event = dict(word_index=len(words), word=word, update_index=row["update_index"],
                                 commit_reason=row["commit_reason"], source_consumed_s=row["source_end_s"],
                                 emitted_s=row["emitted_s"])
                    words.append(event)
                    word_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                word_file.flush()
            wall = replay(wav, args.min_chunk_s, online.update, online.finish, emit, max_frames=frames)
        hypothesis = dict(source=args.audio.name, prediction=" ".join(w["word"] for w in words),
                          delays=[w["source_consumed_s"] * 1000 for w in words],
                          elapsed=[w["emitted_s"] * 1000 for w in words], source_length=duration * 1000)
        (args.out / "hypothesis.jsonl").write_text(json.dumps(hypothesis, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out / "translation.de.txt").write_text(hypothesis["prediction"] + "\n", encoding="utf-8")
        (args.out / "transcript.en.txt").write_text(" ".join(r["english"] for r in rows if r["english"]) + "\n", encoding="utf-8")
        updates = rows[:-1]
        processing = sum(r["process_s"] for r in rows)
        summary = {
            "complete": True, "audio_duration_s": duration, "asr_model": args.asr,
            "min_chunk_s": args.min_chunk_s, "updates": len(updates), "output_words": len(words),
            "agreement_commits": sum(bool(r["english"]) for r in updates),
            "agreement_words": sum(len(r["confirmed_words"]) for r in updates),
            "eof_unconfirmed_words": len(rows[-1]["flushed_words"]),
            "total_processing_s": processing, "rtf_processing": processing / duration,
            "total_asr_s": sum(r["asr_s"] for r in rows), "total_mt_s": sum(r["mt_s"] for r in rows),
            "total_policy_s": sum(r["policy_s"] for r in rows), "replay_wall_s": wall,
            "first_output_s": words[0]["emitted_s"] if words else None,
            "final_tail_s": rows[-1]["emitted_s"] - duration,
            "update_wait_max_s": max(r["update_wait_s"] for r in updates),
            "max_decoded_buffer_s": max(r["decoded_audio_s"] for r in updates),
            "decoded_audio_total_s": sum(r["decoded_audio_s"] for r in updates),
            "buffer_trims": sum(r["trim_at_s"] is not None for r in updates),
            "mt_limit_updates": sum(r["mt_limit_reached"] for r in rows),
            "sampled_peak_replay_rss_mib": max(r["rss_mib"] for r in resources.rows),
            "process_peak_rss_mib_including_load_warmup": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024),
            "resource_sampling_error": resources.error,
            "throttled_before": metadata["throttled_before"], "throttled_after": throttled(),
            "latency_note": "First output, update wait and final tail are diagnostics, not LongYAAL. Adaptive schedules depend on computation.",
        }
        write_json(args.out / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        write_json(args.out / "failure.json", {"complete": False, "error": repr(error), "completed_events": len(rows)})
        raise


if __name__ == "__main__":
    main()
