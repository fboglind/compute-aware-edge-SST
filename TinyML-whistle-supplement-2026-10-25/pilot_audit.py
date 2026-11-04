#!/usr/bin/env python3
"""Audit supplementary replay timing and export consistency without loading models."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text):
    return " ".join(text.split())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-5


def audit(run, manifest, asr, chunk_s):
    summary = read_json(run / "summary.json")
    meta = read_json(run / "metadata.json")
    require(summary["complete"] is True, "Run is incomplete")
    audio_name = manifest["audio_file"]
    require(meta["audio_sha256"] == manifest["prepared_files"][audio_name]["sha256"],
            "Audio hash differs from the reference package")
    require(meta["args"]["asr"] == summary["asr_model"] == asr, "ASR model differs from directory name")
    require(meta["args"]["chunk_s"] == summary["chunk_s"] == chunk_s, "Chunk duration differs from directory name")
    require(meta["args"]["threads"] == 4 and meta["decoding"]["mt_beam"] == 4
            and meta["decoding"]["asr_beam"] == "engine_default", "Unexpected threads or beam settings")
    duration = manifest["audio_duration_s"]
    require(close(summary["audio_duration_s"], duration), "Audio duration differs from reference package")
    chunks = read_jsonl(run / "chunks.jsonl")
    words = read_jsonl(run / "words.jsonl")
    hypotheses = read_jsonl(run / "hypothesis.jsonl")
    require(len(chunks) == summary["chunks"] == math.ceil(duration / chunk_s), "Incorrect chunk count")
    require(len(hypotheses) == 1, "Expected exactly one full-talk hypothesis")
    hyp = hypotheses[0]
    english = normalized((run / "transcript.en.txt").read_text(encoding="utf-8"))
    german = normalized((run / "translation.de.txt").read_text(encoding="utf-8"))
    require(english == normalized(" ".join(c["english"] for c in chunks)), "Transcript and chunks disagree")
    require(german == normalized(" ".join(c["german"] for c in chunks)), "Translation and chunks disagree")
    require(hyp["source"] == audio_name and close(hyp["source_length"], duration * 1000),
            "Hypothesis audio name or duration is incorrect (export uses milliseconds)")
    require(normalized(hyp["prediction"]) == german, "Hypothesis text and translation disagree")
    require(len(words) == len(german.split()) == len(hyp["delays"]) == len(hyp["elapsed"])
            == summary["output_words"], "Words, timestamp arrays or summary counts disagree")
    previous_end = previous_emission = 0.0
    expected_words = []
    for index, chunk in enumerate(chunks):
        start, end = chunk["source_start_s"], chunk["source_end_s"]
        processing_start, emission = chunk["processing_start_s"], chunk["emitted_s"]
        require(chunk["chunk_index"] == index and close(start, previous_end)
                and close(end, min((index + 1) * chunk_s, duration)), "Chunk coverage has a gap or incorrect endpoint")
        require(close(chunk["chunk_duration_s"], end - start), "Incorrect chunk duration")
        require(math.isfinite(processing_start) and math.isfinite(emission)
                and processing_start + 1e-5 >= max(end, previous_emission)
                and emission >= processing_start, "Chunk clock is non-causal or non-monotonic")
        require(close(chunk["process_s"], emission - processing_start)
                and close(chunk["queue_wait_s"], processing_start - end)
                and close(chunk["lag_after_chunk_end_s"], emission - end), "Chunk timing diagnostics disagree")
        require(chunk["words"] == chunk["german"].split(), "Chunk words and German text disagree")
        expected_words.extend((word, index, end, emission) for word in chunk["words"])
        previous_end, previous_emission = end, emission
    require(len(expected_words) == len(words), "Chunk and word event counts disagree")
    for index, (event, expected) in enumerate(zip(words, expected_words)):
        word, chunk_index, consumed, emission = expected
        require(event["word_index"] == index and event["word"] == word
                and event["chunk_index"] == chunk_index, "Word event ordering differs from chunk logs")
        require(close(event["source_consumed_s"], consumed) and close(event["emitted_s"], emission)
                and close(hyp["delays"][index], consumed * 1000)
                and close(hyp["elapsed"][index], emission * 1000), "Word timestamps differ from chunks or use incorrect units")
    total_processing = sum(c["process_s"] for c in chunks)
    require(close(summary["total_processing_s"], total_processing)
            and close(summary["rtf_processing"], total_processing / duration), "Summary processing totals disagree")
    require(close(summary["final_tail_s"], chunks[-1]["emitted_s"] - duration), "Final tail differs from chunk clock")
    return summary, meta, english, german, words
