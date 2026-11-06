#!/usr/bin/env python3
"""Validate saved smoke events and score complete recordings outside inference."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import unicodedata
import wave


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)


def normalize(text):
    # Same lowercase / Unicode-punctuation deletion / whitespace convention as main WER.
    return " ".join("".join(c for c in text.lower() if not unicodedata.category(c).startswith("P")).split())


def validate(run):
    metadata, summary = load(run / "metadata.json"), load(run / "summary.json")
    require(summary["complete"] and not (run / "failure.json").exists(), "Run is incomplete")
    is_la = (run / "updates.jsonl").exists()
    events = rows(run / ("updates.jsonl" if is_la else "chunks.jsonl"))
    word_events = rows(run / "words.jsonl")
    hypothesis = load(run / "hypothesis.jsonl")
    duration = metadata["audio_duration_s"]
    require(events, "Missing update events")
    previous_end = previous_emission = 0
    expected_words = []
    expected_english = []
    index_key = "update_index" if is_la else "chunk_index"
    for index, event in enumerate(events):
        require(event[index_key] == index, "Noncontiguous event indices")
        start, end = event["source_start_s"], event["source_end_s"]
        emitted, processing_start = event["emitted_s"], event["processing_start_s"]
        require(all(math.isfinite(v) for v in [start, end, emitted, processing_start, event["process_s"], event["asr_s"], event["mt_s"]]), "Nonfinite timing")
        require(close(start, previous_end), "Missing or repeated audio")
        require(0 <= start <= end <= duration + 1e-6, "Audio endpoint out of range")
        require(processing_start + 1e-6 >= end, "Read audio before arrival")
        require(processing_start + 1e-6 >= previous_emission, "Sequential worker overlaps itself")
        require(emitted >= processing_start and close(emitted - processing_start, event["process_s"]), "Invalid processing clock")
        require(event["asr_s"] >= 0 and event["mt_s"] >= 0, "Negative stage time")
        require(event["asr_s"] + event["mt_s"] <= event["process_s"] + 1e-6, "Stage time exceeds total processing")
        if is_la:
            eof = index == len(events) - 1
            require(event["event"] == ("eof_flush" if eof else "update"), "Invalid EOF event ordering")
            require(event["commit_reason"] == ("eof_unconfirmed" if eof else "agreement"), "Unlabelled EOF tail")
            source_words = event["flushed_words"] if eof else event["confirmed_words"]
            require(event["english"] == " ".join(w["text"] for w in source_words), "English commit text mismatch")
            if eof:
                require(start == end and event["asr_s"] == 0 and not event["confirmed_words"], "EOF falsely confirmed or decoded")
            else:
                require(end > start, "Agreement update needs new audio")
                require(close(event["decoded_audio_s"], end - event["buffer_start_s"]), "Buffer duration mismatch")
        for word in event["german"].split():
            expected_words.append((word, index, end, emitted))
        if event["english"]:
            expected_english.append(event["english"])
        previous_end, previous_emission = end, emitted
    require(close(previous_end, duration), "Did not consume complete selected audio")
    require(len(expected_words) == len(word_events), "Word log length mismatch")
    for index, (actual, expected) in enumerate(zip(word_events, expected_words)):
        word, event_index, consumed, emitted = expected
        require(actual["word_index"] == index and actual["word"] == word and actual[index_key] == event_index, "Word event mismatch")
        require(close(actual["source_consumed_s"], consumed) and close(actual["emitted_s"], emitted), "Words are not emitted atomically after MT")
    prediction = " ".join(w[0] for w in expected_words)
    require(hypothesis["prediction"] == prediction, "Hypothesis text mismatch")
    require((run / "translation.de.txt").read_text().strip() == prediction, "German export mismatch")
    require((run / "transcript.en.txt").read_text().strip() == " ".join(expected_english), "English export mismatch")
    require(len(hypothesis["delays"]) == len(word_events) == len(hypothesis["elapsed"]), "Timestamp export length mismatch")
    for event, cu, ca in zip(word_events, hypothesis["delays"], hypothesis["elapsed"]):
        require(close(cu, event["source_consumed_s"] * 1000) and close(ca, event["emitted_s"] * 1000), "Timestamp unit mismatch")
    require(close(hypothesis["source_length"], duration * 1000), "Source length unit mismatch")
    processing = sum(e["process_s"] for e in events)
    require(close(processing, summary["total_processing_s"]) and close(processing / duration, summary["rtf_processing"]), "RTF mismatch")
    if is_la:
        require(not events[0]["english"] and not events[0]["german"], "First decode cannot confirm a prefix")
    return metadata, summary, {
        "passed": True, "events": len(events), "words": len(word_events),
        "policy": "english_localagreement2" if is_la else "fixed_chunks",
        "checks": ["complete_audio_coverage", "causal_audio_reads", "sequential_worker", "atomic_word_emissions",
                   "text_export_roundtrip", "timestamp_ms_roundtrip", "processing_rtf"] +
                  (["first_update_unconfirmed", "explicit_eof_flush"] if is_la else []),
    }

