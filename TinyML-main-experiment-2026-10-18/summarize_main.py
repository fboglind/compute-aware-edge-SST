#!/usr/bin/env python3
"""Pool talks within each pass, then report the mean and range of the two passes."""
import argparse
import csv
import itertools
import statistics
from pathlib import Path

from common import GRID, read_json, require, sha, write_json


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def summarize(root):
    from sacrebleu.metrics import CHRF
    root = Path(root)
    report = read_json(root / "scores.json")
    rows = report["runs"]
    expected = {t["id"] for t in report["manifest"]["talks"]}
    segments = {}
    for row in rows:
        path = root / row["segments_file"]
        require(sha(path) == row["segments_sha256"], "Resegmented inputs changed")
        segments[row["key"]] = read_json(path)
    comet = None
    if (root / "comet-scores.json").exists():
        comet = read_json(root / "comet-scores.json")
        require(comet["scores_json_sha256"] == sha(root / "scores.json"), "COMET scores belong to a different evaluation")
        for row in rows:
            require(comet["segments_sha256"][row["key"]] == row["segments_sha256"], "COMET segment hash differs")
    passes = []
    for device, (asr, seconds), rep in itertools.product(("pi", "laptop"), GRID, (1, 2)):
        group = [r for r in rows if (r["device"], r["asr"], r["chunk_s"], r["repetition"]) == (device, asr, seconds, rep)]
        if not group:
            continue
        require(len({r["talk"] for r in group}) == len(group), "Duplicate talk within pass")
        pairs = [s for r in group for s in segments[r["key"]]]
        metric = CHRF(char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False)
        chrf = metric.corpus_score([s["prediction"] for s in pairs], [[s["reference"] for s in pairs]])
        duration = sum(r["audio_duration_s"] for r in group)
        entry = {"device": device, "asr": asr, "chunk_s": seconds, "repetition": rep,
                 "complete_talk_set": {r["talk"] for r in group} == expected,
                 "talks": len(group), "segments": len(pairs), "audio_duration_s": duration,
                 "chrf2": chrf.score, "chrf_signature": str(metric.get_signature()),
                 "wer_pct": 100 * sum(r["wer_errors"] for r in group) / sum(r["wer_reference_words"] for r in group),
                 "rtf_processing": sum(r["total_processing_s"] for r in group) / duration,
                 "asr_rtf": sum(r["total_asr_s"] for r in group) / duration,
                 "mt_rtf": sum(r["total_mt_s"] for r in group) / duration,
                 "mean_first_output_s": mean([r["first_output_s"] for r in group]),
                 "mean_final_tail_s": mean([r["final_tail_s"] for r in group]),
                 "max_final_tail_s": max(r["final_tail_s"] for r in group),
                 "queue_wait_max_s": max(r["queue_wait_max_s"] for r in group),
                 "sampled_peak_replay_rss_mib": max(r["sampled_peak_replay_rss_mib"] for r in group),
                 "process_peak_rss_mib_including_load_warmup": max(r["process_peak_rss_mib_including_load_warmup"] for r in group),
                 "empty_asr_chunks": sum(r["empty_asr_chunks"] for r in group),
                 "mt_limit_chunks": sum(r["mt_limit_chunks"] for r in group),
                 "resource_sampling_errors": sum(bool(r["resource_sampling_error"]) for r in group),
                 "comet": None}
        for clock in ("cu", "ca"):
            values = [v for r in group for v in r["latency"][clock]["segment_scores_s"] if v is not None]
            entry["longyaal_" + clock + "_s"] = mean(values)
            entry["latency_eligible_segments_" + clock] = len(values)
            entry["words_at_or_after_wav_end_" + clock] = sum(r["latency"][clock]["words_at_or_after_wav_end"] for r in group)
        if comet:
            entry["comet"] = mean([s for r in group for s in comet["runs"][r["key"]]["segment_scores"]])
        passes.append(entry)
    configurations = []
    fields = ["chrf2", "comet", "wer_pct", "rtf_processing", "asr_rtf", "mt_rtf", "longyaal_cu_s", "longyaal_ca_s",
              "mean_first_output_s", "mean_final_tail_s", "max_final_tail_s", "queue_wait_max_s",
              "sampled_peak_replay_rss_mib", "process_peak_rss_mib_including_load_warmup"]
    for device, (asr, seconds) in itertools.product(("pi", "laptop"), GRID):
        group = [p for p in passes if (p["device"], p["asr"], p["chunk_s"]) == (device, asr, seconds) and p["complete_talk_set"]]
        entry = {"device": device, "asr": asr, "chunk_s": seconds, "complete_repetitions": len(group), "metrics": {}}
        for field in fields:
            vals = [p[field] for p in group if p[field] is not None]
            entry["metrics"][field] = {"mean": mean(vals), "min": min(vals) if vals else None,
                                        "max": max(vals) if vals else None, "n": len(vals)}
        configurations.append(entry)
    agreement = []
    for talk, (asr, seconds) in itertools.product(sorted(expected), GRID):
        group = [r for r in rows if (r["talk"], r["asr"], r["chunk_s"]) == (talk, asr, seconds)]
        record = {"talk": talk, "asr": asr, "chunk_s": seconds, "runs_present": len(group), "comparisons": []}
        for a, b in itertools.combinations(group, 2):
            same_de = a["german_sha256"] == b["german_sha256"]
            same_schedule = a["cu_schedule_sha256"] == b["cu_schedule_sha256"]
            if same_de and same_schedule:
                cu_a, cu_b = a["longyaal_cu_s"], b["longyaal_cu_s"]
                require(cu_a == cu_b or (cu_a is not None and cu_b is not None and abs(cu_a-cu_b) < 1e-9),
                        "Identical hypotheses and CU schedules produced different CU scores")
            record["comparisons"].append({"a": a["key"], "b": b["key"],
                       "same_asr": a["english_sha256"] == b["english_sha256"], "same_translation": same_de,
                       "same_cu_schedule": same_schedule})
        agreement.append(record)
    write_json(root / "summary.json", {"complete_grid": report["complete_grid"], "comet_complete": comet is not None,
               "aggregation": "Within a pass: corpus chrF2, pooled WER counts, audio-weighted RTF, unweighted eligible-segment LongYAAL, and segment-mean COMET. Between passes: mean and observed min/max; no confidence intervals or independence claim.",
               "independent_content_units": len(expected), "passes": passes, "configurations": configurations})
    write_json(root / "output-agreement.json", agreement)
    if passes:
        with (root / "pass-summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(passes[0]))
            w.writeheader()
            w.writerows(passes)
    print(f"Saved {root / 'summary.json'} and {root / 'pass-summary.csv'}; COMET {'included' if comet else 'pending'}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evaluation", type=Path)
    summarize(p.parse_args().evaluation)
