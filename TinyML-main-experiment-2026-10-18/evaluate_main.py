#!/usr/bin/env python3
"""Audit and score completed main runs; never rerun inference or alter raw logs."""
import argparse
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import jiwer
import yaml
from omnisteval.data import Instance
from omnisteval.scoring import YAALScorer
from sacrebleu.metrics import CHRF

from common import (ROOT, audit_run, completed_attempt, dataset, fingerprint, read_json,
                    read_jsonl, require, sha, utc, validate_plan, write_json)


def chrf_metric():
    return CHRF(char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False)


def latency(rows, segs, duration):
    """Use actual WAV end; retain upstream cutoff sensitivity and eligibility."""
    # Subtract in the SAME millisecond coordinate system as resegmentation.
    # (duration-offset)*1000 can move an exact-end word just below the cutoff
    # through floating-point cancellation and spuriously include a final segment.
    instances = [Instance.from_dict(dict(row, time_to_recording_end=duration * 1000 - seg["offset"] * 1000))
                 for row, seg in zip(rows, segs)]
    original = [Instance.from_dict(row) for row in rows]
    result = {}
    for aware, key in [(False, "cu"), (True, "ca")]:
        scorer = YAALScorer(computation_aware=aware, is_longform=True)
        values = [scorer.compute(x) for x in instances]
        upstream = [scorer.compute(x) for x in original]
        selected = [v for v in values if v is not None]
        old = [v for v in upstream if v is not None]
        require(all(math.isfinite(v) for v in selected + old), "Nonfinite latency")
        timestamps = [d + seg["offset"] * 1000 for row, seg in zip(rows, segs) for d in row["emission_" + key]]
        result[key] = {"longyaal_s": sum(selected) / len(selected) / 1000 if selected else None,
                       "segment_scores_s": [v / 1000 if v is not None else None for v in values],
                       "eligible_segments": len(selected),
                       "excluded_segments_0_based": [i for i, v in enumerate(values) if v is None],
                       "words_at_or_after_wav_end": sum(t >= duration * 1000 for t in timestamps),
                       "upstream_last_reference_end_longyaal_s": sum(old) / len(old) / 1000 if old else None,
                       "upstream_eligible_segments": len(old)}
    common = [(cu, ca) for cu, ca in zip(result["cu"]["segment_scores_s"], result["ca"]["segment_scores_s"])
              if cu is not None and ca is not None]
    result["common_eligible_segments"] = len(common)
    result["ca_minus_cu_common_segments_s"] = sum(ca-cu for cu, ca in common) / len(common) if common else None
    result["cutoff_note"] = "Primary uses the actual full WAV end. Upstream sensitivity uses the last reference end. Each clock may censor different words/segments; CA-CU is not pure compute time."
    return result


def clock_control():
    rows = [{"prediction": "eins zwei drei vier", "reference": "eins zwei drei vier",
             "source_length": 4000, "time_to_recording_end": 20000,
             "emission_cu": [1000, 2000, 3000, 4000], "emission_ca": [1750, 2750, 3750, 4750]}]
    check = latency(rows, [{"offset": 0}], 20)
    require(check["cu"]["longyaal_s"] == 1 and check["ca"]["longyaal_s"] == 1.75, "Clock control failed")
    return {"cu_s": 1, "ca_s": 1.75, "status": "passed"}


def resegment(run, refdir, out, duration):
    segs = yaml.safe_load((refdir / "audio-segments.yaml").read_text())
    sources = (refdir / "en.txt").read_text(encoding="utf-8").splitlines()
    references = (refdir / "de.txt").read_text(encoding="utf-8").splitlines()
    hyp = read_jsonl(run / "hypothesis.jsonl")[0]
    require(len(sources) == len(references) == len(segs), "Reference count mismatch")
    command = [sys.executable, "-m", "omnisteval.cli", "longform", "--speech_segmentation", str(refdir / "audio-segments.yaml"),
               "--ref_sentences_file", str(refdir / "de.txt"), "--source_sentences_file", str(refdir / "en.txt"),
               "--hypothesis_file", str(run / "hypothesis.jsonl"), "--lang", "de", "--word_level",
               "--bleu_tokenizer", "13a", "--output_folder", str(out)]
    if hyp["prediction"].strip():
        p = subprocess.run(command, capture_output=True, text=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / "stdout.txt").write_text(p.stdout)
        (out / "stderr.txt").write_text(p.stderr)
        write_json(out / "command.json", command)
        require(p.returncode == 0, f"OmniSTEval failed; inspect {out / 'stderr.txt'}")
        rows = read_jsonl(out / "instances.resegmented.jsonl")
    else:
        out.mkdir(parents=True)
        # An empty whole-talk prediction has zero words in every segment. Keep
        # every reference in quality and mark latency unavailable, never zero.
        ref_end = max(s["offset"] + s["duration"] for s in segs)
        rows = [{"prediction": "", "reference": ref.strip(), "source_length": seg["duration"] * 1000,
                 "time_to_recording_end": (ref_end - seg["offset"]) * 1000,
                 "emission_cu": [], "emission_ca": []} for ref, seg in zip(references, segs)]
        (out / "instances.resegmented.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    require(len(rows) == len(segs), "Resegmentation changed segment count")
    require([w for row in rows for w in row["prediction"].split()] == hyp["prediction"].split(), "Resegmentation changed words")
    require([r["reference"] for r in rows] == [r.strip() for r in references], "Reference order changed")
    for key, original in [("emission_cu", "delays"), ("emission_ca", "elapsed")]:
        times = [d + s["offset"] * 1000 for row, s in zip(rows, segs) for d in row[key]]
        require(len(times) == len(hyp[original]) and all(abs(a-b) < 1e-6 for a, b in zip(times, hyp[original])), "Timestamp roundtrip failed")
    timing = latency(rows, segs, duration)
    for row, src in zip(rows, sources):
        row["src"] = src.strip()
    write_json(out / "validation.json", {"words_preserved": len(hyp["prediction"].split()), "timestamp_roundtrip": "passed",
               "segments": len(rows), "full_wav_duration_s": duration,
               "last_reference_end_s": max(s["offset"] + s["duration"] for s in segs), "latency": timing})
    return rows, timing


def score_one(run, talk, job, refdir, out):
    summary, meta, en, de, words = audit_run(run, talk, job)
    rows, timing = resegment(run, refdir, out, talk["duration_s"])
    metric = chrf_metric()
    chrf = metric.corpus_score([r["prediction"] for r in rows], [[r["reference"] for r in rows]])
    transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
                               jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    reference_en = " ".join((refdir / "en.txt").read_text(encoding="utf-8").split())
    wer = jiwer.process_words(reference_en, en, reference_transform=transform, hypothesis_transform=transform)
    chunks = read_jsonl(run / "chunks.jsonl")
    resources = read_jsonl(run / "resources.jsonl")
    temp = [r["cpu_temp_c"] for r in resources if r.get("cpu_temp_c") is not None]
    swap = [r["system_swap_used_mib"] for r in resources if r.get("system_swap_used_mib") is not None]
    result = {**job, "audio_duration_s": talk["duration_s"], "quality_segments": len(rows),
              "chrf2": chrf.score, "chrf_signature": str(metric.get_signature()),
              "wer_pct": 100 * wer.wer, "wer_errors": wer.substitutions + wer.deletions + wer.insertions,
              "wer_reference_words": wer.hits + wer.substitutions + wer.deletions,
              "longyaal_cu_s": timing["cu"]["longyaal_s"], "longyaal_ca_s": timing["ca"]["longyaal_s"],
              "latency": timing, "first_output_s": words[0]["emitted_s"] if words else None,
              "queue_wait_p95_s": sorted(c["queue_wait_s"] for c in chunks)[math.ceil(.95 * len(chunks))-1],
              "chunks_queue_over_100ms": sum(c["queue_wait_s"] > .1 for c in chunks),
              "temperature_zone0_max_c": max(temp) if temp else None,
              "system_swap_first_mib": swap[0] if swap else None, "system_swap_max_mib": max(swap) if swap else None,
              "english_sha256": sha(run / "transcript.en.txt"), "german_sha256": sha(run / "translation.de.txt"),
              "cu_schedule_sha256": fingerprint(read_jsonl(run / "hypothesis.jsonl")[0]["delays"]),
              "input_sha256": {p.name: sha(p) for p in sorted(run.iterdir()) if p.is_file()},
              "inference_packages": meta["packages"], "machine": meta["machine"],
              **{k: summary[k] for k in ["rtf_processing", "total_processing_s", "total_asr_s", "total_mt_s", "replay_wall_s",
                 "queue_wait_max_s", "queue_wait_last_s", "final_tail_s", "empty_asr_chunks", "mt_limit_chunks", "output_words",
                 "sampled_peak_replay_rss_mib", "process_peak_rss_mib_including_load_warmup", "resource_sampling_error",
                 "throttled_before", "throttled_after"]}}
    return result, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/main"))
    p.add_argument("--runs", type=Path, default=Path("runs"), help="Contains pi/ and laptop/ plan.json and jobs/")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true", help="Explicitly produce a coverage-labelled interim report")
    args = p.parse_args()
    versions = {n: importlib.metadata.version(n) for n in ["OmniSTEval", "sacrebleu", "jiwer", "PyYAML", "mosestokenizer"]}
    for line in (ROOT / "requirements-evaluation.txt").read_text().splitlines():
        if "==" in line:
            n, v = line.split("==")
            require(versions[n] == v, f"Wrong evaluator version: {n}")
    require(not args.out.exists(), "Evaluation output already exists; choose a new directory")
    m = dataset(args.data, with_audio=False)
    talks = {t["id"]: t for t in m["talks"]}
    ready, missing, plans = [], [], {}
    for device in ("pi", "laptop"):
        root = args.runs / device
        if not (root / "plan.json").exists():
            missing.append({"device": device, "jobs_missing": 48, "reason": "no plan"})
            continue
        plan = read_json(root / "plan.json")
        validate_plan(plan, m)
        require(plan["device"] == device, "Wrong device folder")
        plans[device] = plan
        for job in plan["jobs"]:
            run = completed_attempt(root / "jobs" / job["id"], plan, job, talks[job["talk"]])
            if run is None:
                missing.append({"device": device, "job": job["id"]})
            else:
                ready.append((run, job))
    require(not missing or args.allow_partial, f"Incomplete grid ({len(ready)}/96 runs). Use --allow-partial only for an interim check.")
    require(bool(ready), "No completed, audited runs")
    args.out.mkdir(parents=True)
    report = {"created_utc": utc(), "complete_grid": not missing, "expected_runs": 96, "observed_runs": len(ready),
              "missing": missing, "manifest": m, "plans": plans, "versions": versions, "clock_control": clock_control(),
              "evaluation_source_sha256": {p.name: sha(p) for p in [ROOT / "evaluate_main.py", ROOT / "summarize_main.py", ROOT / "common.py", ROOT / "pilot_audit.py"]},
              "longyaal_cutoff": "actual full WAV end; upstream cutoff sensitivity retained per talk/clock",
              "wer_normalization": "Whole talk: lowercase, delete Unicode punctuation, collapse whitespace; no number normalization.",
              "runs": []}
    for t in m["talks"]:
        ref = args.out / "references" / t["id"]
        ref.mkdir(parents=True)
        for name in ("audio-segments.yaml", "en.txt", "de.txt"):
            shutil.copyfile(args.data / t["reference_dir"] / name, ref / name)
    for run, job in ready:
        key = job["device"] + "/" + job["id"]
        out = args.out / "results" / key
        result, rows = score_one(run, talks[job["talk"]], job, args.out / "references" / job["talk"], out)
        result["key"] = key
        result["segments_file"] = f"results/{key}/segments.json"
        write_json(args.out / result["segments_file"], rows)
        result["segments_sha256"] = sha(args.out / result["segments_file"])
        report["runs"].append(result)
        write_json(args.out / "scores.partial.json", report)
        print(f"{key}: chrF2={result['chrf2']:.2f}; CU/CA={result['longyaal_cu_s']}/{result['longyaal_ca_s']}", flush=True)
    write_json(args.out / "scores.json", report)
    (args.out / "scores.partial.json").unlink()
    from summarize_main import summarize
    summarize(args.out)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}")
