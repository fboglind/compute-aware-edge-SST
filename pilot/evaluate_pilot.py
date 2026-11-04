#!/usr/bin/env python3
"""Reproduce the four MCIF pilot scores without rerunning ASR or translation."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from omnisteval.data import Instance
from omnisteval.scoring import YAALScorer
from sacrebleu.metrics import CHRF

TALK = "JhbtCwcsWY"
GRID = [("tiny", 4), ("tiny", 8), ("base", 4), ("base", 8)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clock_control():
    x = Instance.from_dict({
        "prediction": "eins zwei drei vier", "reference": "eins zwei drei vier",
        "source_length": 4000, "time_to_recording_end": 20000,
        "emission_cu": [1000, 2000, 3000, 4000], "emission_ca": [1750, 2750, 3750, 4750],
    })
    cu = YAALScorer(is_longform=True)([x])
    ca = YAALScorer(computation_aware=True, is_longform=True)([x])
    require(cu == 1000 and ca == 1750, "Known-clock control failed")
    x.emission_ca = x.emission_cu[:]
    zero = YAALScorer(computation_aware=True, is_longform=True)([x])
    require(zero == cu, "Zero-computation control failed")
    return {"status": "passed", "cu_ms": cu, "ca_with_750ms_compute_ms": ca,
            "zero_compute_ca_ms": zero, "recording_cutoff_ms": 20000}


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=base / "input-runs")
    parser.add_argument("--reference-inputs", type=Path, default=base / "official-reference-inputs")
    parser.add_argument("--original-reference", type=Path, default=base / "original-reference")
    parser.add_argument("--out", type=Path, default=base / "recomputed")
    parser.add_argument("--comet-scores", type=Path)
    args = parser.parse_args()
    versions = {x: importlib.metadata.version(x) for x in
                ["OmniSTEval", "sacrebleu", "jiwer", "PyYAML", "mosestokenizer"]}
    require(versions["OmniSTEval"] == "0.1.10" and versions["sacrebleu"] == "2.6.0",
            "Use the pinned evaluator versions in requirements-evaluation.txt")
    args.out.mkdir(parents=True, exist_ok=True)
    refdir = args.out / "references"
    refdir.mkdir(exist_ok=True)
    full_seg = yaml.safe_load((args.reference_inputs / "audio-segments.yaml").read_text())
    all_text = {lang: (args.reference_inputs / f"{lang}.txt").read_text(encoding="utf-8").splitlines()
                for lang in ("en", "de")}
    require(len(full_seg) == len(all_text["en"]) == len(all_text["de"]), "Reference line counts differ")
    indices = [i for i, seg in enumerate(full_seg) if Path(seg["wav"]).name == TALK + ".wav"]
    require(len(indices) == 39, "Expected the 39 official pilot segments")
    segs = [full_seg[i] for i in indices]
    duration = read_json(args.original_reference / "manifest.json")["audio_duration_s"]
    for seg in segs:
        require(seg["offset"] >= 0 and seg["duration"] > 0
                and seg["offset"] + seg["duration"] <= duration + 1e-6, "Invalid reference timing")
    (refdir / "audio-segments.yaml").write_text(yaml.safe_dump(segs, sort_keys=False), encoding="utf-8")
    for lang, lines in all_text.items():
        chosen = [lines[i] for i in indices]
        original = (args.original_reference / f"reference.{lang}.txt").read_text(encoding="utf-8")
        require(" ".join(" ".join(chosen).split()) == " ".join(original.split()),
                f"Timed {lang} reference differs from the original pilot reference")
        (refdir / f"{lang}.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")
    write_json(refdir / "provenance.json", {
        "talk": TALK + ".wav", "reference_lines_1_based": [i + 1 for i in indices],
        "segments": len(segs), "source_files_sha256": {
            name: digest(args.reference_inputs / name) for name in ["audio-segments.yaml", "en.txt", "de.txt"]},
        "reference_clock": "original full-recording clock; no cropping or offset reset",
    })
    audit = subprocess.run([sys.executable, str(base / "score_pilot_grid.py"), "--runs", str(args.runs),
                            "--data", str(args.original_reference), "--out", str(args.out / "whole-talk-audit.json")],
                           capture_output=True, text=True, check=True)
    (args.out / "whole-talk-audit.stdout.txt").write_text(audit.stdout, encoding="utf-8")
    whole = read_json(args.out / "whole-talk-audit.json")
    comet = read_json(args.comet_scores) if args.comet_scores else None
    comet_rows = {r["run"]: r for r in comet["runs"]} if comet else {}
    results, validation = [], {}
    for (label, chunk_s), whole_row in zip(GRID, whole["runs"]):
        name = f"mcif-{TALK}-{label}-{chunk_s}s-mtbeam4-r1"
        run = args.runs / name
        out = args.out / "results" / name
        command = [sys.executable, "-m", "omnisteval.cli", "longform",
                   "--speech_segmentation", str(refdir / "audio-segments.yaml"),
                   "--ref_sentences_file", str(refdir / "de.txt"),
                   "--hypothesis_file", str(run / "hypothesis.jsonl"),
                   "--source_sentences_file", str(refdir / "en.txt"), "--lang", "de",
                   "--word_level", "--bleu_tokenizer", "13a", "--output_folder", str(out)]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        (out / "stderr.txt").write_text(process.stderr, encoding="utf-8")
        write_json(out / "command.json", command)
        rows = read_jsonl(out / "instances.resegmented.jsonl")
        h = read_jsonl(run / "hypothesis.jsonl")[0]
        require(len(rows) == len(segs), "Resegmentation changed reference segment count")
        words = [word for row in rows for word in row["prediction"].split()]
        require(words == h["prediction"].split(), "Resegmentation lost or changed output words")
        for output_key, input_key in [("emission_cu", "delays"), ("emission_ca", "elapsed")]:
            values = [d + seg["offset"] * 1000 for seg, row in zip(segs, rows) for d in row.get(output_key, [])]
            require(len(values) == len(h[input_key]) and
                    all(abs(a - b) < 1e-6 for a, b in zip(values, h[input_key])), "Timestamp roundtrip failed")
        instances = [Instance.from_dict(row) for row in rows]
        actual_end = [Instance.from_dict(dict(row, time_to_recording_end=h["source_length"] - seg["offset"] * 1000))
                      for row, seg in zip(rows, segs)]
        checks = {"segments": len(rows), "output_words_preserved": len(words),
                  "empty_predictions": sum(not row["prediction"].strip() for row in rows),
                  "timestamp_roundtrip": "passed", "source_length_ms": h["source_length"],
                  "upstream_cutoff_ms": (segs[-1]["offset"] + segs[-1]["duration"]) * 1000}
        for aware, key in [(False, "cu"), (True, "ca")]:
            metric = YAALScorer(computation_aware=aware, is_longform=True)
            values = [metric.compute(ins) for ins in instances]
            checks[key] = {"longyaal_ms": metric(instances),
                           "contributing_segments": sum(v is not None for v in values),
                           "excluded_segment_indices": [i for i, v in enumerate(values) if v is None],
                           "full_wav_cutoff_longyaal_ms": metric(actual_end)}
        chrf = CHRF(char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False)
        score = chrf.corpus_score([row["prediction"] for row in rows], [[row["reference"] for row in rows]])
        summary = read_json(run / "summary.json")
        resources = read_jsonl(run / "resources.jsonl")
        temps = [r["cpu_temp_c"] for r in resources if r.get("cpu_temp_c") is not None]
        result = {"run": name, "asr": label + ".en", "chunk_s": chunk_s,
                  "resegmented_chrf2": score.score, "chrf_signature": str(chrf.get_signature()),
                  "longyaal_cu_s": checks["cu"]["longyaal_ms"] / 1000,
                  "longyaal_ca_s": checks["ca"]["longyaal_ms"] / 1000,
                  "longyaal_compute_increase_s": (checks["ca"]["longyaal_ms"] - checks["cu"]["longyaal_ms"]) / 1000,
                  **{k: whole_row[k] for k in ["wer_pct", "whole_talk_chrf2", "rtf_processing", "first_output_s",
                                              "queue_wait_max_s", "final_tail_s", "sampled_peak_replay_rss_mib"]},
                  "max_sampled_cpu_temp_c": max(temps) if temps else None,
                  "throttled_before": summary["throttled_before"], "throttled_after": summary["throttled_after"]}
        if comet:
            for relative in [f"results/{name}/instances.resegmented.jsonl", "references/en.txt", "references/de.txt"]:
                require(digest(args.out / relative) == comet["input_sha256"][relative], "COMET inputs differ")
            result["comet"] = comet_rows[name]["comet"]
        results.append(result)
        validation[name] = checks
    write_json(args.out / "validation.json", validation)
    write_json(args.out / "clock-control.json", clock_control())
    write_json(args.out / "score-summary.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "one reserved pilot talk; one Pi run per configuration",
        "talk_id": TALK, "audio_duration_s": duration, "quality_segments": 39,
        "latency_unit_in_this_summary": "seconds", "omnisteval_raw_latency_unit": "milliseconds",
        "longyaal_aggregation": "OmniSTEval unweighted mean of contributing reference segments; cutoff excludes final segment",
        "versions": versions, "scorer_sha256": digest(Path(__file__)), "runs": results,
        "limitations": ["Pilot tuning evidence only; reserve this talk from main evaluation.",
                        "Whole-talk and resegmented chrF use different segment units and must be labelled separately.",
                        "LongYAAL excludes emissions at/after its recording-end cutoff; retain final-tail diagnostics.",
                        "Raw OmniSTEval 0.1.10 uses the last reference end as cutoff; the full-WAV sensitivity check is recorded.",
                        "Measured elapsed timestamps are already wall-clock emissions; do not apply --fix_simuleval_emission_ca."]})
    for row in results:
        print(f"{row['asr']:7} {row['chunk_s']}s  chrF2={row['resegmented_chrf2']:.2f}  "
              f"LongYAAL CU/CA={row['longyaal_cu_s']:.3f}/{row['longyaal_ca_s']:.3f}s  "
              f"COMET={row.get('comet', 'pending')}")


if __name__ == "__main__":
    main()
