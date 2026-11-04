#!/usr/bin/env python3
"""Score the four MCIF pilot runs on the laptop, without changing run files.

Install: python -m pip install sacrebleu==2.6.0 jiwer==4.0.0
Run: python score_pilot_grid.py --runs pi-runs \
    --data data/mcif-pilot-JhbtCwcsWY --out pi-runs/mcif-pilot-scores.json

Reports whole-talk chrF2 and a diagnostic English WER. Checks timestamp exports
against chunk logs. Does not compute LongYAAL, COMET or resegmented chrF.
"""
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
            and meta["decoding"]["asr_beam"] == 1, "Unexpected threads or beam settings")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("pi-runs"))
    parser.add_argument("--data", type=Path, default=Path("data/mcif-pilot-JhbtCwcsWY"))
    parser.add_argument("--out", type=Path, default=Path("pi-runs/mcif-pilot-scores.json"))
    args = parser.parse_args()
    try:
        import jiwer
        from sacrebleu.metrics import CHRF
    except ImportError:
        parser.exit(2, "Install scoring dependencies: python -m pip install sacrebleu==2.6.0 jiwer==4.0.0\n")
    manifest = read_json(args.data / "manifest.json")
    references = {}
    for lang in ("en", "de"):
        path = args.data / manifest["references"][lang]["file"]
        require(sha256(path) == manifest["prepared_files"][path.name]["sha256"], "Reference hash mismatch: " + str(path))
        references[lang] = normalized(path.read_text(encoding="utf-8"))
        require(bool(references[lang]), "Empty reference: " + str(path))
    transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                               jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    metric = CHRF(char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False)
    rows, consistency = [], None
    for asr, label in (("tiny.en", "tiny"), ("base.en", "base")):
        for chunk_s in (4, 8):
            run = args.runs / f"mcif-{manifest['talk_id']}-{label}-{chunk_s}s-mtbeam4-r1"
            try:
                summary, meta, english, german, words = audit(run, manifest, asr, chunk_s)
                settings = (meta["script_sha256"], meta["models"]["mt"], meta["decoding"])
                if consistency is None:
                    consistency = settings
                require(settings == consistency, "Scripts, MT model or decoding settings changed across runs")
                wer = jiwer.process_words(references["en"], english,
                                           reference_transform=transform, hypothesis_transform=transform)
                chrf = metric.corpus_score([german], [[references["de"]]])
            except (OSError, KeyError, ValueError, TypeError) as error:
                raise ValueError(f"{run}: {error}") from error
            rows.append({
                "run": str(run), "asr_model": asr, "chunk_s": chunk_s,
                "wer_pct": wer.wer * 100, "wer_substitutions": wer.substitutions,
                "wer_deletions": wer.deletions, "wer_insertions": wer.insertions,
                "wer_reference_words": wer.hits + wer.substitutions + wer.deletions,
                "whole_talk_chrf2": chrf.score,
                "rtf_processing": summary["rtf_processing"],
                "mean_lag_after_chunk_end_s": summary["mean_lag_after_chunk_end_s"],
                "queue_wait_max_s": summary["queue_wait_max_s"], "final_tail_s": summary["final_tail_s"],
                "first_output_s": words[0]["emitted_s"] if words else None,
                "sampled_peak_replay_rss_mib": summary["sampled_peak_replay_rss_mib"],
                "timestamp_export_checks": "passed; chunk, word and hypothesis clocks agree",
                "input_sha256": {p.name: sha256(p) for p in (run / "transcript.en.txt", run / "translation.de.txt",
                                                          run / "metadata.json", run / "summary.json",
                                                          run / "chunks.jsonl", run / "words.jsonl", run / "hypothesis.jsonl")},
            })
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One reserved MCIF pilot talk; one run per configuration",
        "dataset_revision": manifest["dataset_revision"], "talk_id": manifest["talk_id"],
        "reference_sha256": {lang: sha256(args.data / manifest["references"][lang]["file"]) for lang in references},
        "scorer_sha256": sha256(Path(__file__)),
        "packages": {name: importlib.metadata.version(name) for name in ("sacrebleu", "jiwer", "rapidfuzz")},
        "wer_normalization": "Whole talk; collapse whitespace, lowercase, delete Unicode punctuation, split on spaces. No number verbalization or model-specific normalization.",
        "chrf_unit": "One whole-talk hypothesis/reference pair per run; collapse whitespace, preserve case; character order 6, word order 0, beta 2.",
        "chrf_signature": str(metric.get_signature()),
        "limitation": "Pilot diagnostics only. No reference resegmentation, COMET or LongYAAL. Timestamp checks are consistency checks, not validation of the complete latency evaluator.",
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ASR       Chunk   WER % (lower)   chrF2 (higher)   RTF    First output s")
    for row in rows:
        first = f"{row['first_output_s']:.2f}" if row["first_output_s"] is not None else "none"
        print(f"{row['asr_model']:<9} {row['chunk_s']:>3}s    {row['wer_pct']:>10.2f}      {row['whole_talk_chrf2']:>10.2f}     {row['rtf_processing']:.3f}   {first:>8}")
    print("\nAll four runs passed timestamp export consistency checks.")
    print("Scores cover one whole pilot talk; LongYAAL, COMET and resegmented chrF remain pending.")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise SystemExit(f"ERROR: {exc}")
