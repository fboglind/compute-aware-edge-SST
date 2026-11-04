"""Deterministic whole-talk WER decomposition matching the existing main scorer."""

import importlib.metadata

from .common import digest, read_json, require, write_json, write_jsonl
from .inputs import validate


def analyze_asr(directory):
    windows, _, manifest = validate(directory)
    require(manifest.get("dataset") == "main", "ASR decomposition currently requires prepare-main")
    try:
        import jiwer
    except ImportError as error:
        raise ValueError("Run asr-errors with the existing .venv-evaluation Python (requires jiwer==4.0.0)") from error
    require(importlib.metadata.version("jiwer") == "4.0.0", "Use pinned jiwer==4.0.0 to match the main scorer")
    reference = read_json(directory / "reference_evidence.json")["talks"]
    scores_record = next(r for r in manifest["reference_inputs"] if r["path"].endswith("/scores.json"))
    scores = {r["key"]: r for r in read_json(scores_record["path"])["runs"]}
    transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
                               jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    by_run = {}
    for w in windows:
        by_run.setdefault(w["run_id"], []).append(w)
    results, alignments, totals = [], [], {}
    for run_id, group in sorted(by_run.items()):
        group.sort(key=lambda w: w["ordinal"])
        talk = group[0]["talk_id"]
        ref = " ".join(reference[talk]["en"].split())
        hyp = " ".join(" ".join(w["source_en"] for w in group).split())
        scored = jiwer.process_words(ref, hyp, reference_transform=transform, hypothesis_transform=transform)
        n = scored.hits + scored.substitutions + scored.deletions
        errors = scored.substitutions + scored.deletions + scored.insertions
        require(errors == scores[run_id]["wer_errors"] and n == scores[run_id]["wer_reference_words"]
                and abs(100 * scored.wer - scores[run_id]["wer_pct"]) < 1e-9,
                f"WER does not reproduce existing evaluation: {run_id}")
        row = {"run_id": run_id, "talk_id": talk, "configuration": group[0]["config"],
               "reference_words": n, "hits": scored.hits, "substitutions": scored.substitutions,
               "deletions": scored.deletions, "insertions": scored.insertions,
               "wer_errors": errors, "wer_pct": 100 * scored.wer,
               "matched_original_score": True}
        results.append(row)
        aggregate = totals.setdefault(row["configuration"], {k: 0 for k in
                           ("talks", "reference_words", "hits", "substitutions", "deletions", "insertions", "wer_errors")})
        aggregate["talks"] += 1
        for k in aggregate:
            if k != "talks":
                aggregate[k] += row[k]
        # Whole-talk word alignment: indexes are normalized word indexes, not audio timestamps.
        alignments.append({"run_id": run_id, "reference_words": scored.references[0],
                           "hypothesis_words": scored.hypotheses[0], "operations": [
                               {"type": a.type, "ref_start": a.ref_start_idx, "ref_end": a.ref_end_idx,
                                "hyp_start": a.hyp_start_idx, "hyp_end": a.hyp_end_idx}
                               for a in scored.alignments[0]]})
    for aggregate in totals.values():
        for name in ("substitutions", "deletions", "insertions", "wer_errors"):
            aggregate[name + "_per_100_reference_words"] = 100 * aggregate[name] / aggregate["reference_words"]
    output = {"label_source": "deterministic edit alignment, no model judge", "jiwer_version": "4.0.0",
              "normalization": "Whole talk: lowercase, delete Unicode punctuation, collapse whitespace; no number normalization",
              "scored_inputs_sha256": scores_record["sha256"], "prepared_manifest_sha256": digest(directory / "manifest.json"),
              "retained_repetition": 1, "runs": results, "by_configuration": totals,
              "interpretation": "WER deletions are token-alignment edits, not semantic translation omissions. No audio was listened to."}
    write_json(directory / "asr_errors.json", output)
    write_jsonl(directory / "asr_alignments.jsonl", alignments)
    return {"runs": len(results), "original_scores_reproduced": len(results), "network_requests": 0,
            "outputs": ["asr_errors.json", "asr_alignments.jsonl"]}
