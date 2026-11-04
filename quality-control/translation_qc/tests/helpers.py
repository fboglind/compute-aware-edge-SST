"""Build complete synthetic run exports without accessing the pilot data or network."""

from translation_qc.common import digest, write_json, write_jsonl
from translation_qc.inputs import AUDIO_SHA256, RUNNER_SHA256, TALK, prepare
from translation_qc.tests.test_inputs import synthetic_chunks


def prepare_synthetic(root):
    runs = root / "runs"
    references = root / "references"
    references.mkdir()
    manifest = {"talk_id": TALK, "audio_duration_s": 50.5, "references": {},
                "prepared_files": {f"{TALK}.wav": {"sha256": AUDIO_SHA256}}}
    for language in ("en", "de"):
        filename = f"reference.{language}.txt"
        (references / filename).write_text("Synthetic reference for tests.\n", encoding="utf-8")
        manifest["references"][language] = {"file": filename}
        manifest["prepared_files"][filename] = {"sha256": digest(references / filename)}
    write_json(references / "manifest.json", manifest)
    for label in ("tiny", "base"):
        for chunk_s in (4, 8):
            run = runs / f"mcif-{TALK}-{label}-{chunk_s}s-mtbeam4-r1"
            run.mkdir(parents=True)
            chunks = synthetic_chunks(chunk_s, 50.5)
            write_jsonl(run / "chunks.jsonl", chunks)
            words = [{"word_index": index, "word": word, "chunk_index": chunk["chunk_index"],
                      "source_consumed_s": chunk["source_end_s"], "emitted_s": chunk["emitted_s"]}
                     for index, (word, chunk) in enumerate((word, chunk) for chunk in chunks for word in chunk["words"])]
            write_jsonl(run / "words.jsonl", words)
            write_jsonl(run / "hypothesis.jsonl", [{"source": f"{TALK}.wav", "source_length": 50500,
                                                     "prediction": " ".join(word["word"] for word in words),
                                                     "delays": [word["source_consumed_s"] * 1000 for word in words],
                                                     "elapsed": [word["emitted_s"] * 1000 for word in words]}])
            for key, filename in (("english", "transcript.en.txt"), ("german", "translation.de.txt")):
                (run / filename).write_text(" ".join(chunk[key] for chunk in chunks) + "\n", encoding="utf-8")
            write_json(run / "summary.json", {"complete": True, "audio_duration_s": 50.5, "chunks": len(chunks),
                                             "asr_model": label + ".en", "chunk_s": chunk_s, "output_words": len(words)})
            write_json(run / "metadata.json", {"args": {"asr": label + ".en", "chunk_s": chunk_s, "threads": 4},
                                              "decoding": {"asr_beam": 1, "mt_beam": 4}, "audio_sha256": AUDIO_SHA256,
                                              "audio_duration_s": 50.5, "script_sha256": {"run_pilot.py": RUNNER_SHA256}})
    out = root / "qc"
    prepare(runs, references, out)
    return out, runs, references
