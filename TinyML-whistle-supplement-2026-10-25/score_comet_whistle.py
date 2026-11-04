#!/usr/bin/env python3
"""Score supplementary Whistle references with pinned COMET; reuse identical triplets exactly."""
import argparse
import importlib.metadata
import math
import os
from pathlib import Path

from common import ROOT, fingerprint, read_json, require, sha, utc, write_json


def build_samples(report, root):
    unique, keys, spans, hashes = [], {}, {}, {}
    for row in report["runs"]:
        path = root / row["segments_file"]
        require(sha(path) == row["segments_sha256"], "Resegmented inputs changed")
        pairs = read_json(path)
        require(len(pairs) == row["quality_segments"], "COMET segment count changed")
        indices = []
        for pair in pairs:
            sample = {"src": pair["src"], "mt": pair["prediction"], "ref": pair["reference"]}
            key = (sample["src"], sample["mt"], sample["ref"])
            if key not in keys:
                keys[key] = len(unique)
                unique.append(sample)
            indices.append(keys[key])
        spans[row["key"]] = indices
        hashes[row["key"]] = row["segments_sha256"]
    return unique, spans, hashes


def load_model(model_dir, encoder_dir):
    import yaml
    from comet.models import RegressionMetric
    manifests = [read_json(root / "download-manifest.json") for root in (model_dir, encoder_dir)]
    for root, manifest, expected_name in zip((model_dir, encoder_dir), manifests,
                                           ("wmt22-comet-da", "xlm-roberta-large-tokenizer")):
        require(manifest == read_json(ROOT / "model-manifests" / (expected_name + ".json")), "COMET manifest differs from pinned pilot model")
        for name, info in manifest["files"].items():
            require(sha(root / name) == info["sha256"], f"Model hash mismatch: {root / name}")
    hp = yaml.safe_load((model_dir / "hparams.yaml").read_text())
    require(hp["class_identifier"] == "regression_metric" and hp["layer_transformation"] == "sparsemax", "Unexpected COMET architecture")
    # Same verified loading method as the retained pilot scorer. The encoder's
    # configuration/tokenizer are local; learned weights come from the checkpoint.
    model = RegressionMetric.load_from_checkpoint(
        checkpoint_path=model_dir / "checkpoints/model.ckpt", map_location="cpu", strict=False,
        load_pretrained_weights=False, pretrained_model=str(encoder_dir.resolve()), local_files_only=True)
    return model, manifests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--encoder-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    require(args.threads > 0 and args.batch_size > 0, "Thread and batch counts must be positive")
    require(not (args.evaluation / "comet-scores.json").exists(), "COMET output already exists; preserve it")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    versions = {}
    for line in (ROOT / "requirements-comet.txt").read_text().splitlines():
        if "==" in line:
            name, version = line.split("==")
            versions[name] = importlib.metadata.version(name)
            require(versions[name] == version, f"Use the pinned COMET environment: {name}=={version}")
    import torch
    import pytorch_lightning as pl
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    pl.seed_everything(0, workers=True)
    report = read_json(args.evaluation / "scores.json")
    samples, spans, hashes = build_samples(report, args.evaluation)
    require(bool(samples), "No COMET inputs")
    model, manifests = load_model(args.model_dir, args.encoder_dir)
    lengths = [len(model.encoder.tokenizer(s[k], truncation=False)["input_ids"]) for s in samples for k in ("src", "mt", "ref")]
    require(max(lengths) <= model.encoder.max_positions, f"COMET would truncate an input ({max(lengths)} tokens); investigate before scoring")
    print(f"Scoring {len(samples)} unique triplets representing {sum(map(len, spans.values()))} segment observations.", flush=True)
    predicted = model.predict(samples, batch_size=args.batch_size, gpus=0, accelerator="cpu", num_workers=0, progress_bar=True)
    values = [float(s) for s in predicted.scores]
    require(len(values) == len(samples) and all(math.isfinite(v) for v in values), "Nonfinite or missing COMET output")
    runs = {key: {"segment_scores": [values[i] for i in indices],
                  "comet": sum(values[i] for i in indices) / len(indices)} for key, indices in spans.items()}
    write_json(args.evaluation / "comet-scores.json", {
        "created_utc": utc(), "scores_json_sha256": sha(args.evaluation / "scores.json"),
        "segments_sha256": hashes, "metric": "Unbabel/wmt22-comet-da", "packages": versions,
        "model": manifests[0], "encoder_config_and_tokenizer": manifests[1],
        "threads": args.threads, "batch_size": args.batch_size, "max_tokens": max(lengths),
        "unique_triplets": len(samples), "total_segment_observations": sum(map(len, spans.values())),
        "deduplication": "Exact identical (official source, hypothesis, reference) triplets share a deterministic score; every segment remains represented in aggregation.",
        "unique_inputs_sha256": fingerprint(samples), "scorer_sha256": sha(Path(__file__)), "runs": runs})
    print("COMET saved. Run summarize_whistle.py in the evaluation environment to update the tables.")


if __name__ == "__main__":
    main()
