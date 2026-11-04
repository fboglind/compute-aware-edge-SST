#!/usr/bin/env python3
"""Score the 39 resegmented pilot pairs for each run with pinned COMET on CPU.

Model preparation instructions are in README.md. This script does not measure
inference speed; Raspberry Pi timings remain those in the original run logs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path(__file__).parent)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import pytorch_lightning as pl
    import yaml
    from comet.models import RegressionMetric

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    pl.seed_everything(0, workers=True)
    model_manifest = json.loads((args.model_dir / "download-manifest.json").read_text())
    encoder_manifest = json.loads((args.encoder_dir / "download-manifest.json").read_text())
    if (model_manifest["repo"], model_manifest["revision"]) != (
        "Unbabel/wmt22-comet-da", "2760a223ac957f30acfb18c8aa649b01cf1d75f2"
    ):
        raise ValueError("Unexpected COMET model revision")
    if (encoder_manifest["repo"], encoder_manifest["revision"]) != (
        "FacebookAI/xlm-roberta-large", "c23d21b0620b635a76227c604d44e43a9f0ee389"
    ):
        raise ValueError("Unexpected encoder/tokenizer revision")
    for root, manifest in [(args.model_dir, model_manifest), (args.encoder_dir, encoder_manifest)]:
        for name, info in manifest["files"].items():
            if sha256(root / name) != info["sha256"]:
                raise ValueError(f"Model file hash mismatch: {name}")
    hp = yaml.safe_load((args.model_dir / "hparams.yaml").read_text())
    if hp["class_identifier"] != "regression_metric" or hp["layer_transformation"] != "sparsemax":
        raise ValueError("Unexpected checkpoint architecture")

    # The same Lightning class loader used by comet.load_from_checkpoint.
    # Only encoder config/tokenizer lookup is redirected to pinned local files;
    # every learned weight and hyperparameter comes from the COMET checkpoint.
    print("Loading pinned COMET checkpoint on CPU...", flush=True)
    model = RegressionMetric.load_from_checkpoint(
        checkpoint_path=args.model_dir / "checkpoints/model.ckpt",
        map_location="cpu", strict=False, load_pretrained_weights=False,
        pretrained_model=str(args.encoder_dir.resolve()), local_files_only=True,
    )
    sources = (args.evaluation / "references/en.txt").read_text(encoding="utf-8").splitlines()
    references = (args.evaluation / "references/de.txt").read_text(encoding="utf-8").splitlines()
    samples, spans, input_hashes = [], [], {}
    for label, seconds in [("tiny", 4), ("tiny", 8), ("base", 4), ("base", 8)]:
        name = f"mcif-JhbtCwcsWY-{label}-{seconds}s-mtbeam4-r1"
        path = args.evaluation / "results" / name / "instances.resegmented.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(sources) or len(rows) != 39:
            raise ValueError(f"Expected 39 aligned pairs: {name}")
        if [row["reference"] for row in rows] != [x.strip() for x in references]:
            raise ValueError(f"Reference order differs: {name}")
        start = len(samples)
        samples.extend({"src": src.strip(), "mt": row["prediction"], "ref": row["reference"]}
                       for src, row in zip(sources, rows))
        spans.append((name, start, len(samples)))
        input_hashes[str(path.relative_to(args.evaluation))] = sha256(path)
    for lang in ("en", "de"):
        path = args.evaluation / "references" / f"{lang}.txt"
        input_hashes[str(path.relative_to(args.evaluation))] = sha256(path)
    lengths = [len(model.encoder.tokenizer(sample[key], truncation=False)["input_ids"])
               for sample in samples for key in ("src", "mt", "ref")]
    max_tokens = max(lengths)
    if max_tokens > model.encoder.max_positions:
        raise ValueError(f"Input would be truncated: {max_tokens} tokens")
    print(f"Scoring {len(samples)} segments; longest input {max_tokens} tokens...", flush=True)
    prediction = model.predict(samples, batch_size=args.batch_size, gpus=0,
                               accelerator="cpu", num_workers=0, progress_bar=True)
    if len(prediction.scores) != len(samples) or not all(math.isfinite(s) for s in prediction.scores):
        raise ValueError("COMET did not return one finite score per input segment")
    runs = []
    for name, start, end in spans:
        scores = prediction.scores[start:end]
        runs.append({"run": name, "segments": len(scores), "comet": sum(scores) / len(scores),
                     "segment_scores": scores})
        print(name, runs[-1]["comet"], flush=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "COMET", "scale": "raw model score, not a percentage",
        "aggregation": "unweighted mean over all 39 reference segments per configuration",
        "source": "official English reference transcript, not the ASR hypothesis",
        "device": "CPU; scoring only, outside the Pi/laptop inference experiment",
        "threads": args.threads, "batch_size": args.batch_size,
        "max_input_tokens_including_special": max_tokens, "truncated_inputs": 0,
        "model": model_manifest, "encoder_config_and_tokenizer": encoder_manifest,
        "packages": {x: importlib.metadata.version(x) for x in
                     ["unbabel-comet", "torch", "transformers", "pytorch-lightning",
                      "torchmetrics", "numpy", "sentencepiece", "huggingface-hub"]},
        "input_sha256": input_hashes, "scorer_sha256": sha256(Path(__file__)), "runs": runs,
    }
    (args.evaluation / "comet-scores.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
