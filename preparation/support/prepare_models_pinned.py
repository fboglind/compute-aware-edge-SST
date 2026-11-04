#!/usr/bin/env python3
"""Reproduce the pilot model artifacts at their recorded Hugging Face revisions."""
import argparse
import importlib.metadata
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_helpers import audit_models, read_json, require, sha256, verify_model


def finish(folder, expected):
    actual = {p.relative_to(folder).as_posix(): sha256(p)
              for p in sorted(folder.rglob("*"))
              if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts)}
    require(actual == expected["files_sha256"],
            "New model artifacts differ from the pilot hashes. Existing models were kept. "
            "Reuse the original models/ directory, or investigate the conversion environment before changing the experiment.")
    manifest = {"repo": expected["repo"], "revision": expected["revision"], "kind": expected["kind"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "preparation_versions": {name: importlib.metadata.version(name)
                    for name in ("ctranslate2", "transformers", "torch", "huggingface-hub")},
                "files_sha256": actual}
    (folder / "pilot-model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("pilot-lock.json"))
    args = parser.parse_args()
    lock = read_json(args.lock)
    root = args.models_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, expected in lock["models"].items():
        result = verify_model(root / name, expected)
        print(f"{name}: {result['status']}", flush=True)
        if result["status"] == "missing":
            missing.append(name)
    if not missing:
        print("All three pilot models already exist and match; no downloads or conversion needed.")
        return
    expected_versions = lock["models"]["mt-en-de-int8"]["preparation_versions"]
    for package, version in expected_versions.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = "missing"
        require(installed == version,
                f"Fresh preparation requires {package}=={version}; found {installed}. "
                "Use the conversion-environment commands in the notebook.")
    from huggingface_hub import snapshot_download
    from ctranslate2.converters import TransformersConverter
    from transformers import MarianTokenizer
    for name in missing:
        expected = lock["models"][name]
        print(f"Preparing {expected['repo']} at {expected['revision']}", flush=True)
        with tempfile.TemporaryDirectory(prefix="prepare-pinned-", dir=root) as tmp:
            stage = Path(tmp) / "model"
            if name.startswith("asr-"):
                snapshot_download(expected["repo"], revision=expected["revision"], local_dir=stage,
                                  allow_patterns=list(expected["files_sha256"]))
                shutil.rmtree(stage / ".cache", ignore_errors=True)
            else:
                converter = TransformersConverter(expected["repo"], revision=expected["revision"], trust_remote_code=False)
                converter.convert(str(stage), quantization="int8")
                tokenizer = MarianTokenizer.from_pretrained(expected["repo"], revision=expected["revision"])
                tokenizer.save_pretrained(stage / "tokenizer")
            finish(stage, expected)
            require(not (root / name).exists(), f"Destination appeared during preparation: {root / name}")
            stage.rename(root / name)
            print(f"Verified and saved {name}", flush=True)
    print("All models match the recorded pilot artifacts.")


if __name__ == "__main__":
    main()
