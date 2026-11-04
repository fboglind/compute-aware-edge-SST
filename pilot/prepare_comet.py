#!/usr/bin/env python3
"""Download the pinned COMET checkpoint and encoder tokenizer/config (about 2.34 GB)."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("comet-models"))
    args = parser.parse_args()
    manifests = Path(__file__).resolve().parent / "model-manifests"
    for name in ("wmt22-comet-da", "xlm-roberta-large-tokenizer"):
        manifest = json.loads((manifests / f"{name}.json").read_text())
        root = args.out / name
        root.mkdir(parents=True, exist_ok=True)
        for filename, expected in manifest["files"].items():
            destination = root / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (destination.is_file() and destination.stat().st_size == expected["bytes"]
                    and file_hash(destination) == expected["sha256"]):
                print("Verified cached", name, filename, flush=True)
                continue
            url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{filename}"
            partial = destination.with_name(destination.name + ".part")
            print("Downloading", name, filename, flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
                while block := response.read(8 * 1024 * 1024):
                    stream.write(block)
            if partial.stat().st_size != expected["bytes"] or file_hash(partial) != expected["sha256"]:
                raise ValueError(f"Download size/hash mismatch: {destination}")
            partial.replace(destination)
        (root / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
