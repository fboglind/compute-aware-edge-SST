#!/usr/bin/env python3
"""Download and verify pinned Whistle weights and both Linux engine artifacts.

Preparation only: this script never loads a model or runs inference.
"""
import argparse
import json
import shutil
import sys
import urllib.request
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runner'))
from whistle_backend import expected_pins, sha, verified_assets


def fetch(repo, revision, filename, destination, expected):
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
    request = urllib.request.Request(url, headers={'User-Agent':'TinyML-Whistle-supplement/1'})
    with urllib.request.urlopen(request, timeout=120) as source, destination.open('wb') as target:
        shutil.copyfileobj(source, target)
    if sha(destination) != expected:
        raise ValueError(f'Download checksum differs: {filename}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('../models/whistle-2026-10-25'))
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        verified_assets(out, all_platforms=True)
        print(f'Existing pinned assets verified: {out}; nothing downloaded.')
        return
    staging = out.with_name(out.name + '.preparing-' + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    pins = expected_pins()
    try:
        weight = pins['weights']
        print('Downloading the pinned 16.9 MB Whistle weights...', flush=True)
        fetch(weight['repo_id'], weight['revision'], weight['filename'],
              staging / 'whistle.cact', weight['sha256'])
        runtime = pins['runtime']
        for arch, record in runtime['platforms'].items():
            print(f'Downloading and checking the {arch} native engine...', flush=True)
            archive = staging / (arch + '.whl')
            fetch(runtime['repo_id'], runtime['revision'], record['filename'],
                  archive, record['wheel_sha256'])
            folder = staging / arch
            folder.mkdir()
            with zipfile.ZipFile(archive) as z:
                (folder / 'libneedle3.so').write_bytes(z.read(runtime['wheel_member']))
            if sha(folder / 'libneedle3.so') != record['library_sha256']:
                raise ValueError(f'Native library checksum differs: {arch}')
            archive.unlink()
        (staging / 'asset-manifest.json').write_text(json.dumps(pins, indent=2) + '\n')
        verified_assets(staging, all_platforms=True)
        staging.rename(out)
        print(f'Prepared {out}; includes both devices. No inference was run.')
    except BaseException:
        print(f'Preparation failed; partial downloads retained at {staging}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
