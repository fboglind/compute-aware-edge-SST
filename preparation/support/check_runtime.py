#!/usr/bin/env python3
"""Print a small, shareable runtime report. Does not download models."""
import importlib.metadata
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return (p.stdout or p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def main():
    report = dict(python=sys.version, executable=sys.executable,
                  machine=platform.machine(), bits=struct.calcsize("P") * 8,
                  os=platform.platform(), cpu_count=os.cpu_count(),
                  disk_free_gib=round(shutil.disk_usage(".").free / 2**30, 2))
    for path, key in [("/proc/device-tree/model", "device"),
                      ("/sys/class/thermal/thermal_zone0/temp", "thermal_zone0_millicelsius")]:
        try:
            report[key] = Path(path).read_text().strip("\0\n")
        except OSError:
            pass
    report["throttled"] = command(["vcgencmd", "get_throttled"])
    report["memory"] = command(["free", "-h"])
    report["packages"] = {}
    for name in ["ctranslate2", "faster-whisper", "transformers", "numpy", "av",
                 "onnxruntime", "tokenizers", "sentencepiece", "psutil"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "MISSING"
    try:
        import ctranslate2
        import faster_whisper  # Verify native/transitive imports too.
        from transformers import MarianTokenizer
        supported = sorted(ctranslate2.get_supported_compute_types("cpu"))
        report["cpu_compute_types"] = supported
        report["imports_ok"] = True
        report["int8_ready"] = any(x in supported for x in ("int8", "int8_float32"))
    except Exception as e:
        report["imports_ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(report, indent=2))
    return 0 if report.get("int8_ready") and report["bits"] == 64 else 1


if __name__ == "__main__":
    raise SystemExit(main())
