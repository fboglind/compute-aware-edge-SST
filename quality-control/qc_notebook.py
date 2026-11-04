"""Notebook interface to the unchanged translation_qc CLI.

Each command runs in a fresh process. Requests, caching and evaluation logic
remain in translation_qc; this module handles paths, resuming and display only.
"""

import getpass
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import warnings
from datetime import datetime, timezone


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def table(rows):
    """Display small tables without adding a pandas dependency."""
    from IPython.display import HTML, display
    rows = list(rows)
    if not rows:
        print("No rows available.")
        return
    columns = list(rows[0])
    cell = lambda value: html.escape(str(value)) if value is not None else "—"
    header = "".join(f"<th>{cell(k)}</th>" for k in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell(row.get(k))}</td>" for k in columns) + "</tr>" for row in rows)
    display(HTML(f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"))


class Workflow:
    def __init__(self, package_root, main_dir, pilot_runs, pilot_reference, asr_python,
                 output_root, main_name="main-api-v2", pilot_name="pilot-api-v2"):
        self.root = Path(package_root).expanduser().resolve()
        self.main_dir = Path(main_dir).expanduser().resolve()
        self.pilot_runs = Path(pilot_runs).expanduser().resolve()
        self.pilot_reference = Path(pilot_reference).expanduser().resolve()
        self.asr_python = Path(asr_python).expanduser().absolute()
        self.output_root = Path(output_root).expanduser().resolve()
        if not (self.root / "translation_qc" / "__main__.py").is_file():
            raise ValueError("Keep this notebook, qc_notebook.py and translation_qc/ together in the extracted folder.")
        for name in (main_name, pilot_name):
            if not name or Path(name).name != name or name in (".", ".."):
                raise ValueError("Output names must be single folder names.")
        if main_name == pilot_name:
            raise ValueError("Main and pilot output names must differ.")
        for protected in (self.main_dir, self.pilot_runs, self.pilot_reference):
            if self.output_root.is_relative_to(protected):
                raise ValueError("Choose an output folder outside the experiment inputs.")
        self.main = self.output_root / main_name
        self.pilot = self.output_root / pilot_name

    def run(self, *arguments, python=None, api_key=None):
        args = [str(a) for a in arguments]
        live = "--live-jev" in args
        if live and not api_key:
            raise ValueError("Use live_judge() for explicit paid API execution.")
        command = [str(python or sys.executable), "-m", "translation_qc", *args]
        env = os.environ.copy()
        env.pop("OPENROUTER_API_KEY", None)
        if live:
            env["OPENROUTER_API_KEY"] = api_key
        record_dir = self.output_root / "notebook-invocations"
        record_dir.mkdir(parents=True, exist_ok=True)
        record_path = record_dir / (uuid.uuid4().hex + ".json")
        record = {"started_utc": datetime.now(timezone.utc).isoformat(), "command": command,
                  "cwd": str(self.root), "live_requested": live,
                  "interface_sha256": {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                                       for name in ("qc_notebook.py", "TinyML_translation_QC.ipynb")
                                       if (self.root / name).is_file()}}
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Running {args[0]} ({'paid API' if live else 'offline'})…", flush=True)
        try:
            result = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True)
            record["returncode"] = result.returncode
            if result.returncode:
                message = (result.stderr or result.stdout).strip()
                if api_key:
                    message = message.replace(api_key, "[REDACTED]")
                raise RuntimeError(message)
            return json.loads(result.stdout)
        finally:
            env.pop("OPENROUTER_API_KEY", None)
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            record_path.write_text(json.dumps(record, indent=2) + "\n")

    def check_environment(self):
        if sys.version_info < (3, 10):
            raise RuntimeError("Select a Python 3.10+ notebook kernel.")
        required = (self.main_dir / "runs", self.main_dir / "evaluation-main-fixed" / "scores.json")
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Not found: {path}. Edit MAIN_DIR in the settings cell.")
        if not self.asr_python.is_file():
            raise FileNotFoundError(f"Not found: {self.asr_python}. Set ASR_PYTHON to your evaluation environment's Python.")
        result = subprocess.run([str(self.asr_python), "-c",
                                 'import importlib.metadata; print(importlib.metadata.version("jiwer"))'],
                                capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != "4.0.0":
            raise RuntimeError("ASR_PYTHON must have jiwer==4.0.0. Use the existing evaluation environment.")
        return [{"Setting": "Notebook Python", "Value": sys.executable},
                {"Setting": "ASR Python / jiwer", "Value": str(self.asr_python) + " / 4.0.0"},
                {"Setting": "Main input", "Value": str(self.main_dir)},
                {"Setting": "Main QC output", "Value": str(self.main)},
                {"Setting": "Pilot inputs present", "Value": self.pilot_available()}]

    def pilot_available(self):
        return self.pilot_runs.is_dir() and (self.pilot_reference / "manifest.json").is_file()

    def _resume(self, directory, dataset):
        if not directory.exists():
            return None
        if not (directory / "manifest.json").is_file():
            raise FileExistsError(f"{directory} exists without a QC manifest. Select a fresh output name; nothing was overwritten.")
        manifest = read_json(directory / "manifest.json")
        if manifest.get("dataset", "pilot") != dataset:
            raise ValueError("Existing output is for a different dataset. Select another output name.")
        self.run("validate", "--input", directory)
        return manifest

    def prepare_main(self):
        manifest = self._resume(self.main, "main")
        if manifest:
            sample = read_json(self.main / "sample_manifest.json")
            score_path = str(self.main_dir / "evaluation-main-fixed" / "scores.json")
            run_paths = [Path(r["path"]) for run in manifest["runs"] for r in run["inputs"]]
            if sample["seed"] != 20261022 or sample["per_talk"] != 2:
                raise ValueError("Existing sample uses different settings. Select a fresh MAIN_OUTPUT_NAME.")
            if not any(r["path"] == score_path for r in manifest["reference_inputs"]) or not all(
                    p.is_relative_to(self.main_dir / "runs") for p in run_paths):
                raise ValueError("Existing output uses different input paths. Use its original paths or a fresh output name.")
            print("Reusing validated main output; saved decisions and notes are preserved.")
            return {k: manifest[k] for k in ("audited_runs", "retained_runs", "window_count", "selected_count", "diagnostic_count")}
        return self.run("prepare-main", "--runs-dir", self.main_dir / "runs", "--evaluation",
                        self.main_dir / "evaluation-main-fixed", "--out", self.main,
                        "--per-talk", 2, "--seed", 20261022)

    def prepare_pilot(self):
        if not self.pilot_available():
            print("Pilot inputs unavailable. Edit PILOT_RUNS / PILOT_REFERENCE before the pilot API check.")
            return None
        manifest = self._resume(self.pilot, "pilot")
        if manifest:
            windows = read_jsonl(self.pilot / "windows.jsonl")
            if not all(w["segmentation"]["target_s"] == 24 and w["segmentation"].get("context_s") == 8
                       and w["selection"]["seed"] == 20261018 and w["selection"]["requested_per_config"] == 2 for w in windows):
                raise ValueError("Existing pilot uses different settings. Select a fresh PILOT_OUTPUT_NAME.")
            if not all(Path(r["path"]).is_relative_to(self.pilot_runs)
                       for run in manifest["runs"] for r in run["inputs"]) or not all(
                    Path(r["path"]).is_relative_to(self.pilot_reference) for r in manifest["reference_inputs"]):
                raise ValueError("Existing pilot uses different input paths. Select a fresh output name.")
            print("Reusing validated pilot output.")
            return {"windows": manifest["window_count"], "selected": manifest["selected_count"]}
        result = self.run("prepare", "--runs-dir", self.pilot_runs, "--reference-dir", self.pilot_reference,
                         "--out", self.pilot, "--target-s", 24, "--context-s", 8, "--per-config", 2)
        return {k: result[k] for k in ("windows", "selected", "out")}

    def plan(self, directory, selection="sample", limit=None):
        manifest = read_json(directory / "manifest.json")
        if limit is None:
            limit = manifest["selected_count"] if selection == "sample" else manifest.get("diagnostic_count", 0)
        if not limit:
            print("No windows in this selection.")
            return None
        return self.run("judge", "--input", directory / "windows.jsonl", "--out", directory,
                        "--dry-run", "--selection", selection, "--limit", limit)

    def live_judge(self, directory, *, enabled=False, selection="sample", limit=3,
                   max_requests=3, max_cost_usd=0.01):
        if enabled is not True:
            print("API cell is off. No key requested and no API calls made.")
            return None
        self.run("validate", "--input", directory)
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("OpenRouter API key (hidden): ").strip()
        if not key:
            raise ValueError("No API key supplied; no request made.")
        try:
            return self.run("judge", "--input", directory / "windows.jsonl", "--out", directory,
                            "--live-jev", "--selection", selection, "--limit", limit,
                            "--max-requests", max_requests, "--retries", 0,
                            "--max-cost-usd", max_cost_usd, api_key=key)
        finally:
            key = None

    def report(self, directory):
        self.run("report", "--input", directory, "--thresholds", 0.5, 0.7, 0.9)
        return read_json(directory / "summary.json")


def show_plan(plan):
    if plan is not None:
        table([{"Selection": plan["selection"], "Windows": plan["windows"],
                "New requests": plan["new_requests"], "Cached": plan["cached_successes"],
                "Estimated USD": f"{plan['rough_cost_usd']:.6f}", "API calls made": 0}])


def show_asr(directory):
    data = read_json(directory / "asr_errors.json")
    table({"Configuration": name, "Reference words": row["reference_words"],
           "Substitutions": row["substitutions"], "Deletions": row["deletions"],
           "Insertions": row["insertions"], "WER %": round(row["wer_errors_per_100_reference_words"], 2)}
          for name, row in data["by_configuration"].items())


def show_summary(summary):
    def fraction(rate):
        return f"{rate['numerator']}/{rate['denominator']}" if rate["denominator"] else "Not judged"
    table({"Configuration": name, "Sample windows": row["windows"], "Judged": row["successful"],
           "Failed": row["failed"], "Omission flags / judged": fraction(row["thresholds"]["0.5"]["suggested_categories"]["meaningful_omission"]),
           "Any flags / judged": fraction(row["thresholds"]["0.5"]["candidate_review_rate"])}
          for name, row in summary["sample_by_configuration"].items())
    table({"Selection": label, "Threshold": threshold,
           "Any flags / judged": fraction(result["candidate_review_rate"])}
          for label, subset in (("Fixed sample", summary["review_subset"]), ("Diagnostics", summary["diagnostic_subset"]))
          for threshold, result in subset["thresholds"].items())


def show_window(directory, window_id):
    from IPython.display import HTML, display
    window = next((w for w in read_jsonl(directory / "windows.jsonl") if w["window_id"] == window_id), None)
    if window is None:
        raise ValueError("Unknown window ID; copy one from the sample or diagnostic table.")
    parts = [f"<p><strong>{html.escape(window_id)}</strong></p>"]
    for side in ("before", "focal", "after"):
        chunks = window["chunks"] if side == "focal" else window.get("context_" + side, [])
        parts.append(f"<h4>{side.title()}</h4>")
        if not chunks:
            parts.append("<p>No chunks in this context.</p>")
            continue
        parts.append("<table><thead><tr><th>Chunk ID</th><th>Audio interval (s)</th>"
                     "<th>English ASR</th><th>German MT</th></tr></thead><tbody>")
        for chunk in chunks:
            values = (chunk["chunk_index"],
                      f'{chunk["source_start_s"]:.3f}–{chunk["source_end_s"]:.3f}',
                      chunk["english"], chunk["german"])
            parts.append("<tr>" + "".join(
                f"<td style='vertical-align:top;white-space:pre-wrap'>{html.escape(str(value))}</td>"
                for value in values) + "</tr>")
        parts.append("</tbody></table>")
    display(HTML("".join(parts)))
