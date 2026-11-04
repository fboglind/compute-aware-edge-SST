# Reproduction commands

These are normalized commands for the preserved implementation, assembled from the original guides, notebook calls and saved run records. They are not a verbatim terminal-history transcript. Accepted example commands are preserved in `evidence/*/runs/pi/jobs/*/attempt-*/attempt.json`. The full run and API histories remain in the author's archives.

**Reading the results does not require a new experiment.** The two inspection notebooks and the offline tests below use included local files. The sections below describe a fresh execution; no experiments were run when this package was assembled. New device timings and remote model responses need not be identical to the recorded ones.

## Notebook inspection

From this directory in a Linux terminal:

```bash
python3 -m venv .venv-inspect
.venv-inspect/bin/python -m pip install -r requirements-inspection.txt
.venv-inspect/bin/python -m ipykernel install --user --name tinyml-inspect --display-name "TinyML inspection"
```

Open either notebook in a Jupyter-compatible application and select **TinyML inspection**. If no notebook application is installed, install JupyterLab into this environment with `.venv-inspect/bin/python -m pip install jupyterlab`, then start it with `.venv-inspect/bin/python -m jupyterlab`. Package installation needs network access; subsequent notebook execution uses only included files and needs no API key. Keep the working directory at the project root or the notebook's folder.

The recorded experiments used Linux, Python 3.12.3 on the laptop and 3.13.5 on the Pi. Historical inference/evaluation requirements remain in their respective folders and are separate from this small inspection environment. Other operating systems have not been validated.

## 1. Directories and environments

Start in the project root on the laptop. All blocks are terminal commands. Set the variables in each new terminal:

```bash
TINYML_ROOT="$PWD"
MAIN="$TINYML_ROOT/TinyML-main-experiment-2026-10-18"
WHISTLE="$TINYML_ROOT/TinyML-whistle-supplement-2026-10-25"
mkdir -p "$TINYML_ROOT/work"
```

`work/` holds newly generated runs and evaluations. `evidence/` and `quality-control/recorded/` retain the submitted observations. Original Linux hosts used Python 3.12 on the laptop and Python 3.13 on the Pi; package pins and per-run versions are preserved. A virtual environment is created separately on each architecture.

| Environment | Work | Original requirements |
|---|---|---|
| `.venv-inspect` | Read inspection notebooks | Root `requirements-inspection.txt` |
| `.venv-prepare` | Download/convert the three main model artifacts | `preparation/support/requirements-conversion.txt`, plus pinned CPU PyTorch |
| `.venv-main` | Main ASR/MT inference | Main `requirements-inference.txt` |
| `.venv-whistle` | Whistle/MT inference | Whistle `requirements-inference.txt` |
| `.venv-evaluation` | WER, chrF2, LongYAAL and tests | Main `requirements-evaluation.txt` |
| `.venv-comet` | Local neural COMET metric | Main `requirements-comet.txt` |
| `.venv-qc` | QC CLI/reviewer | `jiwer==4.0.0`; other core QC imports use the standard library |

Do not combine the inference and COMET dependency sets: their NumPy pins differ. Perl is needed by the tokenizer; FFmpeg is needed only if data preparation has to convert audio. Models and the original MCIF audio are external inputs.

## 2. Main model assets — laptop, downloads/conversion

The project reused verified pilot models. For a new installation the pinned preparer reproduces those artifacts and verifies their recorded file hashes:

```bash
python3 -m venv "$TINYML_ROOT/.venv-prepare"
"$TINYML_ROOT/.venv-prepare/bin/python" -m pip install 'torch==2.8.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
"$TINYML_ROOT/.venv-prepare/bin/python" -m pip install \
  -r "$TINYML_ROOT/preparation/support/requirements-conversion.txt"
"$TINYML_ROOT/.venv-prepare/bin/python" "$TINYML_ROOT/preparation/support/prepare_models_pinned.py" \
  --models-dir "$TINYML_ROOT/models" \
  --lock "$TINYML_ROOT/preparation/support/pilot-lock.json"
```

The output folders are `models/asr-tiny.en`, `models/asr-base.en` and `models/mt-en-de-int8`. A hash mismatch is an unresolved preparation difference, not a reason to alter the expected hashes. Whistle reuses the MT model.

## 3. Prepare the six recordings — laptop, no model inference

Obtain the MCIF long-translation data from the dataset source (<https://huggingface.co/datasets/FBK-MT/MCIF). Set `MCIF_ROOT` to the extracted directory containing `audio/`, `ref/` and `audio-segments.yaml`. The fixed talk list is already in `selection.json`; the early preparation notebook's provisional selection is not used.

```bash
MCIF_ROOT="$HOME/Downloads/mcif-long-trans/mcif-long-trans"
python3 -m venv "$TINYML_ROOT/.venv-evaluation"
"$TINYML_ROOT/.venv-evaluation/bin/python" -m pip install -r "$MAIN/requirements-evaluation.txt"
cd "$MAIN"
"$TINYML_ROOT/.venv-evaluation/bin/python" experiment.py prepare \
  --dataset "$MCIF_ROOT" --out "$TINYML_ROOT/work/data/main"
```

Preparation records complete WAV durations, references and hashes in `work/data/main/manifest.json`. The six talks should total 2,018.0365625 seconds. An existing destination is not overwritten. No reference sentence boundaries are used to chunk inference.

Optional offline regression checks, using synthetic fixtures and retained pilot outputs:

```bash
cd "$MAIN"
"$TINYML_ROOT/.venv-evaluation/bin/python" -m unittest -v test_main
cd "$MAIN/runner"
"$TINYML_ROOT/.venv-evaluation/bin/python" -m unittest -v test_timing
```

## 4. Transfer inputs to the Pi — laptop terminal

Set the current SSH target. These commands name their sources explicitly; they never transfer the terminal's current directory implicitly. Transfer before starting either benchmark.

```bash
PI_TARGET='tinyml2026@CURRENT_PI_IP'
ssh "$PI_TARGET" 'mkdir -p "$HOME/Documents/compute-aware-edge-SST/work/data"'
rsync -a --exclude='__pycache__/' "$MAIN/" \
  "$PI_TARGET:Documents/compute-aware-edge-SST/TinyML-main-experiment-2026-10-18/"
rsync -a "$TINYML_ROOT/models/" \
  "$PI_TARGET:Documents/compute-aware-edge-SST/models/"
rsync -a "$TINYML_ROOT/work/data/main/" \
  "$PI_TARGET:Documents/compute-aware-edge-SST/work/data/main/"
```

No virtual environment is transferred. The same audio/model bytes are used on both devices.

## 5. Main inference — separately on each device

Start in that device's project root and set `TINYML_ROOT` and `MAIN` as in section 1. Set `DEVICE=laptop` locally or `DEVICE=pi` after SSH login. Install the matching environment on that host:

```bash
DEVICE=laptop
python3 -m venv "$TINYML_ROOT/.venv-main"
"$TINYML_ROOT/.venv-main/bin/python" -m pip install -r "$MAIN/requirements-inference.txt"
"$TINYML_ROOT/.venv-main/bin/python" -m pip freeze > "$TINYML_ROOT/work/environment-$DEVICE.txt"
cd "$MAIN"
"$TINYML_ROOT/.venv-main/bin/python" experiment.py run \
  --device "$DEVICE" --data "$TINYML_ROOT/work/data/main" --models "$TINYML_ROOT/models" \
  --out "$TINYML_ROOT/work/main-runs/$DEVICE" --preflight-only \
  --notes 'Record the actual power, cooling, profile and background-workload conditions here'
```

Preflight loads the models and checks a silence/MT example; it is not a measured talk. Run one measured recording first, then resume without `--max-jobs 1`:

```bash
"$TINYML_ROOT/.venv-main/bin/python" experiment.py run \
  --device "$DEVICE" --data "$TINYML_ROOT/work/data/main" --models "$TINYML_ROOT/models" \
  --out "$TINYML_ROOT/work/main-runs/$DEVICE" --max-jobs 1 \
  --notes 'Record the actual conditions here'

"$TINYML_ROOT/.venv-main/bin/python" experiment.py run \
  --device "$DEVICE" --data "$TINYML_ROOT/work/data/main" --models "$TINYML_ROOT/models" \
  --out "$TINYML_ROOT/work/main-runs/$DEVICE" \
  --notes 'Record the actual conditions here'

"$TINYML_ROOT/.venv-main/bin/python" experiment.py status \
  --device "$DEVICE" --data "$TINYML_ROOT/work/data/main" --out "$TINYML_ROOT/work/main-runs/$DEVICE"
```

Expected completion is 48/48 per device. Audio replay alone takes approximately 4.48 hours per device, with loading/warm-up/tails in addition. The devices may run concurrently, with one measured job at a time per host. Keep the recorded power/cooling/profile conditions stable and avoid competing heavy work. A `tmux` session on each host preserves the shell across disconnection; suspend must be prevented separately.

Resume uses the same run command and verifies completed jobs before skipping them. An incomplete attempt remains on disk; after inspecting its logs, `--retry-incomplete` permits a new attempt. No source, manifest or successful result is rewritten to resume.

Collect Pi runs on the laptop after both batches finish:

```bash
mkdir -p "$TINYML_ROOT/work/main-runs/pi"
rsync -a "$PI_TARGET:Documents/compute-aware-edge-SST/work/main-runs/pi/" \
  "$TINYML_ROOT/work/main-runs/pi/"
```

## 6. Main metrics and COMET — laptop, outside benchmark timing

Use the corrected adapter by its absolute path **while the working directory is the main experiment folder**. Keep the adapter outside that folder: the frozen source audit includes its top-level Python files.

```bash
cd "$MAIN"
"$TINYML_ROOT/.venv-evaluation/bin/python" "$TINYML_ROOT/evaluation/evaluate_main_fixed.py" \
  --data "$TINYML_ROOT/work/data/main" --runs "$TINYML_ROOT/work/main-runs" \
  --out "$TINYML_ROOT/work/main-evaluation"
```

This produces WER, chrF2, CU/CA latency, resource diagnostics, coverage and per-run/per-pass outputs. Evaluation requires all 96 accepted runs. A new output directory is required.

COMET is a separate local model evaluation and downloads approximately 2.34 GB of assets if unavailable:

```bash
python3 -m venv "$TINYML_ROOT/.venv-comet"
"$TINYML_ROOT/.venv-comet/bin/python" -m pip install 'torch==2.8.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
"$TINYML_ROOT/.venv-comet/bin/python" -m pip install -r "$MAIN/requirements-comet.txt"
"$TINYML_ROOT/.venv-comet/bin/python" "$MAIN/prepare_comet.py" --out "$TINYML_ROOT/models/comet"
"$TINYML_ROOT/.venv-comet/bin/python" "$MAIN/score_comet_main.py" \
  --evaluation "$TINYML_ROOT/work/main-evaluation" \
  --model-dir "$TINYML_ROOT/models/comet/wmt22-comet-da" \
  --encoder-dir "$TINYML_ROOT/models/comet/xlm-roberta-large-tokenizer"
"$TINYML_ROOT/.venv-evaluation/bin/python" "$MAIN/summarize_main.py" "$TINYML_ROOT/work/main-evaluation"
```

For the recorded observations, `analysis/main/data/` contains the final main scores, COMET results and summary. `evidence/main/evaluation-main-fixed/` contains the six reference extracts, pass summary and one resegmented example. Rebuilding report tables from the saved analysis inputs does not require COMET inference; re-running the full evaluator or summary builder requires the full exports or fresh results.

## 7. Whistle supplement — separate runs

Prepare the pinned native assets on the laptop; this downloads weights/libraries but does not run inference:

```bash
python3 "$WHISTLE/prepare_whistle.py" --out "$TINYML_ROOT/models/whistle-2026-10-25"
rsync -a --exclude='__pycache__/' "$WHISTLE/" \
  "$PI_TARGET:Documents/compute-aware-edge-SST/TinyML-whistle-supplement-2026-10-25/"
rsync -a "$TINYML_ROOT/models/whistle-2026-10-25/" \
  "$PI_TARGET:Documents/compute-aware-edge-SST/models/whistle-2026-10-25/"
```

On each device, set its `TINYML_ROOT`, `WHISTLE` and `DEVICE` again. Install `.venv-whistle` locally and run the same preflight/first-recording/full-batch sequence:

```bash
python3 -m venv "$TINYML_ROOT/.venv-whistle"
"$TINYML_ROOT/.venv-whistle/bin/python" -m pip install -r "$WHISTLE/requirements-inference.txt"
cd "$WHISTLE"
"$TINYML_ROOT/.venv-whistle/bin/python" -m unittest -v test_whistle
"$TINYML_ROOT/.venv-whistle/bin/python" experiment.py run \
  --device "$DEVICE" --data "$TINYML_ROOT/work/data/main" --models "$TINYML_ROOT/models" \
  --whistle-assets "$TINYML_ROOT/models/whistle-2026-10-25" \
  --out "$TINYML_ROOT/work/whistle-runs/$DEVICE" --preflight-only --notes 'Record actual conditions'
```

Repeat the last command with `--max-jobs 1` instead of `--preflight-only`, inspect the completed recording, then repeat without either limiting flag. Expected completion is 24/24 per device, with approximately 2.24 hours of audio replay per device plus overhead. Status uses:

```bash
"$TINYML_ROOT/.venv-whistle/bin/python" experiment.py status \
  --device "$DEVICE" --data "$TINYML_ROOT/work/data/main" --out "$TINYML_ROOT/work/whistle-runs/$DEVICE"
```

Back on the laptop, collect Pi results and evaluate:

```bash
mkdir -p "$TINYML_ROOT/work/whistle-runs/pi"
rsync -a "$PI_TARGET:Documents/compute-aware-edge-SST/work/whistle-runs/pi/" \
  "$TINYML_ROOT/work/whistle-runs/pi/"
"$TINYML_ROOT/.venv-evaluation/bin/python" "$WHISTLE/evaluate_whistle.py" \
  --data "$TINYML_ROOT/work/data/main" --runs "$TINYML_ROOT/work/whistle-runs" \
  --out "$TINYML_ROOT/work/whistle-evaluation"
"$TINYML_ROOT/.venv-comet/bin/python" "$WHISTLE/score_comet_whistle.py" \
  --evaluation "$TINYML_ROOT/work/whistle-evaluation" \
  --model-dir "$TINYML_ROOT/models/comet/wmt22-comet-da" \
  --encoder-dir "$TINYML_ROOT/models/comet/xlm-roberta-large-tokenizer"
"$TINYML_ROOT/.venv-evaluation/bin/python" "$WHISTLE/summarize_whistle.py" "$TINYML_ROOT/work/whistle-evaluation"
```

The original supplement is already complete. Its first laptop recording's AC-power deviation remains documented in `evidence/whistle/audit/closure.json`; it was retained, not replaced. No Jev or detailed-review stage applies to Whistle.

## 8. QC preparation and saved records — laptop

For inspection, use the offline QC notebook. Original manifests deliberately retain historical paths; do not rewrite them to resume elsewhere. To reconstruct windows, first complete the fresh main run and evaluation in sections 5–6 (or restore the full archived exports). The included example runs alone are insufficient for the 96-job QC audit. Prepare a **new** output directory:

```bash
python3 -m venv "$TINYML_ROOT/.venv-qc"
"$TINYML_ROOT/.venv-qc/bin/python" -m pip install jiwer==4.0.0
cd "$TINYML_ROOT/quality-control"
"$TINYML_ROOT/.venv-qc/bin/python" -m translation_qc prepare-main \
  --runs-dir "$TINYML_ROOT/work/main-runs" \
  --evaluation "$TINYML_ROOT/work/main-evaluation" \
  --out "$TINYML_ROOT/work/qc-main" --per-talk 2 --seed 20261022
"$TINYML_ROOT/.venv-qc/bin/python" -m translation_qc asr-errors --input "$TINYML_ROOT/work/qc-main"
"$TINYML_ROOT/.venv-qc/bin/python" -m translation_qc judge \
  --input "$TINYML_ROOT/work/qc-main/windows.jsonl" --out "$TINYML_ROOT/work/qc-main" \
  --dry-run --selection sample --limit 96
```

These commands perform no model/API evaluation. For restored archived exports, substitute their complete run/evaluation directories. The QC audit requires the same talk set and text-identical repetitions; it stops if those prerequisites do not hold.

## 9. Optional fresh API evaluation — incurs charges

The repository includes the recorded outcomes and selected failed responses, so repeating API requests is unnecessary for inspection and will not guarantee identical results. The following commands document the original execution method. Model/service availability and prices may change; the code's historical cost estimates are not billing guarantees.

In the same QC terminal, enter a key without writing it into a notebook or shell-history command:

```bash
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
```

The original workflow first checked three pilot requests; the retained pilot request/response files are under `quality-control/recorded/pilot-api-v2/`. The main and diagnostic calls used:

```bash
"$TINYML_ROOT/.venv-qc/bin/python" -m translation_qc judge \
  --input "$TINYML_ROOT/work/qc-main/windows.jsonl" --out "$TINYML_ROOT/work/qc-main" \
  --live-jev --selection sample --limit 96 --max-requests 100 --retries 0 --max-cost-usd 0.05
"$TINYML_ROOT/.venv-qc/bin/python" -m translation_qc judge \
  --input "$TINYML_ROOT/work/qc-main/windows.jsonl" --out "$TINYML_ROOT/work/qc-main" \
  --live-jev --selection diagnostic --limit 6 --max-requests 6 --retries 0 --max-cost-usd 0.01
"$TINYML_ROOT/.venv-qc/bin/python" -m translation_qc report \
  --input "$TINYML_ROOT/work/qc-main" --thresholds 0.5 0.7 0.9
```

The detailed reviewer is called through `qc_review.py`, with the same explicit controls as the original notebook. After Jev results exist, this terminal block selects cases and previews the requests without sending them:

```bash
export TINYML_ROOT
"$TINYML_ROOT/.venv-qc/bin/python" - <<'PY'
import os
from pathlib import Path
import qc_review as review
root = Path(os.environ['TINYML_ROOT'])
source = root / 'work/qc-main'
target = source / 'api-review-v1'
print(review.prepare(source, target, review.Settings()))
print(review.preview(target))
PY
```

Explicit execution, initially limited to three requests:

```bash
"$TINYML_ROOT/.venv-qc/bin/python" - <<'PY'
import os
from pathlib import Path
import qc_review as review
target = Path(os.environ['TINYML_ROOT']) / 'work/qc-main/api-review-v1'
print(review.run(target, enabled=True, max_requests=3, max_cost_usd=0.50))
PY
```

After inspection, the original full-batch call used `max_requests=36, max_cost_usd=5.00`. Successful requests are reused. Failures stop the batch and are preserved; `retry_failed=True` is an explicit retry decision, not the default. No retry or repair was performed merely to package the repository. `review.report(target)` regenerates comparisons for a freshly prepared local review directory without API calls.

Remove the key from that shell when finished:

```bash
unset OPENROUTER_API_KEY
```

## 10. Rebuild report outputs from saved metrics

The report builder does not load ASR, MT or COMET. It checks saved input hashes and generates tables and figures with NumPy/Matplotlib. Use a copy so the recorded artifacts remain intact:

```bash
"$TINYML_ROOT/.venv-inspect/bin/python" -m pip install numpy matplotlib
test ! -e "$TINYML_ROOT/work/report-rebuild" && \
  cp -a "$TINYML_ROOT/analysis/main" "$TINYML_ROOT/work/report-rebuild"
"$TINYML_ROOT/.venv-inspect/bin/python" "$TINYML_ROOT/work/report-rebuild/build_report_assets.py"
```

The output is in `work/report-rebuild/tables/` and `figures/`. The Whistle result table is displayed directly from the supplied summary in the pipeline walkthrough. The full Whistle report handoff was supplied separately; this archive focuses on code and its supporting evidence.

## 11. Retained development-pilot evaluator

The `pilot/` directory contains the earlier four-configuration Pi pilot, including its exact inference source and saved metric outputs. Its reserved talk is excluded from the main experiment. To recompute the pilot's classical metrics and import the already recorded COMET scores without neural inference, use a new output directory:

```bash
cd "$TINYML_ROOT/pilot"
"$TINYML_ROOT/.venv-evaluation/bin/python" evaluate_pilot.py \
  --out "$TINYML_ROOT/work/pilot-evaluation" --comet-scores comet-scores.json
```

The original pilot COMET scorer and model preparation script are retained here as well. They are not required to inspect the supplied scores. Initial smoke checks were development checks, not additional main-test observations.


## 12. LocalAgreement smoke test

The [LocalAgreement guide](TinyML-localagreement-smoke-2026-10-26/README.md) documents the separate prototype. Four measured smoke runs and the English/German reference transcripts are included. The original 23.6146875-second WAV is an external input; its expected hash is retained in each run's metadata and smoke-check record.

## 13. Offline tests

After creating the evaluation environment in section 3, run from the project root:

```bash
TINYML_ROOT="$PWD"
cd "$TINYML_ROOT/TinyML-main-experiment-2026-10-18"
"$TINYML_ROOT/.venv-evaluation/bin/python" -B -m unittest -v test_main
cd runner
"$TINYML_ROOT/.venv-evaluation/bin/python" -B -m unittest -v test_timing
cd "$TINYML_ROOT/TinyML-whistle-supplement-2026-10-25"
"$TINYML_ROOT/.venv-evaluation/bin/python" -B -m unittest -v test_whistle
cd "$TINYML_ROOT/TinyML-localagreement-smoke-2026-10-26"
"$TINYML_ROOT/.venv-evaluation/bin/python" -B -m unittest -v test_local_agreement test_online
cd "$TINYML_ROOT/quality-control"
"$TINYML_ROOT/.venv-evaluation/bin/python" -B -m unittest discover -s translation_qc/tests -t .
"$TINYML_ROOT/.venv-evaluation/bin/python" -B -m unittest discover -s notebook_tests
"$TINYML_ROOT/.venv-evaluation/bin/python" -B -m unittest discover -s review_tests
```

These tests use temporary files, retained pilot outputs and synthetic clocks/API responses. They do not run models or send API requests. Notebook execution needs the separate [inspection environment](#notebook-inspection) described above.
