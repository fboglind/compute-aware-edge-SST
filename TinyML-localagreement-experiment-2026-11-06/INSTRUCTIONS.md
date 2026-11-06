# LocalAgreement experiment operator instructions

Prepared code only. The commands below are for a future experiment session;
preparation did not execute preflight, inference, dataset replay or scoring.
No Jev/API/manual QA stage is needed. Read [PROTOCOL.md](PROTOCOL.md) first.

## 1. Paths and prerequisites

On each device, set its actual project root and device label. All commands below
use absolute paths so they can be run from any directory:

```bash
export TINYML_ROOT=/absolute/path/to/compute-aware-edge-StreamST
export LA="$TINYML_ROOT/TinyML-localagreement-experiment-2026-11-06"
export DEVICE=laptop                 # pi on the Raspberry Pi
export INFER="$TINYML_ROOT/.venv-main/bin/python"
export EVAL="$TINYML_ROOT/.venv-evaluation/bin/python"
export DATA="$TINYML_ROOT/work/data/main"
export RUNS="$TINYML_ROOT/work/localagreement-runs"
export SCORES="$TINYML_ROOT/work/localagreement-evaluation"
```

Use the existing pinned inference environment; on a Pi with `.venv`, set INFER
to that interpreter instead. Requirements files in this kit pin inference,
evaluation and COMET dependencies. If environments/models/data are absent,
follow the environment, pinned-model and six-talk preparation sections of the
project's [REPRODUCE.md](../REPRODUCE.md). This kit adds no model downloads.
DATA must point to the original prepared directory containing manifest.json,
six WAVs and references. Models are under `$TINYML_ROOT/models`.

Copy this entire kit, the sibling `TinyML-main-experiment-2026-10-18/runner`,
the same prepared dataset/manifest and the existing inference models to the Pi,
preserving sibling directory names. Create environments locally on each device;
do not copy virtualenvs between architectures. The smoke directory is not a
runtime dependency. Model manifests and shared source hashes are validated.

Optional code-only regression checks (no model loading or real-time replay):

```bash
cd "$LA"
"$EVAL" -B -m unittest -v test_local_agreement test_experiment
```

## 2. Freeze and inspect a plan (no inference)

```bash
"$INFER" -B "$LA/experiment.py" plan --device "$DEVICE" --data "$DATA" --out "$RUNS/$DEVICE"
"$INFER" -B "$LA/experiment.py" run --device "$DEVICE" --data "$DATA" --out "$RUNS/$DEVICE" --dry-run
```

Expect 48 jobs/device. Source, settings and data are frozen in plan.json. Preserve
the kit after freezing; do not edit code or protocol to resume an existing plan.
Use a new labelled run root for an intentional protocol change.

## 3. Preflight, first retained job, then remaining jobs

Preflight DOES load models and perform synthetic inference. Run it only when
ready to start the experiment session. Record actual power, cooling, profile
and background workload rather than copying the example notes verbatim.

```bash
"$INFER" -B "$LA/experiment.py" run --device "$DEVICE" --data "$DATA"   --models "$TINYML_ROOT/models" --out "$RUNS/$DEVICE"   --notes 'AC power; describe cooling, profile and background workload' --preflight-only

"$INFER" -B "$LA/experiment.py" run --device "$DEVICE" --data "$DATA"   --models "$TINYML_ROOT/models" --out "$RUNS/$DEVICE"   --notes 'Record actual conditions' --max-jobs 1

"$INFER" -B "$LA/experiment.py" run --device "$DEVICE" --data "$DATA"   --models "$TINYML_ROOT/models" --out "$RUNS/$DEVICE"   --notes 'Record actual conditions'
```

The first successful planned talk is retained, not discarded as warm-up. Model
loading and synthetic warm-up occur outside each measured replay. Expect at
least 4.48 hours/device plus overhead, possibly substantially longer for slow
adaptive decoding. Do not compete with heavy background tasks on either host.

Check progress without inference:

```bash
"$INFER" -B "$LA/experiment.py" status --device "$DEVICE" --data "$DATA" --out "$RUNS/$DEVICE"
```

After interruption, repeat the run command. Valid completed runs are audited
and skipped. Complete output left before a parent interruption is recovered.
For an incomplete attempt, inspect console.log/failure.json, resolve the cause,
and add `--retry-incomplete` to retain old evidence and create a new attempt.
Do not delete failed attempts, change plan hashes, or rerun valid slow results.
Changed code requires a separate plan; it cannot silently resume the old one.

## 4. Gather and evaluate offline

Copy both complete device directories, including plan.json, sessions/, jobs/,
all attempts and completion markers, into `$RUNS/pi` and `$RUNS/laptop` on the
scoring host. Preserve filenames and bytes. Keep the same code and prepared
manifest/references there. Scoring never launches ASR/MT inference.
Choose a NEW SCORES path; existing evaluation output is refused.

```bash
"$EVAL" -B "$LA/evaluate_localagreement.py"   --data "$DATA" --runs "$RUNS" --out "$SCORES"

"$TINYML_ROOT/.venv-comet/bin/python" -B "$LA/score_comet_localagreement.py"   --evaluation "$SCORES"   --model-dir "$TINYML_ROOT/models/comet/wmt22-comet-da"   --encoder-dir "$TINYML_ROOT/models/comet/xlm-roberta-large-tokenizer"

"$EVAL" -B "$LA/summarize_localagreement.py" "$SCORES"
```

Reuse the pinned local COMET checkpoint/tokenizer from prior experiments; if
missing, use the existing main prepare_comet.py procedure in REPRODUCE.md.
Use `--allow-partial` on the evaluator only for an explicitly interim report;
the default requires all 96 accepted runs. COMET is pending until its separate
step completes and the summary is refreshed.

Inspect scores.json, summary.json, pass-summary.csv, output-agreement.json and
per-run validation.json. Report corpus quality, both latency clocks and coverage,
processing RTF, resource warnings, EOF words and final tails together. Legacy
`cu` keys denote consumed audio on the measured CA-adaptive trajectory, not an
independent zero-compute simulation. Update waits are not fixed-chunk queues.
Existing main/Whistle baselines stay in their own output directories; do not
feed their raw fixed-chunk logs into this evaluator.
