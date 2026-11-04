# Reading the implementation

The core flow is: **English audio → automatic speech recognition (ASR) → English text → machine translation (MT) → German text → saved output and timing are evaluated**. The two models run sequentially, forming a **cascade**. The main experiment varies Whisper size and chunk duration across two devices. The quality control (QC) extension evaluates MT fidelity to the ASR text; the Whistle supplement changes the ASR system.

Three units serve different purposes: a **chunk** is a 4- or 8-second audio processing block; a **reference segment** is a supplied comparison unit used for scoring; a **QC window** is a 24-second passage inspected with neighboring context. Reference boundaries do not determine processing chunks. A **reference** is supplied comparison text; a **prediction** (also called a hypothesis) is system output.

The nine places below provide a useful reading order. Supporting setup and reporting files can be consulted when needed.

| Read | File or function | What it establishes |
|---|---|---|
| 1 | `TinyML-main-experiment-2026-10-18/selection.json` and `common.py:jobs_for` | Six fixed talks; model/chunk/device/pass grid and reversed second-pass order |
| 2 | `runner/stream_timing.py:replay` in the main folder | When audio becomes available, how late chunks queue, and when output is committed |
| 3 | `runner/run_pilot.py:Cascade` in the main folder | Actual ASR/MT settings and timed processing |
| 4 | `experiment.py:run_batch` in the main folder | Sequential jobs, recorded commands, completion checks and resumption |
| 5 | `evaluation/evaluate_main_fixed.py` and main `evaluate_main.py` | Empty-output repair, reference resegmentation and metric computation |
| 6 | Main `summarize_main.py` | Different pooling rules for WER, RTF, latency and translation metrics |
| 7 | `quality-control/translation_qc/main_experiment.py`, `jev.py`, and `quality-control/qc_review.py` | QC sample, API requests, review selection, validation and failures |
| 8 | Whistle `runner/whistle_backend.py` and `runner/run_whistle.py` | Native ASR integration and how it differs from the main cascade |
| 9 | `TinyML-localagreement-smoke-2026-10-26` | Exploratory streaming-policy prototype. Run instructions and recorded smoke-test results |

## 1. Selection and repeated measurements

The six full recordings contain 33.63 minutes of unique audio. Reference boundaries are used for evaluation, not to reveal sentence boundaries to the inference system. Main configurations are tiny.en/base.en × 4/8 seconds; each runs twice on each device. This gives 6 × 4 × 2 × 2 = 96 jobs. The pilot recording is excluded from this set.

The second pass reverses configuration and recording order. Repeated text outputs matched within each device/configuration, but processing times varied. Two passes provide timing observations, not twelve independent talks.

## 2. The replay clock

`replay` establishes one origin using a monotonic clock. A chunk ending at audio time 8 seconds cannot be processed before that time. If the previous chunk is still running, it waits. Every chunk is processed; late chunks are not dropped. German output for a chunk becomes available atomically after MT completes.

For a saved chunk:

- `source_end_s` is when its incoming audio is complete.
- `processing_start_s` is when processing actually starts.
- `emitted_s` is when its translation is committed.
- `process_s = emitted_s - processing_start_s`.
- `queue_wait_s = processing_start_s - source_end_s`, allowing tiny floating-point tolerances.

Sleeping for another full chunk duration after every inference would incorrectly add computation to the audio-arrival schedule. The single-clock implementation avoids that error.

## 3. The cascade and settings

`Cascade.transcribe` calls faster-whisper with English forced, beam 1, temperature 0, no external voice activity detection (VAD) and no conditioning on previous text. **Beam** is the number of candidate text sequences retained during decoding: beam 1 keeps one, while beam 4 explores four. The returned segment iterator is consumed inside the timed call: faster-whisper performs work lazily, so merely creating the iterator would undercount ASR time.

`Cascade.translate` tokenizes the English ASR, runs Marian through CTranslate2, and decodes the German output. **Tokenization** converts text into model units, which may be words or parts of words; decoding converts output tokens back into readable text. Empty ASR gives empty MT. Inputs beyond the specified token bound are rejected rather than silently truncated; reaching the MT output limit is logged.

The experiment wrapper explicitly supplies four threads and MT beam 4. The low-level pilot runner retains an older default of MT beam 1; calling that runner without the experiment's arguments would not reproduce the main experiment. Effective compute type was checked as `int8_float32`: 8-bit integer quantization with 32-bit floating-point operations for the unquantized parts. **Warm-up** means preliminary model calls to prepare execution before measurement. Loading and warm-up are outside the replay processing measure.

## 4. Jobs, logs and resumption

The runner stores a fixed plan and hashes of source, inputs and outputs. An accepted job has an `attempt.json`, output files, an `exit.json` and a `complete.json` marker. Resume checks the accepted output before skipping it. Partial attempts are preserved in the separate supporting archive; they were not accepted as completed measurements. The repository keeps one complete accepted example from each of the main and Whistle experiments, with the main example used in the walkthrough.

The measured runner, evaluation and QC implementations are retained without refactoring. Historical absolute command paths inside saved records describe the original execution environment.

## 5. Evaluation and the empty-output correction

**Word error rate (WER)** compares the recognized English against normalized English references: (substitutions + deletions + insertions) / reference-word count. Lower is better. One substitution, one deletion and one insertion against ten reference words gives 30% WER. A word deletion in this text alignment is not automatically a meaningful translation omission: that requires assessing what information the translation preserves.

**chrF2** measures character-sequence overlap with the German reference. **COMET** is a learned translation-quality metric using source text, translation and reference; here its source is the official English reference, not the ASR transcript. Higher is better for both; neither is a percentage of correctly translated meaning. OmniSTEval resegments the continuous output into units for reference-based evaluation while retaining emission timestamps.

Some reference segments received no translated words, so their output timestamp lists should be empty. The original evaluator instead encountered missing lists for these empty predictions. Each list contains one timestamp per output word: computation-unaware (CU) timing uses the chunk's audio end; computation-aware (CA) timing uses actual output availability, including processing and queueing. `evaluate_main_fixed.py` supplies empty lists only when the prediction has zero words. It still rejects nonempty text without valid timestamps. Empty translations remain in quality scoring; unavailable latency is not replaced with zero. The adapter remains in its original separate location under `evaluation/`.

## 6. Processing and latency are different quantities

**Real-time factor (RTF)** divides processing seconds by audio seconds and excludes deliberate audio-arrival waits: 2 seconds of processing for 10 seconds of audio gives 0.2. Stage RTFs use the corresponding ASR or MT time. **LongYAAL** is the translation-delay metric calculated from output timing and reference-based evaluation; lower means less measured delay. It is measured in seconds and is distinct from processing RTF.

For the walkthrough's first chunk, audio spans 0–4 seconds and German output becomes available at about 5.584 seconds. Every German word in that chunk receives CU time 4.000 and CA time 5.584 because the words are released together. These lists of word timestamps are inputs to LongYAAL, not the final scores. Both clocks include waiting for incoming audio. Eligibility can differ at recording end, so CA − CU is not pure computation time and CA is not CU + RTF.

Within each pass, WER pools error/reference-word counts; RTF pools processing/audio duration; chrF2 is corpus-level; COMET averages segments; LongYAAL averages eligible segments. The summary then reports means and observed ranges over two passes. An unweighted average of recording WERs or RTFs would answer a different question.

## 7. The exploratory QC stage

`prepare_main` verifies the 96 main jobs and retains one of each identical-text repetition. It forms aligned 24-second passages with neighboring context. Two passages per talk, selected with seed 20261022, produce 96 configuration outputs; six targeted diagnostics are kept separate.

Jev (Typesafe's `typesafe/jev-1.13-20260917`) is the first model screen and returns category scores used to flag candidates. The later detailed review uses Anthropic's Claude Sonnet 4.5 (`anthropic/claude-sonnet-4.5`) to provide category verdicts and quoted evidence. Its selection was designed after seeing Jev's results: 16 high-score, eight borderline, six unflagged and six diagnostic cases. Source/target quotations are checked against the supplied text. A well-formed HTTP response can therefore still fail local validation.

Saved decisions, final detailed-review outcomes and four failed response examples remain in `quality-control/recorded/`; full invocation and attempt histories are in the supporting archive. The 33 successful reviews do not turn this selected sample into an accuracy, recall or population error-rate estimate. The author's limited German inspection is recorded separately from model judgments. Neither stage supplies expert fluency validation.


## 8. Whistle

Whistle replaces the ASR component while retaining outer chunking and the Marian MT stage. Its native C library is called in the same process via `ctypes`; native ASR beam/thread controls are not exposed through the selected API. Compiled defaults and shipped quantization are retained. This is a system comparison, not an architecture-only ablation.

The supplement has 24 jobs per device and automated evaluation only. One first-pass laptop job began without AC power; it remains in the reported results with the deviation documented. High Pi MT time and higher 8-second laptop ASR cost were observed but not causally resolved.

## A short review session

Read the two inspection notebooks alongside items 2, 3, 5 and 7 above. The useful outcome is being able to explain one saved chunk's timestamps, why WER and RTF use different denominators, what the evaluator repair changes, and why the QC reviewer is not ground truth. The remaining files support those core operations.


## Saved report inputs

`analysis/main/build_report_assets.py:build` reads `main-scores.json`, `main-summary.json`, `comet-scores.json`, `asr-errors.json`, `run-audit.json` and `output-agreement.json` from its `data/` folder. The timing extract is needed for the queue-recovery figure and operational table; it is not a packaging check. `collect_run_audit.py` is the original extractor and requires the full raw main runs from the supporting archive if used again. Building report assets from the included extract does not require those raw runs.

`pilot/plot_results.py` reads the retained `pilot/score-summary.json`; pilot scoring sources and their input runs, references and resegmented outputs are retained. Whistle tables in the walkthrough read `evidence/whistle/whistle-summary.json`. The laptop timeline and closure record preserve the observed power-condition deviation.

The QC notebook reads saved results directly. Its `review_outcomes.jsonl` contains the final record for each of 36 cases (33 successful, three failed); `failed_review_examples.jsonl` preserves all four failed responses. There were 37 attempts because one failed case was tried twice. The unchanged experimental reviewer expects its original complete attempt log when resuming or rebuilding its report; that log is in the supporting archive. The notebook does neither operation.

The author's original manual-inspection note is retained unchanged. The later handoff records the fourth case as uncertain with unclear English and corrects its window identifier to `ERmKpJPPDc/laptop/ERmKpJPPDc-tiny-4s-r1/w005/120.000000-144.000000`. These observations remain separate from accepted API judgments.


## Repository layout

| Location | Contents |
|---|---|
| `TinyML-main-experiment-2026-10-18/` | Main runner, scheduler, evaluation, configurations and test fixtures |
| `evaluation/` | Corrected adapter for empty resegmented predictions |
| `TinyML-localagreement-smoke-2026-10-26/` | LocalAgreement prototype, tests, references and four recorded smoke runs |
| `TinyML-whistle-supplement-2026-10-25/` | Whistle integration, experiment code, configuration and licences |
| `pilot/` | Pilot code, references, scores and figure |
| `preparation/` | Model preparation helpers, identifiers and dependencies |
| `notebooks/` | Offline pipeline walkthrough |
| `quality-control/` | Jev/reviewer implementation, prompts, sampling, notebook and saved outcomes |
| `analysis/main/` | Report builder, numeric inputs, tables and figures |
| `evidence/` | Recorded plans, dataset manifest, complete example runs, references and Whistle scores |
| `reports/` | Final report, presentation, demo and original proposal |

## External components

- MCIF recordings and references: FBK-MT and the original talk creators, [MCIF dataset](https://huggingface.co/datasets/FBK-MT/MCIF). Reference extracts are included; audio is supplied separately.
- Whisper inference: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2).
- German translation: [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de), converted for CTranslate2.
- Reference resegmentation and latency: [OmniSTEval](https://github.com/pe-trik/OmniSTEval).
- Automatic quality metrics: [sacreBLEU](https://github.com/mjpost/sacrebleu), [JiWER](https://github.com/jitsi/jiwer) and [COMET](https://github.com/Unbabel/COMET).
- Whistle and needle3: Cactus Compute, [Whistle](https://huggingface.co/Cactus-Compute/whistle) and [needle3](https://huggingface.co/Cactus-Compute/needle3). The supplied [licence](TinyML-whistle-supplement-2026-10-25/LICENSE-needle.txt), [third-party notice](TinyML-whistle-supplement-2026-10-25/THIRD-PARTY-NOTICE.md) and native API header are retained.

The Translation QC methodology was inspired by GeekLink’s [How GeekLink Uses Jev to Check AI-Translated Subtitles](https://geeklink.dev/blog/jev-subtitle-translation-qc/) and its associated [repository](https://github.com/GeekLinkDev/jev-subtitle-translator). Their source–translation screening approach was adapted to the task of English ASR output and German translations.

Pinned model identities and preparation are described in [preparation/README.md](preparation/README.md). QC model/provider identifiers remain in the saved decisions and reviewer settings.
