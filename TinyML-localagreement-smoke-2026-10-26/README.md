# LocalAgreement smoke test

This separate supplement implements **LocalAgreement-2 on English Whisper output**, then translates each newly confirmed span once with the existing Marian English–German model. It does not apply agreement to German hypotheses.

The corrected 23.6146875-second smoke recording has been run on the laptop with tiny.en at 1-second and 2-second minimum update intervals, alongside a fresh fixed 4-second baseline. All three saved runs passed log validation and whole-excerpt scoring. See [RESULTS.md](RESULTS.md). A subsequent Pi tiny.en run at a 2-second minimum update interval also passed validation and scoring; see [RESULTS-pi.md](RESULTS-pi.md). No full MCIF evaluation was run for this supplement.

## Run from this project

Prepare the inference and evaluation environments and pinned models using [REPRODUCE.md](../REPRODUCE.md). Place the original smoke WAV at `work/smoke.wav` in the project root; audio is not included. The reference transcripts are included under `references/`. Start these commands in the project root:

```bash
cd TinyML-localagreement-smoke-2026-10-26
../.venv-evaluation/bin/python -B -m unittest -v test_local_agreement test_online
../.venv-main/bin/python run_localagreement.py ../work/smoke.wav --asr tiny.en --min-chunk-s 2 --models-dir ../models --out ../work/localagreement/laptop-smoke-tiny-la2-r2
```

Use a fresh output directory for every run. `--out` refuses existing directories. `--asr base.en` is also supported, but was not measured in this smoke test. The default minimum update interval is 1 second; use the explicit flag above for 2 seconds. `--buffer-trim-s` defaults to 15 seconds. Four inference threads, CPU INT8 requested, ASR beam 1 and MT beam 4 match the main experiment. Effective compute types are recorded at runtime.

Optional fixed baseline, using the existing frozen runner:

```bash
../.venv-main/bin/python ../TinyML-main-experiment-2026-10-18/runner/run_pilot.py ../work/smoke.wav --asr tiny.en --chunk-s 4 --threads 4 --mt-beam 4 --models-dir ../models --out ../work/localagreement/laptop-smoke-tiny-fixed4-r2
```

Score and check the new runs outside inference timing:

```bash
../.venv-evaluation/bin/python check_smoke.py ../work/localagreement/laptop-smoke-tiny-fixed4-r2 ../work/localagreement/laptop-smoke-tiny-la2-r2 --reference-audio ../work/smoke.wav --english-ref references/smoke.en.txt --german-ref references/smoke.de.txt
```

This writes `smoke-check.json` in each run. Repeating the check refreshes that derived file, preserving raw inference logs. It checks audio hashes, complete audio coverage, causal reads, sequential processing, atomic word emissions, text/timestamp round trips, RTF, and explicit EOF handling. It computes whole-excerpt WER and chrF2 using the existing jiwer 4.0.0 / SacreBLEU 2.6.0 evaluation environment. It refuses to score a truncated prefix against the full recording references.

## Replay and agreement

1. Wait until at least the configured amount of **new** audio has arrived, or the recording ends. If processing ran longer, read all audio that has arrived. No future samples are passed to ASR; the worker never skips accumulated audio.
2. Append those samples to the retained buffer and re-decode the entire buffer with word timestamps. The lazy decoder and timestamp alignment are inside the ASR timer.
3. Remove the already committed part using timestamps and a nearby matching suffix of up to five words. Commit the exact longest common prefix of the two successive unconfirmed hypotheses. Case and punctuation matter. The first hypothesis cannot confirm anything; an empty hypothesis breaks the agreement chain.
4. Translate the newly confirmed English span once. Emit its entire German translation atomically after MT, without revisions or interpolated word times. An update may emit no text.
5. Once the buffer exceeds 15 seconds, trim at the latest confirmed Whisper segment boundary before its final segment. Retain unconfirmed audio. Use up to 200 confirmed words from before the buffer as the explicit ASR prompt.
6. At EOF, flush the latest unconfirmed tail once without re-decoding the same audio. Translate it separately and label it `eof_unconfirmed`. It must not be counted as agreement-confirmed text.

The trim threshold is a trigger, **not a hard maximum**. If there is no safe confirmed segment, the buffer remains intact and may exceed 30 seconds. This implementation does not guarantee bounded latency for difficult long recordings. Small timestamp shifts and repeated words near a commit boundary remain heuristic alignment cases; exact text agreement is evidence of stability, not correctness.

## Relationship to the paper and current implementations

The main reference is Macháček, Dabre and Bojar (2023), *Turning Whisper into Real-Time Transcription System*, arXiv v2 dated 21 September 2023. Section 3 supplies the growing buffer, LocalAgreement-2, timestamp overlap handling, adaptive updates, and confirmed-history prompt. [Paper](https://aclanthology.org/2023.ijcnlp-demo.3/).

The [authors' current implementation](https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py), reviewed online on 26 October 2026, also provides segment trimming with a 15-second default. We use that option instead of the paper's primary sentence trimming, avoiding an additional sentence segmenter. We retain the paper's 200-word prompt limit; the inspected current upstream code uses an approximately 200-character limit. Our overlap check prefers the longest matching 1–5-word suffix. This is an independent project implementation, not a bit-for-bit upstream reproduction. No upstream package is downloaded or imported at runtime. Each run records hashes of the actual local inference source.

Recent alternatives checked:

- [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) exposes a LocalAgreement backend policy, but adds a server/UI and other optional runtimes unnecessary for this smoke test.
- [SimulStreaming](https://github.com/ufal/SimulStreaming) uses an attention-based AlignAtt policy. Substituting it would test a different policy and require a separate implementation decision.

The original paper evaluated Whisper large-v2, whereas this smoke uses the project's pinned tiny.en and Marian CPU cascade. Word timestamps and an explicit confirmed-history prompt differ from the disjoint-chunk baseline. The smoke evaluates this complete adaptation, not the isolated cost of the prefix comparison.

## Measurements and artifacts

| File | Contents |
| --- | --- |
| `metadata.json` | Audio/model/source hashes, package versions, decoding policy, load/warm-up timing, compute types, starting power snapshot |
| `updates.jsonl` | Every decode and the EOF event, full timestamped English hypothesis, committed/pending text, buffer trims, ASR/MT/policy duration |
| `words.jsonl` | German output events with consumed-audio and measured emission times |
| `hypothesis.jsonl` | German text, `delays`, `elapsed`, source length; timestamps in milliseconds |
| `transcript.en.txt`, `translation.de.txt` | Committed output plus the explicitly logged EOF tail |
| `resources.jsonl`, `summary.json` | Resource samples, processing RTF, first output, final tail, repeated decoded-audio duration |
| `smoke-check.json` | Offline validation, WER, chrF2, scorer versions and input/reference hashes |
| `failure.json` | Failure details and completed event count if loading, warm-up or replay fails |

RTF is summed processing time divided by original audio duration, including repeated ASR work and EOF MT, excluding model loading, warm-up and deliberate waits. The adaptive scheduler uses one wall clock; ASR, agreement and MT run sequentially. First output and final tail are diagnostics, not LongYAAL.

**Clock caveat:** `delays` records consumed audio on the actual computation-aware adaptive trajectory. A slower machine can change update endpoints and hypotheses. This is not an independently simulated zero-compute schedule. A future formal CU/CA experiment must specify whether it compares clocks on one trajectory or runs a separate computation-unaware simulation. No LongYAAL or COMET values are claimed here, and the frozen fixed-chunk evaluator is not automatically reused for these logs.

## Pi use and future development

Copy this directory alongside the existing `TinyML-main-experiment-2026-10-18/runner` and `models` directories on the Pi, keeping the same relative layout. Use the Pi's existing `.venv` in place of `.venv-main`, and pass the actual path to its corrected smoke WAV (it may be `../data/smoke.wav` rather than the laptop's `../data/data/smoke.wav`). The runner requires mono 16 kHz signed 16-bit PCM and verifies model hashes before loading. Its imports reuse the frozen runner's model loading, Marian translation and resource sampling helpers.

For a longer unscored diagnostic, `--max-audio-s 60` can limit the existing held-out pilot talk to its first minute. Do not use whole-talk references for that prefix. A formal MCIF comparison, a Pi/laptop grid, target-side agreement, and a sentence-buffered MT policy are outside the completed smoke scope.

The observed German fragmentation suggests a useful next experiment: accumulate confirmed English into suitable translation units before MT. That changes translation timing and policy; the current implementation deliberately logs and measures translation of each new confirmed span, as selected for this smoke.
