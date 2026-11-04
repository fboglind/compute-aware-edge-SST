# Pi LocalAgreement smoke results — 26 October 2026 (UTC)

The Pi run completed and passed saved-log validation. On the corrected 23.6146875-second recording, LocalAgreement-2 with a 2-second minimum update interval achieved **4.92% English WER**, **76.82 German chrF2**, and **0.880 processing RTF**. First German output arrived after 5.00 seconds, and replay finished 3.38 seconds after the audio ended. These are observations from one development smoke run, not a full-corpus evaluation or evidence of sustained long-form throughput.

Run: [`pi-smoke-tiny-la2-r1`](runs/pi-smoke-tiny-la2-r1/summary.json). Recorded creation time: `2026-10-26T22:38:21.058003+00:00` (23:38 in Stockholm). The 16 unit tests passed in the supplied Pi terminal output. The copied run was subsequently validated and scored on the laptop, outside inference timing.

## Configuration and provenance

- ASR: pinned `tiny.en`; MT: pinned Marian English–German; four inference threads; ASR beam 1 and MT beam 4.
- Effective compute type: `int8_float32` for both models. Inference uses faster-whisper 1.2.1 and CTranslate2 4.8.2; Transformers 4.57.6 supplies the Marian tokenizer.
- Pi environment: aarch64 Linux, kernel `6.18.50+rpt-rpi-2712`, Python 3.13.5. The exact board model is not recorded in this run's metadata.
- Agreement operates on two successive English hypotheses. Each newly confirmed span is translated independently; there is no German agreement or revision. The minimum update interval is 2 seconds, with a 15-second buffer-trim trigger.
- Audio SHA-256: `7c11580d4cc52ae57e367501fab81f9b5bb04ce30aef12011da9653a3397fcaf`.
- Recorded audio hash, model manifests, inference source hashes, decoding settings, and effective compute types match the laptop's corresponding 2-second run. The Pi's recorded inference source hashes also match the current local files.

Full configuration and model revisions are in [metadata.json](runs/pi-smoke-tiny-la2-r1/metadata.json).

## Measurements and laptop comparison

Both columns describe one run on the same recording with the same policy settings. The laptop results are documented separately in [RESULTS.md](RESULTS.md).

| Measurement | Pi | Laptop |
| --- | ---: | ---: |
| English WER (%) | 4.92 | 4.92 |
| German chrF2 | 76.82 | 74.99 |
| Processing RTF | 0.880 | 0.298 |
| Total processing (s) | 20.79 | 7.03 |
| ASR processing (s) | 18.30 | 6.49 |
| MT processing (s) | 2.48 | 0.54 |
| First German output (s) | 5.00 | 4.37 |
| Final tail after audio ends (s) | 3.38 | 0.91 |
| Replay wall time (s) | 26.99 | 24.52 |
| ASR updates | 12 | 12 |
| Nonempty agreement commits | 10 | 11 |
| Agreement-confirmed English words | 52 | 58 |
| English words flushed unconfirmed at EOF | 11 | 5 |
| German output words | 55 | 55 |
| Total overlapping audio decoded (s) | 104.80 | 104.20 |
| Largest decoded buffer (s) | 16.07 | 16.00 |
| Buffer trims | 1 | 1 |
| Maximum update wait (s) | 0.874 | 0.00047 |
| Sampled peak replay RSS (MiB) | 417.22 | 365.36 |

Processing RTF divides summed processing time, including repeated ASR work and EOF translation, by original audio duration. It excludes loading, warm-up, and deliberate waits. Pi model loading took 1.68 seconds and warm-up took 1.03 seconds. RTF below one does not imply that every update kept up or that output had no delay.

The Pi summary records no MT limit hits or resource-sampling errors. Its before/after throttle snapshots both read `throttled=0x0`. No continuous power audit was performed. Sampled RSS can miss peaks; the separate process-peak field uses a different measurement source and is not substituted for the sampled replay value.

## Output observations

The final English transcripts are byte-identical between these Pi and laptop runs. WER comprises 60 hits, one substitution, no deletions, and two insertions against 61 reference words. Both outputs include the final “Thank you.”, which is absent from the supplied reference; the references and outputs were left unchanged for scoring.

The German outputs differ. For example, the Pi emits “Es wird verschiedene Modelle mit der gleichen Aufnahme vergleichen.”, whereas the laptop emits “Es wird verschiedene Modelle mit vergleichen Die gleiche Aufnahme.” The Pi output still contains grammatical problems such as “Ein kleines Modell kann schneller”. These are text observations, not a formal human assessment.

The Pi chrF2 score is 1.83 points higher in this pair of runs. The adaptive scheduler depends on computation time, and the logs show different agreement commits and EOF tails. Identical final English text therefore does not imply identical translation units or German output. This single comparison does not establish a hardware effect on translation quality.

## Validation and scoring

[smoke-check.json](runs/pi-smoke-tiny-la2-r1/smoke-check.json) records `validation.passed: true`. The checker verified complete audio coverage, causal reads, sequential processing, atomic German word emissions, text exports, timestamp conversion to milliseconds, reconstructed RTF, a nonemitting first decode, exercised agreement, and an explicit unconfirmed EOF flush. It validated 13 events: 12 updates and one EOF event, with 55 German word emissions. The reference audio hash and duration match the run.

Scoring used jiwer 4.0.0 with lowercase, Unicode-punctuation deletion and whitespace collapse, and SacreBLEU 2.6.0 chrF2 with character order 6, word order 0, beta 2, case-sensitive and ignoring whitespace. Exact scores, reference hashes, and scored artifact hashes are retained in `smoke-check.json`.

To repeat validation and scoring on the laptop, from this supplement directory:

```bash
../TinyML-main-experiment-2026-10-18/.venv-evaluation/bin/python \
  check_smoke.py runs/pi-smoke-tiny-la2-r1 \
  --reference-audio ../data/data/smoke.wav \
  --english-ref ../smoke.en.txt \
  --german-ref ../smoke.de.txt
```

Inspect the saved [English transcript](runs/pi-smoke-tiny-la2-r1/transcript.en.txt), [German translation](runs/pi-smoke-tiny-la2-r1/translation.de.txt), and [update log](runs/pi-smoke-tiny-la2-r1/updates.jsonl) for the underlying output and timing evidence.

## Scope

This report covers only the Pi tiny.en run with a 2-second minimum update interval. No matched Pi fixed-4-second baseline or Pi 1-second LocalAgreement run was evaluated here. First output, update wait, and final tail are diagnostics, not LongYAAL. No COMET, formal CU/CA comparison, full MCIF evaluation, or repeated-run hardware benchmark is claimed.
