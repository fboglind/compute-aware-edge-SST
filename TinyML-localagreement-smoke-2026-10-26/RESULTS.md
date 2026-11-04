# Laptop LocalAgreement smoke results — 26 October 2026

The implementation and saved-event checks passed. On this recording, neither LocalAgreement setting improved whole-excerpt German chrF2 over the matched fixed-chunk baseline. The shorter interval emitted its first German text earlier but required substantially more processing.

One run per setting, executed sequentially on the laptop using the corrected **23.6146875 s** smoke WAV. Audio SHA-256: `7c11580d4cc52ae57e367501fab81f9b5bb04ce30aef12011da9653a3397fcaf`. All runs used the same pinned tiny.en and Marian models, four inference threads, ASR beam 1, MT beam 4, and effective `int8_float32` for both models. Software: faster-whisper 1.2.1, CTranslate2 4.8.2. These are development observations on a short recording, not research-corpus or Pi results.

| Policy | English WER (%) | German chrF2 | Processing RTF | First German output (s) | Final tail (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 4 s | 4.92 | 78.03 | 0.110 | 4.36 | 0.46 |
| LA-2, minimum update 1 s | 4.92 | 68.60 | 0.683 | 3.54 | 1.28 |
| LA-2, minimum update 2 s | 4.92 | 74.99 | 0.298 | 4.37 | 0.91 |

`LA-2` means agreement between **two successive hypotheses**, not a two-second interval. The update interval is a separate setting. WER is identical numerically despite different English errors. The references were not edited. In particular, both LA outputs end in “Thank you.” / “Danke.”, which is absent from the supplied reference; this text remains in output and scoring. Its correspondence to the recording was not separately checked by listening.

The 1-second run performed 24 ASR updates, made 21 nonempty agreement commits, and released no additional words at EOF. It re-decoded a total of 183.91 seconds of overlapping audio. The 2-second run performed 12 updates, made 11 nonempty agreement commits, and flushed five unconfirmed ASR words at EOF. It re-decoded 104.20 seconds of overlapping audio. Each run exercised one buffer trim, at audio time 12.86 s; their largest decoded buffers were 15.00 and 16.00 s. EOF translation time is included in processing totals.

All three processing RTFs are below one over this short recording. That does not establish sustained long-form throughput or bounded delays. The 1-second run had a maximum update wait of about 0.30 s before consuming accumulated audio; this adaptive-loop diagnostic is different from the fixed baseline's per-chunk queue wait. First output does not summarize latency of the rest of the text, and no LongYAAL was computed.

## Text observations

The fixed baseline splits the opening into “Sprachübersetzungen. System.” The two-second LA run instead emits “Sprachübersetzungssystem.”, but later produces the fragmented “Es wird verschiedene Modelle mit vergleichen Die gleiche Aufnahme.” The one-second run includes “Eine kleine Modell kann sein schneller”. Thus stabilizing English alone does not supply Marian with complete grammatical translation units. These are inspection examples, not a formal human quality evaluation or an established causal explanation for every score difference.

## Validation and provenance

- **16 automated tests passed**: successive agreement/revision, immutable commits, timestamp overlap, empty updates, punctuation, EOF, safe trims, buffer continuity, confirmed-history prompts, MT routing, fast/slow causal replay, truncated input selection, and detection of corrupted timestamp/word logs.
- All three completed runs passed saved-log validation: complete audio coverage, no future audio, sequential work, output timing and text round trips, millisecond export conversion, and RTF reconstruction. The LA checks also verify a nonemitting first decode, actual agreement commits, and a distinct EOF event.
- No MT output limit hits or resource-sampling errors were recorded. Sampled replay RSS maxima were 340.27 MiB for fixed chunks and 365.26 / 365.36 MiB for LA 1 s / 2 s; sampling can miss peaks.
- Both LA starting snapshots recorded `AC/online = 1`. The reused fixed runner does not record this snapshot. No continuous power or background-load audit was performed; do not treat the one-pass timings as a controlled hardware benchmark.
- The LA source hashes match the implementation after testing. Main runner files were reused without edits. Model and audio hashes agree across the three runs. The complete run metadata retains exact package versions and platform details.
- Scoring used jiwer 4.0.0 with lowercase, Unicode-punctuation deletion and whitespace collapse; no number normalization. chrF2 used SacreBLEU 2.6.0, character order 6, word order 0, beta 2, case-sensitive, ignoring whitespace. The full metric signature and reference hashes are in each `smoke-check.json`.

Saved runs:

- [Fixed 4 s](runs/laptop-smoke-tiny-fixed4-r1/smoke-check.json)
- [LA-2 / 1 s updates](runs/laptop-smoke-tiny-la1-r1/smoke-check.json)
- [LA-2 / 2 s updates](runs/laptop-smoke-tiny-la2-r1/smoke-check.json)

The original 96-run experiment and 48-run Whistle supplement remain separate. Their historical handoffs correctly describe LocalAgreement as not yet implemented at the time they were written; this dated supplement records the later smoke implementation and its limited evidence. Base.en, Pi execution, longer pilot recordings, COMET and formal CU/CA evaluation remain unmeasured for this policy.
