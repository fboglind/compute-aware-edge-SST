# Main experiment protocol — 18 October 2026

## Questions and fixed comparison

Measure how computation changes the apparent quality–latency trade-off of the existing English-speech → English-transcript → German-text cascade on the Pi 5, and compare the same configurations on the ThinkPad CPU. This is a feasibility study of the tested hardware and software stack.

The four configurations are Whisper `tiny.en`/`base.en` × disjoint 4/8-second chunks. Keep OPUS-MT EN→DE, four CPU threads, ASR beam 1, MT beam 4, temperature 0, no VAD, no previous-text conditioning, no overlap/revision and effective `int8_float32`. Reuse the exact pilot weights and runner. No LocalAgreement, alternative ASR, quantization sweep or LLM judge is added to this grid.

## Content selection

Exclude the entire development talk `JhbtCwcsWY`. Sort the other 20 MCIF talk IDs by SHA-256 of UTF-8 `20261018:` followed by the ID. Among prefixes whose summed final reference endpoints fall between 30 and 40 minutes, choose the prefix closest to 35 minutes. This rule uses IDs and durations only; it does not inspect model output, quality, topics or apparent difficulty.

| Selected talk | Reference segments | Original reference lines (1-based) | Last reference end (s) |
|---|---:|---|---:|
| rxrToXvRyM | 52 | 31–82 | 425.608 |
| EqmWoxNDIr | 46 | 704–749 | 299.062 |
| UOlPKyCVgg | 32 | 585–616 | 298.185 |
| krJSAnVcGR | 43 | 844–886 | 302.657 |
| ERmKpJPPDc | 48 | 656–703 | 340.254 |
| xhjBHGVOyQ | 45 | 163–207 | 346.498 |
| Total | 266 | — | 2012.264 |

The total above is a **duration proxy**, not a claim that the six WAV durations were verified here. Local preparation checks actual full-recording durations, records original and prepared audio hashes, requires a total of 30–40 minutes, and freezes the final manifest before inference. It rejects missing, incompatible or mismatched inputs. Full recordings preserve leading/trailing silence, internal pauses and the original clock. All excerpt offsets are zero. Conversion to mono 16 kHz signed 16-bit PCM, when needed, happens before measurement.

The same prepared files and manifest are copied to both devices. The inference runner only receives the WAV path; reference text and timestamps are used solely for evaluation. Six talks provide six content units, not hundreds of independent speech samples. The selection remains a small, technical-talk sample and is not claimed to be representative of every speaker/domain.

## Repetitions and order

One job is one complete talk replay in a fresh inference process. Each configuration is run on all six talks twice per device: 48 jobs/device and 96 total. Audio arrives at normal speed. Total replay time is approximately 8 × the selected audio duration per device, plus loading/warm-up, final tails and other overhead. Backlog can extend this estimate.

| Device / pass | Configuration block order |
|---|---|
| Pi, pass 1 | tiny/4 → tiny/8 → base/4 → base/8 |
| Pi, pass 2 | base/8 → base/4 → tiny/8 → tiny/4 |
| Laptop, pass 1 | base/4 → base/8 → tiny/4 → tiny/8 |
| Laptop, pass 2 | tiny/8 → tiny/4 → base/8 → base/4 |

Talks use the selection order in pass 1 and reverse order in pass 2. This balances block position across devices/passes without allowing manual reordering after results are seen. It cannot eliminate all thermal or time effects. Keep power, cooling and laptop profile stable and record actual conditions in `--notes`. Every invocation records hardware, versions, governor/profile information where available and sensor labels. Each job retains model/input/source hashes.

There is no artificial cooldown between jobs. Fresh process loading and the pilot warm-up are retained outside the replay clock. A pause or interruption is visible in session timestamps and notes. The first `--max-jobs 1` job is an ordinary main measurement, retained when the batch resumes. Completed results are never replaced by a faster attempt. Partial failed jobs may be retried explicitly, preserving their original evidence.

## Measurements

The unchanged runner retains a continuous monotonic replay clock. All words from a translated chunk are committed together after MT. CU timestamps equal consumed-audio endpoints; CA timestamps equal actual elapsed emission time. No interpolated word times are introduced. Deliberate arrival waits, loading and warm-up are outside processing RTF. Sequential processing includes audio read/conversion, ASR iteration and MT; event-log overhead remains outside the processing duration but within the replay clock.

Report the following together:

- Quality: reference-resegmented chrF2 and `Unbabel/wmt22-comet-da`, plus whole-talk normalized English WER as an ASR diagnostic.
- Latency: CU and CA LongYAAL, eligible segment counts and post-cutoff word counts; first output and final tail separately.
- Throughput/queues: processing RTF; ASR/MT totals; maximum, final and per-talk p95 queue wait; number of chunks waiting over 100 ms as a diagnostic, not a universal real-time threshold. Inspect the queue trajectory before asserting sustained backlog.
- Resources: sampled replay RSS and process lifetime peak RSS separately; resource sampling errors; raw before/after Pi throttle flags and thermal readings. Historical throttle flags are not automatically evidence of throttling during a run.

Temperature is the runner's `thermal_zone0` reading, whose type is saved per session. It must not automatically be called CPU temperature on the laptop. CPU0 frequency is not an all-core mean. There is no measured energy, power draw or CPU-utilization percentage in this experiment.

## Evaluation and aggregation

Pin OmniSTEval 0.1.10 and the other pilot scoring versions. Use its existing long-form reference resegmentation independently for each talk/run. Check preservation of every emitted word and round-trip reconstruction of both timestamp arrays. Never apply `--fix_simuleval_emission_ca` to these already measured elapsed times.

For primary LongYAAL, explicitly use the **actual full WAV end** relative to each reference segment, in milliseconds. Compute `duration_ms - offset_ms` using the same coordinate arithmetic as resegmentation. Retain the upstream last-reference-end score as a sensitivity check. Exact-end emissions remain excluded under the pinned metric's rule. The regression suite guards against floating-point arithmetic accidentally including a final segment.

Empty predictions remain in quality. If no segment contributes to latency, report missing latency rather than zero. CU and CA may retain different segments or words; report that coverage. Their difference is not always a pure measure of processing time. Retain final-tail/queue diagnostics because LongYAAL can censor slow outputs at recording end.

Within each device/configuration/pass, pool the talks as follows:

| Measure | Aggregation |
|---|---|
| chrF2 | One corpus calculation across all 266 reference-aligned pairs, not a mean of talk-level chrF |
| COMET | Mean of all 266 segment scores, retaining empty translations |
| WER | Sum errors / sum reference words across talks |
| RTF | Sum processing time / sum audio duration |
| LongYAAL | Mean across all contributing reference segments, not an unweighted mean of talk means |
| First output / final tail | Talk means with maximum final tail retained |
| Peak RSS / maximum queue | Maximum across the six talk-runs within a pass |

Then report the mean and observed min/max of the two complete passes. Keep pass-level and per-talk results. These are repeated performance measurements of the same content, not 12 new independent talks. No significance test or confidence interval is asserted automatically. Any later uncertainty analysis should resample whole talks and preserve pairing.

Compare exact English/German output hashes across repetitions/devices. Identical German hypotheses and source-consumption schedules must produce the same CU LongYAAL. Different hypotheses remain separately scored; do not copy Pi quality to laptop runs by assumption. COMET's exact-triplet deduplication reduces evaluation work without dropping repeated observations.

## Completion and exclusions

Require all 96 accepted runs for the default final evaluation. An explicit interim evaluation must list missing jobs. Raw outputs are immutable and all attempts are retained. Completion checks enforce input/model/source identities, settings, precision, coverage, ordering, causality, timestamps and processing totals.

Resource sampling warnings, empty ASR chunks, output-limit events, throttling and output differences remain visible in the results. Inspect them before making substantive claims; do not silently filter completed runs. Record any protocol deviation and its reason independently of which configuration looked best.

## Source and attribution

MCIF is distributed under CC BY 4.0; attribute FBK-MT, the MCIF authors and original talk creators. The supplied reference fixtures are retained from the uploaded pilot evaluation bundle and are used for regression checks. They are not new main measurements.

- [MCIF dataset](https://huggingface.co/datasets/FBK-MT/MCIF)
- [OmniSTEval](https://github.com/pe-trik/OmniSTEval), pinned local 0.1.10 source inspected for cutoff behavior
- [CTranslate2 quantization and CPU compute types](https://opennmt.net/CTranslate2/quantization.html)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [COMET checkpoint](https://huggingface.co/Unbabel/wmt22-comet-da)
