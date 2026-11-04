# Supplementary Whistle cascade protocol

## Purpose and scope

Explore a recently available compact ASR implementation as a supplementary system alongside the completed Whisper cascades. This experiment does not replace or extend the frozen main comparison's grid. It asks what happens to observed ASR quality, downstream German translation quality, processing cost and measured latency when the ASR implementation is replaced by Whistle under the same outer chunking and MT pipeline.

Six MCIF recordings: 2,018.0365625 seconds (33.63394 minutes), 266 reference segments, same prepared 16 kHz mono signed-16-bit WAVs and official English/German references as the main experiment. The original pilot talk remains excluded. Check all input hashes against the existing manifest; do not reconvert or resample the data.

| Item | Fixed choice |
|---|---|
| Devices | Existing Raspberry Pi 5 and ThinkPad laptop |
| ASR | Pinned `Cactus-Compute/whistle` `.cact` weights; full shipped default depth |
| Runtime | Published `needle3` native Linux libraries from release artifacts labelled 3.3.3; pinned per architecture |
| Input language | Explicit English (`en`) |
| ASR call | `needle_transcribe` once per independent chunk; no keyword bias, no word alignment request |
| Native ASR decoding/thread controls | Compiled defaults; public C API has no beam/thread/depth argument |
| External chunking | Disjoint 4 s and 8 s chunks; last chunk uses actual remaining duration |
| MT | Existing Helsinki-NLP/opus-mt-en-de CTranslate2 model and tokenizer, identical hashes |
| MT runtime/settings | Same CTranslate2 version; CPU `int8_float32`, beam 4, four intra-op threads, one inter-op worker |
| MT bounds | Reject inputs exceeding 512 tokens; maximum output 256 tokens, limit flag logged |
| Replay | One sequential worker; normal-speed audio arrival; no future lookahead or dropped late chunks |
| Emission | Complete German chunk translation committed atomically after MT |
| Warm-up | Per-job synthetic-silence ASR call of chunk duration and one English MT sentence |
| Repetitions | Two complete passes per device: 6 recordings × 2 chunk sizes × 2 repetitions = 24 jobs |
| Evaluation | WER, chrF2, COMET, LongYAAL CU/CA, ASR/MT/total processing RTF and resource diagnostics |
| Quality control | No Jev, API review or manual inspection stage |

On the Pi, pass 1 uses 4 s then 8 s. On the laptop it uses 8 s then 4 s. Pass 2 reverses configuration and recording order on each device. Record power, cooling, profile and background-workload notes for every session. Run the devices simultaneously if useful, while avoiding competing heavy tasks on either host.

## Timing and memory

Reuse the unchanged `stream_timing.py` from the main experiment. For each chunk, read its samples only after its audio end time; start processing after any previous chunk finishes. Record actual start/completion times, queue wait and ASR/MT stage durations. Loading and warm-up are recorded separately, outside replay timing.

The C library is called inside the replay process through `ctypes`, with weights loaded once per recording. Per-chunk ASR timing includes PCM conversion, native inference and response parsing. MT timing includes tokenization, inference and decoding. This avoids per-chunk process startup/model loading. Native diagnostics such as first-token time or decode tokens/s are retained in `asr_response`, but do not replace wall-clock stage measurements or LongYAAL.

CU uses chunk audio-end emissions; CA uses actual translation-completion emissions. RTF is total processing time divided by audio duration, excluding deliberate playback waits. Both CU and CA include waiting for audio. Their difference is not pure inference time, and CA is not CU + RTF. Preserve clock-specific eligibility and end-of-recording cutoff sensitivity exactly as in the main evaluator.

Sample process RSS and system temperature/swap every five seconds plus start/end. The process contains both native Whistle and CTranslate2 MT. Memory includes the Python binding and mapped model; sampled RSS can miss short peaks. Process thread counts are recorded as diagnostics, not interpreted as the number of active ASR compute threads. Preserve before/after Pi throttle flags. No energy/power measurement is added.

## Evaluation and aggregation

Use the same reference resegmentation and pinned evaluator dependencies as the main experiment. Include the validated fix that supplies empty timestamp arrays only for empty predictions; nonempty outputs with missing timestamps remain errors. Keep empty/poor outputs in quality metrics. Do not drop repetitive or slow cases from primary supplementary summaries.

Pool WER error counts over normalized English reference words; report corpus chrF2 and segment-mean COMET. Pool processing time by audio duration. LongYAAL is the mean over eligible segments with separate CU/CA coverage. Summarize each repetition and then report mean and observed range over the two passes. Check output identity between repetitions; do not assume Whistle is deterministic before seeing the logs. Two timing repetitions are not independent additional audio.

First use a preflight and one complete planned recording on each device. Their timing is not used to tune model settings. That successful recording remains part of the fixed run grid. Stop on a failed job, retain its logs, and resume only after resolving the cause. Successful jobs are skipped only after hash and timing audits pass. A changed model, code or policy requires a separately labelled run rather than changing a frozen plan.

## Limits to report

This is a supplementary **system comparison**, not an isolated architecture ablation. Besides ASR weights/architecture, Whistle changes the quantization format, native kernels and decoding policy. The main ASR used Whisper beam 1 and four CTranslate2 threads; the pinned Whistle C API does not expose equivalent controls. Environment variables used for MT are not a claim that Whistle obeys a four-thread setting. Native silence/noise suppression and other compiled heuristics remain part of Whistle's defaults; no external VAD is added.

The shipped model card describes 2–4-bit quantization. Record the shipped Cactus Quants format and artifact hashes rather than relabelling it INT8 or asserting one uniform arithmetic precision. Its 16.9 MB weight-file size is not the cascade's working memory. The vendor's published speed comparisons use different hardware/settings and cannot predict this experiment's speed-up.

Whistle also supports its own agreement-based live-streaming policy. This experiment deliberately uses the same outer fixed chunks as the main cascade; it does not measure that native streaming policy, contextual carry-over, keyword biasing, a shallower decoder, fine-tuning or microphone capture. The six recordings belong to one domain. No training-data exclusion audit for MCIF has been performed. Automatic quality scores provide no new expert/manual adequacy or fluency validation.

## Pinned identities and sources

Exact weight, wheel and library SHA-256 digests are in `whistle-runtime.json` and copied into the frozen plan and each run's metadata. Preparation downloads both architecture-specific libraries; the same model bytes are used on both devices.

- Weights: `Cactus-Compute/whistle`, revision `741118687e1f008f06d1351a402a731274789790`, `whistle.cact`, 16,919,407 bytes.
- Engine assets: `Cactus-Compute/needle3`, revision `d6c7d6a2e9655505e0f8e23339356dd1705900eb`, manylinux2014 x86_64/aarch64 artifacts labelled 3.3.3. Only their `needle/libneedle3.so` member is extracted; the Python wrapper is not installed.
- The public C header is retained as `runner/needle.h`, with its checksum. The binding uses only its documented entry points.
- Original source-file hashes and adaptation provenance are retained in `source-provenance.json`. The original main-experiment files are unchanged.

Primary sources checked 25 October 2026:

- Model card: <https://huggingface.co/Cactus-Compute/whistle>
- Pinned model files: <https://huggingface.co/Cactus-Compute/whistle/tree/741118687e1f008f06d1351a402a731274789790>
- Pinned engine files: <https://huggingface.co/Cactus-Compute/needle3/tree/d6c7d6a2e9655505e0f8e23339356dd1705900eb>
- Pinned C header: <https://huggingface.co/Cactus-Compute/needle3/blob/d6c7d6a2e9655505e0f8e23339356dd1705900eb/linux-x86_64/needle.h>
- Official project: <https://github.com/cactus-compute/needle>
- Device/deployment documentation is linked from the model card.
