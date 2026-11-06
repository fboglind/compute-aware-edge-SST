# LocalAgreement supplementary experiment — 6 November 2026

## Fixed design

Use the same six full MCIF recordings and frozen manifest as the main experiment:
rxrToXvRyM, EqmWoxNDIr, UOlPKyCVgg, krJSAnVcGR, ERmKpJPPDc, xhjBHGVOyQ.
The development talk JhbtCwcsWY remains excluded. Expected total is
2,018.0365625 seconds, 266 reference segments. Verify the original prepared
mono 16 kHz signed-16-bit PCM bytes and reference hashes; do not reselect, crop,
reconvert, or use reference boundaries during inference.

Grid: tiny.en/base.en × minimum new-audio intervals 1 s/2 s; two passes on each
of the existing Pi 5 and ThinkPad CPUs. This is LocalAgreement-2 in all four
configurations: 1 s and 2 s describe update intervals, not agreement depth.
Six talks × four configurations × two passes = 48 jobs/device, 96 total.
The normal-speed replay floor is 8 × 33.634 minutes, about 4.48 h/device;
loading, warm-up, backlog and EOF processing extend it.

Pi pass 1: tiny/1, tiny/2, base/1, base/2. Laptop pass 1: base/1, base/2,
tiny/1, tiny/2. Pass 2 reverses configuration blocks and talk order. Pass 1
talks use the manifest order. One talk/job runs in a fresh process. No cooldown
is added. Keep power, cooling and workload stable and record notes per session.
Retain the first planned run, interruptions, failed attempts and all valid slow
outputs. Resume skips only hash-verified and audited completions. A changed
implementation, model or manifest requires a separate labelled plan/output.

## Exact smoke policy

Keep the smoke runner and algorithm unchanged. Use pinned tiny.en/base.en and
Marian EN→DE weights, four CPU threads, ASR beam 1, MT beam 4, effective
int8_float32, temperature 0, no VAD and no previous-text conditioning. Request
word timestamps and supply up to 200 confirmed words preceding the retained
buffer as an explicit prompt. Minimum update intervals are lower bounds:
consume all audio that has arrived when the sequential worker becomes free.
No future audio, dropped backlog, overlap between workers or timestamp
interpolation is allowed.

Agreement applies to exact English word prefixes in successive unconfirmed
hypotheses, including case and punctuation. Translate each newly confirmed
span independently once, then emit all German words atomically after MT.
There is no German agreement or sentence accumulation. Trim at a safe confirmed
Whisper segment boundary after the buffer exceeds 15 s, retaining the last
segment and unconfirmed audio. The threshold is not a hard cap: difficult
recordings can grow beyond 30 s. At EOF translate the latest pending tail
once, without re-decoding, labelled eof_unconfirmed. Retain this output in
quality and report its word count separately from agreement-confirmed words.

## Clocks and measurements

Score both timestamp arrays from the SAME measured adaptive trajectory.
The legacy `cu` output key means consumed-audio timestamps on that trajectory;
`ca` means actual elapsed emission times. CU is not a separately simulated
zero-compute LocalAgreement run. Compute speed can change update endpoints,
hypotheses, translation units and quality. Never present CU as a device-neutral
schedule or CA minus CU as isolated inference cost. Compare to existing fixed
baselines as a complete system adaptation, not a prefix-comparison ablation.

Processing RTF sums ASR, policy, MT and measured replay processing over the
original audio duration, including repeated ASR and EOF MT, excluding model
loading, warm-up and deliberate arrival waits. Report stage totals, decoded
seconds, maximum retained decode buffer, trim count, agreement/EOF word counts,
MT output-limit events, first output and final tail. `update_wait_s` measures
lateness relative to the next minimum new-audio endpoint; it is not the fixed
baseline's chunk queue. Do not relabel it as fixed-chunk queue wait.

Retain sampled replay RSS, lifetime peak RSS, sampling errors, raw throttle
flags, temperature sensor labels and power information. Historical throttle
flags do not establish throttling during a run. No energy or power measurement
or CPU utilization claim is added.

## Offline scoring

Use the main pinned evaluation and COMET environments. Audit audio, model,
source and configuration identities; contiguous causal reads, sequential
processing, exact agreement replay, explicit EOF, atomic word emissions,
text/timestamp round trips and processing totals before accepting a run.
Empty outputs and runs with no confirmed prefix remain valid observations,
not quality-based exclusions. Resource or output-limit warnings remain visible.

Use whole-talk normalized English WER, reference-resegmented corpus chrF2,
Unbabel/wmt22-comet-da and LongYAAL for both clocks. Preserve all German words
and round-trip both timestamp arrays through resegmentation. Missing timestamp
arrays may become [] only for empty predictions; never invent nonempty times.
Do not apply SimulEval CA correction to measured elapsed timestamps.

Primary LongYAAL uses actual WAV end, subtracting offsets in the same
millisecond coordinates as resegmentation. Keep last-reference-end sensitivity,
clock-specific eligible segment counts and words at/after WAV end. Exact-end
words remain excluded by the pinned metric. Missing latency remains null.
Retain tail diagnostics because recording-end censoring can hide late output.

Within each pass pool the six talks: corpus chrF2; WER errors/reference words;
segment-mean COMET; summed processing/audio duration; mean across eligible
LongYAAL segments; maximum RSS and update wait. Then report mean and observed
min/max of complete passes. Keep per-talk and pass-level scores and output
identity comparisons. Six talks are six content units; timing repetitions are
not independent content. No automatic significance or confidence claims.
All 96 accepted runs are required for final evaluation; --allow-partial is an
explicit coverage-labelled interim analysis. COMET exact-triplet deduplication
retains every segment observation. No Jev/API/manual quality review is required.

## Provenance and limitations

This adapts the existing local smoke implementation and main scoring procedure;
no new upstream algorithm is substituted. Fragmented translation units, heuristic
timestamp alignment, unbounded retained audio and content-dependent confirmation
are inherited limitations. The domain/sample and possible training overlap
limits of prior experiments remain. Attribute MCIF to FBK-MT, its authors and
original talk creators under CC BY 4.0. Existing smoke documentation retains the
algorithm references and attribution; source-provenance.json pins local inputs.
