# TinyML error analysis protocol — main-api-v2

Status: implemented and checked offline on 22 October 2026; no live model results yet. This is an exploratory analysis specified after the main experiment and aggregate scores were available, **not a preregistration**. Known output-limit anomalies had already been inspected as log events. Earlier informal chat observations are excluded from formal labels/results.

## Questions and scope

1. Which word edit types contribute to the existing ASR WER?
2. How often does the API evaluator flag lost/changed meaning, entity/quantity/term errors, or unsupported additions in German relative to the actual English MT input?
3. Do logged output-limit cases warrant investigation, separately from the fixed sample?

Question 1 is deterministic transcript comparison. Questions 2–3 use saved API responses only. No translation quality judgments are authored in chat. The unit is a configuration's output for a focal source-time window, not a sentence or independently sampled speaker.

## Corpus and repetition handling

Use the frozen six-talk main experiment: 2,018.0365625 seconds (33.63 minutes) of unique audio and 266 reference segments. Eight configurations combine two devices, two ASR models and two chunk sizes; two repetitions give 96 jobs. Verify every job, its completion record, scored inputs, reference hashes, export clocks and inspected runner version.

Compare repetitions at chunk level: source start/end and exact English/German strings must match. Retain repetition 1 for quality only if this passes. The supplied archive passes for all 48 pairs. Keep all 96 timing measurements in the original experiment; this package does not recompute them. Repeated identical text is not independent quality evidence. Keep both devices as configuration outputs even where text is identical.

The uploaded archive contains text/logs and references, not audio. The adapter verifies recorded audio fingerprints against the frozen allowlist; it cannot rehash or listen to unavailable audio here. No listening-based claim is made.

## Windows and selection

Segmentation version: `common-24s-core-plus-8s-context-v1`.

- Start at source time zero. Form non-overlapping 24-second focal windows from complete 4-second or 8-second chunks; never split or discard a chunk.
- Retain the shorter final window. Both chunk sizes share focal source-time boundaries.
- Supply up to eight seconds of preceding/following whole-chunk English/German context. Context is not an additional scored unit.
- Pair the eight outputs by talk and focal interval. This is time pairing, not guaranteed sentence/proposition alignment.

The inventory has 87 shared passages and 696 configuration outputs. Select **two passages per talk** using the lowest SHA-256 ranks of canonical `{seed, passage_id}` values, seed **20261022**. Selection ignores output text, scores and warnings. Use the same passages for all configurations: 12 source passages, 96 outputs. Save selection parameters and IDs before API calls. Equal numbers per talk give equal talk representation, not a duration-weighted sample. These are not 96 independent observations.

The separate diagnostic selection includes windows with a logged MT output-limit event or nonempty English/empty German. There are six in the supplied archive. Selection deliberately uses an observed failure indicator; do not pool it into fixed-sample prevalence estimates. Membership can overlap; retain both labels and do not add their counts.

## ASR measurement

Pin `jiwer==4.0.0`. Concatenate each retained run's English in chunk order. Compare with the official whole-talk English using the original normalization: lowercase, remove Unicode punctuation, collapse whitespace, no number normalization.

Record reference words N, hits H, substitutions S, deletions D and insertions I. Verify `(S+D+I)/N` plus integer error/reference counts against every retained original score. Report per-talk and per-configuration counts; pooled WER uses summed errors/summed reference words. Save normalized word edit alignments for locating candidate spans.

Word indexes are not audio timestamps. Minimum-edit alignments can admit alternative explanations. A WER deletion is a token edit, **not a verified semantic omission**. Do not combine ASR deletion counts with Jev flags into one omission rate.

## API fidelity method

Use OpenRouter Decisions, `typesafe/jev-1.13`, question version `mt-fidelity-context-v2`, schema version `openrouter-decisions-2026-10-18-v1`. Four `noul` questions share each request:

| Key | Question |
| --- | --- |
| `meaningful_omission` | Is substantive focal English meaning missing from the German? |
| `meaning_change_or_negation` | Is a proposition, relation or negation changed? |
| `entity_or_quantity_error` | Is an entity, quantity or term rendered with substantively different meaning? |
| `unsupported_addition` | Does focal German assert unsupported substantive content? |

`jev.py` contains exact instructions/criteria; each invocation archives the payload. Accept faithful paraphrase, meaning-preserving compression, reordering and German grammar. Read chunks together. Use neighboring context to avoid flagging content translated across a boundary. Do not count neighbor-only errors or assert focal errors when boundary ownership is unclear.

Only English/German text, language labels, chunk indexes and rubric enter the payload. Device/ASR labels, scores, paths, hashes, times and warnings stay local. Chunk grouping/text can still reveal processing properties; this is not complete experimental blinding. Whole-talk references are not sent as the primary source or assigned to windows.

Check/develop requests on the separate pilot talk with 24-second windows/eight-second context. Three initial live requests check schema and operation only. If a prompt changes, change its version and prepare fresh main artifacts **before** main evaluation. Preserve earlier drafts separately. After main judging begins, revisions are separately reported exploratory analyses, not replacements chosen for favorable results.

## Interpretation and author review

Preserve raw probabilities. Primary screening threshold: **0.5**; sensitivity thresholds: **0.7, 0.9**, without tuning on main results. Report per-category and any-category flags, successful-judgment denominators and coverage for each configuration/talk. Flags are **model-suggested candidates**, not verified errors. Low scores do not prove adequacy. These values are not calibrated real-world error probabilities.

Keep the sample, diagnostic set and later expansions separate. Unsuccessful, skipped and unjudged windows remain visible, never negative labels. With incomplete coverage, identify missing paired outputs and avoid rankings driven by selective success. Six talks/12 selected passages support descriptive paired comparisons; this initial analysis makes no significance or population-wide quality claim.

No fluency score is produced. At A2 German, the author may inspect English recognition, consult parallel text and record supported meaning observations with uncertainty. Label these as author review, not expert German reference judgments. Qualified bilingual validation would require documented reviewer competence, independent/score-hidden labeling and adjudication. Agreement with assisted author notes does not establish evaluator accuracy.

## Reproducibility

Save input hashes; code hashes; selection/segmentation/prompt/schema versions; requested/returned model IDs; complete credential-free requests; raw responses; UTC times; HTTP status; provider/usage/cost where supplied; retries/errors; and window-response mappings. Cache exact payloads plus endpoint/method versions, preserving each mapping. Cache reuse is not another independent judgment or paid call.

Freeze the first successful returned model ID; stop on subsequent changes. A returned alias cannot prove unchanged internal weights. Python/source hashes support saved-evidence reanalysis. Fresh API evaluations can differ despite identical requests: distinguish reanalysis of saved responses from repeated evaluation. No deterministic seed/temperature setting is claimed for this API.

Live execution checks preparation code hashes and input/artifact hashes. Each invocation retains its own settings/requests. HTTP attempts, including failures, are append-only. Request/cost limits are per invocation. No key is saved. QC API duration is unrelated to system computation time, RTF or LongYAAL.

## Still required for end-to-end semantic omissions

This package does not align reference meaning units to hypothesis windows or judge German directly against original speech meaning. Existing word/time alignment is not assumed to be semantic alignment. An extension needs an audited alignment/unit protocol, a separate versioned API evaluator with evidence spans, controls and uncertainty handling, and review appropriate to the author's language competence.

Generative explanation export/import remains a scaffold, not a completed API runner. Neither it nor chat-produced labels enter primary results. Existing chrF, COMET and latency measurements remain unchanged and complement these limited diagnostics. This package is not the full report.
