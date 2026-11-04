# Translation QC summary

Six-talk main experiment; one text-identical repetition retained per device/configuration; MT fidelity relative to ASR

Inventory: 696 windows; seeded review subset: 96.
Jev successes: 102; failed: 0; unjudged: 594; budget-skipped: 0.

Model-based German judgments; no expert German fluency or adequacy validation.

| Configuration | Inventory | Selected | Jev successful | Failed | Unjudged | Budget-skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| laptop/base.en/4s | 87 | 12 | 12 | 0 | 75 | 0 |
| laptop/base.en/8s | 87 | 12 | 15 | 0 | 72 | 0 |
| laptop/tiny.en/4s | 87 | 12 | 14 | 0 | 73 | 0 |
| laptop/tiny.en/8s | 87 | 12 | 12 | 0 | 75 | 0 |
| pi/base.en/4s | 87 | 12 | 12 | 0 | 75 | 0 |
| pi/base.en/8s | 87 | 12 | 12 | 0 | 75 | 0 |
| pi/tiny.en/4s | 87 | 12 | 13 | 0 | 74 | 0 |
| pi/tiny.en/8s | 87 | 12 | 12 | 0 | 75 | 0 |

Fixed-sample model flags (successful judgments only; not validated error rates):

| Configuration | Sample inventory | Judged | Omission flags | Any category flags |
| --- | ---: | ---: | ---: | ---: |
| laptop/base.en/4s | 12 | 12 | 0 | 10 |
| laptop/base.en/8s | 12 | 12 | 0 | 11 |
| laptop/tiny.en/4s | 12 | 12 | 0 | 11 |
| laptop/tiny.en/8s | 12 | 12 | 0 | 10 |
| pi/base.en/4s | 12 | 12 | 0 | 10 |
| pi/base.en/8s | 12 | 12 | 0 | 10 |
| pi/tiny.en/4s | 12 | 12 | 0 | 10 |
| pi/tiny.en/8s | 12 | 12 | 0 | 9 |

Fixed sample threshold sensitivity:

- Threshold 0.5: 81/96 (84.4%)
- Threshold 0.7: 52/96 (54.2%)
- Threshold 0.9: 0/96 (0.0%)

Targeted diagnostic subset threshold sensitivity:

- Threshold 0.5: 4/6 (66.7%)
- Threshold 0.7: 1/6 (16.7%)
- Threshold 0.9: 0/6 (0.0%)
A passage can belong to both selections; these subsets must not be added together.

Provider usage/cost (null means not returned, never zero-filled):

```json
{"fields":{"cost":{"known_sum":0.0083034,"reported_attempts":102,"unknown_attempts":0},"input_tokens":{"known_sum":197700,"reported_attempts":102,"unknown_attempts":0},"output_tokens":{"known_sum":8670,"reported_attempts":102,"unknown_attempts":0}},"http_attempts":102,"providers":{"TypeSafe":102},"resolved_models":{"typesafe/jev-1.13-20260917":102},"scope":"All recorded HTTP attempts, including retries, failures and previous prompt versions; cache hits are not billed again."}
```

Human labels: human_labels.csv. Review: review.md / review.csv.
Optional LLM explanations: llm_annotations.jsonl, separate from human labels.

- Review rates are model suggestions, not measured translation error rates or calibrated probabilities.
- ASR omissions absent from both texts can escape Jev; source is the actual MT input.
- Reference evidence is whole-talk and unaligned; attribution remains uncertain.
- Main windows are paired in source time, not guaranteed semantic alignment. No causal boundary effect is established.
- API failures and unjudged windows are excluded from semantic-rate denominators.
- Human labels and optional imported LLM explanations remain separate from Jev probabilities.
- No significance claims; chrF, COMET and LongYAAL remain unchanged. QC API time is not inference latency.
