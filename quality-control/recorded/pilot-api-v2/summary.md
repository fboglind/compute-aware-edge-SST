# Translation QC summary

Development pilot; MT fidelity relative to ASR, not original-audio fidelity

Inventory: 60 windows; seeded review subset: 8.
Jev successes: 3; failed: 0; unjudged: 57; budget-skipped: 0.

Model-based German judgments; no expert German fluency or adequacy validation.

| Configuration | Inventory | Selected | Jev successful | Failed | Unjudged | Budget-skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base.en/4s | 15 | 2 | 0 | 0 | 15 | 0 |
| base.en/8s | 15 | 2 | 0 | 0 | 15 | 0 |
| tiny.en/4s | 15 | 2 | 2 | 0 | 13 | 0 |
| tiny.en/8s | 15 | 2 | 1 | 0 | 14 | 0 |

Fixed-sample model flags (successful judgments only; not validated error rates):

| Configuration | Sample inventory | Judged | Omission flags | Any category flags |
| --- | ---: | ---: | ---: | ---: |
| base.en/4s | 2 | 0 | 0 | 0 |
| base.en/8s | 2 | 0 | 0 | 0 |
| tiny.en/4s | 2 | 2 | 0 | 2 |
| tiny.en/8s | 2 | 1 | 0 | 1 |

Fixed sample threshold sensitivity:

- Threshold 0.5: 3/3 (100.0%)
- Threshold 0.7: 3/3 (100.0%)
- Threshold 0.9: 0/3 (0.0%)

Targeted diagnostic subset threshold sensitivity:

- Threshold 0.5: 0/0 (not available)
- Threshold 0.7: 0/0 (not available)
- Threshold 0.9: 0/0 (not available)
A passage can belong to both selections; these subsets must not be added together.

Provider usage/cost (null means not returned, never zero-filled):

```json
{"fields":{"cost":{"known_sum":0.000238854,"reported_attempts":3,"unknown_attempts":0},"input_tokens":{"known_sum":5687,"reported_attempts":3,"unknown_attempts":0},"output_tokens":{"known_sum":255,"reported_attempts":3,"unknown_attempts":0}},"http_attempts":3,"providers":{"TypeSafe":3},"resolved_models":{"typesafe/jev-1.13-20260917":3},"scope":"All recorded HTTP attempts, including retries, failures and previous prompt versions; cache hits are not billed again."}
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
