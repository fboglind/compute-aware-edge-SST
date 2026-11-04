# Detailed API review

`qc_review.py:select`, `request_payload`, `validate_review` and `run` implement selection, prompts/schema, evidence validation and API execution. The reviewer was designed after seeing the Jev results. Its selected-case comparisons are not accuracy estimates.

The [offline QC notebook](TinyML_translation_QC.ipynb) reads 36 final outcomes: 33 validated successes and three failed cases. Four failed response examples preserve the repeated failure; the full 37-attempt event history is in the supporting archive. `review_manifest.json`, `review_requests.jsonl`, `review_summary.json` and `review_comparisons.csv` preserve settings, selection, prompts and reported comparisons.
