# LocalAgreement experiment preparation

Prepared only; no new inference or experiment measurements have been run.

The separate operator document is [INSTRUCTIONS.md](INSTRUCTIONS.md). The frozen
method is [PROTOCOL.md](PROTOCOL.md). Source ancestry is in `source-provenance.json`.
This folder is an untracked working deliverable; no Git staging or commit is needed.

`experiment.py` plans, preflights, runs, resumes and audits the grid.
`run_localagreement.py` and `local_agreement.py` are byte-identical copies of the
smoke implementation. They reuse the existing sibling main experiment's
`runner/run_pilot.py` and `runner/stream_timing.py`; keep that directory layout.
`evaluate_localagreement.py`, `score_comet_localagreement.py` and
`summarize_localagreement.py` provide offline scoring and aggregation.

No Jev, LLM/API judging or manual QA stage is included.
