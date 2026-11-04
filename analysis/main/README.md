# Report tables and figures

`build_report_assets.py:build` produced the included tables and figures from the six JSON inputs in `data/`: main scores and summary, COMET scores, ASR error counts, run timing extracts and repeated-output comparisons. These inputs are retained unchanged. NumPy and Matplotlib are needed to run the builder; neither is needed by the inspection notebooks.

For optional figure/table regeneration, run `python build_report_assets.py` from a copy of this folder under `work/report-rebuild` so saved report assets remain intact. This reads saved metrics without running models. The original script also writes a consistency record. See [REPRODUCE.md](../../REPRODUCE.md) for the commands.

`collect_run_audit.py` extracted the included timing data from full main-run logs. Repeating that extraction requires the separately archived raw runs. QC results and the author's inspection notes are retained under [quality-control/recorded](../../quality-control/recorded/).

See the [code guide](../../CODE_GUIDE.md) for metric pooling and latency eligibility.
