# QC implementation

`main_experiment.py:prepare_main` constructs the sample; `jev.py` implements the first API stage; `reports.py` aggregates its decisions. Prompts, thresholds and sampling are unchanged. See the [QC notebook](../TinyML_translation_QC.ipynb) for offline inspection and the [code guide](../../CODE_GUIDE.md) for interpretation. The experimental CLI can call paid services; it is not executed by the notebook.
