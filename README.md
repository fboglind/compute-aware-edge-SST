# Computation-aware streaming speech translation

Cascaded English-to-German speech translation on a Raspberry Pi 5 and a ThinkPad laptop, using Whisper and Marian MT. Project for the course TinyML at Stockholm University/DSV Fall 2026. Compares translation quality, latency and processing throughput across two ASR model sizes and 4- or 8-second audio chunks.

The main experiment covers six recordings and 96 runs. Supplementary work includes Whistle ASR, LocalAgreement and exploratory translation quality control (Jev reviewer -> Sonnet 5.5 evaluator + manual inspection).

- [Final report](reports/report.pdf), [presentation](reports/presentation.pdf) and [demo](reports/fixed-4s-vs-8s.mp4)
- [Results and figures](analysis/main/README.md)
- [Setup, reproduction and tests](REPRODUCE.md)
- [Code guide and dependencies](CODE_GUIDE.md)
- [Pipeline walkthrough](notebooks/01_pipeline_walkthrough.ipynb) and [QC notebook](quality-control/TinyML_translation_QC.ipynb)

The notebooks read included results without running models or calling APIs. [Recorded evidence](evidence/README.md) includes aggregate results and selected complete runs. Audio and model weights are obtained separately; setup instructions are in the reproduction guide.
