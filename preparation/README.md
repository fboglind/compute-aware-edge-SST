# Model identities and preparation

The main experiment reused the pilot's prepared models. [pilot-lock.json](support/pilot-lock.json) and the experiment's [expected-models.json](../TinyML-main-experiment-2026-10-18/expected-models.json) retain exact identities.

| Component | Repository | Revision |
|---|---|---|
| Whisper tiny.en | `Systran/faster-whisper-tiny.en` | `0d3d19a32d3338f10357c0889762bd8d64bbdeba` |
| Whisper base.en | `Systran/faster-whisper-base.en` | `3d3d5dee26484f91867d81cb899cfcf72b96be6c` |
| Marian English–German | `Helsinki-NLP/opus-mt-en-de` | `6183067f769a302e3861815543b9f312c71b0ca4` |
| COMET | `Unbabel/wmt22-comet-da` | `2760a223ac957f30acfb18c8aa649b01cf1d75f2` |
| Whistle | `Cactus-Compute/whistle` | `741118687e1f008f06d1351a402a731274789790` |
| needle3 | `Cactus-Compute/needle3` | `d6c7d6a2e9655505e0f8e23339356dd1705900eb` |

`support/prepare_models_pinned.py` uses `prepare_helpers.py` to obtain preconverted Whisper artifacts and convert Marian to CTranslate2 INT8. Main inference used CPU `int8_float32`, four threads, ASR beam 1 and MT beam 4. The preparation dependencies are in [requirements-conversion.txt](support/requirements-conversion.txt); the recorded conversion environment also used CPU PyTorch `2.8.0+cpu`.

The preparation entry point, from the project root with those dependencies installed, is:

```bash
python preparation/support/prepare_models_pinned.py --models-dir models --lock preparation/support/pilot-lock.json
```

This command downloads/converts models and is not needed for offline reading. `prepare_comet.py` in each experiment/pilot folder preserves COMET and encoder-tokenizer preparation; their `model-manifests/` folders identify both assets. Whistle's `prepare_whistle.py` and [whistle-runtime.json](../TinyML-whistle-supplement-2026-10-25/whistle-runtime.json) identify the shipped quantized weights and architecture-specific native libraries (release 3.3.3). Whistle reuses Marian.

The remaining `support/` sources are used by the preparation helpers or their tests. The obsolete transfer/checksum utility and its two transfer-command helpers are excluded. Downloaded models and audio are not included. See [REPRODUCE.md](../REPRODUCE.md) for complete environment setup, including the required CPU PyTorch version.
