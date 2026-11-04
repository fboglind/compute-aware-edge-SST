# Recorded evidence

This is a curated collection of recorded results and complete examples. It is not the full raw-run export.

| Location | Scope |
|---|---|
| `../analysis/main/data/` | Scores, COMET, ASR errors, output comparisons and timing extracts covering all 96 main jobs |
| `main/data-manifest.json` | Frozen six-talk selection, recording/reference hashes and conversion records; WAVs are external |
| `main/runs/{pi,laptop}/plan.json` | Recorded 48-job plan for each device |
| `main/runs/pi/jobs/rxrToXvRyM-tiny-4s-r1/` | Complete accepted `attempt-001` with its completion marker, source/model identities, output hashes and logs |
| `main/evaluation-main-fixed/` | Reference extracts for all six talks, pass summary and resegmented output for the main example |
| `whistle/whistle-scores.json`, `whistle/whistle-summary.json`, `whistle/whistle-pass-summary.csv` | Scores and summaries covering all 48 Whistle jobs |
| `whistle/comet-scores.json`, `whistle/output-agreement.json` | Recorded neural scores and repeated-output comparisons |
| `whistle/runs/{pi,laptop}/plan.json` | Recorded 24-job plan for each device |
| `whistle/runs/pi/jobs/rxrToXvRyM-whistle-4s-r1/` | Complete accepted `attempt-002`; the interrupted first attempt remains in the author's archive |
| `whistle/audit/` | Experiment closure and laptop power/timing deviation |
| `../pilot/` | Four complete development-pilot runs, references and scoring outputs |
| `../TinyML-localagreement-smoke-2026-10-26/runs/` | Four complete smoke runs, including the fixed-chunk baseline |
| `../quality-control/recorded/` | Saved QC samples, decisions, final review outcomes, selected failures and manual notes |

Original records are retained byte for byte, including historical absolute paths. Completion markers can be checked against their included accepted attempts. Plans describe the entire measured grid even though only one raw example per experiment is included here.

Re-evaluating the entire main or Whistle grid, or reconstructing the main QC sample, requires full archived exports or fresh runs. Use [REPRODUCE.md](../REPRODUCE.md) and write new outputs under `work/`.
