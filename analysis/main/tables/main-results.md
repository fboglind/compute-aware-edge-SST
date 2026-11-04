| Device | ASR / chunk | chrF2 | COMET | WER (%) | CU (s) | CA (s) | RTF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pi | tiny.en / 4s | 46.68 | 0.559 | 36.02 | 2.15 | 3.53 | 0.322 |
| pi | tiny.en / 8s | 50.74 | 0.626 | 25.61 | 4.20 | 6.04 | 0.228 |
| pi | base.en / 4s | 50.26 | 0.602 | 26.36 | 2.12 | 4.20 | 0.516 |
| pi | base.en / 8s | 52.28 | 0.659 | 22.92 | 4.28 | 7.12 | 0.350 |
| laptop | tiny.en / 4s | 45.53 | 0.556 | 40.91 | 2.19 | 2.59 | 0.097 |
| laptop | tiny.en / 8s | 50.52 | 0.629 | 26.18 | 4.20 | 4.67 | 0.059 |
| laptop | base.en / 4s | 49.92 | 0.603 | 26.95 | 2.13 | 2.82 | 0.173 |
| laptop | base.en / 8s | 51.31 | 0.650 | 29.39 | 4.22 | 5.02 | 0.100 |

Main experiment over six talks (33.63 minutes, 266 reference segments). Quality is corpus chrF2, segment-mean COMET and pooled WER. Timing is the mean of two complete passes; RTF is audio-weighted processing time, excluding deliberate playback waits. CU and CA are LongYAAL in seconds, with clock-specific eligible segments. The repeated outputs were identical within each device/configuration.
