## Case 1



**krJSAnVcGR/pi/krJSAnVcGR-base-4s-r1/w007/168.000000-192.000000**

|                   Category |  Jev | API reviewer |     Comparison |                                                  Explanation |                                                     Evidence |
| -------------------------: | ---: | -------: | -------------: | -----------------------------------------------------------: | -----------------------------------------------------------: |
|        meaningful_omission | 0.22 |  present |  API reviewer_only | The word 'pre-trained' at the start of chunk 47 is completely omitted from the German translation, which jumps directly to 'Hier ist ein Beispiel'. This omission loses the connection to the previous discussion about pre-training. | [{"chunk_id":47,"scope":"focal","source_quote":"pre-trained","target_quote":null}] |
| meaning_change_or_negation | 0.34 |  present |  API reviewer_only | In chunk 42, 'in frequency time' becomes 'in der Frequenz' (in the frequency), losing the temporal aspect. In chunk 43, the grammatical structure breaks down: 'We both knowledge types available' is fragmented in German as 'Wir beide Wissenstypen nur zur...verfügbar', changing the intended meaning about knowledge type availability. | [{"chunk_id":42,"scope":"focal","source_quote":"in frequency time","target_quote":"in der Frequenz"},{"chunk_id":43,"scope":"focal","source_quote":"We both knowledge types available","target_quote":"Wir beide Wissenstypen nur zur Schlussfolgerung Zeit verfügbar"}] |
|   entity_or_quantity_error |  0.5 |  present |      both_flag | The English term 'inference time' is a technical term in machine learning referring to when a model makes predictions. The German translation 'Schlussfolgerung Zeit' (conclusion time) changes the technical meaning, as the standard German term would be 'Inferenzzeit' or 'Vorhersagezeit'. | [{"chunk_id":43,"scope":"focal","source_quote":"inference time","target_quote":"Schlussfolgerung Zeit"}] |
|       unsupported_addition | 0.19 |   absent | both_unflagged | No substantive content in the focal German chunks appears unsupported by the focal or neighboring English text. The German translations, while sometimes awkward or incomplete, do not add new claims beyond what is present in the source. |                                                           [] |

#### Before

English ASR

```
My background knowledge is assumed to be available at pre-trained time. Second, there's a back one both setting.
```

German MT

```
Mein Hintergrundwissen wird zur vorschulischen Zeit als verfügbar angenommen. Zweitens, es gibt eine hintere eine beide Einstellung.
```

#### Focal

English ASR

```
both at pre-trained time and in frequency time. Lastly, the back one. and the current setting. We both knowledge types available only at the inference time. This last setting is especially interesting. Since it's similar... the case with the background noise necessary to solve a task is not part of the pre-trained data. for example, because new occupations have developed since the time of the day. pre-trained. Here's an example of how we can
```

German MT

```
sowohl zur vortrainierten Zeit als auch in der Frequenz. Schließlich die hintere. und die aktuelle Einstellung. Wir beide Wissenstypen nur zur Schlussfolgerung Zeit verfügbar. Diese letzte Einstellung ist besonders interessant. Da es ähnlich ist... der Fall mit dem Hintergrundrauschen, der zur Lösung einer Aufgabe erforderlich ist, nicht Teil der vortrainierten Daten ist. Zum Beispiel, weil sich seit der Tageszeit neue Berufe entwickelt haben. Hier ist ein Beispiel, wie wir
```

#### After

English ASR

```
control the availability of facts and true sources. setting. We assumed that to back up knowledge, potassium-seq elected seats in government
```

German MT

```
die Verfügbarkeit von Fakten und wahren Quellen zu kontrollieren. Wir nahmen an, dass, um das Wissen zu sichern, Kalium-Seq gewählt Sitze in der Regierung
```



## Manual inspection

### Question: “is pre-trained” represented anywhere in the German, including neighboring chunks?

Window ID: **krJSAnVcGR/pi/krJSAnVcGR-base-4s-r1/w007/168.000000-192.000000**
Inspector and date: F
English information being checked:
Relevant German text and chunk IDs:
Observation: found / **appears missing** / uncertain
English source clarity: clear / **partly clear** / unclear
Reason:  The earlier occurrence of “pre-trained” is represented by “vortrainierten” in chunk 42. However, the API reviewer refers to a separate occurrence at the beginning of chunk 47, whose German translation starts directly with “Hier ist ein Beispiel”. This later occurrence appears missing. Its contribution to the intended meaning remains uncertain because the surrounding English ASR output is fragmented.



__________________________________________

## Case 2



**xhjBHGVOyQ/laptop/xhjBHGVOyQ-tiny-8s-r1/w002/48.000000-72.000000**

|                   Category |  Jev | API reviewer |     Comparison |                                                  Explanation |                                                     Evidence |
| -------------------------: | ---: | -------: | -------------: | -----------------------------------------------------------: | -----------------------------------------------------------: |
|        meaningful_omission | 0.22 |  present |  API reviewer_only | The English 'multimodal protein models' (plural) is translated as 'multimodalen Proteins' (singular genitive), omitting the explicit reference to 'models' as a separate noun. The German construction suggests 'tuning of a multimodal protein' rather than 'tuning multimodal protein models'. | [{"chunk_id":6,"scope":"focal","source_quote":"models","target_quote":null}] |
| meaning_change_or_negation | 0.66 |  present |      both_flag | In chunk 6, 'instruction tuning' (a technical term for a training method) is translated as 'Anleitung zum Tuning' (guide/instruction for tuning), changing the meaning from a technical process to guidance about tuning. In chunk 8, 'instruction data set' becomes 'Befehlsdatensatz' (command dataset), which shifts the meaning from training instructions to commands. | [{"chunk_id":6,"scope":"focal","source_quote":"instruction tuning a multimodal protein models","target_quote":"Anleitung zum Tuning eines multimodalen Proteins"},{"chunk_id":8,"scope":"focal","source_quote":"instruction data set","target_quote":"Befehlsdatensatz"}] |
|   entity_or_quantity_error | 0.88 |  present |      both_flag | The English ASR contains 'consider-revolt discrepancy' which appears to be a recognition error (likely 'considerable' was intended). The German translates this as 'Diskrepanz zwischen revolt und revolt' (discrepancy between revolt and revolt), which is a different formulation that changes the structure. Additionally, 'LP' in English is preserved as 'LP' in German, but without context it's unclear if this technical abbreviation is correctly handled. | [{"chunk_id":8,"scope":"focal","source_quote":"consider-revolt discrepancy","target_quote":"Diskrepanz zwischen revolt und revolt"},{"chunk_id":8,"scope":"focal","source_quote":"LP","target_quote":"LP"}] |
|       unsupported_addition | 0.49 |   absent | both_unflagged | No substantive content in the German translation appears unsupported by the English source. The translations are attempts to render the English ASR content, even where that content contains apparent recognition errors. |                                                           [] |

#### Before

English ASR

```
are improving the serial shot performance on language only task, where computer vision and multimodal task has been left out.
```

German MT

```
Verbesserung der seriellen Schussleistung nur auf Sprachaufgabe, bei der Computersicht und multimodale Aufgabe ausgelassen wurde.
```

#### Focal

English ASR

```
Therefore, in this work, we want to investigate whether instruction tuning a multimodal protein models can actually... improve the generalization to unseen multimodal tasks. Additionally, at the time of our research. We discovered a consider-revolt discrepancy in availability of instruction data set between LP and multi-modal.
```

German MT

```
Deshalb wollen wir in dieser Arbeit untersuchen, ob eine Anleitung zum Tuning eines multimodalen Proteins tatsächlich... Verbesserung der Verallgemeinerung zu unsichtbaren multimodalen Aufgaben. Darüber hinaus zum Zeitpunkt unserer Forschung. Wir entdeckten eine Diskrepanz zwischen revolt und revolt in der Verfügbarkeit von Befehlsdatensatz zwischen LP und multimodal.
```

#### After

English ASR

```
It's a very existing model with 1600 lunged only instruction tasks. However, there is no large scale publicly available.
```

German MT

```
Es ist ein sehr existierendes Modell mit 1600 nur gelungenen Lehraufgaben. Es ist jedoch kein großes öffentliches Format verfügbar.
```

## Manual inspection

### Question: Is the reference to “models” preserved?

Window ID: **xhjBHGVOyQ/laptop/xhjBHGVOyQ-tiny-8s-r1/w002/48.000000-72.000000**
Inspector and date: **F**
English information being checked:
Relevant German text and chunk IDs:
Observation: found / **appears missing** / uncertain
English source clarity: clear / **partly clear** / unclear
Reason: The English phrase “multimodal protein models” contains a reference to models. The corresponding German phrase, “eines multimodalen Proteins”, does not preserve that reference. The English source is partly unclear, but “models” appears missing from the translation.

_________________________

## Case 3

**UOlPKyCVgg/laptop/UOlPKyCVgg-base-4s-r1/w011/264.000000-288.000000**

|                   Category |  Jev | API reviewer |     Comparison |                                                  Explanation |                                                     Evidence |
| -------------------------: | ---: | -------: | -------------: | -----------------------------------------------------------: | -----------------------------------------------------------: |
|        meaningful_omission | 0.47 |  present |  API reviewer_only | The phrase 'I'm supposed to globally use the res...' in chunk 66 is omitted from the German translation. This appears to be a substantive statement about intended usage that is not recovered in the focal or neighboring German chunks. | [{"chunk_id":66,"scope":"focal","source_quote":"I'm supposed to globally use the res...","target_quote":null}] |
| meaning_change_or_negation | 0.27 |   absent | both_unflagged | The focal chunks preserve the core propositions from English to German. While the ASR appears fragmented and incomplete, the German translation faithfully renders the available English content without changing substantive meaning or introducing negations. |                                                           [] |
|   entity_or_quantity_error | 0.51 |   absent |       jev_only | The quantity '9 of the 11' is correctly translated as '9 der 11'. No entity or quantity error is present. | [{"chunk_id":66,"scope":"focal","source_quote":"9 of the 11","target_quote":"9 der 11"}] |
|       unsupported_addition | 0.23 |   absent | both_unflagged | All content in the focal German chunks corresponds to content in the focal English chunks. No substantive assertions appear in German that lack support in the English source. |                                                           [] |

#### Before

English ASR

```
from stability issues. our proper system of a better performance.
```

German MT

```
von Stabilitätsfragen. Unser richtiges System von besserer Leistung.
```

#### Focal

English ASR

```
9 of the 11 don't use the task. I'm supposed to globally use the res... of the generic model here, come on back. Also observing that the special data is the Most research data is better but it doesn't scale well. All the pre-trained model obtained from Nachos are freely available. and on the eugene face and all the training strips are on the...
```

German MT

```
9 der 11 benutzen die Aufgabe nicht. des generischen Modells hier, komm zurück. Auch zu beobachten, dass die besonderen Daten sind die Die meisten Forschungsdaten sind besser, aber sie skalieren nicht gut. Alle von Nachos erhaltenen vortrainierten Modelle sind frei erhältlich. und auf dem eugenen Gesicht und alle Trainingsstreifen sind auf dem...
```

#### After

English ASR

```
our GitHub repository. So thank you for. for this presentation and we are looking forward to.
```

German MT

```
Also danke für unser GitHub-Repository. für diese Präsentation und wir freuen uns darauf.
```

## Manual inspection

### Question: Does the allegedly omitted fragment contain recoverable, meaningful information?

Window ID:
Inspector and date: **F**
English information being checked:
Relevant German text and chunk IDs:
Observation: found / appears missing / **uncertain**
English source clarity: clear / partly clear / **unclear**
Reason: TThe English fragment “I'm supposed to globally use the res...” is incomplete and difficult to interpret. It appears absent from the German translation, but I could not determine confidently what meaningful information was lost.
______________________________

## Case 4


Window ID: ERmKpJPPDc/laptop/ERmKpJPPDc-tiny-4s-r1/w005/120.000000-144.000000

Inspector: F

English information being checked: The percentage expression in chunk 35, beginning “30% of the total”.

Observation: uncertain

English source clarity: unclear

Reason: The English source repeats “total” more than 100 times, and the German translation similarly repeats “Gesamtsumme”. The repetition makes the intended source message unclear, so I could not confidently assess the meaningful omission.

Review status: The automated review failed validation. This manual observation is retained separately from the accepted automated results.
