"""Injected errors are synthetic plumbing fixtures, not natural-pilot accuracy evidence."""

CASES = (
    {"name": "faithful_control", "source_en": "We tested three models, but only two produced usable results.",
     "target_de": "Wir testeten drei Modelle, aber nur zwei lieferten brauchbare Ergebnisse.", "injected_category": None},
    {"name": "removed_major_clause", "source_en": "We tested three models, but only two produced usable results.",
     "target_de": "Wir testeten drei Modelle.", "injected_category": "meaningful_omission"},
    {"name": "reversed_negation", "source_en": "The model did not improve accuracy.",
     "target_de": "Das Modell verbesserte die Genauigkeit.", "injected_category": "meaning_change_or_negation"},
    {"name": "altered_number", "source_en": "We tested three models.",
     "target_de": "Wir testeten fünf Modelle.", "injected_category": "entity_or_quantity_error"},
)
