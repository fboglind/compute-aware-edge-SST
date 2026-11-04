"""Validated empty-segment repair from the main evaluation adapter."""
import math

def normalize_empty_timestamps(rows):
    normalized, changes = [], []
    for i, original in enumerate(rows):
        row = dict(original)
        prediction = row.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"Resegmented row {i}: missing or invalid prediction")
        words = len(prediction.split())
        added = []
        for key in ("emission_cu", "emission_ca"):
            values = row.get(key)
            if values is None:
                if words:
                    raise ValueError(f"Resegmented row {i}: missing {key} for {words} output words; refusing to invent timestamps")
                row[key] = []
                added.append(key)
            elif not isinstance(values, list) or len(values) != words:
                raise ValueError(f"Resegmented row {i}: {key} count does not match output words")
            elif any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError(f"Resegmented row {i}: invalid numeric value in {key}")
        if added:
            changes.append({"row_0_based": i, "fields_set_to_empty_arrays": added})
        normalized.append(row)
    return normalized, changes
