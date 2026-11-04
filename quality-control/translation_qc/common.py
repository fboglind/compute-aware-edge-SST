"""Small strict readers and reproducible exports."""

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def normalized(text):
    return " ".join(text.split())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_record(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def now():
    return datetime.now(timezone.utc).isoformat()


def reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


def read_json(path):
    try:
        return parse_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def read_jsonl(path):
    result = []
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                require(bool(line.strip()), "Blank JSONL record")
                row = parse_json(line)
                require(isinstance(row, dict), "Expected a JSON object")
                result.append(row)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(canonical(row) + "\n")
    temporary.replace(path)


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(canonical(row) + "\n")
        output.flush()


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
