from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from datasets import Dataset


REQUIRED_FIELDS = {"instruction", "context", "response"}


def read_jsonl(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset file not found: {source}")

    records: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing required fields: {sorted(missing)}"
                )

            normalized = {
                "instruction": str(record["instruction"]).strip(),
                "context": str(record.get("context", "")).strip(),
                "response": str(record["response"]).strip(),
            }

            if not normalized["instruction"] or not normalized["response"]:
                continue

            records.append(normalized)

    return records


def deduplicate(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, str]] = []

    for record in records:
        key = (
            record["instruction"].casefold(),
            record["context"].casefold(),
            record["response"].casefold(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(record)

    return unique


def to_hf_dataset(records: list[dict[str, str]]) -> Dataset:
    if not records:
        raise ValueError("Cannot create a dataset from zero records")
    return Dataset.from_list(records)
