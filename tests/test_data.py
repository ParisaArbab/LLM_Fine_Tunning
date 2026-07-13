import json
from pathlib import Path

import pytest

from llm_finetuning.data import deduplicate, read_jsonl


def test_read_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "instruction": "Test",
                "context": "",
                "response": "Answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = read_jsonl(path)
    assert len(records) == 1
    assert records[0]["instruction"] == "Test"


def test_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"instruction": "Test"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_jsonl(path)


def test_deduplicate() -> None:
    records = [
        {"instruction": "A", "context": "", "response": "B"},
        {"instruction": "a", "context": "", "response": "b"},
    ]
    assert len(deduplicate(records)) == 1
