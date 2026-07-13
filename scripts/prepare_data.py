from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from llm_finetuning.data import deduplicate, read_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split an instruction dataset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")

    records = deduplicate(read_jsonl(args.input))
    if len(records) < 2:
        raise ValueError("At least two valid examples are required")

    random.Random(args.seed).shuffle(records)
    validation_size = max(1, round(len(records) * args.validation_ratio))

    validation_records = records[:validation_size]
    train_records = records[validation_size:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "validation.jsonl", validation_records)

    logger.info("Prepared %d training examples", len(train_records))
    logger.info("Prepared %d validation examples", len(validation_records))


if __name__ == "__main__":
    main()
