from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    model_name: str
    train_file: str
    validation_file: str
    output_dir: str

    seed: int = 42
    max_seq_length: int = 1024
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_nested_quant: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 5
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    report_to: str = "none"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            values: dict[str, Any] = yaml.safe_load(handle)

        return cls(**values)
