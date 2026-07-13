from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer

from llm_finetuning.config import TrainConfig
from llm_finetuning.modeling import (
    build_lora_config,
    load_base_model,
    load_tokenizer,
)
from llm_finetuning.prompts import build_training_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LLM with LoRA or QLoRA")
    parser.add_argument("--config", required=True, help="Path to a YAML training config")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    set_seed(config.seed)

    tokenizer = load_tokenizer(config.model_name)
    model = load_base_model(config)

    if config.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": config.train_file,
            "validation": config.validation_file,
        },
    )

    eos_token = tokenizer.eos_token or ""

    def format_example(example: dict[str, str]) -> str:
        return build_training_text(
            instruction=example["instruction"],
            context=example.get("context", ""),
            response=example["response"],
            eos_token=eos_token,
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16 and torch.cuda.is_available(),
        fp16=config.fp16 and torch.cuda.is_available(),
        report_to=[] if config.report_to == "none" else [config.report_to],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=build_lora_config(config),
        formatting_func=format_example,
    )

    logger.info("Starting training with model %s", config.model_name)
    train_result = trainer.train()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = {
        "train_metrics": train_result.metrics,
        "model_name": config.model_name,
        "use_4bit": config.use_4bit,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "target_modules": config.target_modules,
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Adapter and tokenizer saved to %s", config.output_dir)


if __name__ == "__main__":
    main()
