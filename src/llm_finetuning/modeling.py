from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from llm_finetuning.config import TrainConfig

logger = logging.getLogger(__name__)


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config: TrainConfig):
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
    }

    if config.use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA training requires a CUDA-compatible GPU")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_torch_dtype(config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
        model_kwargs.update(
            {
                "quantization_config": quantization_config,
                "device_map": "auto",
            }
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.config.use_cache = False

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def build_lora_config(config: TrainConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )


def load_model_for_inference(
    base_model: str,
    adapter_path: str | None = None,
    device_map: str = "auto",
):
    tokenizer = load_tokenizer(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        torch_dtype="auto",
    )

    if adapter_path:
        logger.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer
