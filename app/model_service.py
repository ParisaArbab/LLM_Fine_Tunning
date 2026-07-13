from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from llm_finetuning.modeling import load_model_for_inference
from llm_finetuning.prompts import build_prompt

logger = logging.getLogger(__name__)


@dataclass
class ModelService:
    base_model: str
    adapter_path: str | None
    device_map: str = "auto"

    def __post_init__(self) -> None:
        logger.info("Loading base model: %s", self.base_model)
        self.model, self.tokenizer = load_model_for_inference(
            base_model=self.base_model,
            adapter_path=self.adapter_path,
            device_map=self.device_map,
        )

    @torch.inference_mode()
    def generate(
        self,
        instruction: str,
        context: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        prompt = build_prompt(instruction, context)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        do_sample = temperature > 0
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_args["temperature"] = temperature

        output = self.model.generate(**inputs, **generation_args)
        generated_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
