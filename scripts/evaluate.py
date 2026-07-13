from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from llm_finetuning.data import read_jsonl
from llm_finetuning.evaluation import exact_match, token_f1
from llm_finetuning.modeling import load_model_for_inference
from llm_finetuning.prompts import build_prompt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    instruction: str,
    context: str,
    max_new_tokens: int,
) -> str:
    prompt = build_prompt(instruction, context)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned adapter")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    records = read_jsonl(args.data_path)[: args.max_examples]
    model, tokenizer = load_model_for_inference(
        args.base_model,
        args.adapter_path,
    )

    results = []
    exact_scores = []
    f1_scores = []

    for record in tqdm(records, desc="Evaluating"):
        prediction = generate_answer(
            model=model,
            tokenizer=tokenizer,
            instruction=record["instruction"],
            context=record["context"],
            max_new_tokens=args.max_new_tokens,
        )
        em = exact_match(prediction, record["response"])
        f1 = token_f1(prediction, record["response"])

        exact_scores.append(em)
        f1_scores.append(f1)
        results.append(
            {
                "instruction": record["instruction"],
                "context": record["context"],
                "reference": record["response"],
                "prediction": prediction,
                "exact_match": em,
                "token_f1": f1,
            }
        )

    report = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "number_of_examples": len(results),
        "average_exact_match": sum(exact_scores) / len(exact_scores),
        "average_token_f1": sum(f1_scores) / len(f1_scores),
        "examples": results,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Evaluation report saved to %s", output_path)


if __name__ == "__main__":
    main()
