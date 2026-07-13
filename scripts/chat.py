from __future__ import annotations

import argparse

import torch

from llm_finetuning.modeling import load_model_for_inference
from llm_finetuning.prompts import build_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the fine-tuned model")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    args = parser.parse_args()

    model, tokenizer = load_model_for_inference(
        args.base_model,
        args.adapter_path,
    )

    print("Type 'exit' to stop.")

    while True:
        instruction = input("\nInstruction: ").strip()
        if instruction.casefold() in {"exit", "quit"}:
            break

        context = input("Context, optional: ").strip()
        prompt = build_prompt(instruction, context)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = output[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
