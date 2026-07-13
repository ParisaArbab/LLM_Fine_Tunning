from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful software-engineering assistant. "
    "Give accurate, practical, secure, and easy-to-follow answers."
)


def build_prompt(instruction: str, context: str = "") -> str:
    instruction = instruction.strip()
    context = context.strip()

    if not instruction:
        raise ValueError("instruction must not be empty")

    if context:
        return (
            f"### System\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction\n{instruction}\n\n"
            f"### Context\n{context}\n\n"
            "### Response\n"
        )

    return (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### Instruction\n{instruction}\n\n"
        "### Response\n"
    )


def build_training_text(instruction: str, context: str, response: str, eos_token: str) -> str:
    response = response.strip()
    if not response:
        raise ValueError("response must not be empty")
    return f"{build_prompt(instruction, context)}{response}{eos_token}"
