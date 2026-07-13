import pytest

from llm_finetuning.prompts import build_prompt, build_training_text


def test_build_prompt_with_context() -> None:
    prompt = build_prompt("Explain the bug", "x = 1 / 0")
    assert "### Instruction" in prompt
    assert "### Context" in prompt
    assert "x = 1 / 0" in prompt
    assert prompt.endswith("### Response\n")


def test_build_prompt_without_context() -> None:
    prompt = build_prompt("Explain LoRA")
    assert "### Context" not in prompt
    assert "Explain LoRA" in prompt


def test_empty_instruction_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_prompt("   ")


def test_training_text_adds_response_and_eos() -> None:
    text = build_training_text("Question", "", "Answer", "<eos>")
    assert text.endswith("Answer<eos>")
