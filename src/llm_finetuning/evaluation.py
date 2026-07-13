from __future__ import annotations

import math
import re
from collections import Counter


def normalize_text(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0

    overlap = Counter(predicted_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())

    if common == 0:
        return 0.0

    precision = common / len(predicted_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def safe_perplexity(loss: float) -> float:
    if loss > 20:
        return float("inf")
    return math.exp(loss)
