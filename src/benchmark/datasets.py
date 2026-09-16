from __future__ import annotations

from itertools import cycle


def make_synthetic_prompt(target_tokens: int, seed_text: str) -> str:
    # Approximate token length without binding prompt generation to a tokenizer.
    words = seed_text.strip().split()
    if not words:
        words = ["benchmark"]
    target_words = max(8, int(target_tokens * 0.75))
    repeated = []
    for word in cycle(words):
        repeated.append(word)
        if len(repeated) >= target_words:
            break
    return " ".join(repeated)


def build_prompts(base_prompts: list[str], input_len: int, count: int) -> list[str]:
    prompts = []
    for i, prompt in zip(range(count), cycle(base_prompts)):
        prompts.append(make_synthetic_prompt(input_len, f"{prompt} Request id {i}."))
    return prompts

