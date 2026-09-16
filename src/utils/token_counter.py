from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer


@lru_cache(maxsize=4)
def get_tokenizer(model_path: str, trust_remote_code: bool = True):
    return AutoTokenizer.from_pretrained(
        str(Path(model_path)),
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )


def count_tokens(text: str, model_path: str, trust_remote_code: bool = True) -> int:
    tokenizer = get_tokenizer(model_path, trust_remote_code)
    return len(tokenizer.encode(text, add_special_tokens=False))

