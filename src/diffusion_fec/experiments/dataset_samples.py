"""Dataset loading helpers for validation runners."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from diffusion_fec.data.text_samples import load_text_records, tokenize_text_records
from diffusion_fec.types import TokenSample


FAKE_DATASET_TOKENIZER_NAME = "fake-deterministic-char-tokenizer"


def fake_tokenize_text(text: str, *, vocab_size: int) -> list[int]:
    """Tokenize text deterministically for model-free dataset validation."""

    if vocab_size <= 16:
        raise ValueError("vocab_size must be greater than 16")
    return [
        3 + (ord(character) % (vocab_size - 3))
        for character in text
    ]


def load_dataset_token_samples(
    *,
    dataset_path: str | Path,
    tokenize: Callable[[str], Sequence[int]],
    tokenizer_name: str,
    sample_count: int | None = None,
    seed: int = 0,
    min_tokens: int = 1,
    max_tokens: int | None = None,
    dataset_label: str | None = None,
) -> tuple[tuple[TokenSample, ...], dict[str, Any]]:
    """Load records, tokenize them, and return samples plus manifest metadata."""

    path = Path(dataset_path)
    records = load_text_records(path)
    samples = tokenize_text_records(
        records,
        tokenize=tokenize,
        tokenizer_name=tokenizer_name,
        sample_count=sample_count,
        seed=seed,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    info = {
        "dataset_label": dataset_label or path.stem,
        "path": str(path),
        "record_count": len(records),
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "token_counts": [len(sample.token_ids) for sample in samples],
        "tokenizer_name": tokenizer_name,
        "selection_seed": seed,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "text_source": "loaded_text_records",
    }
    return samples, info
