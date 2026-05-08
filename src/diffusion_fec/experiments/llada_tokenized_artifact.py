"""Build LLaDA-tokenized dataset artifacts without loading model weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diffusion_fec.data.tokenized_samples import write_tokenized_sample_artifact
from diffusion_fec.models.llada import LLADA_1_5_MODEL_ID, LLaDAAdapter


class LLaDATokenizedArtifactUnavailable(RuntimeError):
    """Raised when the LLaDA tokenizer artifact builder cannot run."""


def build_llada_tokenized_dataset_artifact(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    model_id: str = LLADA_1_5_MODEL_ID,
    dataset_label: str | None = None,
    sample_count: int | None = None,
    seed: int = 0,
    min_tokens: int = 1,
    max_tokens: int | None = None,
    source_dataset_manifest_path: str | Path | None = None,
    local_files_only: bool = False,
    tokenizer_adapter: LLaDAAdapter | None = None,
) -> dict[str, Any]:
    """Tokenize a text dataset with the LLaDA tokenizer and write an artifact."""

    tokenizer = tokenizer_adapter or _load_tokenizer(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    return write_tokenized_sample_artifact(
        dataset_path=dataset_path,
        output_path=output_path,
        tokenize=lambda text: tokenizer.tokenize(text, add_special_tokens=False),
        tokenizer_name=tokenizer.model_id,
        model_id=model_id,
        vocab_size=tokenizer.vocab_size,
        sample_count=sample_count,
        seed=seed,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        dataset_label=dataset_label,
        source_dataset_manifest_path=source_dataset_manifest_path,
        add_special_tokens=False,
    )


def _load_tokenizer(*, model_id: str, local_files_only: bool) -> LLaDAAdapter:
    try:
        return LLaDAAdapter.from_pretrained(
            model_id,
            load_model=False,
            config_kwargs={"local_files_only": local_files_only},
            tokenizer_kwargs={"local_files_only": local_files_only},
        )
    except Exception as exc:
        raise LLaDATokenizedArtifactUnavailable(
            f"Could not load LLaDA tokenizer/config for {model_id!r}. "
            "Install optional Hugging Face dependencies and confirm the tokenizer "
            "is cached when using local-files-only mode."
        ) from exc
