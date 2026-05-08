"""Pre-tokenized sample artifacts for fair strategy comparisons."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from diffusion_fec.data.text_samples import load_text_records, tokenize_text_records
from diffusion_fec.types import TokenSample


TOKENIZED_SAMPLE_ARTIFACT_KIND = "pretokenized_token_samples"
TOKENIZED_SAMPLE_ARTIFACT_FORMAT_VERSION = 1


def write_tokenized_sample_artifact(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    tokenize: Callable[[str], Sequence[int]],
    tokenizer_name: str,
    model_id: str,
    vocab_size: int,
    sample_count: int | None = None,
    seed: int = 0,
    min_tokens: int = 1,
    max_tokens: int | None = None,
    dataset_label: str | None = None,
    source_dataset_manifest_path: str | Path | None = None,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    """Build and write a JSON artifact containing already-tokenized samples."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    dataset = Path(dataset_path)
    output = Path(output_path)
    records = load_text_records(dataset)
    samples = tokenize_text_records(
        records,
        tokenize=tokenize,
        tokenizer_name=tokenizer_name,
        sample_count=sample_count,
        seed=seed,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    _validate_samples(samples=samples, vocab_size=vocab_size)

    artifact = {
        "artifact_kind": TOKENIZED_SAMPLE_ARTIFACT_KIND,
        "format_version": TOKENIZED_SAMPLE_ARTIFACT_FORMAT_VERSION,
        "tokenization": {
            "tokenizer_name": tokenizer_name,
            "model_id": model_id,
            "vocab_size": vocab_size,
            "add_special_tokens": add_special_tokens,
        },
        "source_dataset": _source_dataset_info(
            dataset_path=dataset,
            dataset_label=dataset_label,
            manifest_path=source_dataset_manifest_path,
        ),
        "selection": {
            "sample_count_requested": sample_count,
            "sample_count_written": len(samples),
            "selection_seed": seed,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
        },
        "samples": [_sample_record(sample) for sample in samples],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "artifact": artifact,
        "output_path": str(output),
        "artifact_sha256": sha256_file(output),
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "token_counts": [len(sample.token_ids) for sample in samples],
    }


def load_tokenized_sample_artifact(
    path: str | Path,
    *,
    expected_vocab_size: int | None = None,
    expected_model_id: str | None = None,
    expected_tokenizer_name: str | None = None,
) -> tuple[tuple[TokenSample, ...], dict[str, Any]]:
    """Load pre-tokenized samples and return TokenSample objects plus metadata."""

    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("artifact_kind") != TOKENIZED_SAMPLE_ARTIFACT_KIND:
        raise ValueError("tokenized sample artifact has unsupported artifact_kind")
    if artifact.get("format_version") != TOKENIZED_SAMPLE_ARTIFACT_FORMAT_VERSION:
        raise ValueError("tokenized sample artifact has unsupported format_version")

    tokenization = artifact.get("tokenization")
    if not isinstance(tokenization, dict):
        raise ValueError("tokenized sample artifact is missing tokenization metadata")
    vocab_size = int(tokenization.get("vocab_size"))
    if vocab_size <= 0:
        raise ValueError("tokenized sample artifact vocab_size must be positive")
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise ValueError(
            "tokenized sample artifact vocab_size does not match expected vocab_size "
            f"({vocab_size} != {expected_vocab_size})"
        )
    model_id = str(tokenization.get("model_id", ""))
    tokenizer_name = str(tokenization.get("tokenizer_name", ""))
    if expected_model_id is not None and model_id != expected_model_id:
        raise ValueError(
            "tokenized sample artifact model_id does not match expected model_id "
            f"({model_id!r} != {expected_model_id!r})"
        )
    if expected_tokenizer_name is not None and tokenizer_name != expected_tokenizer_name:
        raise ValueError(
            "tokenized sample artifact tokenizer_name does not match expected tokenizer_name "
            f"({tokenizer_name!r} != {expected_tokenizer_name!r})"
        )

    raw_samples = artifact.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("tokenized sample artifact samples must be a list")
    samples = tuple(_token_sample_from_record(record, tokenizer_name=tokenizer_name) for record in raw_samples)
    _validate_samples(samples=samples, vocab_size=vocab_size)
    if not samples:
        raise ValueError("tokenized sample artifact must contain at least one sample")

    selection = artifact.get("selection", {})
    source_dataset = artifact.get("source_dataset", {})
    info = {
        "dataset_label": source_dataset.get("label") or artifact_path.stem,
        "path": str(artifact_path),
        "tokenized_artifact_path": str(artifact_path),
        "tokenized_artifact_sha256": sha256_file(artifact_path),
        "artifact_format_version": artifact.get("format_version"),
        "source_dataset": dict(source_dataset),
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "token_counts": [len(sample.token_ids) for sample in samples],
        "tokenizer_name": tokenizer_name,
        "model_id": model_id,
        "vocab_size": vocab_size,
        "selection_seed": selection.get("selection_seed"),
        "min_tokens": selection.get("min_tokens"),
        "max_tokens": selection.get("max_tokens"),
        "text_source": "pretokenized_token_samples",
    }
    return samples, info


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_dataset_info(
    *,
    dataset_path: Path,
    dataset_label: str | None,
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    manifest = None
    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        manifest = {
            "path": str(manifest_file),
            "sha256": sha256_file(manifest_file),
        }
    return {
        "label": dataset_label or dataset_path.stem,
        "path": str(dataset_path),
        "sha256": sha256_file(dataset_path),
        "manifest": manifest,
    }


def _sample_record(sample: TokenSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "text": sample.text,
        "token_ids": list(sample.token_ids),
        "token_count": len(sample.token_ids),
        "tokenizer_name": sample.tokenizer_name,
    }


def _token_sample_from_record(record: Any, *, tokenizer_name: str) -> TokenSample:
    if not isinstance(record, dict):
        raise ValueError("tokenized sample records must be objects")
    sample_id = record.get("sample_id")
    text = record.get("text")
    token_ids = record.get("token_ids")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("tokenized sample record is missing sample_id")
    if not isinstance(text, str):
        raise ValueError("tokenized sample record is missing text")
    if not isinstance(token_ids, list):
        raise ValueError("tokenized sample record token_ids must be a list")
    declared_count = record.get("token_count")
    if declared_count is not None and int(declared_count) != len(token_ids):
        raise ValueError("tokenized sample record token_count does not match token_ids")
    return TokenSample(
        sample_id=sample_id,
        text=text,
        token_ids=tuple(int(token_id) for token_id in token_ids),
        tokenizer_name=str(record.get("tokenizer_name") or tokenizer_name),
    )


def _validate_samples(*, samples: Sequence[TokenSample], vocab_size: int) -> None:
    for sample in samples:
        if sample.tokenizer_name == "":
            raise ValueError("sample tokenizer_name must be non-empty")
        for token_id in sample.token_ids:
            if token_id >= vocab_size:
                raise ValueError(
                    f"sample {sample.sample_id!r} contains token_id {token_id} "
                    f"outside vocab_size={vocab_size}"
                )
