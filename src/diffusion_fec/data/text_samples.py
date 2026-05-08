"""Deterministic text-record loading and token sample construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from random import Random
from typing import Any

from diffusion_fec.types import TokenSample


@dataclass(frozen=True)
class TextRecord:
    """One source text record before tokenization."""

    record_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.record_id):
            raise ValueError("record_id must be non-empty")
        if not isinstance(self.text, str):
            raise TypeError("text must be a str")
        object.__setattr__(self, "record_id", str(self.record_id))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


def load_text_records(path: str | Path) -> tuple[TextRecord, ...]:
    """Load text records from JSON, JSONL, or plain text."""

    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_records(data_path)
    if suffix == ".json":
        return _load_json_records(data_path)
    return _load_plain_text_records(data_path)


def tokenize_text_records(
    records: Iterable[TextRecord],
    *,
    tokenize: Callable[[str], Sequence[int]],
    tokenizer_name: str,
    sample_count: int | None = None,
    seed: int = 0,
    min_tokens: int = 1,
    max_tokens: int | None = None,
) -> tuple[TokenSample, ...]:
    """Tokenize and deterministically select text records."""

    if sample_count is not None and sample_count < 0:
        raise ValueError("sample_count must be non-negative when set")
    if min_tokens < 0:
        raise ValueError("min_tokens must be non-negative")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive when set")
    if max_tokens is not None and min_tokens > max_tokens:
        raise ValueError("min_tokens cannot exceed max_tokens")

    candidates: list[TokenSample] = []
    for record in records:
        token_ids = tuple(int(token_id) for token_id in tokenize(record.text))
        if max_tokens is not None:
            token_ids = token_ids[:max_tokens]
        if len(token_ids) < min_tokens:
            continue
        candidates.append(
            TokenSample(
                sample_id=record.record_id,
                text=record.text,
                token_ids=token_ids,
                tokenizer_name=tokenizer_name,
            )
        )

    rng = Random(seed)
    ordered = list(candidates)
    rng.shuffle(ordered)
    if sample_count is not None:
        ordered = ordered[:sample_count]
    return tuple(sorted(ordered, key=lambda sample: sample.sample_id))


def _load_jsonl_records(path: Path) -> tuple[TextRecord, ...]:
    records: list[TextRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(_record_from_json_value(data, default_id=f"line-{line_index:06d}"))
    return tuple(records)


def _load_json_records(path: Path) -> tuple[TextRecord, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("samples", "records", "data"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("JSON text dataset must be a list or contain samples/records/data")
    return tuple(
        _record_from_json_value(item, default_id=f"item-{index:06d}")
        for index, item in enumerate(data)
    )


def _load_plain_text_records(path: Path) -> tuple[TextRecord, ...]:
    records: list[TextRecord] = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text:
            continue
        records.append(TextRecord(record_id=f"line-{line_index:06d}", text=text))
    return tuple(records)


def _record_from_json_value(data: Any, *, default_id: str) -> TextRecord:
    if isinstance(data, str):
        return TextRecord(record_id=default_id, text=data)
    if not isinstance(data, dict):
        raise ValueError("text dataset items must be strings or objects")
    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError("text dataset object is missing string field 'text'")
    record_id = data.get("sample_id", data.get("id", data.get("record_id", default_id)))
    metadata = {
        key: value
        for key, value in data.items()
        if key not in {"sample_id", "id", "record_id", "text"}
    }
    return TextRecord(record_id=str(record_id), text=text, metadata=metadata)
