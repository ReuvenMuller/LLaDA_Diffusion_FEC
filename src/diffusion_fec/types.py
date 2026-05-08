"""Serializable core data types for token-erasure recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STATE_KNOWN = "known"
STATE_MISSING = "missing"
STATE_UNGUIDED = "unguided"
VALID_ENTRY_STATES = frozenset({STATE_KNOWN, STATE_MISSING, STATE_UNGUIDED})


def _validate_token_id(token_id: int, field_name: str = "token_id") -> None:
    if not isinstance(token_id, int):
        raise TypeError(f"{field_name} must be an int")
    if token_id < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_position(position: int, field_name: str = "position") -> None:
    if not isinstance(position, int):
        raise TypeError(f"{field_name} must be an int")
    if position < 0:
        raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True)
class TokenSample:
    """A clean source message after tokenization."""

    sample_id: str
    text: str
    token_ids: tuple[int, ...]
    tokenizer_name: str

    def __post_init__(self) -> None:
        if not str(self.sample_id):
            raise ValueError("sample_id must be non-empty")
        if not isinstance(self.text, str):
            raise TypeError("text must be a str")
        if not str(self.tokenizer_name):
            raise ValueError("tokenizer_name must be non-empty")
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "tokenizer_name", str(self.tokenizer_name))
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        for index, token_id in enumerate(self.token_ids):
            _validate_token_id(token_id, f"token_ids[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "tokenizer_name": self.tokenizer_name,
        }


@dataclass(frozen=True)
class Packet:
    """A transmitted packet with explicit source-token positions."""

    source_id: str
    wire_id: int
    kind: str
    token_ids: tuple[int, ...]
    token_positions: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.source_id):
            raise ValueError("source_id must be non-empty")
        if not isinstance(self.wire_id, int):
            raise TypeError("wire_id must be an int")
        if self.wire_id < 0:
            raise ValueError("wire_id must be non-negative")
        if not str(self.kind):
            raise ValueError("kind must be non-empty")
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "token_positions", tuple(self.token_positions))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if len(self.token_ids) != len(self.token_positions):
            raise ValueError("token_ids and token_positions must have the same length")
        if len(set(self.token_positions)) != len(self.token_positions):
            raise ValueError("token_positions must not contain duplicates within a packet")
        for index, token_id in enumerate(self.token_ids):
            _validate_token_id(token_id, f"token_ids[{index}]")
        for index, position in enumerate(self.token_positions):
            _validate_position(position, f"token_positions[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "wire_id": self.wire_id,
            "kind": self.kind,
            "token_ids": list(self.token_ids),
            "token_positions": list(self.token_positions),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReconstructionEntry:
    """The receiver's state for one target token position."""

    position: int
    state: str
    token_id: int | None = None
    hash_value: int | None = None
    fixed: bool = False

    def __post_init__(self) -> None:
        _validate_position(self.position)
        if self.state not in VALID_ENTRY_STATES:
            states = ", ".join(sorted(VALID_ENTRY_STATES))
            raise ValueError(f"state must be one of: {states}")
        if self.token_id is not None:
            _validate_token_id(self.token_id)
        if self.hash_value is not None:
            if not isinstance(self.hash_value, int):
                raise TypeError("hash_value must be an int when set")
            if self.hash_value < 0:
                raise ValueError("hash_value must be non-negative")

        if self.state == STATE_KNOWN:
            if self.token_id is None:
                raise ValueError("known entries require token_id")
            if self.hash_value is not None:
                raise ValueError("known entries must not carry hash_value")
            if not self.fixed:
                raise ValueError("known entries must be fixed")
        elif self.state == STATE_MISSING:
            if self.token_id is not None:
                raise ValueError("missing entries must not carry token_id")
            if self.hash_value is None:
                raise ValueError("missing entries require hash_value")
            if self.fixed:
                raise ValueError("missing entries must be editable")
        elif self.state == STATE_UNGUIDED:
            if self.token_id is not None:
                raise ValueError("unguided entries must not carry token_id")
            if self.hash_value is not None:
                raise ValueError("unguided entries must not carry hash_value")
            if self.fixed:
                raise ValueError("unguided entries must be editable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "state": self.state,
            "token_id": self.token_id,
            "hash_value": self.hash_value,
            "fixed": self.fixed,
        }


@dataclass(frozen=True)
class ReconstructionPlan:
    """The full receiver-side reconstruction state for one target sequence."""

    entries: tuple[ReconstructionEntry, ...]
    total_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.total_tokens, int):
            raise TypeError("total_tokens must be an int")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        object.__setattr__(self, "entries", tuple(self.entries))
        if len(self.entries) != self.total_tokens:
            raise ValueError("entries length must equal total_tokens")

        positions = [entry.position for entry in self.entries]
        expected_positions = list(range(self.total_tokens))
        if positions != expected_positions:
            raise ValueError("entries must be ordered by position and cover 0..total_tokens-1")

    @property
    def known_count(self) -> int:
        return sum(entry.state == STATE_KNOWN for entry in self.entries)

    @property
    def missing_count(self) -> int:
        return sum(entry.state in {STATE_MISSING, STATE_UNGUIDED} for entry in self.entries)

    @property
    def hash_guided_count(self) -> int:
        return sum(entry.state == STATE_MISSING for entry in self.entries)

    @property
    def unguided_count(self) -> int:
        return sum(entry.state == STATE_UNGUIDED for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "total_tokens": self.total_tokens,
            "known_count": self.known_count,
            "missing_count": self.missing_count,
            "hash_guided_count": self.hash_guided_count,
            "unguided_count": self.unguided_count,
        }


@dataclass(frozen=True)
class ConfidenceStat:
    """Per-position confidence and decision diagnostics."""

    position: int
    state: str
    selected_token_id: int | None = None
    top1_probability: float | None = None
    top2_probability: float | None = None
    margin: float | None = None
    candidate_count: int | None = None
    commit_step: int | None = None
    was_fixed: bool = False
    was_restored: bool = False
    hash_value: int | None = None

    def __post_init__(self) -> None:
        _validate_position(self.position)
        if self.state not in VALID_ENTRY_STATES:
            states = ", ".join(sorted(VALID_ENTRY_STATES))
            raise ValueError(f"state must be one of: {states}")
        if self.selected_token_id is not None:
            _validate_token_id(self.selected_token_id, "selected_token_id")
        if self.candidate_count is not None and self.candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if self.commit_step is not None and self.commit_step < 0:
            raise ValueError("commit_step must be non-negative")
        if self.hash_value is not None and self.hash_value < 0:
            raise ValueError("hash_value must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "state": self.state,
            "selected_token_id": self.selected_token_id,
            "top1_probability": self.top1_probability,
            "top2_probability": self.top2_probability,
            "margin": self.margin,
            "candidate_count": self.candidate_count,
            "commit_step": self.commit_step,
            "was_fixed": self.was_fixed,
            "was_restored": self.was_restored,
            "hash_value": self.hash_value,
        }


@dataclass(frozen=True)
class StepSummary:
    """Aggregate diagnostics for one denoising step."""

    step: int
    still_masked_count: int
    committed_count: int
    average_confidence: float | None = None
    hash_guided_committed_count: int = 0
    unguided_committed_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.still_masked_count < 0:
            raise ValueError("still_masked_count must be non-negative")
        if self.committed_count < 0:
            raise ValueError("committed_count must be non-negative")
        if self.hash_guided_committed_count < 0:
            raise ValueError("hash_guided_committed_count must be non-negative")
        if self.unguided_committed_count < 0:
            raise ValueError("unguided_committed_count must be non-negative")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "still_masked_count": self.still_masked_count,
            "committed_count": self.committed_count,
            "average_confidence": self.average_confidence,
            "hash_guided_committed_count": self.hash_guided_committed_count,
            "unguided_committed_count": self.unguided_committed_count,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class DecodingResult:
    """Model output and diagnostics for one reconstruction attempt."""

    reconstructed_text: str
    reconstructed_tokens: tuple[int, ...]
    decode_latency_sec: float
    steps: int
    fixed_token_count: int
    editable_token_count: int
    hash_guided_token_count: int
    confidence_stats: tuple[ConfidenceStat, ...] = field(default_factory=tuple)
    step_summaries: tuple[StepSummary, ...] = field(default_factory=tuple)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.reconstructed_text, str):
            raise TypeError("reconstructed_text must be a str")
        object.__setattr__(self, "reconstructed_tokens", tuple(self.reconstructed_tokens))
        object.__setattr__(self, "confidence_stats", tuple(self.confidence_stats))
        object.__setattr__(self, "step_summaries", tuple(self.step_summaries))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        for index, token_id in enumerate(self.reconstructed_tokens):
            _validate_token_id(token_id, f"reconstructed_tokens[{index}]")
        if self.decode_latency_sec < 0:
            raise ValueError("decode_latency_sec must be non-negative")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.fixed_token_count < 0:
            raise ValueError("fixed_token_count must be non-negative")
        if self.editable_token_count < 0:
            raise ValueError("editable_token_count must be non-negative")
        if self.hash_guided_token_count < 0:
            raise ValueError("hash_guided_token_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reconstructed_text": self.reconstructed_text,
            "reconstructed_tokens": list(self.reconstructed_tokens),
            "decode_latency_sec": self.decode_latency_sec,
            "steps": self.steps,
            "fixed_token_count": self.fixed_token_count,
            "editable_token_count": self.editable_token_count,
            "hash_guided_token_count": self.hash_guided_token_count,
            "confidence_stats": [stat.to_dict() for stat in self.confidence_stats],
            "step_summaries": [summary.to_dict() for summary in self.step_summaries],
            "diagnostics": dict(self.diagnostics),
        }
