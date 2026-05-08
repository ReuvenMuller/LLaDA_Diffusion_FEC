"""Overhead accounting shared by classical baselines."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Any


def token_bit_width_for_vocab(vocab_size: int) -> int:
    """Return the minimum fixed bit width needed for one token ID."""

    if not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an int")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return max(1, ceil(log2(max(vocab_size, 2))))


def estimate_hash_overhead_ratio(*, hash_bits: int, vocab_size: int) -> float:
    """Estimate hash metadata overhead as token-equivalent ratio."""

    if not isinstance(hash_bits, int):
        raise TypeError("hash_bits must be an int")
    if hash_bits <= 0:
        raise ValueError("hash_bits must be positive")
    return hash_bits / token_bit_width_for_vocab(vocab_size)


def repair_token_overhead_ratio(
    *,
    total_tokens: int,
    tokens_per_packet: int,
    repair_packet_count: int,
) -> float:
    """Return repair-token budget divided by source-token count."""

    _validate_nonnegative_int(total_tokens, "total_tokens")
    _validate_positive_int(tokens_per_packet, "tokens_per_packet")
    _validate_nonnegative_int(repair_packet_count, "repair_packet_count")
    if total_tokens == 0:
        return 0.0
    return (repair_packet_count * tokens_per_packet) / total_tokens


def token_equivalent_overhead(
    *,
    metadata_bits: int,
    vocab_size: int,
) -> float:
    """Convert metadata bits to token-equivalent budget."""

    _validate_nonnegative_int(metadata_bits, "metadata_bits")
    return metadata_bits / token_bit_width_for_vocab(vocab_size)


def metadata_token_equivalent_overhead_ratio(
    *,
    metadata_bits: int,
    total_tokens: int,
    vocab_size: int,
) -> float:
    """Return metadata bit budget as token-equivalent ratio."""

    _validate_nonnegative_int(total_tokens, "total_tokens")
    if total_tokens == 0:
        return 0.0
    return token_equivalent_overhead(
        metadata_bits=metadata_bits,
        vocab_size=vocab_size,
    ) / total_tokens


def select_closest_repair_count(
    *,
    total_tokens: int,
    tokens_per_packet: int,
    target_overhead_ratio: float,
    max_repair_count: int,
) -> int:
    """Pick the repair-packet count closest to a target token-overhead ratio."""

    _validate_nonnegative_int(total_tokens, "total_tokens")
    _validate_positive_int(tokens_per_packet, "tokens_per_packet")
    _validate_nonnegative_int(max_repair_count, "max_repair_count")
    if target_overhead_ratio < 0.0:
        raise ValueError("target_overhead_ratio must be non-negative")

    best_repair_count = 0
    best_score: tuple[float, float, int] | None = None
    for repair_count in range(max_repair_count + 1):
        actual_ratio = repair_token_overhead_ratio(
            total_tokens=total_tokens,
            tokens_per_packet=tokens_per_packet,
            repair_packet_count=repair_count,
        )
        gap = abs(actual_ratio - target_overhead_ratio)
        overshoot = max(0.0, actual_ratio - target_overhead_ratio)
        score = (gap, overshoot, repair_count)
        if best_score is None or score < best_score:
            best_score = score
            best_repair_count = repair_count
    return best_repair_count


@dataclass(frozen=True)
class OverheadSummary:
    """Serializable overhead record for a baseline encoding."""

    target_hash_bits: int | None
    vocab_size: int | None
    token_bit_width: int | None
    target_overhead_ratio: float | None
    repair_packet_count: int
    repair_token_budget: int
    actual_repair_token_overhead_ratio: float
    metadata_bits_total: int = 0

    def __post_init__(self) -> None:
        if self.target_hash_bits is not None:
            _validate_positive_int(self.target_hash_bits, "target_hash_bits")
        if self.vocab_size is not None:
            _validate_positive_int(self.vocab_size, "vocab_size")
        if self.token_bit_width is not None:
            _validate_positive_int(self.token_bit_width, "token_bit_width")
        if self.target_overhead_ratio is not None and self.target_overhead_ratio < 0.0:
            raise ValueError("target_overhead_ratio must be non-negative")
        _validate_nonnegative_int(self.repair_packet_count, "repair_packet_count")
        _validate_nonnegative_int(self.repair_token_budget, "repair_token_budget")
        if self.actual_repair_token_overhead_ratio < 0.0:
            raise ValueError("actual_repair_token_overhead_ratio must be non-negative")
        _validate_nonnegative_int(self.metadata_bits_total, "metadata_bits_total")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_hash_bits": self.target_hash_bits,
            "vocab_size": self.vocab_size,
            "token_bit_width": self.token_bit_width,
            "target_overhead_ratio": self.target_overhead_ratio,
            "repair_packet_count": self.repair_packet_count,
            "repair_token_budget": self.repair_token_budget,
            "actual_repair_token_overhead_ratio": self.actual_repair_token_overhead_ratio,
            "metadata_bits_total": self.metadata_bits_total,
        }


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
