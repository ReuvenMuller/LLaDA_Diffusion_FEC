import pytest

from diffusion_fec.baselines.overhead import (
    OverheadSummary,
    estimate_hash_overhead_ratio,
    metadata_token_equivalent_overhead_ratio,
    repair_token_overhead_ratio,
    select_closest_repair_count,
    token_bit_width_for_vocab,
    token_equivalent_overhead,
)


def test_token_bit_width_for_vocab_uses_minimum_fixed_width() -> None:
    assert token_bit_width_for_vocab(2) == 1
    assert token_bit_width_for_vocab(3) == 2
    assert token_bit_width_for_vocab(128) == 7
    assert token_bit_width_for_vocab(126464) == 17


def test_estimate_hash_overhead_ratio_matches_genfec_accounting() -> None:
    assert estimate_hash_overhead_ratio(hash_bits=4, vocab_size=128) == pytest.approx(4 / 7)


def test_metadata_token_equivalent_overhead_converts_bits_to_tokens() -> None:
    assert token_equivalent_overhead(metadata_bits=16, vocab_size=128) == pytest.approx(16 / 7)
    assert metadata_token_equivalent_overhead_ratio(
        metadata_bits=16,
        total_tokens=8,
        vocab_size=128,
    ) == pytest.approx((16 / 7) / 8)


def test_select_closest_repair_count_prefers_smallest_overshoot() -> None:
    repair_count = select_closest_repair_count(
        total_tokens=16,
        tokens_per_packet=4,
        target_overhead_ratio=0.50,
        max_repair_count=4,
    )

    assert repair_count == 2
    assert repair_token_overhead_ratio(
        total_tokens=16,
        tokens_per_packet=4,
        repair_packet_count=repair_count,
    ) == 0.5


def test_overhead_summary_serializes() -> None:
    summary = OverheadSummary(
        target_hash_bits=4,
        vocab_size=128,
        token_bit_width=7,
        target_overhead_ratio=4 / 7,
        repair_packet_count=2,
        repair_token_budget=8,
        actual_repair_token_overhead_ratio=0.5,
    )

    assert summary.to_dict()["target_hash_bits"] == 4
    assert summary.to_dict()["metadata_bits_total"] == 0
