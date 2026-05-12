from diffusion_fec.baselines.sparse_fountain_xor import (
    SPARSE_FOUNTAIN_XOR_SCHEME,
    SparseFountainXorConfig,
    build_sparse_equation_specs,
    encode_sparse_fountain_xor,
)
from diffusion_fec.baselines.xor_equations import equations_from_sparse_fountain_packets
from diffusion_fec.channels.packet_loss import PacketLossChannelConfig, apply_packet_loss_channel
from diffusion_fec.types import TokenSample


def _sample(length: int = 32) -> TokenSample:
    return TokenSample(
        sample_id="s",
        text="",
        token_ids=tuple(range(10, 10 + length)),
        tokenizer_name="fake",
    )


def test_sparse_equations_are_deterministic_for_same_seed() -> None:
    config = SparseFountainXorConfig(
        xor_overhead_bits_per_token=4,
        vocab_size=128,
        random_seed=3,
    )

    first = build_sparse_equation_specs(total_tokens=32, config=config)
    second = build_sparse_equation_specs(total_tokens=32, config=config)

    assert first == second


def test_sparse_equations_change_with_seed() -> None:
    first = build_sparse_equation_specs(
        total_tokens=32,
        config=SparseFountainXorConfig(
            xor_overhead_bits_per_token=4,
            vocab_size=128,
            random_seed=3,
        ),
    )
    second = build_sparse_equation_specs(
        total_tokens=32,
        config=SparseFountainXorConfig(
            xor_overhead_bits_per_token=4,
            vocab_size=128,
            random_seed=4,
        ),
    )

    assert first != second


def test_sparse_encoder_respects_strict_budget_and_reports_coverage() -> None:
    sample = _sample(128)
    config = SparseFountainXorConfig(
        xor_overhead_bits_per_token=4,
        vocab_size=126_464,
        random_seed=0,
        max_coverage_degree=8,
    )

    encoded = encode_sparse_fountain_xor(
        sample,
        tokens_per_packet=4,
        config=config,
    )

    assert encoded.diagnostics.repair_token_budget == 30
    assert encoded.diagnostics.equation_count <= encoded.diagnostics.repair_token_budget
    assert encoded.diagnostics.coverage_pass_degree == 5
    assert encoded.diagnostics.coverage_possible is True
    assert encoded.diagnostics.coverage_zero_count == 0
    assert encoded.diagnostics.actual_mean_degree >= 2
    assert encoded.overhead.repair_token_budget == 30
    assert encoded.overhead.repair_packet_count == encoded.diagnostics.equation_count
    assert encoded.overhead.actual_repair_token_overhead_ratio == (
        encoded.diagnostics.equation_count / 128
    )


def test_sparse_encoder_reports_impossible_full_coverage_without_exceeding_budget() -> None:
    config = SparseFountainXorConfig(
        xor_overhead_bits_per_token=1,
        vocab_size=126_464,
        random_seed=0,
        max_coverage_degree=2,
    )

    encoded = encode_sparse_fountain_xor(
        _sample(128),
        tokens_per_packet=4,
        config=config,
    )

    assert encoded.diagnostics.repair_token_budget == 7
    assert encoded.diagnostics.equation_count == 7
    assert encoded.diagnostics.coverage_possible is False
    assert encoded.diagnostics.coverage_zero_count > 0


def test_sparse_equations_are_reconstructed_from_seed_config() -> None:
    sample = _sample(16)
    config = SparseFountainXorConfig(
        xor_overhead_bits_per_token=4,
        vocab_size=128,
        random_seed=5,
    )
    encoded = encode_sparse_fountain_xor(
        sample,
        tokens_per_packet=4,
        config=config,
    )

    equations = equations_from_sparse_fountain_packets(
        encoded.parity_packets,
        total_tokens=len(sample.token_ids),
        config=config,
    )

    assert len(equations) == encoded.diagnostics.equation_count
    assert equations[0].positions == encoded.equation_specs[0].positions


def test_dropped_sparse_repair_packets_do_not_contribute_equations() -> None:
    sample = _sample(16)
    config = SparseFountainXorConfig(
        xor_overhead_bits_per_token=4,
        vocab_size=128,
        random_seed=5,
    )
    encoded = encode_sparse_fountain_xor(
        sample,
        tokens_per_packet=4,
        config=config,
    )
    loss = apply_packet_loss_channel(
        encoded.packets,
        config=PacketLossChannelConfig(mode="burst", burst_start_wire_id=4, burst_length=100),
    )

    equations = equations_from_sparse_fountain_packets(
        loss.received,
        total_tokens=len(sample.token_ids),
        config=config,
    )

    assert all(packet.kind != SPARSE_FOUNTAIN_XOR_SCHEME for packet in loss.received)
    assert equations == ()
