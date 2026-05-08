from diffusion_fec.baselines.xor_parity import (
    XOR_PARITY_METADATA_KEY,
    XorParityConfig,
    encode_xor_parity,
    reconstruct_xor_parity,
)
from diffusion_fec.coding.packetizer import (
    SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
    WIRE_INTERLEAVING_MATRIX,
    SourceLayoutConfig,
    WireInterleavingConfig,
)
from diffusion_fec.types import STATE_KNOWN, STATE_UNGUIDED, TokenSample


def make_sample(token_ids=None) -> TokenSample:
    return TokenSample(
        sample_id="sample-1",
        text="fake",
        token_ids=tuple(token_ids or [10, 11, 12, 13, 14, 15, 16, 17]),
        tokenizer_name="fake-tokenizer",
    )


def test_xor_parity_encoding_adds_repair_packets_with_metadata() -> None:
    encoded = encode_xor_parity(
        make_sample(),
        tokens_per_packet=2,
        config=XorParityConfig(data_packets_per_stripe=2),
    )

    assert encoded.source_packet_count == 4
    assert encoded.extra_packet_count == 2
    assert [packet.kind for packet in encoded.packets] == [
        "data",
        "data",
        "data",
        "data",
        "parity",
        "parity",
    ]
    first_parity = encoded.parity_packets[0]
    assert first_parity.token_ids == (10 ^ 12, 11 ^ 13)
    assert first_parity.metadata[XOR_PARITY_METADATA_KEY]["stripe_member_lengths"] == [
        2,
        2,
    ]
    assert encoded.overhead.repair_packet_count == 2
    assert encoded.overhead.repair_token_budget == 4


def test_xor_parity_recovers_one_missing_source_packet_in_stripe() -> None:
    encoded = encode_xor_parity(
        make_sample(),
        tokens_per_packet=2,
        config=XorParityConfig(data_packets_per_stripe=2),
    )
    received = [
        packet
        for packet in encoded.packets
        if not (packet.kind == "data" and packet.metadata["source_packet_index"] == 0)
    ]

    plan = reconstruct_xor_parity(
        total_tokens=8,
        received_packets=received,
        tokens_per_packet=2,
    )

    assert [entry.state for entry in plan.entries] == [STATE_KNOWN] * 8
    assert [entry.token_id for entry in plan.entries[:2]] == [10, 11]
    assert plan.unguided_count == 0


def test_xor_parity_leaves_stripe_unguided_when_two_members_are_missing() -> None:
    encoded = encode_xor_parity(
        make_sample(),
        tokens_per_packet=2,
        config=XorParityConfig(data_packets_per_stripe=2),
    )
    received = [
        packet
        for packet in encoded.packets
        if not (
            packet.kind == "data"
            and packet.metadata["source_packet_index"] in {0, 1}
        )
    ]

    plan = reconstruct_xor_parity(
        total_tokens=8,
        received_packets=received,
        tokens_per_packet=2,
    )

    assert [entry.state for entry in plan.entries[:4]] == [STATE_UNGUIDED] * 4
    assert [entry.state for entry in plan.entries[4:]] == [STATE_KNOWN] * 4


def test_xor_parity_preserves_source_layout_positions() -> None:
    encoded = encode_xor_parity(
        make_sample(),
        tokens_per_packet=4,
        source_layout=SourceLayoutConfig(
            mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
            chunk_size=1,
        ),
        config=XorParityConfig(data_packets_per_stripe=2),
    )
    received = [
        packet
        for packet in encoded.packets
        if not (packet.kind == "data" and packet.metadata["source_packet_index"] == 0)
    ]

    plan = reconstruct_xor_parity(
        total_tokens=8,
        received_packets=received,
        tokens_per_packet=4,
    )

    assert [entry.token_id for entry in plan.entries] == [10, 11, 12, 13, 14, 15, 16, 17]


def test_xor_parity_assigns_wire_ids_across_data_and_repair_packets() -> None:
    encoded = encode_xor_parity(
        make_sample(token_ids=list(range(10))),
        tokens_per_packet=1,
        config=XorParityConfig(data_packets_per_stripe=2),
        wire_interleaving=WireInterleavingConfig(
            mode=WIRE_INTERLEAVING_MATRIX,
            span=4,
        ),
    )

    assert sorted(packet.wire_id for packet in encoded.packets) == list(range(len(encoded.packets)))
    assert [packet.kind for packet in sorted(encoded.packets, key=lambda packet: packet.wire_id)][:4] == [
        "data",
        "data",
        "data",
        "parity",
    ]


def test_xor_parity_can_resolve_matched_hash_overhead() -> None:
    encoded = encode_xor_parity(
        make_sample(token_ids=list(range(16))),
        tokens_per_packet=4,
        config=XorParityConfig(
            data_packets_per_stripe=4,
            target_hash_bits=4,
            vocab_size=128,
        ),
    )

    assert encoded.overhead.target_overhead_ratio == 4 / 7
    assert encoded.overhead.actual_repair_token_overhead_ratio >= 0.0
