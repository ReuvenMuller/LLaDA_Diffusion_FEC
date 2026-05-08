from diffusion_fec.baselines.lt_fountain import (
    LT_FOUNTAIN_METADATA_KEY,
    LT_FOUNTAIN_SCHEME,
    LTFountainConfig,
    encode_lt_fountain,
    reconstruct_lt_fountain,
)
from diffusion_fec.coding.packetizer import SOURCE_PACKET_INDEX_METADATA_KEY, packetize_contiguous
from diffusion_fec.types import Packet, STATE_KNOWN, TokenSample


def make_sample(token_ids=None) -> TokenSample:
    return TokenSample(
        sample_id="sample-1",
        text="fake",
        token_ids=tuple(token_ids or [10, 11, 12, 13, 14, 15, 16, 17]),
        tokenizer_name="fake-tokenizer",
    )


def test_lt_fountain_encoding_is_seeded_and_records_neighbors() -> None:
    config = LTFountainConfig(
        repair_rate=0.75,
        random_seed=11,
        degree_values=(1, 2),
        degree_weights=(1.0, 0.0),
    )

    first = encode_lt_fountain(make_sample(), tokens_per_packet=2, config=config)
    second = encode_lt_fountain(make_sample(), tokens_per_packet=2, config=config)

    assert first.extra_packet_count == 3
    assert [packet.to_dict() for packet in first.repair_packets] == [
        packet.to_dict()
        for packet in second.repair_packets
    ]
    metadata = first.repair_packets[0].metadata[LT_FOUNTAIN_METADATA_KEY]
    assert metadata["scheme"] == LT_FOUNTAIN_SCHEME
    assert metadata["neighbor_source_packet_indices"]


def test_lt_fountain_coverage_aware_singletons_recover_single_loss() -> None:
    config = LTFountainConfig(
        repair_rate=1.0,
        random_seed=3,
        coverage_aware=True,
        degree_values=(1,),
        degree_weights=(1.0,),
    )
    encoded = encode_lt_fountain(make_sample(), tokens_per_packet=2, config=config)
    missing_source = encoded.repair_packets[0].metadata[LT_FOUNTAIN_METADATA_KEY][
        "neighbor_source_packet_indices"
    ][0]
    received = [
        packet
        for packet in encoded.packets
        if not (
            packet.kind == "data"
            and packet.metadata[SOURCE_PACKET_INDEX_METADATA_KEY] == missing_source
        )
    ]

    plan = reconstruct_lt_fountain(
        total_tokens=8,
        received_packets=received,
        tokens_per_packet=2,
    )

    assert [entry.state for entry in plan.entries] == [STATE_KNOWN] * 8
    assert [entry.token_id for entry in plan.entries] == list(make_sample().token_ids)


def test_lt_fountain_peeling_can_recover_multi_round_dependencies() -> None:
    sample = make_sample(token_ids=[10, 11, 12])
    source_packets = packetize_contiguous(sample, tokens_per_packet=1)
    source_2 = source_packets[2]
    repair_a = Packet(
        source_id="sample-1",
        wire_id=3,
        kind="lt_repair",
        token_ids=(11 ^ 12,),
        token_positions=(0,),
        metadata={
            LT_FOUNTAIN_METADATA_KEY: {
                "scheme": LT_FOUNTAIN_SCHEME,
                "repair_index": 0,
                "neighbor_source_packet_indices": [1, 2],
                "neighbor_lengths": [1, 1],
                "neighbor_token_positions": [[1], [2]],
                "tokens_per_packet": 1,
            }
        },
    )
    repair_b = Packet(
        source_id="sample-1",
        wire_id=4,
        kind="lt_repair",
        token_ids=(10 ^ 11,),
        token_positions=(0,),
        metadata={
            LT_FOUNTAIN_METADATA_KEY: {
                "scheme": LT_FOUNTAIN_SCHEME,
                "repair_index": 1,
                "neighbor_source_packet_indices": [0, 1],
                "neighbor_lengths": [1, 1],
                "neighbor_token_positions": [[0], [1]],
                "tokens_per_packet": 1,
            }
        },
    )

    plan = reconstruct_lt_fountain(
        total_tokens=3,
        received_packets=[source_2, repair_a, repair_b],
        tokens_per_packet=1,
    )

    assert [entry.state for entry in plan.entries] == [STATE_KNOWN] * 3
    assert [entry.token_id for entry in plan.entries] == [10, 11, 12]


def test_lt_fountain_matched_overhead_records_budget() -> None:
    encoded = encode_lt_fountain(
        make_sample(token_ids=list(range(16))),
        tokens_per_packet=4,
        config=LTFountainConfig(
            target_hash_bits=4,
            vocab_size=128,
            random_seed=5,
        ),
    )

    assert encoded.overhead.target_overhead_ratio == 4 / 7
    assert encoded.overhead.repair_packet_count == encoded.extra_packet_count
    assert encoded.overhead.repair_token_budget == encoded.extra_packet_count * 4
