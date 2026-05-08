from diffusion_fec.baselines.streaming_window import (
    STREAMING_WINDOW_METADATA_KEY,
    STREAMING_WINDOW_SCHEME,
    StreamingWindowConfig,
    encode_streaming_window,
    reconstruct_streaming_window,
)
from diffusion_fec.coding.packetizer import SOURCE_PACKET_INDEX_METADATA_KEY, packetize_contiguous
from diffusion_fec.types import Packet, STATE_KNOWN, STATE_UNGUIDED, TokenSample


def make_sample(token_ids=None) -> TokenSample:
    return TokenSample(
        sample_id="sample-1",
        text="fake",
        token_ids=tuple(token_ids or [10, 11, 12, 13, 14, 15, 16, 17]),
        tokenizer_name="fake-tokenizer",
    )


def test_streaming_window_encoding_adds_overlapping_repair_packets() -> None:
    encoded = encode_streaming_window(
        make_sample(),
        tokens_per_packet=2,
        config=StreamingWindowConfig(window_size=2, window_stride=1),
    )

    assert encoded.source_packet_count == 4
    assert encoded.extra_packet_count == 3
    assert encoded.repair_packets[0].metadata[STREAMING_WINDOW_METADATA_KEY]["scheme"] == (
        STREAMING_WINDOW_SCHEME
    )
    assert encoded.repair_packets[0].metadata[STREAMING_WINDOW_METADATA_KEY][
        "neighbor_source_packet_indices"
    ] == [0, 1]


def test_streaming_window_recovers_one_missing_packet() -> None:
    encoded = encode_streaming_window(
        make_sample(),
        tokens_per_packet=2,
        config=StreamingWindowConfig(window_size=2, window_stride=1),
    )
    received = [
        packet
        for packet in encoded.packets
        if not (packet.kind == "data" and packet.metadata[SOURCE_PACKET_INDEX_METADATA_KEY] == 0)
    ]

    plan = reconstruct_streaming_window(
        total_tokens=8,
        received_packets=received,
        tokens_per_packet=2,
    )

    assert [entry.state for entry in plan.entries] == [STATE_KNOWN] * 8
    assert [entry.token_id for entry in plan.entries] == list(make_sample().token_ids)


def test_streaming_window_peeling_can_recover_multi_round_dependencies() -> None:
    sample = make_sample(token_ids=[10, 11, 12])
    source_packets = packetize_contiguous(sample, tokens_per_packet=1)
    source_2 = source_packets[2]
    repair_a = Packet(
        source_id="sample-1",
        wire_id=3,
        kind="streaming_window_repair",
        token_ids=(11 ^ 12,),
        token_positions=(0,),
        metadata={
            STREAMING_WINDOW_METADATA_KEY: {
                "scheme": STREAMING_WINDOW_SCHEME,
                "repair_index": 0,
                "window_start_source_packet_index": 1,
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
        kind="streaming_window_repair",
        token_ids=(10 ^ 11,),
        token_positions=(0,),
        metadata={
            STREAMING_WINDOW_METADATA_KEY: {
                "scheme": STREAMING_WINDOW_SCHEME,
                "repair_index": 1,
                "window_start_source_packet_index": 0,
                "neighbor_source_packet_indices": [0, 1],
                "neighbor_lengths": [1, 1],
                "neighbor_token_positions": [[0], [1]],
                "tokens_per_packet": 1,
            }
        },
    )

    plan = reconstruct_streaming_window(
        total_tokens=3,
        received_packets=[source_2, repair_a, repair_b],
        tokens_per_packet=1,
    )

    assert [entry.state for entry in plan.entries] == [STATE_KNOWN] * 3
    assert [entry.token_id for entry in plan.entries] == [10, 11, 12]


def test_streaming_window_leaves_unresolved_multi_loss_unguided() -> None:
    encoded = encode_streaming_window(
        make_sample(),
        tokens_per_packet=2,
        config=StreamingWindowConfig(window_size=2, window_stride=2),
    )
    received = [
        packet
        for packet in encoded.packets
        if not (
            packet.kind == "data"
            and packet.metadata[SOURCE_PACKET_INDEX_METADATA_KEY] in {0, 1}
        )
    ]

    plan = reconstruct_streaming_window(
        total_tokens=8,
        received_packets=received,
        tokens_per_packet=2,
    )

    assert [entry.state for entry in plan.entries[:4]] == [STATE_UNGUIDED] * 4
    assert [entry.state for entry in plan.entries[4:]] == [STATE_KNOWN] * 4


def test_streaming_window_matched_overhead_records_budget() -> None:
    encoded = encode_streaming_window(
        make_sample(token_ids=list(range(16))),
        tokens_per_packet=4,
        config=StreamingWindowConfig(
            window_size=3,
            target_hash_bits=4,
            vocab_size=128,
        ),
    )

    assert encoded.overhead.target_overhead_ratio == 4 / 7
    assert encoded.overhead.repair_packet_count == encoded.extra_packet_count
