import pytest

from diffusion_fec.channels.burst_loss import apply_burst_loss
from diffusion_fec.coding.packetizer import (
    SOURCE_LAYOUT_METADATA_KEY,
    SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
    SOURCE_PACKET_INDEX_METADATA_KEY,
    WIRE_INTERLEAVING_MATRIX,
    SourceLayoutConfig,
    WireInterleavingConfig,
    build_reconstruction_plan,
    packetize_contiguous,
    packetize_sample,
)
from diffusion_fec.types import (
    Packet,
    STATE_KNOWN,
    STATE_MISSING,
    STATE_UNGUIDED,
    TokenSample,
)


def make_sample(token_ids=None) -> TokenSample:
    token_ids = token_ids or [100, 101, 102, 103, 104, 105, 106]
    return TokenSample(
        sample_id="sample-1",
        text="fake",
        token_ids=token_ids,
        tokenizer_name="fake-tokenizer",
    )


def test_contiguous_packetization_covers_every_position_once() -> None:
    packets = packetize_sample(make_sample(), tokens_per_packet=3)

    positions = [
        position
        for packet in packets
        for position in packet.token_positions
    ]
    assert positions == list(range(7))
    assert len(set(positions)) == 7
    assert [packet.metadata[SOURCE_PACKET_INDEX_METADATA_KEY] for packet in packets] == [
        0,
        1,
        2,
    ]


def test_contiguous_packetization_assigns_deterministic_wire_ids() -> None:
    packets = packetize_contiguous(make_sample(), tokens_per_packet=3)

    assert [packet.wire_id for packet in packets] == [0, 1, 2]
    assert [list(packet.token_ids) for packet in packets] == [
        [100, 101, 102],
        [103, 104, 105],
        [106],
    ]
    assert [list(packet.token_positions) for packet in packets] == [
        [0, 1, 2],
        [3, 4, 5],
        [6],
    ]


def test_packetization_rejects_invalid_packet_size() -> None:
    with pytest.raises(ValueError, match="tokens_per_packet must be positive"):
        packetize_contiguous(make_sample(), tokens_per_packet=0)


def test_round_robin_chunk_layout_covers_every_position_once_and_disperses_neighbors() -> None:
    sample = make_sample(token_ids=[100, 101, 102, 103, 104, 105, 106, 107])

    packets = packetize_sample(
        sample,
        tokens_per_packet=4,
        source_layout=SourceLayoutConfig(
            mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
            chunk_size=1,
        ),
    )

    positions = [
        position
        for packet in packets
        for position in packet.token_positions
    ]
    assert sorted(positions) == list(range(8))
    assert len(set(positions)) == 8
    assert [list(packet.token_positions) for packet in packets] == [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    ]
    assert packets[0].metadata[SOURCE_LAYOUT_METADATA_KEY]["mode"] == (
        SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS
    )


def test_round_robin_chunk_layout_rejects_invalid_chunk_sizes() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SourceLayoutConfig(
            mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
            chunk_size=0,
        )

    with pytest.raises(ValueError, match="cannot exceed tokens_per_packet"):
        packetize_sample(
            make_sample(),
            tokens_per_packet=2,
            source_layout=SourceLayoutConfig(
                mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
                chunk_size=3,
            ),
        )

    with pytest.raises(ValueError, match="must evenly divide tokens_per_packet"):
        packetize_sample(
            make_sample(),
            tokens_per_packet=3,
            source_layout=SourceLayoutConfig(
                mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
                chunk_size=2,
            ),
        )


def test_packet_level_wire_interleaving_assigns_deterministic_wire_ids() -> None:
    sample = make_sample(token_ids=list(range(10)))

    packets = packetize_sample(
        sample,
        tokens_per_packet=1,
        wire_interleaving=WireInterleavingConfig(
            mode=WIRE_INTERLEAVING_MATRIX,
            span=4,
        ),
    )

    assert [packet.wire_id for packet in packets] == [0, 3, 6, 8, 1, 4, 7, 9, 2, 5]
    assert [
        packet.metadata[SOURCE_PACKET_INDEX_METADATA_KEY]
        for packet in sorted(packets, key=lambda packet: packet.wire_id)
    ] == [0, 4, 8, 1, 5, 9, 2, 6, 3, 7]


def test_burst_loss_operates_over_wire_ids_not_source_positions() -> None:
    sample = make_sample(token_ids=list(range(8)))
    contiguous_packets = packetize_sample(sample, tokens_per_packet=1)
    wire_interleaved_packets = packetize_sample(
        sample,
        tokens_per_packet=1,
        wire_interleaving=WireInterleavingConfig(
            mode=WIRE_INTERLEAVING_MATRIX,
            span=4,
        ),
    )

    contiguous_burst = apply_burst_loss(
        contiguous_packets,
        burst_start_wire_id=0,
        burst_length=2,
    )
    interleaved_burst = apply_burst_loss(
        wire_interleaved_packets,
        burst_start_wire_id=0,
        burst_length=2,
    )

    assert [
        position
        for packet in contiguous_burst.dropped
        for position in packet.token_positions
    ] == [0, 1]
    assert [
        position
        for packet in interleaved_burst.dropped
        for position in packet.token_positions
    ] == [0, 4]


def test_source_layout_changes_erased_token_geometry_without_wire_interleaving() -> None:
    sample = make_sample(token_ids=list(range(8)))
    contiguous_packets = packetize_sample(sample, tokens_per_packet=4)
    source_interleaved_packets = packetize_sample(
        sample,
        tokens_per_packet=4,
        source_layout=SourceLayoutConfig(
            mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
            chunk_size=1,
        ),
    )

    contiguous_burst = apply_burst_loss(
        contiguous_packets,
        burst_start_wire_id=0,
        burst_length=1,
    )
    source_interleaved_burst = apply_burst_loss(
        source_interleaved_packets,
        burst_start_wire_id=0,
        burst_length=1,
    )

    assert [
        position
        for packet in contiguous_burst.dropped
        for position in packet.token_positions
    ] == [0, 1, 2, 3]
    assert [
        position
        for packet in source_interleaved_burst.dropped
        for position in packet.token_positions
    ] == [0, 2, 4, 6]


def test_reconstruction_plan_marks_known_hash_guided_and_unguided_positions() -> None:
    packets = packetize_contiguous(make_sample(), tokens_per_packet=3)
    received_packets = [packets[0], packets[2]]
    plan = build_reconstruction_plan(
        total_tokens=7,
        received_packets=received_packets,
        hash_metadata={3: 9, 4: 10},
    )

    states = [entry.state for entry in plan.entries]
    assert states == [
        STATE_KNOWN,
        STATE_KNOWN,
        STATE_KNOWN,
        STATE_MISSING,
        STATE_MISSING,
        STATE_UNGUIDED,
        STATE_KNOWN,
    ]
    assert [entry.token_id for entry in plan.entries if entry.state == STATE_KNOWN] == [
        100,
        101,
        102,
        106,
    ]
    assert plan.known_count == 4
    assert plan.hash_guided_count == 2
    assert plan.unguided_count == 1
    assert plan.missing_count == 3


def test_reconstruction_plan_raises_for_conflicting_duplicate_positions() -> None:
    packet_a = Packet(
        source_id="sample-1",
        wire_id=0,
        kind="data",
        token_ids=[100],
        token_positions=[0],
    )
    packet_b = Packet(
        source_id="sample-1",
        wire_id=1,
        kind="data",
        token_ids=[999],
        token_positions=[0],
    )

    with pytest.raises(ValueError, match="conflicting token IDs"):
        build_reconstruction_plan(total_tokens=1, received_packets=[packet_a, packet_b])
