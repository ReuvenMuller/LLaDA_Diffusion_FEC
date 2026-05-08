import pytest

from diffusion_fec.coding.packetizer import (
    build_reconstruction_plan,
    packetize_contiguous,
)
from diffusion_fec.types import (
    Packet,
    STATE_KNOWN,
    STATE_MISSING,
    STATE_UNGUIDED,
    TokenSample,
)


def make_sample() -> TokenSample:
    return TokenSample(
        sample_id="sample-1",
        text="fake",
        token_ids=[100, 101, 102, 103, 104, 105, 106],
        tokenizer_name="fake-tokenizer",
    )


def test_contiguous_packetization_covers_every_position_once() -> None:
    packets = packetize_contiguous(make_sample(), tokens_per_packet=3)

    positions = [
        position
        for packet in packets
        for position in packet.token_positions
    ]
    assert positions == list(range(7))
    assert len(set(positions)) == 7


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
