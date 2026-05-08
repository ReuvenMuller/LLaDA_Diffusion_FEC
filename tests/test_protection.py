import pytest

from diffusion_fec.coding.packetizer import packetize_contiguous
from diffusion_fec.coding.protection import (
    LOOKBACK_HASH_METADATA_KEY,
    LOOKBACK_1_SCHEME,
    LookbackHashMetadata,
    PacketRef,
    PositionHash,
    attach_lookback_hashes,
    extract_received_hash_metadata,
)
from diffusion_fec.coding.token_hash import build_token_hash_map
from diffusion_fec.types import Packet, TokenSample


def make_packets():
    sample = TokenSample(
        sample_id="sample-1",
        text="synthetic",
        token_ids=[10, 11, 12, 13],
        tokenizer_name="fake",
    )
    return packetize_contiguous(sample, tokens_per_packet=1)


def make_hash_map():
    return build_token_hash_map(
        vocab_size=32,
        hash_bits=4,
        decode_token=lambda token_id: f"token-{token_id}",
    )


def test_packet_i_carries_hashes_for_packet_i_minus_one() -> None:
    packets = make_packets()
    token_hash = make_hash_map()

    protected_packets = attach_lookback_hashes(packets, token_hash)

    assert LOOKBACK_HASH_METADATA_KEY not in protected_packets[0].metadata
    for index in range(1, len(protected_packets)):
        metadata = LookbackHashMetadata.from_dict(
            protected_packets[index].metadata[LOOKBACK_HASH_METADATA_KEY]
        )
        assert metadata.scheme == LOOKBACK_1_SCHEME
        assert metadata.protecting_packet == PacketRef(
            source_id="sample-1",
            wire_id=index,
        )
        assert metadata.protected_packet == PacketRef(
            source_id="sample-1",
            wire_id=index - 1,
        )
        assert metadata.hashes == (
            PositionHash(
                position=index - 1,
                hash_value=token_hash.bucket_for_token(10 + index - 1),
            ),
        )


def test_attach_lookback_hashes_does_not_mutate_input_packets() -> None:
    packets = make_packets()
    original_metadata = [dict(packet.metadata) for packet in packets]

    protected_packets = attach_lookback_hashes(packets, make_hash_map())

    assert [packet.metadata for packet in packets] == original_metadata
    assert protected_packets is not packets
    assert protected_packets[1] is not packets[1]


def test_extract_received_hash_metadata_uses_surviving_packets_only() -> None:
    protected_packets = attach_lookback_hashes(make_packets(), make_hash_map())

    received_hashes = extract_received_hash_metadata([protected_packets[2]])

    assert set(received_hashes) == {1}


def test_extract_received_hash_metadata_allows_duplicate_identical_hashes() -> None:
    metadata = LookbackHashMetadata(
        scheme=LOOKBACK_1_SCHEME,
        protecting_packet=PacketRef(source_id="sample-1", wire_id=1),
        protected_packet=PacketRef(source_id="sample-1", wire_id=0),
        hashes=(PositionHash(position=0, hash_value=3),),
    ).to_dict()
    packet_a = Packet(
        source_id="sample-1",
        wire_id=1,
        kind="data",
        token_ids=[11],
        token_positions=[1],
        metadata={LOOKBACK_HASH_METADATA_KEY: metadata},
    )
    packet_b = Packet(
        source_id="sample-1",
        wire_id=2,
        kind="data",
        token_ids=[12],
        token_positions=[2],
        metadata={
            LOOKBACK_HASH_METADATA_KEY: {
                **metadata,
                "protecting_packet": {"source_id": "sample-1", "wire_id": 2},
            }
        },
    )

    assert extract_received_hash_metadata([packet_a, packet_b]) == {0: 3}


def test_extract_received_hash_metadata_raises_for_conflicting_hashes() -> None:
    packet_a = Packet(
        source_id="sample-1",
        wire_id=1,
        kind="data",
        token_ids=[11],
        token_positions=[1],
        metadata={
            LOOKBACK_HASH_METADATA_KEY: LookbackHashMetadata(
                scheme=LOOKBACK_1_SCHEME,
                protecting_packet=PacketRef(source_id="sample-1", wire_id=1),
                protected_packet=PacketRef(source_id="sample-1", wire_id=0),
                hashes=(PositionHash(position=0, hash_value=3),),
            ).to_dict()
        },
    )
    packet_b = Packet(
        source_id="sample-1",
        wire_id=2,
        kind="data",
        token_ids=[12],
        token_positions=[2],
        metadata={
            LOOKBACK_HASH_METADATA_KEY: LookbackHashMetadata(
                scheme=LOOKBACK_1_SCHEME,
                protecting_packet=PacketRef(source_id="sample-1", wire_id=2),
                protected_packet=PacketRef(source_id="sample-1", wire_id=0),
                hashes=(PositionHash(position=0, hash_value=4),),
            ).to_dict()
        },
    )

    with pytest.raises(ValueError, match="conflicting hash metadata"):
        extract_received_hash_metadata([packet_a, packet_b])


def test_extract_received_hash_metadata_validates_carrier_identity() -> None:
    packet = Packet(
        source_id="sample-1",
        wire_id=1,
        kind="data",
        token_ids=[11],
        token_positions=[1],
        metadata={
            LOOKBACK_HASH_METADATA_KEY: LookbackHashMetadata(
                scheme=LOOKBACK_1_SCHEME,
                protecting_packet=PacketRef(source_id="sample-1", wire_id=99),
                protected_packet=PacketRef(source_id="sample-1", wire_id=0),
                hashes=(PositionHash(position=0, hash_value=3),),
            ).to_dict()
        },
    )

    with pytest.raises(ValueError, match="protecting packet does not match"):
        extract_received_hash_metadata([packet])
