import pytest

from diffusion_fec.channels.packet_loss import (
    CHANNEL_BURST,
    CHANNEL_GILBERT_ELLIOTT,
    CHANNEL_RANDOM_IID,
    PacketLossChannelConfig,
    apply_packet_loss_channel,
    resolve_burst_length,
    resolve_packet_loss_channel_config,
)
from diffusion_fec.coding.packetizer import packetize_contiguous
from diffusion_fec.types import Packet, TokenSample


def make_packets():
    sample = TokenSample(
        sample_id="sample-1",
        text="fake",
        token_ids=list(range(8)),
        tokenizer_name="fake-tokenizer",
    )
    return packetize_contiguous(sample, tokens_per_packet=1)


def wire_ids(packets):
    return [packet.wire_id for packet in packets]


def make_data_and_repair_packets():
    packets = list(make_packets())
    packets.extend(
        [
            Packet(
                source_id="sample-1",
                wire_id=8,
                kind="parity",
                token_ids=(0,),
                token_positions=(0,),
            ),
            Packet(
                source_id="sample-1",
                wire_id=9,
                kind="parity",
                token_ids=(0,),
                token_positions=(1,),
            ),
        ]
    )
    return tuple(packets)


def test_packet_loss_config_applies_iid_random_loss() -> None:
    result = apply_packet_loss_channel(
        make_packets(),
        config=PacketLossChannelConfig(
            mode=CHANNEL_RANDOM_IID,
            loss_rate=0.0,
            seed=1,
        ),
    )

    assert wire_ids(result.received) == list(range(8))
    assert result.dropped == ()


def test_packet_loss_config_applies_burst_loss() -> None:
    result = apply_packet_loss_channel(
        make_packets(),
        config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=2,
            burst_length=3,
        ),
    )

    assert wire_ids(result.dropped) == [2, 3, 4]
    assert wire_ids(result.received) == [0, 1, 5, 6, 7]


def test_burst_channel_requires_length_when_applied() -> None:
    with pytest.raises(ValueError, match="burst channel requires burst_length"):
        apply_packet_loss_channel(
            make_packets(),
            config=PacketLossChannelConfig(mode=CHANNEL_BURST),
        )


def test_burst_loss_rate_resolves_against_all_transmitted_packets() -> None:
    packets = make_data_and_repair_packets()
    config = PacketLossChannelConfig(
        mode=CHANNEL_BURST,
        burst_start_wire_id=0,
        burst_loss_rate=0.5,
    )

    resolved = resolve_packet_loss_channel_config(packets, config=config)
    result = apply_packet_loss_channel(packets, config=config)

    assert resolve_burst_length(
        total_transmitted_packet_count=len(packets),
        burst_loss_rate=0.5,
    ) == 5
    assert resolved.burst_length == 5
    assert resolved.resolved_burst_length == 5
    assert wire_ids(result.dropped) == [0, 1, 2, 3, 4]
    assert len(result.dropped) == len(packets) // 2


def test_fixed_burst_length_behavior_is_unchanged_with_repair_packets() -> None:
    result = apply_packet_loss_channel(
        make_data_and_repair_packets(),
        config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=8,
            burst_length=1,
        ),
    )

    assert wire_ids(result.dropped) == [8]


def test_gilbert_elliott_channel_is_seeded_and_stateful() -> None:
    config = PacketLossChannelConfig(
        mode=CHANNEL_GILBERT_ELLIOTT,
        seed=3,
        good_loss_rate=0.0,
        bad_loss_rate=1.0,
        good_to_bad_rate=1.0,
        bad_to_good_rate=0.0,
        initial_state="good",
    )

    first = apply_packet_loss_channel(make_packets(), config=config)
    second = apply_packet_loss_channel(make_packets(), config=config)

    assert wire_ids(first.received) == [0]
    assert wire_ids(first.dropped) == [1, 2, 3, 4, 5, 6, 7]
    assert wire_ids(second.received) == wire_ids(first.received)
    assert wire_ids(second.dropped) == wire_ids(first.dropped)


def test_channel_config_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError, match="good_to_bad_rate"):
        PacketLossChannelConfig(
            mode=CHANNEL_GILBERT_ELLIOTT,
            good_to_bad_rate=1.5,
        )
