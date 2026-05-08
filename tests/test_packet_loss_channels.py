import pytest

from diffusion_fec.channels.packet_loss import (
    CHANNEL_BURST,
    CHANNEL_GILBERT_ELLIOTT,
    CHANNEL_RANDOM_IID,
    PacketLossChannelConfig,
    apply_packet_loss_channel,
)
from diffusion_fec.coding.packetizer import packetize_contiguous
from diffusion_fec.types import TokenSample


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
