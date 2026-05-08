from diffusion_fec.channels.random_loss import apply_random_loss
from diffusion_fec.coding.packetizer import packetize_contiguous
from diffusion_fec.types import TokenSample


def make_packets():
    sample = TokenSample(
        sample_id="sample-1",
        text="fake",
        token_ids=list(range(12)),
        tokenizer_name="fake-tokenizer",
    )
    return packetize_contiguous(sample, tokens_per_packet=1)


def wire_ids(packets):
    return [packet.wire_id for packet in packets]


def test_loss_rate_zero_keeps_all_packets() -> None:
    packets = make_packets()
    result = apply_random_loss(packets, loss_rate=0.0, seed=123)

    assert wire_ids(result.received) == list(range(12))
    assert result.dropped == ()


def test_loss_rate_one_drops_all_packets() -> None:
    packets = make_packets()
    result = apply_random_loss(packets, loss_rate=1.0, seed=123)

    assert result.received == ()
    assert wire_ids(result.dropped) == list(range(12))


def test_same_seed_reproduces_received_and_dropped_wire_ids() -> None:
    packets = make_packets()
    first = apply_random_loss(list(reversed(packets)), loss_rate=0.4, seed=7)
    second = apply_random_loss(packets, loss_rate=0.4, seed=7)

    assert wire_ids(first.received) == wire_ids(second.received)
    assert wire_ids(first.dropped) == wire_ids(second.dropped)


def test_different_seeds_can_produce_different_outcomes() -> None:
    packets = make_packets()
    first = apply_random_loss(packets, loss_rate=0.5, seed=1)
    second = apply_random_loss(packets, loss_rate=0.5, seed=2)

    assert wire_ids(first.dropped) != wire_ids(second.dropped)
