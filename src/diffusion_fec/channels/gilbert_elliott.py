"""Gilbert-Elliott packet-erasure channel."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from diffusion_fec.channels.random_loss import RandomLossResult
from diffusion_fec.types import Packet


GE_STATE_GOOD = "good"
GE_STATE_BAD = "bad"


def apply_gilbert_elliott_loss(
    packets: Sequence[Packet],
    *,
    good_loss_rate: float,
    bad_loss_rate: float,
    good_to_bad_rate: float,
    bad_to_good_rate: float,
    seed: int | None = None,
    initial_state: str = GE_STATE_GOOD,
) -> RandomLossResult:
    """Drop packets from a two-state bursty channel in wire-ID order."""

    _validate_rate(good_loss_rate, "good_loss_rate")
    _validate_rate(bad_loss_rate, "bad_loss_rate")
    _validate_rate(good_to_bad_rate, "good_to_bad_rate")
    _validate_rate(bad_to_good_rate, "bad_to_good_rate")
    if initial_state not in {GE_STATE_GOOD, GE_STATE_BAD}:
        raise ValueError("initial_state must be 'good' or 'bad'")

    ordered_packets = tuple(sorted(packets, key=lambda packet: packet.wire_id))
    wire_ids = [packet.wire_id for packet in ordered_packets]
    if len(set(wire_ids)) != len(wire_ids):
        raise ValueError("packets must have unique wire_id values")

    rng = Random(seed)
    state = initial_state
    received: list[Packet] = []
    dropped: list[Packet] = []
    for packet in ordered_packets:
        loss_rate = good_loss_rate if state == GE_STATE_GOOD else bad_loss_rate
        if rng.random() < loss_rate:
            dropped.append(packet)
        else:
            received.append(packet)

        transition_rate = (
            good_to_bad_rate if state == GE_STATE_GOOD else bad_to_good_rate
        )
        if rng.random() < transition_rate:
            state = GE_STATE_BAD if state == GE_STATE_GOOD else GE_STATE_GOOD

    return RandomLossResult(received=tuple(received), dropped=tuple(dropped))


def _validate_rate(value: float, name: str) -> None:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
