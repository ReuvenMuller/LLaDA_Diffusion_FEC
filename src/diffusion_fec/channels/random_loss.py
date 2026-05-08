"""Seeded IID packet-erasure channel."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Sequence

from diffusion_fec.types import Packet


@dataclass(frozen=True)
class RandomLossResult:
    """Surviving and dropped packets from a random packet-erasure channel."""

    received: tuple[Packet, ...]
    dropped: tuple[Packet, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "received": [packet.to_dict() for packet in self.received],
            "dropped": [packet.to_dict() for packet in self.dropped],
        }


def apply_random_loss(
    packets: Sequence[Packet],
    *,
    loss_rate: float,
    seed: int | None = None,
) -> RandomLossResult:
    """Drop packets independently with probability ``loss_rate`` in wire-ID order."""

    if not isinstance(loss_rate, int | float):
        raise TypeError("loss_rate must be numeric")
    if loss_rate < 0.0 or loss_rate > 1.0:
        raise ValueError("loss_rate must be between 0.0 and 1.0")

    ordered_packets = tuple(sorted(packets, key=lambda packet: packet.wire_id))
    wire_ids = [packet.wire_id for packet in ordered_packets]
    if len(set(wire_ids)) != len(wire_ids):
        raise ValueError("packets must have unique wire_id values")

    rng = Random(seed)
    received: list[Packet] = []
    dropped: list[Packet] = []
    for packet in ordered_packets:
        if rng.random() < loss_rate:
            dropped.append(packet)
        else:
            received.append(packet)

    return RandomLossResult(received=tuple(received), dropped=tuple(dropped))
