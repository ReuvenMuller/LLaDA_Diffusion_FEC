"""Deterministic contiguous burst packet-erasure channel."""

from __future__ import annotations

from collections.abc import Sequence

from diffusion_fec.channels.random_loss import RandomLossResult
from diffusion_fec.types import Packet


def apply_burst_loss(
    packets: Sequence[Packet],
    *,
    burst_start_wire_id: int,
    burst_length: int,
) -> RandomLossResult:
    """Drop a contiguous burst of packets in wire-ID order."""

    if not isinstance(burst_start_wire_id, int):
        raise TypeError("burst_start_wire_id must be an int")
    if not isinstance(burst_length, int):
        raise TypeError("burst_length must be an int")
    if burst_start_wire_id < 0:
        raise ValueError("burst_start_wire_id must be non-negative")
    if burst_length < 0:
        raise ValueError("burst_length must be non-negative")

    ordered_packets = tuple(sorted(packets, key=lambda packet: packet.wire_id))
    wire_ids = [packet.wire_id for packet in ordered_packets]
    if len(set(wire_ids)) != len(wire_ids):
        raise ValueError("packets must have unique wire_id values")

    burst_wire_ids = set(
        range(burst_start_wire_id, burst_start_wire_id + burst_length)
    )
    received: list[Packet] = []
    dropped: list[Packet] = []
    for packet in ordered_packets:
        if packet.wire_id in burst_wire_ids:
            dropped.append(packet)
        else:
            received.append(packet)
    return RandomLossResult(received=tuple(received), dropped=tuple(dropped))

