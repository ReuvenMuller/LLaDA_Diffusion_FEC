"""Packet-loss diagnostics for artifact rows and events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from diffusion_fec.channels.random_loss import RandomLossResult
from diffusion_fec.types import Packet


DATA_PACKET_KIND = "data"


def compute_packet_loss_diagnostics(
    *,
    loss_result: RandomLossResult,
    source_token_count: int,
    channel_lost_position_count: int,
) -> dict[str, Any]:
    """Return explicit packet/token loss rates from a channel result.

    Repair packets are every transmitted packet whose kind is not ``data``.
    They are included in wire-packet loss but excluded from source-token loss.
    """

    if source_token_count < 0:
        raise ValueError("source_token_count must be non-negative")
    if channel_lost_position_count < 0:
        raise ValueError("channel_lost_position_count must be non-negative")
    received_packets = tuple(loss_result.received)
    dropped_packets = tuple(loss_result.dropped)
    transmitted_packets = (*received_packets, *dropped_packets)
    total_count = len(transmitted_packets)
    received_count = len(received_packets)
    dropped_count = len(dropped_packets)
    source_packet_count = _count_packets(transmitted_packets, source=True)
    repair_packet_count = total_count - source_packet_count
    dropped_data_count = _count_packets(dropped_packets, source=True)
    dropped_repair_count = dropped_count - dropped_data_count
    return {
        "total_transmitted_packet_count": total_count,
        "received_packet_count": received_count,
        "dropped_packet_count": dropped_count,
        "actual_wire_packet_loss_rate": _rate(dropped_count, total_count),
        "source_packet_count": source_packet_count,
        "dropped_data_packet_count": dropped_data_count,
        "dropped_repair_packet_count": dropped_repair_count,
        "actual_data_packet_loss_rate": _rate(dropped_data_count, source_packet_count),
        "actual_repair_packet_loss_rate": _rate(dropped_repair_count, repair_packet_count),
        "source_token_count": int(source_token_count),
        "channel_lost_position_count": int(channel_lost_position_count),
        "actual_source_token_loss_rate": _rate(
            channel_lost_position_count,
            source_token_count,
        ),
    }


def _count_packets(packets: Sequence[Packet], *, source: bool) -> int:
    return sum((packet.kind == DATA_PACKET_KIND) == source for packet in packets)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
