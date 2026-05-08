"""Streaming-window XOR repair baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

from diffusion_fec.baselines.overhead import (
    OverheadSummary,
    estimate_hash_overhead_ratio,
    repair_token_overhead_ratio,
    token_bit_width_for_vocab,
)
from diffusion_fec.coding.packetizer import (
    SOURCE_PACKET_INDEX_METADATA_KEY,
    WIRE_INTERLEAVING_MATRIX,
    SourceLayoutConfig,
    WireInterleavingConfig,
    build_reconstruction_plan,
    packetize_sample,
)
from diffusion_fec.types import Packet, ReconstructionPlan, TokenSample


STREAMING_WINDOW_SCHEME = "streaming_window"
STREAMING_WINDOW_METADATA_KEY = "streaming_window"


@dataclass(frozen=True)
class StreamingWindowConfig:
    """Streaming-window XOR baseline configuration."""

    window_size: int = 5
    window_stride: int = 1
    target_hash_bits: int | None = None
    vocab_size: int | None = None
    target_overhead_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.window_size, int):
            raise TypeError("window_size must be an int")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not isinstance(self.window_stride, int):
            raise TypeError("window_stride must be an int")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if self.target_hash_bits is not None:
            if not isinstance(self.target_hash_bits, int):
                raise TypeError("target_hash_bits must be an int when set")
            if self.target_hash_bits <= 0:
                raise ValueError("target_hash_bits must be positive")
        if self.vocab_size is not None:
            if not isinstance(self.vocab_size, int):
                raise TypeError("vocab_size must be an int when set")
            if self.vocab_size <= 0:
                raise ValueError("vocab_size must be positive")
        if self.target_overhead_ratio is not None and self.target_overhead_ratio < 0.0:
            raise ValueError("target_overhead_ratio must be non-negative")
        if self.target_hash_bits is not None and self.vocab_size is None:
            raise ValueError("vocab_size is required when target_hash_bits is set")

    def resolved_target_overhead_ratio(self) -> float | None:
        if self.target_overhead_ratio is not None:
            return self.target_overhead_ratio
        if self.target_hash_bits is None:
            return None
        if self.vocab_size is None:
            raise ValueError("vocab_size is required when target_hash_bits is set")
        return estimate_hash_overhead_ratio(
            hash_bits=self.target_hash_bits,
            vocab_size=self.vocab_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "target_hash_bits": self.target_hash_bits,
            "vocab_size": self.vocab_size,
            "target_overhead_ratio": self.target_overhead_ratio,
        }


@dataclass(frozen=True)
class StreamingWindowEncoded:
    """Encoded packet set and overhead record for streaming-window repair."""

    packets: tuple[Packet, ...]
    source_packets: tuple[Packet, ...]
    repair_packets: tuple[Packet, ...]
    overhead: OverheadSummary
    config: StreamingWindowConfig

    @property
    def source_packet_count(self) -> int:
        return len(self.source_packets)

    @property
    def extra_packet_count(self) -> int:
        return len(self.repair_packets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "packets": [packet.to_dict() for packet in self.packets],
            "source_packets": [packet.to_dict() for packet in self.source_packets],
            "repair_packets": [packet.to_dict() for packet in self.repair_packets],
            "source_packet_count": self.source_packet_count,
            "extra_packet_count": self.extra_packet_count,
            "overhead": self.overhead.to_dict(),
            "config": self.config.to_dict(),
        }


def encode_streaming_window(
    sample: TokenSample,
    *,
    tokens_per_packet: int,
    config: StreamingWindowConfig | None = None,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
) -> StreamingWindowEncoded:
    """Encode source packets plus streaming-window XOR repair packets."""

    config = config or StreamingWindowConfig()
    source_layout = source_layout or SourceLayoutConfig()
    wire_interleaving = wire_interleaving or WireInterleavingConfig()
    source_packets = packetize_sample(
        sample,
        tokens_per_packet=tokens_per_packet,
        source_layout=source_layout,
        wire_interleaving=WireInterleavingConfig(),
    )
    stride = _resolve_stride(
        source_packet_count=len(source_packets),
        total_tokens=len(sample.token_ids),
        tokens_per_packet=tokens_per_packet,
        config=config,
    )
    repair_packets = _build_repair_packets(
        source_id=sample.sample_id,
        source_packets=source_packets,
        tokens_per_packet=tokens_per_packet,
        config=config,
        stride=stride,
    )
    packets = _assign_wire_ids(
        packets=[*source_packets, *repair_packets],
        wire_interleaving=wire_interleaving,
    )
    source_count = len(source_packets)
    source_packets = tuple(packets[:source_count])
    repair_packets = tuple(packets[source_count:])
    return StreamingWindowEncoded(
        packets=tuple(packets),
        source_packets=source_packets,
        repair_packets=repair_packets,
        overhead=_overhead_summary(
            total_tokens=len(sample.token_ids),
            tokens_per_packet=tokens_per_packet,
            repair_packet_count=len(repair_packets),
            config=config,
        ),
        config=config,
    )


def reconstruct_streaming_window(
    *,
    total_tokens: int,
    received_packets: Sequence[Packet],
    tokens_per_packet: int,
) -> ReconstructionPlan:
    """Build a known/unguided plan after streaming-window peeling repair."""

    recovered_packets: dict[int, Packet] = {}
    repair_packets: list[Packet] = []
    for packet in received_packets:
        if packet.kind == "data":
            recovered_packets[_source_packet_index(packet)] = packet
        elif packet.kind == "streaming_window_repair":
            repair_packets.append(packet)

    progress = True
    while progress:
        progress = False
        for repair_packet in sorted(repair_packets, key=lambda packet: packet.wire_id):
            metadata = _repair_metadata(repair_packet)
            neighbors = list(metadata["neighbor_source_packet_indices"])
            unresolved = [
                source_packet_index
                for source_packet_index in neighbors
                if source_packet_index not in recovered_packets
            ]
            if len(unresolved) != 1:
                continue
            missing_source_packet_index = unresolved[0]
            missing_neighbor_offset = neighbors.index(missing_source_packet_index)
            recovered_tokens = list(repair_packet.token_ids)
            for source_packet_index in neighbors:
                if source_packet_index == missing_source_packet_index:
                    continue
                source_packet = recovered_packets[source_packet_index]
                recovered_tokens = [
                    left ^ right
                    for left, right in zip(
                        recovered_tokens,
                        _pad_tokens(source_packet.token_ids, tokens_per_packet),
                    )
                ]
            neighbor_lengths = list(metadata["neighbor_lengths"])
            neighbor_positions = list(metadata["neighbor_token_positions"])
            recovered_tokens = recovered_tokens[: neighbor_lengths[missing_neighbor_offset]]
            recovered_packets[missing_source_packet_index] = Packet(
                source_id=repair_packet.source_id,
                wire_id=repair_packet.wire_id,
                kind="data",
                token_ids=tuple(recovered_tokens),
                token_positions=tuple(neighbor_positions[missing_neighbor_offset]),
                metadata={
                    SOURCE_PACKET_INDEX_METADATA_KEY: missing_source_packet_index,
                    "recovered_by": STREAMING_WINDOW_SCHEME,
                    "repair_wire_id": repair_packet.wire_id,
                },
            )
            progress = True

    return build_reconstruction_plan(
        total_tokens=total_tokens,
        received_packets=tuple(recovered_packets.values()),
    )


def _build_repair_packets(
    *,
    source_id: str,
    source_packets: Sequence[Packet],
    tokens_per_packet: int,
    config: StreamingWindowConfig,
    stride: int,
) -> list[Packet]:
    repair_packets: list[Packet] = []
    for start_index in range(0, len(source_packets), stride):
        neighbors = list(range(start_index, min(start_index + config.window_size, len(source_packets))))
        if len(neighbors) < 2:
            continue
        repair_tokens = [0] * tokens_per_packet
        neighbor_lengths: list[int] = []
        neighbor_positions: list[list[int]] = []
        for source_packet_index in neighbors:
            packet = source_packets[source_packet_index]
            repair_tokens = [
                left ^ right
                for left, right in zip(repair_tokens, _pad_tokens(packet.token_ids, tokens_per_packet))
            ]
            neighbor_lengths.append(len(packet.token_ids))
            neighbor_positions.append(list(packet.token_positions))
        repair_packets.append(
            Packet(
                source_id=source_id,
                wire_id=len(source_packets) + len(repair_packets),
                kind="streaming_window_repair",
                token_ids=tuple(repair_tokens),
                token_positions=tuple(range(tokens_per_packet)),
                metadata={
                    STREAMING_WINDOW_METADATA_KEY: {
                        "scheme": STREAMING_WINDOW_SCHEME,
                        "repair_index": len(repair_packets),
                        "window_start_source_packet_index": start_index,
                        "neighbor_source_packet_indices": neighbors,
                        "neighbor_lengths": neighbor_lengths,
                        "neighbor_token_positions": neighbor_positions,
                        "tokens_per_packet": tokens_per_packet,
                    }
                },
            )
        )
    return repair_packets


def _resolve_stride(
    *,
    source_packet_count: int,
    total_tokens: int,
    tokens_per_packet: int,
    config: StreamingWindowConfig,
) -> int:
    if source_packet_count <= 1:
        return config.window_stride
    target_overhead_ratio = config.resolved_target_overhead_ratio()
    if target_overhead_ratio is None:
        return config.window_stride
    best_stride = 1
    best_score: tuple[float, float, int, float] | None = None
    for stride in range(1, source_packet_count + 1):
        repair_count = _repair_count_for_stride(
            source_packet_count=source_packet_count,
            window_size=config.window_size,
            stride=stride,
        )
        actual_ratio = repair_token_overhead_ratio(
            total_tokens=total_tokens,
            tokens_per_packet=tokens_per_packet,
            repair_packet_count=repair_count,
        )
        gap = abs(actual_ratio - target_overhead_ratio)
        overshoot = max(0.0, actual_ratio - target_overhead_ratio)
        score = (gap, overshoot, repair_count, actual_ratio)
        if best_score is None or score < best_score:
            best_score = score
            best_stride = stride
    return best_stride


def _repair_count_for_stride(
    *,
    source_packet_count: int,
    window_size: int,
    stride: int,
) -> int:
    repair_count = 0
    for start_index in range(0, source_packet_count, stride):
        if len(range(start_index, min(start_index + window_size, source_packet_count))) >= 2:
            repair_count += 1
    return repair_count


def _overhead_summary(
    *,
    total_tokens: int,
    tokens_per_packet: int,
    repair_packet_count: int,
    config: StreamingWindowConfig,
) -> OverheadSummary:
    token_bit_width = (
        None if config.vocab_size is None else token_bit_width_for_vocab(config.vocab_size)
    )
    target_overhead_ratio = config.resolved_target_overhead_ratio()
    return OverheadSummary(
        target_hash_bits=config.target_hash_bits,
        vocab_size=config.vocab_size,
        token_bit_width=token_bit_width,
        target_overhead_ratio=target_overhead_ratio,
        repair_packet_count=repair_packet_count,
        repair_token_budget=repair_packet_count * tokens_per_packet,
        actual_repair_token_overhead_ratio=repair_token_overhead_ratio(
            total_tokens=total_tokens,
            tokens_per_packet=tokens_per_packet,
            repair_packet_count=repair_packet_count,
        ),
        metadata_bits_total=0,
    )


def _assign_wire_ids(
    *,
    packets: Sequence[Packet],
    wire_interleaving: WireInterleavingConfig,
) -> tuple[Packet, ...]:
    wire_order = _build_wire_order(len(packets), wire_interleaving)
    packet_index_to_wire_id = {
        packet_index: wire_id
        for wire_id, packet_index in enumerate(wire_order)
    }
    return tuple(
        Packet(
            source_id=packet.source_id,
            wire_id=packet_index_to_wire_id[packet_index],
            kind=packet.kind,
            token_ids=packet.token_ids,
            token_positions=packet.token_positions,
            metadata=dict(packet.metadata),
        )
        for packet_index, packet in enumerate(packets)
    )


def _build_wire_order(
    packet_count: int,
    wire_interleaving: WireInterleavingConfig,
) -> list[int]:
    if (
        wire_interleaving.mode != WIRE_INTERLEAVING_MATRIX
        or wire_interleaving.span <= 1
        or packet_count <= 1
    ):
        return list(range(packet_count))
    span = min(wire_interleaving.span, packet_count)
    rows = ceil(packet_count / span)
    matrix: list[list[int | None]] = [[None for _ in range(span)] for _ in range(rows)]
    packet_index = 0
    for row in range(rows):
        for column in range(span):
            if packet_index < packet_count:
                matrix[row][column] = packet_index
                packet_index += 1
    wire_order: list[int] = []
    for column in range(span):
        for row in range(rows):
            value = matrix[row][column]
            if value is not None:
                wire_order.append(value)
    return wire_order


def _pad_tokens(token_ids: Sequence[int], tokens_per_packet: int) -> list[int]:
    return [*token_ids, *([0] * (tokens_per_packet - len(token_ids)))]


def _source_packet_index(packet: Packet) -> int:
    value = packet.metadata.get(SOURCE_PACKET_INDEX_METADATA_KEY)
    if value is None:
        raise ValueError("data packet is missing source_packet_index metadata")
    return int(value)


def _repair_metadata(packet: Packet) -> dict[str, Any]:
    metadata = packet.metadata.get(STREAMING_WINDOW_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError("streaming-window repair packet is missing metadata")
    if metadata.get("scheme") != STREAMING_WINDOW_SCHEME:
        raise ValueError("streaming-window repair packet metadata has unsupported scheme")
    return metadata
