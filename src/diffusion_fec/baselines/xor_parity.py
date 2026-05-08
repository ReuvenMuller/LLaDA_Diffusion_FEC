"""XOR parity matched-overhead baseline."""

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


XOR_PARITY_SCHEME = "xor_parity"
XOR_PARITY_METADATA_KEY = "xor_parity"


@dataclass(frozen=True)
class XorParityConfig:
    """XOR parity baseline configuration."""

    data_packets_per_stripe: int = 4
    stripe_stride: int | None = None
    target_hash_bits: int | None = None
    vocab_size: int | None = None
    target_overhead_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data_packets_per_stripe, int):
            raise TypeError("data_packets_per_stripe must be an int")
        if self.data_packets_per_stripe < 2:
            raise ValueError("data_packets_per_stripe must be at least 2")
        if self.stripe_stride is not None:
            if not isinstance(self.stripe_stride, int):
                raise TypeError("stripe_stride must be an int when set")
            if self.stripe_stride <= 0:
                raise ValueError("stripe_stride must be positive")
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
            "data_packets_per_stripe": self.data_packets_per_stripe,
            "stripe_stride": self.stripe_stride,
            "target_hash_bits": self.target_hash_bits,
            "vocab_size": self.vocab_size,
            "target_overhead_ratio": self.target_overhead_ratio,
        }


@dataclass(frozen=True)
class XorParityEncoded:
    """Encoded packet set and overhead record for XOR parity."""

    packets: tuple[Packet, ...]
    source_packets: tuple[Packet, ...]
    parity_packets: tuple[Packet, ...]
    overhead: OverheadSummary
    config: XorParityConfig

    @property
    def source_packet_count(self) -> int:
        return len(self.source_packets)

    @property
    def extra_packet_count(self) -> int:
        return len(self.parity_packets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "packets": [packet.to_dict() for packet in self.packets],
            "source_packets": [packet.to_dict() for packet in self.source_packets],
            "parity_packets": [packet.to_dict() for packet in self.parity_packets],
            "source_packet_count": self.source_packet_count,
            "extra_packet_count": self.extra_packet_count,
            "overhead": self.overhead.to_dict(),
            "config": self.config.to_dict(),
        }


def encode_xor_parity(
    sample: TokenSample,
    *,
    tokens_per_packet: int,
    config: XorParityConfig | None = None,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
) -> XorParityEncoded:
    """Encode source packets plus XOR parity repair packets."""

    config = config or XorParityConfig()
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
    parity_packets = _build_parity_packets(
        source_id=sample.sample_id,
        source_packets=source_packets,
        tokens_per_packet=tokens_per_packet,
        config=config,
        stride=stride,
    )
    packets = _assign_wire_ids(
        packets=[*source_packets, *parity_packets],
        wire_interleaving=wire_interleaving,
    )
    source_count = len(source_packets)
    source_packets = tuple(packets[:source_count])
    parity_packets = tuple(packets[source_count:])
    return XorParityEncoded(
        packets=tuple(packets),
        source_packets=source_packets,
        parity_packets=parity_packets,
        overhead=_overhead_summary(
            total_tokens=len(sample.token_ids),
            tokens_per_packet=tokens_per_packet,
            repair_packet_count=len(parity_packets),
            config=config,
        ),
        config=config,
    )


def reconstruct_xor_parity(
    *,
    total_tokens: int,
    received_packets: Sequence[Packet],
    tokens_per_packet: int,
) -> ReconstructionPlan:
    """Build a known/unguided plan after XOR repair."""

    recovered_packets: dict[int, Packet] = {}
    parity_packets: list[Packet] = []
    for packet in received_packets:
        if packet.kind == "data":
            recovered_packets[_source_packet_index(packet)] = packet
        elif packet.kind == "parity":
            parity_packets.append(packet)

    for parity_packet in sorted(parity_packets, key=lambda packet: packet.wire_id):
        metadata = _parity_metadata(parity_packet)
        member_indices = list(metadata["stripe_member_source_packet_indices"])
        missing_members = [
            source_packet_index
            for source_packet_index in member_indices
            if source_packet_index not in recovered_packets
        ]
        if len(missing_members) != 1:
            continue

        missing_source_packet_index = missing_members[0]
        missing_member_offset = member_indices.index(missing_source_packet_index)
        recovered_tokens = list(parity_packet.token_ids)
        for source_packet_index in member_indices:
            if source_packet_index == missing_source_packet_index:
                continue
            source_packet = recovered_packets[source_packet_index]
            padded_tokens = _pad_tokens(source_packet.token_ids, tokens_per_packet)
            recovered_tokens = [
                left ^ right
                for left, right in zip(recovered_tokens, padded_tokens)
            ]

        member_lengths = list(metadata["stripe_member_lengths"])
        member_positions = list(metadata["stripe_member_token_positions"])
        recovered_tokens = recovered_tokens[: member_lengths[missing_member_offset]]
        recovered_packets[missing_source_packet_index] = Packet(
            source_id=parity_packet.source_id,
            wire_id=parity_packet.wire_id,
            kind="data",
            token_ids=tuple(recovered_tokens),
            token_positions=tuple(member_positions[missing_member_offset]),
            metadata={
                SOURCE_PACKET_INDEX_METADATA_KEY: missing_source_packet_index,
                "recovered_by": XOR_PARITY_SCHEME,
                "repair_wire_id": parity_packet.wire_id,
            },
        )

    return build_reconstruction_plan(
        total_tokens=total_tokens,
        received_packets=tuple(recovered_packets.values()),
    )


def _build_parity_packets(
    *,
    source_id: str,
    source_packets: Sequence[Packet],
    tokens_per_packet: int,
    config: XorParityConfig,
    stride: int,
) -> list[Packet]:
    parity_packets: list[Packet] = []
    for stripe_id, start_index in enumerate(range(0, len(source_packets), stride)):
        stripe_packets = source_packets[start_index:start_index + config.data_packets_per_stripe]
        if len(stripe_packets) < 2:
            continue
        parity_tokens = [0] * tokens_per_packet
        member_indices: list[int] = []
        member_lengths: list[int] = []
        member_positions: list[list[int]] = []
        for packet in stripe_packets:
            parity_tokens = [
                left ^ right
                for left, right in zip(parity_tokens, _pad_tokens(packet.token_ids, tokens_per_packet))
            ]
            member_indices.append(_source_packet_index(packet))
            member_lengths.append(len(packet.token_ids))
            member_positions.append(list(packet.token_positions))

        parity_packets.append(
            Packet(
                source_id=source_id,
                wire_id=len(source_packets) + len(parity_packets),
                kind="parity",
                token_ids=tuple(parity_tokens),
                token_positions=tuple(range(tokens_per_packet)),
                metadata={
                    XOR_PARITY_METADATA_KEY: {
                        "scheme": XOR_PARITY_SCHEME,
                        "stripe_id": stripe_id,
                        "stripe_member_source_packet_indices": member_indices,
                        "stripe_member_lengths": member_lengths,
                        "stripe_member_token_positions": member_positions,
                        "tokens_per_packet": tokens_per_packet,
                    }
                },
            )
        )
    return parity_packets


def _resolve_stride(
    *,
    source_packet_count: int,
    total_tokens: int,
    tokens_per_packet: int,
    config: XorParityConfig,
) -> int:
    if source_packet_count <= 1:
        return config.stripe_stride or config.data_packets_per_stripe
    target_overhead_ratio = config.resolved_target_overhead_ratio()
    if target_overhead_ratio is None:
        return config.stripe_stride or config.data_packets_per_stripe

    best_stride = 1
    best_score: tuple[float, float, int, float] | None = None
    for stride in range(1, source_packet_count + 1):
        repair_count = _repair_count_for_stride(
            source_packet_count=source_packet_count,
            data_packets_per_stripe=config.data_packets_per_stripe,
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
    data_packets_per_stripe: int,
    stride: int,
) -> int:
    repair_count = 0
    for start_index in range(0, source_packet_count, stride):
        if len(range(start_index, min(start_index + data_packets_per_stripe, source_packet_count))) >= 2:
            repair_count += 1
    return repair_count


def _overhead_summary(
    *,
    total_tokens: int,
    tokens_per_packet: int,
    repair_packet_count: int,
    config: XorParityConfig,
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


def _parity_metadata(packet: Packet) -> dict[str, Any]:
    metadata = packet.metadata.get(XOR_PARITY_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError("parity packet is missing xor parity metadata")
    if metadata.get("scheme") != XOR_PARITY_SCHEME:
        raise ValueError("parity packet metadata has unsupported scheme")
    return metadata
