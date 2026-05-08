"""LT/fountain-style matched-overhead baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil
from random import Random
from typing import Any

from diffusion_fec.baselines.overhead import (
    OverheadSummary,
    estimate_hash_overhead_ratio,
    repair_token_overhead_ratio,
    select_closest_repair_count,
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


LT_FOUNTAIN_SCHEME = "lt_fountain"
LT_FOUNTAIN_METADATA_KEY = "lt_fountain"


@dataclass(frozen=True)
class LTFountainConfig:
    """Seeded LT/fountain-style baseline configuration."""

    repair_rate: float = 0.25
    random_seed: int = 7
    target_hash_bits: int | None = None
    vocab_size: int | None = None
    target_overhead_ratio: float | None = None
    coverage_aware: bool = False
    degree_values: tuple[int, ...] = (1, 2, 3, 4)
    degree_weights: tuple[float, ...] = (0.35, 0.30, 0.20, 0.15)

    def __post_init__(self) -> None:
        if self.repair_rate < 0.0:
            raise ValueError("repair_rate must be non-negative")
        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an int")
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
        object.__setattr__(self, "degree_values", tuple(self.degree_values))
        object.__setattr__(self, "degree_weights", tuple(self.degree_weights))
        if len(self.degree_values) != len(self.degree_weights):
            raise ValueError("degree_values and degree_weights must have the same length")
        if not self.degree_values:
            raise ValueError("degree_values must be non-empty")
        for degree in self.degree_values:
            if not isinstance(degree, int):
                raise TypeError("degree_values must contain ints")
            if degree <= 0:
                raise ValueError("degree_values must be positive")
        for weight in self.degree_weights:
            if weight < 0.0:
                raise ValueError("degree_weights must be non-negative")
        if sum(self.degree_weights) <= 0.0:
            raise ValueError("degree_weights must sum to a positive value")

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
            "repair_rate": self.repair_rate,
            "random_seed": self.random_seed,
            "target_hash_bits": self.target_hash_bits,
            "vocab_size": self.vocab_size,
            "target_overhead_ratio": self.target_overhead_ratio,
            "coverage_aware": self.coverage_aware,
            "degree_values": list(self.degree_values),
            "degree_weights": list(self.degree_weights),
        }


@dataclass(frozen=True)
class LTFountainEncoded:
    """Encoded packet set and overhead record for LT/fountain repair."""

    packets: tuple[Packet, ...]
    source_packets: tuple[Packet, ...]
    repair_packets: tuple[Packet, ...]
    overhead: OverheadSummary
    config: LTFountainConfig

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


def encode_lt_fountain(
    sample: TokenSample,
    *,
    tokens_per_packet: int,
    config: LTFountainConfig | None = None,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
) -> LTFountainEncoded:
    """Encode source packets plus seeded LT/fountain repair packets."""

    config = config or LTFountainConfig()
    source_layout = source_layout or SourceLayoutConfig()
    wire_interleaving = wire_interleaving or WireInterleavingConfig()
    source_packets = packetize_sample(
        sample,
        tokens_per_packet=tokens_per_packet,
        source_layout=source_layout,
        wire_interleaving=WireInterleavingConfig(),
    )
    repair_count = _repair_count(
        source_packet_count=len(source_packets),
        total_tokens=len(sample.token_ids),
        tokens_per_packet=tokens_per_packet,
        config=config,
    )
    rng = Random(config.random_seed + len(source_packets))
    neighbor_sets = _neighbor_sets(
        rng=rng,
        source_packet_count=len(source_packets),
        repair_count=repair_count,
        config=config,
    )
    repair_packets = _build_repair_packets(
        source_id=sample.sample_id,
        source_packets=source_packets,
        tokens_per_packet=tokens_per_packet,
        neighbor_sets=neighbor_sets,
    )
    packets = _assign_wire_ids(
        packets=[*source_packets, *repair_packets],
        wire_interleaving=wire_interleaving,
    )
    source_count = len(source_packets)
    source_packets = tuple(packets[:source_count])
    repair_packets = tuple(packets[source_count:])
    return LTFountainEncoded(
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


def reconstruct_lt_fountain(
    *,
    total_tokens: int,
    received_packets: Sequence[Packet],
    tokens_per_packet: int,
) -> ReconstructionPlan:
    """Build a known/unguided plan after LT/fountain peeling repair."""

    recovered_packets: dict[int, Packet] = {}
    repair_packets: list[Packet] = []
    for packet in received_packets:
        if packet.kind == "data":
            recovered_packets[_source_packet_index(packet)] = packet
        elif packet.kind == "lt_repair":
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
                    "recovered_by": LT_FOUNTAIN_SCHEME,
                    "repair_wire_id": repair_packet.wire_id,
                },
            )
            progress = True

    return build_reconstruction_plan(
        total_tokens=total_tokens,
        received_packets=tuple(recovered_packets.values()),
    )


def _repair_count(
    *,
    source_packet_count: int,
    total_tokens: int,
    tokens_per_packet: int,
    config: LTFountainConfig,
) -> int:
    if source_packet_count <= 0:
        return 0
    target_overhead_ratio = config.resolved_target_overhead_ratio()
    if target_overhead_ratio is not None:
        return select_closest_repair_count(
            total_tokens=total_tokens,
            tokens_per_packet=tokens_per_packet,
            target_overhead_ratio=target_overhead_ratio,
            max_repair_count=source_packet_count,
        )
    if config.repair_rate == 0.0:
        return 0
    return max(1, ceil(source_packet_count * config.repair_rate))


def _neighbor_sets(
    *,
    rng: Random,
    source_packet_count: int,
    repair_count: int,
    config: LTFountainConfig,
) -> list[list[int]]:
    if source_packet_count <= 0 or repair_count <= 0:
        return []
    if config.coverage_aware:
        return _coverage_aware_neighbor_sets(
            rng=rng,
            source_packet_count=source_packet_count,
            repair_count=repair_count,
            config=config,
        )
    return [
        _sample_neighbors(rng, source_packet_count, config)
        for _ in range(repair_count)
    ]


def _sample_neighbors(
    rng: Random,
    source_packet_count: int,
    config: LTFountainConfig,
) -> list[int]:
    degree = rng.choices(
        list(config.degree_values),
        weights=list(config.degree_weights),
        k=1,
    )[0]
    degree = min(max(1, degree), source_packet_count)
    return sorted(rng.sample(range(source_packet_count), degree))


def _coverage_aware_neighbor_sets(
    *,
    rng: Random,
    source_packet_count: int,
    repair_count: int,
    config: LTFountainConfig,
) -> list[list[int]]:
    target_degrees = [
        min(max(1, rng.choices(list(config.degree_values), weights=list(config.degree_weights), k=1)[0]), source_packet_count)
        for _ in range(repair_count)
    ]
    total_capacity = sum(target_degrees)
    repair_order = list(range(repair_count))
    rng.shuffle(repair_order)
    cursor = 0
    while total_capacity < source_packet_count:
        repair_id = repair_order[cursor % repair_count]
        if target_degrees[repair_id] < source_packet_count:
            target_degrees[repair_id] += 1
            total_capacity += 1
        cursor += 1

    neighbor_sets = [set() for _ in range(repair_count)]
    remaining_capacity = list(target_degrees)
    source_order = list(range(source_packet_count))
    rng.shuffle(source_order)
    cursor = 0
    for source_packet_index in source_order:
        for _ in range(repair_count):
            repair_id = repair_order[cursor % repair_count]
            cursor += 1
            if remaining_capacity[repair_id] <= 0:
                continue
            neighbor_sets[repair_id].add(source_packet_index)
            remaining_capacity[repair_id] -= 1
            break
    for repair_id, add_count in enumerate(remaining_capacity):
        if add_count <= 0:
            continue
        candidates = [
            source_packet_index
            for source_packet_index in range(source_packet_count)
            if source_packet_index not in neighbor_sets[repair_id]
        ]
        neighbor_sets[repair_id].update(rng.sample(candidates, min(add_count, len(candidates))))
    return [sorted(neighbors) for neighbors in neighbor_sets]


def _build_repair_packets(
    *,
    source_id: str,
    source_packets: Sequence[Packet],
    tokens_per_packet: int,
    neighbor_sets: Sequence[Sequence[int]],
) -> list[Packet]:
    repair_packets: list[Packet] = []
    for repair_index, neighbors in enumerate(neighbor_sets):
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
                wire_id=len(source_packets) + repair_index,
                kind="lt_repair",
                token_ids=tuple(repair_tokens),
                token_positions=tuple(range(tokens_per_packet)),
                metadata={
                    LT_FOUNTAIN_METADATA_KEY: {
                        "scheme": LT_FOUNTAIN_SCHEME,
                        "repair_index": repair_index,
                        "neighbor_source_packet_indices": list(neighbors),
                        "neighbor_lengths": neighbor_lengths,
                        "neighbor_token_positions": neighbor_positions,
                        "tokens_per_packet": tokens_per_packet,
                    }
                },
            )
        )
    return repair_packets


def _overhead_summary(
    *,
    total_tokens: int,
    tokens_per_packet: int,
    repair_packet_count: int,
    config: LTFountainConfig,
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
    metadata = packet.metadata.get(LT_FOUNTAIN_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError("LT repair packet is missing metadata")
    if metadata.get("scheme") != LT_FOUNTAIN_SCHEME:
        raise ValueError("LT repair packet metadata has unsupported scheme")
    return metadata
