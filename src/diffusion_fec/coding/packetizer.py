"""Packetization and reconstruction-plan helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

from diffusion_fec.types import (
    Packet,
    ReconstructionEntry,
    ReconstructionPlan,
    STATE_KNOWN,
    STATE_MISSING,
    STATE_UNGUIDED,
    TokenSample,
)


SOURCE_LAYOUT_CONTIGUOUS = "contiguous"
SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS = "round_robin_chunks"
WIRE_INTERLEAVING_NONE = "none"
WIRE_INTERLEAVING_MATRIX = "matrix"
SOURCE_PACKET_INDEX_METADATA_KEY = "source_packet_index"
SOURCE_LAYOUT_METADATA_KEY = "source_layout"
WIRE_INTERLEAVING_METADATA_KEY = "wire_interleaving"


@dataclass(frozen=True)
class SourceLayoutConfig:
    """How absolute source-token positions are grouped into packets."""

    mode: str = SOURCE_LAYOUT_CONTIGUOUS
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {SOURCE_LAYOUT_CONTIGUOUS, SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS}:
            raise ValueError(
                "source layout mode must be 'contiguous' or 'round_robin_chunks'"
            )
        if self.chunk_size is not None:
            if not isinstance(self.chunk_size, int):
                raise TypeError("source layout chunk_size must be an int when set")
            if self.chunk_size <= 0:
                raise ValueError("source layout chunk_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "chunk_size": self.chunk_size,
        }


@dataclass(frozen=True)
class WireInterleavingConfig:
    """How source packet indices are assigned to wire IDs."""

    mode: str = WIRE_INTERLEAVING_NONE
    span: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {WIRE_INTERLEAVING_NONE, WIRE_INTERLEAVING_MATRIX}:
            raise ValueError("wire interleaving mode must be 'none' or 'matrix'")
        if not isinstance(self.span, int):
            raise TypeError("wire interleaving span must be an int")
        if self.span <= 0:
            raise ValueError("wire interleaving span must be positive")

    @property
    def enabled(self) -> bool:
        return self.mode == WIRE_INTERLEAVING_MATRIX and self.span > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "span": self.span,
        }


def packetize_contiguous(
    sample: TokenSample,
    tokens_per_packet: int,
    *,
    packet_kind: str = "data",
) -> list[Packet]:
    """Split a token sample into deterministic contiguous source packets."""

    return packetize_sample(
        sample,
        tokens_per_packet=tokens_per_packet,
        source_layout=SourceLayoutConfig(mode=SOURCE_LAYOUT_CONTIGUOUS),
        wire_interleaving=WireInterleavingConfig(),
        packet_kind=packet_kind,
    )


def packetize_sample(
    sample: TokenSample,
    tokens_per_packet: int,
    *,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
    packet_kind: str = "data",
) -> list[Packet]:
    """Packetize a sample with explicit source layout and wire ordering."""

    if not isinstance(tokens_per_packet, int):
        raise TypeError("tokens_per_packet must be an int")
    if tokens_per_packet <= 0:
        raise ValueError("tokens_per_packet must be positive")

    source_layout = source_layout or SourceLayoutConfig()
    wire_interleaving = wire_interleaving or WireInterleavingConfig()
    _validate_layout_for_packet_size(source_layout, tokens_per_packet)

    position_groups = _build_source_position_groups(
        total_tokens=len(sample.token_ids),
        tokens_per_packet=tokens_per_packet,
        source_layout=source_layout,
    )
    source_index_to_wire_id = _source_index_to_wire_id(
        packet_count=len(position_groups),
        wire_interleaving=wire_interleaving,
    )

    packets: list[Packet] = []
    for source_packet_index, token_positions in enumerate(position_groups):
        token_ids = tuple(sample.token_ids[position] for position in token_positions)
        packets.append(
            Packet(
                source_id=sample.sample_id,
                wire_id=source_index_to_wire_id[source_packet_index],
                kind=packet_kind,
                token_ids=token_ids,
                token_positions=tuple(token_positions),
                metadata={
                    SOURCE_PACKET_INDEX_METADATA_KEY: source_packet_index,
                    SOURCE_LAYOUT_METADATA_KEY: source_layout.to_dict(),
                    WIRE_INTERLEAVING_METADATA_KEY: wire_interleaving.to_dict(),
                },
            )
        )
    return packets


def _validate_layout_for_packet_size(
    source_layout: SourceLayoutConfig,
    tokens_per_packet: int,
) -> None:
    if source_layout.mode != SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS:
        return
    chunk_size = source_layout.chunk_size or 1
    if chunk_size > tokens_per_packet:
        raise ValueError("source layout chunk_size cannot exceed tokens_per_packet")
    if tokens_per_packet % chunk_size != 0:
        raise ValueError("source layout chunk_size must evenly divide tokens_per_packet")


def _build_source_position_groups(
    *,
    total_tokens: int,
    tokens_per_packet: int,
    source_layout: SourceLayoutConfig,
) -> list[list[int]]:
    if source_layout.mode == SOURCE_LAYOUT_CONTIGUOUS:
        return [
            list(range(start, min(start + tokens_per_packet, total_tokens)))
            for start in range(0, total_tokens, tokens_per_packet)
        ]

    chunk_size = source_layout.chunk_size or 1
    packet_count = ceil(total_tokens / tokens_per_packet)
    position_groups: list[list[int]] = [[] for _ in range(packet_count)]
    chunks = [
        list(range(start, min(start + chunk_size, total_tokens)))
        for start in range(0, total_tokens, chunk_size)
    ]
    for chunk_index, chunk_positions in enumerate(chunks):
        packet_index = chunk_index % packet_count
        position_groups[packet_index].extend(chunk_positions)
    return position_groups


def _source_index_to_wire_id(
    *,
    packet_count: int,
    wire_interleaving: WireInterleavingConfig,
) -> dict[int, int]:
    wire_order = _build_wire_order(packet_count, wire_interleaving)
    return {
        source_packet_index: wire_id
        for wire_id, source_packet_index in enumerate(wire_order)
    }


def _build_wire_order(
    packet_count: int,
    wire_interleaving: WireInterleavingConfig,
) -> list[int]:
    if not wire_interleaving.enabled or packet_count <= 1:
        return list(range(packet_count))

    span = min(wire_interleaving.span, packet_count)
    rows = ceil(packet_count / span)
    matrix: list[list[int | None]] = [[None for _ in range(span)] for _ in range(rows)]

    source_packet_index = 0
    for row in range(rows):
        for column in range(span):
            if source_packet_index < packet_count:
                matrix[row][column] = source_packet_index
                source_packet_index += 1

    wire_order: list[int] = []
    for column in range(span):
        for row in range(rows):
            value = matrix[row][column]
            if value is not None:
                wire_order.append(value)
    return wire_order


def build_reconstruction_plan(
    *,
    total_tokens: int,
    received_packets: Sequence[Packet],
    hash_metadata: Mapping[int, int] | None = None,
) -> ReconstructionPlan:
    """Build a receiver-side plan from surviving packets and optional hash metadata."""

    if not isinstance(total_tokens, int):
        raise TypeError("total_tokens must be an int")
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")

    known_tokens: dict[int, int] = {}
    for packet in sorted(received_packets, key=lambda item: item.wire_id):
        if len(packet.token_ids) != len(packet.token_positions):
            raise ValueError("packet token_ids and token_positions must have the same length")
        for token_id, position in zip(packet.token_ids, packet.token_positions):
            if position >= total_tokens:
                raise ValueError(f"packet position {position} is outside total_tokens")
            existing = known_tokens.get(position)
            if existing is not None and existing != token_id:
                raise ValueError(
                    f"conflicting token IDs for received position {position}: "
                    f"{existing} != {token_id}"
                )
            known_tokens[position] = token_id

    hash_by_position = dict(hash_metadata or {})
    for position, hash_value in hash_by_position.items():
        if not isinstance(position, int):
            raise TypeError("hash metadata positions must be ints")
        if position < 0 or position >= total_tokens:
            raise ValueError(f"hash metadata position {position} is outside total_tokens")
        if not isinstance(hash_value, int):
            raise TypeError("hash metadata values must be ints")
        if hash_value < 0:
            raise ValueError("hash metadata values must be non-negative")

    entries: list[ReconstructionEntry] = []
    for position in range(total_tokens):
        if position in known_tokens:
            entries.append(
                ReconstructionEntry(
                    position=position,
                    state=STATE_KNOWN,
                    token_id=known_tokens[position],
                    fixed=True,
                )
            )
        elif position in hash_by_position:
            entries.append(
                ReconstructionEntry(
                    position=position,
                    state=STATE_MISSING,
                    hash_value=hash_by_position[position],
                    fixed=False,
                )
            )
        else:
            entries.append(
                ReconstructionEntry(
                    position=position,
                    state=STATE_UNGUIDED,
                    fixed=False,
                )
            )

    return ReconstructionPlan(entries=tuple(entries), total_tokens=total_tokens)
