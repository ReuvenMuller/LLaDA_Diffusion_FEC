"""Packetization and reconstruction-plan helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from diffusion_fec.types import (
    Packet,
    ReconstructionEntry,
    ReconstructionPlan,
    STATE_KNOWN,
    STATE_MISSING,
    STATE_UNGUIDED,
    TokenSample,
)


def packetize_contiguous(
    sample: TokenSample,
    tokens_per_packet: int,
    *,
    packet_kind: str = "data",
) -> list[Packet]:
    """Split a token sample into deterministic contiguous source packets."""

    if not isinstance(tokens_per_packet, int):
        raise TypeError("tokens_per_packet must be an int")
    if tokens_per_packet <= 0:
        raise ValueError("tokens_per_packet must be positive")

    packets: list[Packet] = []
    for wire_id, start in enumerate(range(0, len(sample.token_ids), tokens_per_packet)):
        end = min(start + tokens_per_packet, len(sample.token_ids))
        token_ids = sample.token_ids[start:end]
        token_positions = tuple(range(start, end))
        packets.append(
            Packet(
                source_id=sample.sample_id,
                wire_id=wire_id,
                kind=packet_kind,
                token_ids=token_ids,
                token_positions=token_positions,
            )
        )
    return packets


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
