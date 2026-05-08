"""Hash-protection metadata encoders for token packets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from diffusion_fec.coding.token_hash import TokenHashMap
from diffusion_fec.types import Packet


LOOKBACK_HASH_METADATA_KEY = "lookback_hashes"
LOOKBACK_1_SCHEME = "lookback_1"


@dataclass(frozen=True)
class PacketRef:
    """Stable packet identity for metadata records."""

    source_id: str
    wire_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "wire_id": self.wire_id,
        }


@dataclass(frozen=True)
class PositionHash:
    """Hash metadata for one absolute target-token position."""

    position: int
    hash_value: int

    def __post_init__(self) -> None:
        if not isinstance(self.position, int):
            raise TypeError("position must be an int")
        if self.position < 0:
            raise ValueError("position must be non-negative")
        if not isinstance(self.hash_value, int):
            raise TypeError("hash_value must be an int")
        if self.hash_value < 0:
            raise ValueError("hash_value must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "hash_value": self.hash_value,
        }


@dataclass(frozen=True)
class LookbackHashMetadata:
    """Hashes carried by one packet for a previous packet."""

    scheme: str
    protecting_packet: PacketRef
    protected_packet: PacketRef
    hashes: tuple[PositionHash, ...]

    def __post_init__(self) -> None:
        if self.scheme != LOOKBACK_1_SCHEME:
            raise ValueError(f"scheme must be {LOOKBACK_1_SCHEME!r}")
        object.__setattr__(self, "hashes", tuple(self.hashes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "protecting_packet": self.protecting_packet.to_dict(),
            "protected_packet": self.protected_packet.to_dict(),
            "hashes": [position_hash.to_dict() for position_hash in self.hashes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LookbackHashMetadata":
        return cls(
            scheme=data["scheme"],
            protecting_packet=_packet_ref_from_dict(data["protecting_packet"]),
            protected_packet=_packet_ref_from_dict(data["protected_packet"]),
            hashes=tuple(PositionHash(**item) for item in data["hashes"]),
        )


def attach_lookback_hashes(
    packets: Sequence[Packet],
    token_hash_map: TokenHashMap,
) -> list[Packet]:
    """Return packets with lookback-1 hash metadata attached."""

    ordered_packets = tuple(sorted(packets, key=lambda packet: packet.wire_id))
    _validate_unique_wire_ids(ordered_packets)
    protected_packets: list[Packet] = []

    previous_packet: Packet | None = None
    for packet in ordered_packets:
        metadata = dict(packet.metadata)
        if previous_packet is not None:
            lookback_metadata = LookbackHashMetadata(
                scheme=LOOKBACK_1_SCHEME,
                protecting_packet=_packet_ref(packet),
                protected_packet=_packet_ref(previous_packet),
                hashes=tuple(
                    PositionHash(
                        position=position,
                        hash_value=token_hash_map.bucket_for_token(token_id),
                    )
                    for token_id, position in zip(
                        previous_packet.token_ids,
                        previous_packet.token_positions,
                    )
                ),
            )
            metadata[LOOKBACK_HASH_METADATA_KEY] = lookback_metadata.to_dict()

        protected_packets.append(
            Packet(
                source_id=packet.source_id,
                wire_id=packet.wire_id,
                kind=packet.kind,
                token_ids=packet.token_ids,
                token_positions=packet.token_positions,
                metadata=metadata,
            )
        )
        previous_packet = packet

    return protected_packets


def extract_received_hash_metadata(received_packets: Sequence[Packet]) -> dict[int, int]:
    """Extract position hashes carried by surviving packets only."""

    hash_metadata: dict[int, int] = {}
    for packet in sorted(received_packets, key=lambda item: item.wire_id):
        raw_metadata = packet.metadata.get(LOOKBACK_HASH_METADATA_KEY)
        if raw_metadata is None:
            continue
        lookback_metadata = _coerce_lookback_metadata(raw_metadata)
        if lookback_metadata.protecting_packet != _packet_ref(packet):
            raise ValueError("lookback metadata protecting packet does not match carrier packet")
        for position_hash in lookback_metadata.hashes:
            existing = hash_metadata.get(position_hash.position)
            if existing is not None and existing != position_hash.hash_value:
                raise ValueError(
                    f"conflicting hash metadata for position {position_hash.position}: "
                    f"{existing} != {position_hash.hash_value}"
                )
            hash_metadata[position_hash.position] = position_hash.hash_value
    return hash_metadata


def _packet_ref(packet: Packet) -> PacketRef:
    return PacketRef(source_id=packet.source_id, wire_id=packet.wire_id)


def _packet_ref_from_dict(data: dict[str, Any]) -> PacketRef:
    return PacketRef(source_id=str(data["source_id"]), wire_id=int(data["wire_id"]))


def _coerce_lookback_metadata(data: Any) -> LookbackHashMetadata:
    if isinstance(data, LookbackHashMetadata):
        return data
    if isinstance(data, dict):
        return LookbackHashMetadata.from_dict(data)
    raise TypeError("lookback metadata must be a dict or LookbackHashMetadata")


def _validate_unique_wire_ids(packets: Sequence[Packet]) -> None:
    wire_ids = [packet.wire_id for packet in packets]
    if len(set(wire_ids)) != len(wire_ids):
        raise ValueError("packets must have unique wire_id values")
