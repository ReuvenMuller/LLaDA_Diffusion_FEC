"""Coding utilities for packetizing and protecting token streams."""

from diffusion_fec.coding.packetizer import (
    build_reconstruction_plan,
    packetize_contiguous,
)
from diffusion_fec.coding.protection import (
    LOOKBACK_HASH_METADATA_KEY,
    LOOKBACK_1_SCHEME,
    LookbackHashMetadata,
    PacketRef,
    PositionHash,
    attach_lookback_hashes,
    extract_received_hash_metadata,
)
from diffusion_fec.coding.token_hash import TokenHashMap, build_token_hash_map

__all__ = [
    "LOOKBACK_1_SCHEME",
    "LOOKBACK_HASH_METADATA_KEY",
    "LookbackHashMetadata",
    "PacketRef",
    "PositionHash",
    "TokenHashMap",
    "attach_lookback_hashes",
    "build_reconstruction_plan",
    "build_token_hash_map",
    "extract_received_hash_metadata",
    "packetize_contiguous",
]
