"""Coding utilities for packetizing and protecting token streams."""

from diffusion_fec.coding.packetizer import (
    SOURCE_LAYOUT_CONTIGUOUS,
    SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
    WIRE_INTERLEAVING_MATRIX,
    WIRE_INTERLEAVING_NONE,
    SourceLayoutConfig,
    WireInterleavingConfig,
    build_reconstruction_plan,
    packetize_sample,
    packetize_contiguous,
)
from diffusion_fec.coding.hash_profiles import (
    build_and_save_hash_profile,
    load_hash_profile,
    load_or_build_hash_profile,
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
from diffusion_fec.coding.token_hash import (
    TokenHashMap,
    build_token_hash_map,
    token_hash_map_from_token_to_bucket,
)

__all__ = [
    "LOOKBACK_1_SCHEME",
    "LOOKBACK_HASH_METADATA_KEY",
    "LookbackHashMetadata",
    "PacketRef",
    "PositionHash",
    "SOURCE_LAYOUT_CONTIGUOUS",
    "SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS",
    "SourceLayoutConfig",
    "TokenHashMap",
    "WIRE_INTERLEAVING_MATRIX",
    "WIRE_INTERLEAVING_NONE",
    "WireInterleavingConfig",
    "attach_lookback_hashes",
    "build_and_save_hash_profile",
    "build_reconstruction_plan",
    "build_token_hash_map",
    "extract_received_hash_metadata",
    "load_hash_profile",
    "load_or_build_hash_profile",
    "packetize_contiguous",
    "packetize_sample",
    "token_hash_map_from_token_to_bucket",
]
