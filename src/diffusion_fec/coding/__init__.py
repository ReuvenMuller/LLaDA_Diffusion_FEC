"""Coding utilities for packetizing and protecting token streams."""

from diffusion_fec.coding.packetizer import (
    build_reconstruction_plan,
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
    "TokenHashMap",
    "attach_lookback_hashes",
    "build_and_save_hash_profile",
    "build_reconstruction_plan",
    "build_token_hash_map",
    "extract_received_hash_metadata",
    "load_hash_profile",
    "load_or_build_hash_profile",
    "packetize_contiguous",
    "token_hash_map_from_token_to_bucket",
]
