"""Classical matched-overhead baseline codecs."""

from diffusion_fec.baselines.overhead import (
    OverheadSummary,
    estimate_hash_overhead_ratio,
    repair_token_overhead_ratio,
    select_closest_repair_count,
    token_bit_width_for_vocab,
)
from diffusion_fec.baselines.lt_fountain import (
    LT_FOUNTAIN_SCHEME,
    LTFountainConfig,
    LTFountainEncoded,
    encode_lt_fountain,
    reconstruct_lt_fountain,
)
from diffusion_fec.baselines.xor_parity import (
    XOR_PARITY_SCHEME,
    XorParityConfig,
    XorParityEncoded,
    encode_xor_parity,
    reconstruct_xor_parity,
)

__all__ = [
    "OverheadSummary",
    "LT_FOUNTAIN_SCHEME",
    "LTFountainConfig",
    "LTFountainEncoded",
    "XOR_PARITY_SCHEME",
    "XorParityConfig",
    "XorParityEncoded",
    "encode_xor_parity",
    "encode_lt_fountain",
    "estimate_hash_overhead_ratio",
    "reconstruct_xor_parity",
    "reconstruct_lt_fountain",
    "repair_token_overhead_ratio",
    "select_closest_repair_count",
    "token_bit_width_for_vocab",
]
