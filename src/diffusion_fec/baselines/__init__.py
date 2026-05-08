"""Classical matched-overhead baseline codecs."""

from diffusion_fec.baselines.overhead import (
    OverheadSummary,
    estimate_hash_overhead_ratio,
    repair_token_overhead_ratio,
    select_closest_repair_count,
    token_bit_width_for_vocab,
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
    "XOR_PARITY_SCHEME",
    "XorParityConfig",
    "XorParityEncoded",
    "encode_xor_parity",
    "estimate_hash_overhead_ratio",
    "reconstruct_xor_parity",
    "repair_token_overhead_ratio",
    "select_closest_repair_count",
    "token_bit_width_for_vocab",
]
