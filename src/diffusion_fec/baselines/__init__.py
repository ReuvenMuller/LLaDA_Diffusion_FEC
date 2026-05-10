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
from diffusion_fec.baselines.streaming_window import (
    STREAMING_WINDOW_SCHEME,
    StreamingWindowConfig,
    StreamingWindowEncoded,
    encode_streaming_window,
    reconstruct_streaming_window,
)
from diffusion_fec.baselines.xor_parity import (
    XOR_PARITY_SCHEME,
    XorParityConfig,
    XorParityEncoded,
    encode_xor_parity,
    reconstruct_xor_parity,
)
from diffusion_fec.baselines.xor_equations import (
    ParityCandidateFilter,
    XorAuditResult,
    XorPeelResult,
    XorTokenEquation,
    audit_xor_equations,
    equations_from_parity_packets,
    known_tokens_from_data_packets,
    peel_xor_equations,
)

__all__ = [
    "OverheadSummary",
    "LT_FOUNTAIN_SCHEME",
    "LTFountainConfig",
    "LTFountainEncoded",
    "STREAMING_WINDOW_SCHEME",
    "StreamingWindowConfig",
    "StreamingWindowEncoded",
    "XOR_PARITY_SCHEME",
    "ParityCandidateFilter",
    "XorAuditResult",
    "XorParityConfig",
    "XorParityEncoded",
    "XorPeelResult",
    "XorTokenEquation",
    "audit_xor_equations",
    "encode_xor_parity",
    "encode_lt_fountain",
    "encode_streaming_window",
    "equations_from_parity_packets",
    "estimate_hash_overhead_ratio",
    "known_tokens_from_data_packets",
    "peel_xor_equations",
    "reconstruct_xor_parity",
    "reconstruct_lt_fountain",
    "reconstruct_streaming_window",
    "repair_token_overhead_ratio",
    "select_closest_repair_count",
    "token_bit_width_for_vocab",
]
