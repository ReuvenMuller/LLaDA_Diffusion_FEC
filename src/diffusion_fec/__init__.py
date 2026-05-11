"""Core package for LLaDA-style diffusion FEC experiments."""

from diffusion_fec.coding.token_hash import TokenHashMap, build_token_hash_map
from diffusion_fec.decoding.constraints import ConstraintMasks, build_constraint_masks
from diffusion_fec.decoding.llada_diffusion import (
    DiffusionDecodingConfig,
    decode_masked_diffusion,
)
from diffusion_fec.experiments.smoke import SmokeRecoveryCase, run_smoke_recovery_case
from diffusion_fec.metrics.token_metrics import (
    ChannelLostPositionMetrics,
    channel_lost_source_positions,
    compute_channel_lost_position_metrics,
    compute_token_metrics,
    TokenMetrics,
)
from diffusion_fec.models.llada import LLaDAAdapter
from diffusion_fec.types import (
    ConfidenceStat,
    DecodingResult,
    Packet,
    ReconstructionEntry,
    ReconstructionPlan,
    StepSummary,
    TokenSample,
)

__all__ = [
    "ConfidenceStat",
    "ConstraintMasks",
    "ChannelLostPositionMetrics",
    "DecodingResult",
    "DiffusionDecodingConfig",
    "LLaDAAdapter",
    "Packet",
    "ReconstructionEntry",
    "ReconstructionPlan",
    "SmokeRecoveryCase",
    "StepSummary",
    "TokenHashMap",
    "TokenMetrics",
    "TokenSample",
    "build_constraint_masks",
    "build_token_hash_map",
    "channel_lost_source_positions",
    "compute_channel_lost_position_metrics",
    "decode_masked_diffusion",
    "compute_token_metrics",
    "run_smoke_recovery_case",
]
