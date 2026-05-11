"""Small experiment orchestration helpers."""

from diffusion_fec.experiments.classical_micro_eval import (
    run_lt_fountain_micro_eval,
    run_streaming_window_micro_eval,
    run_xor_parity_micro_eval,
)
from diffusion_fec.experiments.micro_eval import (
    MICRO_EVAL_MODEL_HASH,
    MICRO_EVAL_MODEL_ONLY,
    run_synthetic_micro_eval,
)
from diffusion_fec.experiments.llada_micro_eval import (
    RealLLaDAMicroEvalUnavailable,
    run_real_llada_micro_eval,
)
from diffusion_fec.experiments.hybrid_eval import (
    HYBRID_MODE_ITERATIVE_PEEL,
    HYBRID_MODE_PARITY_FILTER,
    HYBRID_MODE_PRE_PEEL_ONLY,
    run_hybrid_xor_hash_micro_eval,
    run_real_llada_hybrid_xor_hash_micro_eval,
)
from diffusion_fec.experiments.smoke import SmokeRecoveryCase, run_smoke_recovery_case
from diffusion_fec.experiments.sweep import (
    SyntheticSweepConfig,
    SweepRunSpec,
    build_synthetic_sweep_config,
    run_synthetic_sweep,
)

__all__ = [
    "MICRO_EVAL_MODEL_HASH",
    "MICRO_EVAL_MODEL_ONLY",
    "HYBRID_MODE_ITERATIVE_PEEL",
    "HYBRID_MODE_PARITY_FILTER",
    "HYBRID_MODE_PRE_PEEL_ONLY",
    "RealLLaDAMicroEvalUnavailable",
    "SmokeRecoveryCase",
    "SweepRunSpec",
    "SyntheticSweepConfig",
    "build_synthetic_sweep_config",
    "run_real_llada_micro_eval",
    "run_real_llada_hybrid_xor_hash_micro_eval",
    "run_hybrid_xor_hash_micro_eval",
    "run_lt_fountain_micro_eval",
    "run_synthetic_micro_eval",
    "run_synthetic_sweep",
    "run_streaming_window_micro_eval",
    "run_smoke_recovery_case",
    "run_xor_parity_micro_eval",
]
