"""Model adapter interfaces and implementations."""

from diffusion_fec.models.base import MaskedDiffusionModel
from diffusion_fec.models.llada import (
    LLADA_1_5_MODEL_ID,
    LLADA_1_5_DEFAULT_EOS_TOKEN_ID,
    LLADA_1_5_DEFAULT_MASK_TOKEN_ID,
    LLADA_1_5_DEFAULT_PAD_TOKEN_ID,
    LLADA_1_5_DEFAULT_VOCAB_SIZE,
    LLaDAAdapter,
)

__all__ = [
    "LLADA_1_5_DEFAULT_EOS_TOKEN_ID",
    "LLADA_1_5_DEFAULT_MASK_TOKEN_ID",
    "LLADA_1_5_DEFAULT_PAD_TOKEN_ID",
    "LLADA_1_5_DEFAULT_VOCAB_SIZE",
    "LLADA_1_5_MODEL_ID",
    "LLaDAAdapter",
    "MaskedDiffusionModel",
]
