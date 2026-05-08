"""Packet-erasure channel simulators."""

from diffusion_fec.channels.random_loss import RandomLossResult, apply_random_loss

__all__ = [
    "RandomLossResult",
    "apply_random_loss",
]
