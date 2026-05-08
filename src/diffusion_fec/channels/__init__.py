"""Packet-erasure channel simulators."""

from diffusion_fec.channels.burst_loss import apply_burst_loss
from diffusion_fec.channels.gilbert_elliott import apply_gilbert_elliott_loss
from diffusion_fec.channels.packet_loss import (
    CHANNEL_BURST,
    CHANNEL_GILBERT_ELLIOTT,
    CHANNEL_RANDOM_IID,
    PacketLossChannelConfig,
    apply_packet_loss_channel,
)
from diffusion_fec.channels.random_loss import RandomLossResult, apply_random_loss

__all__ = [
    "CHANNEL_BURST",
    "CHANNEL_GILBERT_ELLIOTT",
    "CHANNEL_RANDOM_IID",
    "PacketLossChannelConfig",
    "RandomLossResult",
    "apply_burst_loss",
    "apply_gilbert_elliott_loss",
    "apply_packet_loss_channel",
    "apply_random_loss",
]
