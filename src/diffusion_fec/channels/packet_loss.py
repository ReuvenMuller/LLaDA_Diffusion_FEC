"""Serializable packet-loss channel configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from diffusion_fec.channels.burst_loss import apply_burst_loss
from diffusion_fec.channels.gilbert_elliott import (
    GE_STATE_BAD,
    GE_STATE_GOOD,
    apply_gilbert_elliott_loss,
)
from diffusion_fec.channels.random_loss import RandomLossResult, apply_random_loss
from diffusion_fec.types import Packet


CHANNEL_RANDOM_IID = "random_iid"
CHANNEL_BURST = "burst"
CHANNEL_GILBERT_ELLIOTT = "gilbert_elliott"


@dataclass(frozen=True)
class PacketLossChannelConfig:
    """Config for packet-erasure channels used by smoke and micro-eval runners."""

    mode: str = CHANNEL_RANDOM_IID
    loss_rate: float = 0.5
    seed: int | None = None
    burst_start_wire_id: int = 0
    burst_length: int | None = None
    good_loss_rate: float = 0.0
    bad_loss_rate: float = 1.0
    good_to_bad_rate: float = 0.05
    bad_to_good_rate: float = 0.5
    initial_state: str = GE_STATE_GOOD

    def __post_init__(self) -> None:
        if self.mode not in {
            CHANNEL_RANDOM_IID,
            CHANNEL_BURST,
            CHANNEL_GILBERT_ELLIOTT,
        }:
            raise ValueError(
                "channel mode must be 'random_iid', 'burst', or 'gilbert_elliott'"
            )
        _validate_rate(self.loss_rate, "loss_rate")
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("seed must be an int when set")
        if not isinstance(self.burst_start_wire_id, int):
            raise TypeError("burst_start_wire_id must be an int")
        if self.burst_start_wire_id < 0:
            raise ValueError("burst_start_wire_id must be non-negative")
        if self.burst_length is not None:
            if not isinstance(self.burst_length, int):
                raise TypeError("burst_length must be an int when set")
            if self.burst_length < 0:
                raise ValueError("burst_length must be non-negative")
        _validate_rate(self.good_loss_rate, "good_loss_rate")
        _validate_rate(self.bad_loss_rate, "bad_loss_rate")
        _validate_rate(self.good_to_bad_rate, "good_to_bad_rate")
        _validate_rate(self.bad_to_good_rate, "bad_to_good_rate")
        if self.initial_state not in {GE_STATE_GOOD, GE_STATE_BAD}:
            raise ValueError("initial_state must be 'good' or 'bad'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "loss_rate": self.loss_rate,
            "seed": self.seed,
            "burst_start_wire_id": self.burst_start_wire_id,
            "burst_length": self.burst_length,
            "good_loss_rate": self.good_loss_rate,
            "bad_loss_rate": self.bad_loss_rate,
            "good_to_bad_rate": self.good_to_bad_rate,
            "bad_to_good_rate": self.bad_to_good_rate,
            "initial_state": self.initial_state,
        }


def apply_packet_loss_channel(
    packets: Sequence[Packet],
    *,
    config: PacketLossChannelConfig,
) -> RandomLossResult:
    """Apply the configured packet-erasure channel."""

    if config.mode == CHANNEL_RANDOM_IID:
        return apply_random_loss(
            packets,
            loss_rate=config.loss_rate,
            seed=config.seed,
        )
    if config.mode == CHANNEL_BURST:
        if config.burst_length is None:
            raise ValueError("burst channel requires burst_length")
        return apply_burst_loss(
            packets,
            burst_start_wire_id=config.burst_start_wire_id,
            burst_length=config.burst_length,
        )
    return apply_gilbert_elliott_loss(
        packets,
        good_loss_rate=config.good_loss_rate,
        bad_loss_rate=config.bad_loss_rate,
        good_to_bad_rate=config.good_to_bad_rate,
        bad_to_good_rate=config.bad_to_good_rate,
        seed=config.seed,
        initial_state=config.initial_state,
    )


def _validate_rate(value: float, name: str) -> None:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
