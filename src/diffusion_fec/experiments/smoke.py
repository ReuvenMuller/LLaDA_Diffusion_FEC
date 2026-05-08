"""Tiny end-to-end recovery harness for smoke experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diffusion_fec.channels.random_loss import RandomLossResult, apply_random_loss
from diffusion_fec.coding.packetizer import (
    build_reconstruction_plan,
    packetize_contiguous,
)
from diffusion_fec.coding.protection import (
    LOOKBACK_1_SCHEME,
    attach_lookback_hashes,
    extract_received_hash_metadata,
)
from diffusion_fec.coding.token_hash import TokenHashMap
from diffusion_fec.decoding.llada_diffusion import (
    DiffusionDecodingConfig,
    decode_masked_diffusion,
)
from diffusion_fec.metrics.token_metrics import TokenMetrics, compute_token_metrics
from diffusion_fec.types import DecodingResult, Packet, ReconstructionPlan, TokenSample


@dataclass(frozen=True)
class SmokeRecoveryCase:
    """Artifacts from one tiny packet-loss recovery run."""

    sample: TokenSample
    loss_result: RandomLossResult
    hash_metadata: dict[int, int]
    protection_mode: str
    oracle_hash_metadata: bool
    mask_token_id: int | None
    reconstruction_plan: ReconstructionPlan
    decoding_result: DecodingResult

    @property
    def metrics(self) -> TokenMetrics:
        return compute_token_metrics(
            original_tokens=self.sample.token_ids,
            reconstructed_tokens=self.decoding_result.reconstructed_tokens,
            reconstruction_plan=self.reconstruction_plan,
            mask_token_id=self.mask_token_id,
        )

    @property
    def exact_token_match(self) -> bool:
        return self.metrics.exact_match

    @property
    def known_positions_preserved(self) -> bool:
        return self.metrics.known_position_preserved

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "loss_result": self.loss_result.to_dict(),
            "hash_metadata": dict(self.hash_metadata),
            "protection_mode": self.protection_mode,
            "oracle_hash_metadata": self.oracle_hash_metadata,
            "reconstruction_plan": self.reconstruction_plan.to_dict(),
            "decoding_result": self.decoding_result.to_dict(),
            "metrics": self.metrics.to_dict(),
            "exact_token_match": self.exact_token_match,
            "known_positions_preserved": self.known_positions_preserved,
        }


def run_smoke_recovery_case(
    *,
    sample: TokenSample,
    model: object,
    config: DiffusionDecodingConfig,
    tokens_per_packet: int,
    loss_rate: float,
    seed: int,
    token_hash_map: TokenHashMap | None = None,
    protection_mode: str = "none",
    oracle_hash_metadata: bool = False,
    prompt_token_ids: list[int] | None = None,
) -> SmokeRecoveryCase:
    """Run packetization, random loss, reconstruction planning, and decoding."""

    packets = packetize_contiguous(sample, tokens_per_packet=tokens_per_packet)
    transmitted_packets = _apply_protection_mode(
        packets=packets,
        token_hash_map=token_hash_map,
        protection_mode=protection_mode,
        oracle_hash_metadata=oracle_hash_metadata,
    )
    loss_result = apply_random_loss(transmitted_packets, loss_rate=loss_rate, seed=seed)
    hash_metadata = _received_hash_metadata(
        sample=sample,
        loss_result=loss_result,
        token_hash_map=token_hash_map,
        protection_mode=protection_mode,
        oracle_hash_metadata=oracle_hash_metadata,
    )
    plan = build_reconstruction_plan(
        total_tokens=len(sample.token_ids),
        received_packets=loss_result.received,
        hash_metadata=hash_metadata,
    )
    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=config,
        token_hash_map=token_hash_map,
        prompt_token_ids=prompt_token_ids,
    )
    return SmokeRecoveryCase(
        sample=sample,
        loss_result=loss_result,
        hash_metadata=hash_metadata,
        protection_mode=protection_mode,
        oracle_hash_metadata=oracle_hash_metadata,
        mask_token_id=config.mask_token_id,
        reconstruction_plan=plan,
        decoding_result=result,
    )


def _apply_protection_mode(
    *,
    packets: list[Packet],
    token_hash_map: TokenHashMap | None,
    protection_mode: str,
    oracle_hash_metadata: bool,
) -> list[Packet]:
    if protection_mode == "none":
        return packets
    if protection_mode == LOOKBACK_1_SCHEME:
        if oracle_hash_metadata:
            raise ValueError("lookback_1 protection must not set oracle_hash_metadata=True")
        if token_hash_map is None:
            raise ValueError("protection_mode='lookback_1' requires token_hash_map")
        return attach_lookback_hashes(packets, token_hash_map)
    raise ValueError("protection_mode must be 'none' or 'lookback_1'")


def _received_hash_metadata(
    *,
    sample: TokenSample,
    loss_result: RandomLossResult,
    token_hash_map: TokenHashMap | None,
    protection_mode: str,
    oracle_hash_metadata: bool,
) -> dict[int, int]:
    if protection_mode == LOOKBACK_1_SCHEME:
        hash_metadata = extract_received_hash_metadata(loss_result.received)
    else:
        hash_metadata = _oracle_hash_metadata_for_dropped_positions(
            sample=sample,
            loss_result=loss_result,
            token_hash_map=token_hash_map,
            oracle_hash_metadata=oracle_hash_metadata,
        )
    return _filter_known_position_hash_metadata(
        hash_metadata=hash_metadata,
        loss_result=loss_result,
    )


def _filter_known_position_hash_metadata(
    *,
    hash_metadata: dict[int, int],
    loss_result: RandomLossResult,
) -> dict[int, int]:
    known_positions = {
        position
        for packet in loss_result.received
        for position in packet.token_positions
    }
    return {
        position: hash_value
        for position, hash_value in hash_metadata.items()
        if position not in known_positions
    }


def _oracle_hash_metadata_for_dropped_positions(
    *,
    sample: TokenSample,
    loss_result: RandomLossResult,
    token_hash_map: TokenHashMap | None,
    oracle_hash_metadata: bool,
) -> dict[int, int]:
    if token_hash_map is None:
        if oracle_hash_metadata:
            raise ValueError("oracle_hash_metadata=True requires token_hash_map")
        return {}
    if not oracle_hash_metadata:
        raise ValueError(
            "token_hash_map in the smoke harness requires oracle_hash_metadata=True; "
            "this mode derives hashes from dropped source tokens for decoder validation only"
        )

    hash_metadata: dict[int, int] = {}
    for packet in loss_result.dropped:
        for token_id, position in zip(packet.token_ids, packet.token_positions):
            if sample.token_ids[position] != token_id:
                raise ValueError("dropped packet token does not match source sample")
            hash_metadata[position] = token_hash_map.bucket_for_token(token_id)
    return hash_metadata
