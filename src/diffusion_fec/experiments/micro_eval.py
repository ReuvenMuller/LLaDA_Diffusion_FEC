"""Model-free synthetic micro-eval runner.

This runner uses a fake deterministic model to exercise the packetization,
protection, reconstruction, decoding, metrics, and artifact path. It is
engineering validation only, not a research result.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from diffusion_fec.baselines.overhead import (
    metadata_token_equivalent_overhead_ratio,
    token_bit_width_for_vocab,
    token_equivalent_overhead,
)
from diffusion_fec.channels.packet_loss import (
    CHANNEL_RANDOM_IID,
    PacketLossChannelConfig,
)
from diffusion_fec.coding.hash_profiles import DEFAULT_HASH_MAP_MODE, load_or_build_hash_profile
from diffusion_fec.coding.packetizer import SourceLayoutConfig, WireInterleavingConfig
from diffusion_fec.coding.protection import LOOKBACK_1_SCHEME, LOOKBACK_HASH_METADATA_KEY
from diffusion_fec.decoding.llada_diffusion import DiffusionDecodingConfig
from diffusion_fec.experiments.logging import write_run_artifacts
from diffusion_fec.experiments.smoke import SmokeRecoveryCase, run_smoke_recovery_case
from diffusion_fec.types import TokenSample


MICRO_EVAL_MODEL_LABEL = "FakeDeterministicMicroEvalModel"
MICRO_EVAL_MODEL_ONLY = "model_only"
MICRO_EVAL_MODEL_HASH = "model_hash"
MICRO_EVAL_WARNING = (
    "Synthetic micro-eval outputs are engineering validation only and are not "
    "research claims."
)
DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS = (8, 16, 32)
DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET = 4
DEFAULT_MICRO_EVAL_HASH_BITS = 4
DEFAULT_MICRO_EVAL_VOCAB_SIZE = 128
DEFAULT_MICRO_EVAL_STEPS = 4
DEFAULT_MASK_TOKEN_ID = 0
DEFAULT_EOS_TOKEN_ID = 1
DEFAULT_PAD_TOKEN_ID = 2


@dataclass
class FakeForwardOutput:
    logits: list[list["_TargetPositionLogits"]]


class _TargetPositionLogits:
    """Sparse sequence-like logits for large tokenizer vocabularies."""

    def __init__(self, *, vocab_size: int, target_token_id: int | None, score: float):
        self.vocab_size = vocab_size
        self.target_token_id = target_token_id
        self.score = score

    def __len__(self) -> int:
        return self.vocab_size

    def __getitem__(self, token_id: int) -> float:
        if self.target_token_id is not None and int(token_id) == self.target_token_id:
            return self.score
        return 0.0


class FakeDeterministicMicroEvalModel:
    """Tiny oracle-like model for local pipeline validation only."""

    def __init__(self, target_tokens: tuple[int, ...], vocab_size: int):
        self.target_tokens = target_tokens
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        sequence_length = len(input_ids[0])
        logits = [
            [
                _TargetPositionLogits(
                    vocab_size=self.vocab_size,
                    target_token_id=(
                        self.target_tokens[position]
                        if position < len(self.target_tokens)
                        else None
                    ),
                    score=100.0 - position,
                )
                for position in range(sequence_length)
            ]
        ]
        return FakeForwardOutput(logits=logits)

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)


def run_synthetic_micro_eval(
    *,
    output_dir: str | Path,
    sample_lengths: Sequence[int] = DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    samples: Sequence[TokenSample] | None = None,
    dataset_info: dict[str, Any] | None = None,
    loss_rate: float = 0.5,
    seed: int = 0,
    tokens_per_packet: int = DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    mode: str = MICRO_EVAL_MODEL_HASH,
    hash_bits: int = DEFAULT_MICRO_EVAL_HASH_BITS,
    vocab_size: int = DEFAULT_MICRO_EVAL_VOCAB_SIZE,
    steps: int = DEFAULT_MICRO_EVAL_STEPS,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
    channel_config: PacketLossChannelConfig | None = None,
    hash_profile_dir: str | Path | None = None,
    build_hash_profile: bool = False,
    hash_map_mode: str = DEFAULT_HASH_MAP_MODE,
    hash_profile_name: str = "fake_micro_eval_v1",
) -> dict[str, Any]:
    """Run deterministic synthetic micro-eval cases and write artifacts."""

    source_layout = source_layout or SourceLayoutConfig()
    wire_interleaving = wire_interleaving or WireInterleavingConfig()
    channel_config = channel_config or PacketLossChannelConfig(
        mode=CHANNEL_RANDOM_IID,
        loss_rate=loss_rate,
        seed=seed,
    )
    samples = None if samples is None else tuple(samples)
    sample_lengths = (
        tuple(len(sample.token_ids) for sample in samples)
        if samples is not None
        else tuple(sample_lengths)
    )
    if samples is not None:
        _validate_token_samples(samples=samples, vocab_size=vocab_size)
    _validate_micro_eval_config(
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        tokens_per_packet=tokens_per_packet,
        mode=mode,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
        steps=steps,
    )

    protection_mode = LOOKBACK_1_SCHEME if mode == MICRO_EVAL_MODEL_HASH else "none"
    token_hash_map = None
    hash_profile_info = _unused_hash_profile_info(hash_bits=hash_bits, hash_map_mode=hash_map_mode)
    if mode == MICRO_EVAL_MODEL_HASH:
        token_hash_map, hash_profile_info = load_or_build_hash_profile(
            profile_dir=hash_profile_dir,
            profile_name=hash_profile_name,
            vocab_size=vocab_size,
            hash_bits=hash_bits,
            decode_token=lambda token_id: f"fake-token-{token_id}",
            excluded_token_ids={
                DEFAULT_MASK_TOKEN_ID,
                DEFAULT_EOS_TOKEN_ID,
                DEFAULT_PAD_TOKEN_ID,
            },
            salt="fake-micro-eval",
            map_mode=hash_map_mode,
            model_id=MICRO_EVAL_MODEL_LABEL,
            tokenizer_name="fake-deterministic-tokenizer",
            build_if_missing=build_hash_profile,
        )

    config = DiffusionDecodingConfig(
        mask_token_id=DEFAULT_MASK_TOKEN_ID,
        eos_token_id=DEFAULT_EOS_TOKEN_ID,
        pad_token_id=DEFAULT_PAD_TOKEN_ID,
        vocab_size=vocab_size,
        steps=steps,
        block_length=max(tokens_per_packet, 1),
    )
    run_id = _run_id(
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        mode=mode,
        hash_bits=hash_bits,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
    )
    manifest = _manifest(
        run_id=run_id,
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        mode=mode,
        protection_mode=protection_mode,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
        steps=steps,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        hash_profile_info=hash_profile_info,
        dataset_info=dataset_info,
    )

    result_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for case_index, sample_length in enumerate(sample_lengths):
        sample = (
            samples[case_index]
            if samples is not None
            else synthetic_sample(
                sample_index=case_index,
                token_count=sample_length,
                vocab_size=vocab_size,
            )
        )
        model = FakeDeterministicMicroEvalModel(
            target_tokens=sample.token_ids,
            vocab_size=vocab_size,
        )
        case_seed = seed + case_index
        case_channel_config = replace(channel_config, seed=case_seed)
        case = run_smoke_recovery_case(
            sample=sample,
            model=model,
            config=config,
            tokens_per_packet=tokens_per_packet,
            loss_rate=loss_rate,
            seed=case_seed,
            token_hash_map=token_hash_map,
            protection_mode=protection_mode,
            source_layout=source_layout,
            wire_interleaving=wire_interleaving,
            channel_config=case_channel_config,
        )
        case_id = f"case{case_index:04d}"
        result_rows.append(
            _result_row(
                run_id=run_id,
                case_id=case_id,
                case=case,
                sample_index=case_index,
                case_seed=case_seed,
                loss_rate=loss_rate,
                tokens_per_packet=tokens_per_packet,
                mode=mode,
                protection_mode=protection_mode,
                hash_bits=hash_bits,
                source_layout=source_layout,
                wire_interleaving=wire_interleaving,
                channel_config=case_channel_config,
                hash_profile_source=hash_profile_info.get("source", ""),
                vocab_size=vocab_size,
            )
        )
        events.append(
            _event(
                run_id=run_id,
                case_id=case_id,
                case=case,
                mode=mode,
            )
        )

    write_run_artifacts(
        output_dir=output_dir,
        manifest=manifest,
        result_rows=result_rows,
        events=events,
    )
    return {
        "run_id": run_id,
        "manifest": manifest,
        "result_rows": result_rows,
        "events": events,
    }


def synthetic_sample(
    *,
    sample_index: int,
    token_count: int,
    vocab_size: int,
) -> TokenSample:
    """Build a deterministic synthetic token sample."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if vocab_size <= 16:
        raise ValueError("vocab_size must be greater than 16 for synthetic samples")

    start = 5 + (sample_index * token_count)
    token_ids = tuple(
        3 + ((start + offset - 3) % (vocab_size - 3))
        for offset in range(token_count)
    )
    return TokenSample(
        sample_id=f"synthetic-len{token_count:04d}-case{sample_index:04d}",
        text=" ".join(f"token{token_id}" for token_id in token_ids),
        token_ids=token_ids,
        tokenizer_name="fake-deterministic-tokenizer",
    )


def _validate_micro_eval_config(
    *,
    sample_lengths: Sequence[int],
    loss_rate: float,
    tokens_per_packet: int,
    mode: str,
    hash_bits: int,
    vocab_size: int,
    steps: int,
) -> None:
    if not sample_lengths:
        raise ValueError("sample_lengths must be non-empty")
    for sample_length in sample_lengths:
        if not isinstance(sample_length, int):
            raise TypeError("sample_lengths must contain ints")
        if sample_length <= 0:
            raise ValueError("sample_lengths must contain positive values")
    if loss_rate < 0.0 or loss_rate > 1.0:
        raise ValueError("loss_rate must be between 0.0 and 1.0")
    if tokens_per_packet <= 0:
        raise ValueError("tokens_per_packet must be positive")
    if mode not in {MICRO_EVAL_MODEL_ONLY, MICRO_EVAL_MODEL_HASH}:
        raise ValueError("mode must be 'model_only' or 'model_hash'")
    if hash_bits not in {4, 8, 16}:
        raise ValueError("hash_bits must be 4, 8, or 16")
    if vocab_size <= 16:
        raise ValueError("vocab_size must be greater than 16 for synthetic samples")
    if steps <= 0:
        raise ValueError("steps must be positive")


def _validate_token_samples(*, samples: Sequence[TokenSample], vocab_size: int) -> None:
    if not samples:
        raise ValueError("samples must be non-empty when supplied")
    for sample in samples:
        if not isinstance(sample, TokenSample):
            raise TypeError("samples must contain TokenSample objects")
        for token_id in sample.token_ids:
            if token_id >= vocab_size:
                raise ValueError(
                    f"sample {sample.sample_id!r} contains token_id {token_id} "
                    f"outside vocab_size={vocab_size}"
                )


def _unused_hash_profile_info(*, hash_bits: int, hash_map_mode: str) -> dict[str, Any]:
    return {
        "source": "not_used",
        "profile_dir": None,
        "profile_name": None,
        "map_mode": hash_map_mode,
        "hash_bits": hash_bits,
        "file": None,
        "reason": "model_only mode carries no hash metadata",
    }


def _run_id(
    *,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    mode: str,
    hash_bits: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
) -> str:
    lengths = "-".join(str(length) for length in sample_lengths)
    source_chunk = source_layout.chunk_size if source_layout.chunk_size is not None else "default"
    return (
        f"fake-micro-eval|{mode}|hash{hash_bits}|loss{loss_rate:g}|"
        f"lengths{lengths}|tpp{tokens_per_packet}|seed{seed}|"
        f"source-{source_layout.mode}-chunk{source_chunk}|"
        f"wire-{wire_interleaving.mode}-span{wire_interleaving.span}"
    )


def _strategy(mode: str) -> str:
    if mode == MICRO_EVAL_MODEL_HASH:
        return "FakeMicroEval_LookbackHash_NoPrompt"
    return "FakeMicroEval_ModelOnly_NoPrompt"


def _manifest(
    *,
    run_id: str,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    mode: str,
    protection_mode: str,
    hash_bits: int,
    vocab_size: int,
    steps: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    hash_profile_info: dict[str, Any],
    dataset_info: dict[str, Any] | None,
) -> dict[str, Any]:
    sample_generation = {
        "type": "provided_token_samples" if dataset_info else "deterministic_synthetic_token_ids",
        "special_token_ids_excluded_from_samples": [
            DEFAULT_MASK_TOKEN_ID,
            DEFAULT_EOS_TOKEN_ID,
            DEFAULT_PAD_TOKEN_ID,
        ],
    }
    if dataset_info:
        sample_generation["dataset"] = dict(dataset_info)
    return {
        "run_id": run_id,
        "runner": "synthetic_micro_eval",
        "model_label": MICRO_EVAL_MODEL_LABEL,
        "model_kind": "fake_deterministic_micro_eval_model",
        "not_a_research_claim": True,
        "not_a_research_claim_warning": MICRO_EVAL_WARNING,
        "micro_eval": True,
        "strategy": _strategy(mode),
        "config": {
            "sample_lengths": list(sample_lengths),
            "loss_rate": loss_rate,
            "seed": seed,
            "tokens_per_packet": tokens_per_packet,
            "mode": mode,
            "protection_mode": protection_mode,
            "oracle_hash_metadata": False,
            "hash_bits": hash_bits,
            "vocab_size": vocab_size,
            "mask_token_id": DEFAULT_MASK_TOKEN_ID,
            "eos_token_id": DEFAULT_EOS_TOKEN_ID,
            "pad_token_id": DEFAULT_PAD_TOKEN_ID,
            "steps": steps,
            "source_layout": source_layout.to_dict(),
            "wire_interleaving": wire_interleaving.to_dict(),
            "channel": channel_config.to_dict(),
            "sample_generation": sample_generation,
            "decoder": {
                "type": "fake_deterministic_diffusion_shaped_decoder",
                "block_length": max(tokens_per_packet, 1),
            },
        },
        "hash_profile": hash_profile_info,
        "artifacts": {
            "manifest": "run_manifest.json",
            "results": "results.csv",
            "events": "events.jsonl",
        },
    }


def _result_row(
    *,
    run_id: str,
    case_id: str,
    case: SmokeRecoveryCase,
    sample_index: int,
    case_seed: int,
    loss_rate: float,
    tokens_per_packet: int,
    mode: str,
    protection_mode: str,
    hash_bits: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    hash_profile_source: str,
    vocab_size: int,
) -> dict[str, Any]:
    metrics = case.metrics.to_dict()
    plan = case.reconstruction_plan
    packet_count = len(case.loss_result.received) + len(case.loss_result.dropped)
    overhead = _hash_metadata_overhead(
        case=case,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
    )
    return {
        "run_id": run_id,
        "case_id": case_id,
        "sample_id": case.sample.sample_id,
        "model_label": MICRO_EVAL_MODEL_LABEL,
        "strategy": _strategy(mode),
        "protection_mode": protection_mode,
        "oracle_hash_metadata": False,
        "hash_bits": hash_bits,
        "source_layout": source_layout.mode,
        "source_chunk_size": source_layout.chunk_size,
        "wire_interleaving": wire_interleaving.mode,
        "wire_interleaving_span": wire_interleaving.span,
        "channel_mode": channel_config.mode,
        "burst_start_wire_id": channel_config.burst_start_wire_id,
        "burst_length": channel_config.burst_length,
        "ge_good_loss_rate": channel_config.good_loss_rate,
        "ge_bad_loss_rate": channel_config.bad_loss_rate,
        "ge_good_to_bad_rate": channel_config.good_to_bad_rate,
        "ge_bad_to_good_rate": channel_config.bad_to_good_rate,
        "ge_initial_state": channel_config.initial_state,
        "loss_rate": loss_rate,
        "seed": case_seed,
        "tokens_per_packet": tokens_per_packet,
        "source_token_count": len(case.sample.token_ids),
        "known_count": plan.known_count,
        "missing_count": plan.missing_count,
        "hash_guided_count": plan.hash_guided_count,
        "unguided_count": plan.unguided_count,
        "received_packet_count": len(case.loss_result.received),
        "dropped_packet_count": len(case.loss_result.dropped),
        "source_packet_count": packet_count,
        "extra_packet_count": 0,
        "repair_packet_count": 0,
        "repair_token_budget": 0,
        "target_overhead_ratio": overhead["hash_metadata_token_equivalent_overhead_ratio"],
        "actual_repair_token_overhead_ratio": 0.0,
        "token_bit_width": overhead["token_bit_width"],
        "hash_metadata_count": overhead["hash_metadata_count"],
        "hash_metadata_bit_count": overhead["hash_metadata_bit_count"],
        "hash_metadata_token_equivalent": overhead["hash_metadata_token_equivalent"],
        "hash_metadata_token_equivalent_overhead_ratio": overhead[
            "hash_metadata_token_equivalent_overhead_ratio"
        ],
        "total_overhead_ratio": overhead["hash_metadata_token_equivalent_overhead_ratio"],
        "hash_profile_source": hash_profile_source,
        "decode_latency_sec": 0.0,
        "decoder_steps": case.decoding_result.steps,
        "model_forward_calls": case.decoding_result.diagnostics.get("model_forward_calls", ""),
        **metrics,
        "sample_index": sample_index,
    }


def _event(
    *,
    run_id: str,
    case_id: str,
    case: SmokeRecoveryCase,
    mode: str,
) -> dict[str, Any]:
    return {
        "event_type": "micro_eval_case",
        "run_id": run_id,
        "case_id": case_id,
        "model_label": MICRO_EVAL_MODEL_LABEL,
        "strategy": _strategy(mode),
        "case": _normalize_case_for_artifacts(case.to_dict()),
    }


def _normalize_case_for_artifacts(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    decoding_result = dict(normalized["decoding_result"])
    decoding_result["decode_latency_sec"] = 0.0
    normalized["decoding_result"] = decoding_result
    return normalized


def _hash_metadata_overhead(
    *,
    case: SmokeRecoveryCase,
    hash_bits: int,
    vocab_size: int,
) -> dict[str, Any]:
    metadata_count = _transmitted_hash_metadata_count(case)
    metadata_bits = metadata_count * hash_bits
    token_bit_width = token_bit_width_for_vocab(vocab_size)
    token_equivalent = token_equivalent_overhead(
        metadata_bits=metadata_bits,
        vocab_size=vocab_size,
    )
    ratio = metadata_token_equivalent_overhead_ratio(
        metadata_bits=metadata_bits,
        total_tokens=len(case.sample.token_ids),
        vocab_size=vocab_size,
    )
    return {
        "token_bit_width": token_bit_width,
        "hash_metadata_count": metadata_count,
        "hash_metadata_bit_count": metadata_bits,
        "hash_metadata_token_equivalent": token_equivalent,
        "hash_metadata_token_equivalent_overhead_ratio": ratio,
    }


def _transmitted_hash_metadata_count(case: SmokeRecoveryCase) -> int:
    if case.protection_mode != LOOKBACK_1_SCHEME:
        return 0
    return sum(
        _packet_hash_metadata_count(packet)
        for packet in (*case.loss_result.received, *case.loss_result.dropped)
    )


def _packet_hash_metadata_count(packet) -> int:
    raw_metadata = packet.metadata.get(LOOKBACK_HASH_METADATA_KEY)
    if raw_metadata is None:
        return 0
    if isinstance(raw_metadata, dict):
        hashes = raw_metadata.get("hashes", ())
        return len(hashes)
    hashes = getattr(raw_metadata, "hashes", ())
    return len(tuple(hashes))
