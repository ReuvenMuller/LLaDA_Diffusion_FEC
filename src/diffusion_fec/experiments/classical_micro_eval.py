"""Synthetic micro-eval runners for classical baselines."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from diffusion_fec.baselines.xor_parity import (
    XOR_PARITY_SCHEME,
    XorParityConfig,
    encode_xor_parity,
    reconstruct_xor_parity,
)
from diffusion_fec.baselines.lt_fountain import (
    LT_FOUNTAIN_SCHEME,
    LTFountainConfig,
    encode_lt_fountain,
    reconstruct_lt_fountain,
)
from diffusion_fec.baselines.streaming_window import (
    STREAMING_WINDOW_SCHEME,
    StreamingWindowConfig,
    encode_streaming_window,
    reconstruct_streaming_window,
)
from diffusion_fec.channels.packet_loss import (
    CHANNEL_RANDOM_IID,
    PacketLossChannelConfig,
    apply_packet_loss_channel,
)
from diffusion_fec.coding.packetizer import SourceLayoutConfig, WireInterleavingConfig
from diffusion_fec.experiments.logging import write_run_artifacts
from diffusion_fec.experiments.micro_eval import (
    DEFAULT_MICRO_EVAL_HASH_BITS,
    DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    DEFAULT_MICRO_EVAL_VOCAB_SIZE,
    DEFAULT_MASK_TOKEN_ID,
    MICRO_EVAL_WARNING,
    synthetic_sample,
)
from diffusion_fec.metrics.token_metrics import TokenMetrics, compute_token_metrics
from diffusion_fec.types import ReconstructionPlan


XOR_PARITY_MODEL_LABEL = "ClassicalXORParity"
XOR_PARITY_BASELINE_FAMILY = "xor_parity"
LT_FOUNTAIN_MODEL_LABEL = "ClassicalLTFountain"
LT_FOUNTAIN_BASELINE_FAMILY = "lt_fountain"
STREAMING_WINDOW_MODEL_LABEL = "ClassicalStreamingWindow"
STREAMING_WINDOW_BASELINE_FAMILY = "streaming_window"


def run_xor_parity_micro_eval(
    *,
    output_dir: str | Path,
    sample_lengths: Sequence[int] = DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    loss_rate: float = 0.5,
    seed: int = 0,
    tokens_per_packet: int = DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    hash_bits: int = DEFAULT_MICRO_EVAL_HASH_BITS,
    vocab_size: int = DEFAULT_MICRO_EVAL_VOCAB_SIZE,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
    channel_config: PacketLossChannelConfig | None = None,
    data_packets_per_stripe: int = 4,
    stripe_stride: int | None = None,
) -> dict[str, Any]:
    """Run deterministic synthetic XOR parity baseline cases and write artifacts."""

    source_layout = source_layout or SourceLayoutConfig()
    wire_interleaving = wire_interleaving or WireInterleavingConfig()
    channel_config = channel_config or PacketLossChannelConfig(
        mode=CHANNEL_RANDOM_IID,
        loss_rate=loss_rate,
        seed=seed,
    )
    sample_lengths = tuple(sample_lengths)
    _validate_config(
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
    )
    xor_config = XorParityConfig(
        data_packets_per_stripe=data_packets_per_stripe,
        stripe_stride=stripe_stride,
        target_hash_bits=hash_bits,
        vocab_size=vocab_size,
    )
    run_id = _run_id(
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        xor_config=xor_config,
    )
    manifest = _manifest(
        run_id=run_id,
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        xor_config=xor_config,
    )

    result_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for case_index, sample_length in enumerate(sample_lengths):
        sample = synthetic_sample(
            sample_index=case_index,
            token_count=sample_length,
            vocab_size=vocab_size,
        )
        encoded = encode_xor_parity(
            sample,
            tokens_per_packet=tokens_per_packet,
            config=xor_config,
            source_layout=source_layout,
            wire_interleaving=wire_interleaving,
        )
        case_seed = seed + case_index
        case_channel_config = replace(channel_config, seed=case_seed)
        loss_result = apply_packet_loss_channel(
            encoded.packets,
            config=case_channel_config,
        )
        plan = reconstruct_xor_parity(
            total_tokens=len(sample.token_ids),
            received_packets=loss_result.received,
            tokens_per_packet=tokens_per_packet,
        )
        reconstructed_tokens = _tokens_from_plan(plan)
        metrics = compute_token_metrics(
            original_tokens=sample.token_ids,
            reconstructed_tokens=reconstructed_tokens,
            reconstruction_plan=plan,
            mask_token_id=DEFAULT_MASK_TOKEN_ID,
        )
        case_id = f"case{case_index:04d}"
        result_rows.append(
            _result_row(
                run_id=run_id,
                case_id=case_id,
                sample_id=sample.sample_id,
                case_seed=case_seed,
                loss_rate=loss_rate,
                tokens_per_packet=tokens_per_packet,
                hash_bits=hash_bits,
                source_token_count=len(sample.token_ids),
                plan=plan,
                metrics=metrics,
                received_packet_count=len(loss_result.received),
                dropped_packet_count=len(loss_result.dropped),
                source_layout=source_layout,
                wire_interleaving=wire_interleaving,
                channel_config=case_channel_config,
                encoded=encoded,
                model_label=XOR_PARITY_MODEL_LABEL,
                strategy=_strategy(hash_bits),
                baseline_family=XOR_PARITY_BASELINE_FAMILY,
                protection_mode=XOR_PARITY_SCHEME,
            )
        )
        events.append(
            {
                "event_type": "xor_parity_micro_eval_case",
                "run_id": run_id,
                "case_id": case_id,
                "model_label": XOR_PARITY_MODEL_LABEL,
                "strategy": _strategy(hash_bits),
                "sample": sample.to_dict(),
                "encoded": encoded.to_dict(),
                "loss_result": loss_result.to_dict(),
                "reconstruction_plan": plan.to_dict(),
                "reconstructed_tokens": list(reconstructed_tokens),
                "metrics": metrics.to_dict(),
            }
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


def run_lt_fountain_micro_eval(
    *,
    output_dir: str | Path,
    sample_lengths: Sequence[int] = DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    loss_rate: float = 0.5,
    seed: int = 0,
    tokens_per_packet: int = DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    hash_bits: int = DEFAULT_MICRO_EVAL_HASH_BITS,
    vocab_size: int = DEFAULT_MICRO_EVAL_VOCAB_SIZE,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
    channel_config: PacketLossChannelConfig | None = None,
    repair_rate: float = 0.25,
    lt_random_seed: int = 7,
    coverage_aware: bool = False,
) -> dict[str, Any]:
    """Run deterministic synthetic LT/fountain baseline cases and write artifacts."""

    source_layout = source_layout or SourceLayoutConfig()
    wire_interleaving = wire_interleaving or WireInterleavingConfig()
    channel_config = channel_config or PacketLossChannelConfig(
        mode=CHANNEL_RANDOM_IID,
        loss_rate=loss_rate,
        seed=seed,
    )
    sample_lengths = tuple(sample_lengths)
    _validate_config(
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
    )
    lt_config = LTFountainConfig(
        repair_rate=repair_rate,
        random_seed=lt_random_seed,
        target_hash_bits=hash_bits,
        vocab_size=vocab_size,
        coverage_aware=coverage_aware,
    )
    run_id = _lt_run_id(
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        lt_config=lt_config,
    )
    manifest = _lt_manifest(
        run_id=run_id,
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        lt_config=lt_config,
    )

    result_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for case_index, sample_length in enumerate(sample_lengths):
        sample = synthetic_sample(
            sample_index=case_index,
            token_count=sample_length,
            vocab_size=vocab_size,
        )
        encoded = encode_lt_fountain(
            sample,
            tokens_per_packet=tokens_per_packet,
            config=lt_config,
            source_layout=source_layout,
            wire_interleaving=wire_interleaving,
        )
        case_seed = seed + case_index
        case_channel_config = replace(channel_config, seed=case_seed)
        loss_result = apply_packet_loss_channel(
            encoded.packets,
            config=case_channel_config,
        )
        plan = reconstruct_lt_fountain(
            total_tokens=len(sample.token_ids),
            received_packets=loss_result.received,
            tokens_per_packet=tokens_per_packet,
        )
        reconstructed_tokens = _tokens_from_plan(plan)
        metrics = compute_token_metrics(
            original_tokens=sample.token_ids,
            reconstructed_tokens=reconstructed_tokens,
            reconstruction_plan=plan,
            mask_token_id=DEFAULT_MASK_TOKEN_ID,
        )
        case_id = f"case{case_index:04d}"
        result_rows.append(
            _result_row(
                run_id=run_id,
                case_id=case_id,
                sample_id=sample.sample_id,
                case_seed=case_seed,
                loss_rate=loss_rate,
                tokens_per_packet=tokens_per_packet,
                hash_bits=hash_bits,
                source_token_count=len(sample.token_ids),
                plan=plan,
                metrics=metrics,
                received_packet_count=len(loss_result.received),
                dropped_packet_count=len(loss_result.dropped),
                source_layout=source_layout,
                wire_interleaving=wire_interleaving,
                channel_config=case_channel_config,
                encoded=encoded,
                model_label=LT_FOUNTAIN_MODEL_LABEL,
                strategy=_lt_strategy(hash_bits, coverage_aware),
                baseline_family=LT_FOUNTAIN_BASELINE_FAMILY,
                protection_mode=LT_FOUNTAIN_SCHEME,
            )
        )
        events.append(
            {
                "event_type": "lt_fountain_micro_eval_case",
                "run_id": run_id,
                "case_id": case_id,
                "model_label": LT_FOUNTAIN_MODEL_LABEL,
                "strategy": _lt_strategy(hash_bits, coverage_aware),
                "sample": sample.to_dict(),
                "encoded": encoded.to_dict(),
                "loss_result": loss_result.to_dict(),
                "reconstruction_plan": plan.to_dict(),
                "reconstructed_tokens": list(reconstructed_tokens),
                "metrics": metrics.to_dict(),
            }
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


def run_streaming_window_micro_eval(
    *,
    output_dir: str | Path,
    sample_lengths: Sequence[int] = DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    loss_rate: float = 0.5,
    seed: int = 0,
    tokens_per_packet: int = DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    hash_bits: int = DEFAULT_MICRO_EVAL_HASH_BITS,
    vocab_size: int = DEFAULT_MICRO_EVAL_VOCAB_SIZE,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
    channel_config: PacketLossChannelConfig | None = None,
    window_size: int = 5,
    window_stride: int = 1,
) -> dict[str, Any]:
    """Run deterministic synthetic streaming-window baseline cases and write artifacts."""

    source_layout = source_layout or SourceLayoutConfig()
    wire_interleaving = wire_interleaving or WireInterleavingConfig()
    channel_config = channel_config or PacketLossChannelConfig(
        mode=CHANNEL_RANDOM_IID,
        loss_rate=loss_rate,
        seed=seed,
    )
    sample_lengths = tuple(sample_lengths)
    _validate_config(
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
    )
    stream_config = StreamingWindowConfig(
        window_size=window_size,
        window_stride=window_stride,
        target_hash_bits=hash_bits,
        vocab_size=vocab_size,
    )
    run_id = _stream_run_id(
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        stream_config=stream_config,
    )
    manifest = _stream_manifest(
        run_id=run_id,
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        stream_config=stream_config,
    )

    result_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for case_index, sample_length in enumerate(sample_lengths):
        sample = synthetic_sample(
            sample_index=case_index,
            token_count=sample_length,
            vocab_size=vocab_size,
        )
        encoded = encode_streaming_window(
            sample,
            tokens_per_packet=tokens_per_packet,
            config=stream_config,
            source_layout=source_layout,
            wire_interleaving=wire_interleaving,
        )
        case_seed = seed + case_index
        case_channel_config = replace(channel_config, seed=case_seed)
        loss_result = apply_packet_loss_channel(
            encoded.packets,
            config=case_channel_config,
        )
        plan = reconstruct_streaming_window(
            total_tokens=len(sample.token_ids),
            received_packets=loss_result.received,
            tokens_per_packet=tokens_per_packet,
        )
        reconstructed_tokens = _tokens_from_plan(plan)
        metrics = compute_token_metrics(
            original_tokens=sample.token_ids,
            reconstructed_tokens=reconstructed_tokens,
            reconstruction_plan=plan,
            mask_token_id=DEFAULT_MASK_TOKEN_ID,
        )
        case_id = f"case{case_index:04d}"
        result_rows.append(
            _result_row(
                run_id=run_id,
                case_id=case_id,
                sample_id=sample.sample_id,
                case_seed=case_seed,
                loss_rate=loss_rate,
                tokens_per_packet=tokens_per_packet,
                hash_bits=hash_bits,
                source_token_count=len(sample.token_ids),
                plan=plan,
                metrics=metrics,
                received_packet_count=len(loss_result.received),
                dropped_packet_count=len(loss_result.dropped),
                source_layout=source_layout,
                wire_interleaving=wire_interleaving,
                channel_config=case_channel_config,
                encoded=encoded,
                model_label=STREAMING_WINDOW_MODEL_LABEL,
                strategy=_stream_strategy(hash_bits),
                baseline_family=STREAMING_WINDOW_BASELINE_FAMILY,
                protection_mode=STREAMING_WINDOW_SCHEME,
            )
        )
        events.append(
            {
                "event_type": "streaming_window_micro_eval_case",
                "run_id": run_id,
                "case_id": case_id,
                "model_label": STREAMING_WINDOW_MODEL_LABEL,
                "strategy": _stream_strategy(hash_bits),
                "sample": sample.to_dict(),
                "encoded": encoded.to_dict(),
                "loss_result": loss_result.to_dict(),
                "reconstruction_plan": plan.to_dict(),
                "reconstructed_tokens": list(reconstructed_tokens),
                "metrics": metrics.to_dict(),
            }
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


def _validate_config(
    *,
    sample_lengths: Sequence[int],
    loss_rate: float,
    tokens_per_packet: int,
    hash_bits: int,
    vocab_size: int,
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
    if hash_bits not in {4, 8, 16}:
        raise ValueError("hash_bits must be 4, 8, or 16")
    if vocab_size <= 16:
        raise ValueError("vocab_size must be greater than 16 for synthetic samples")


def _tokens_from_plan(plan: ReconstructionPlan) -> tuple[int, ...]:
    return tuple(
        entry.token_id if entry.token_id is not None else DEFAULT_MASK_TOKEN_ID
        for entry in plan.entries
    )


def _run_id(
    *,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    hash_bits: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    xor_config: XorParityConfig,
) -> str:
    lengths = "-".join(str(length) for length in sample_lengths)
    source_chunk = source_layout.chunk_size if source_layout.chunk_size is not None else "default"
    stride = xor_config.stripe_stride if xor_config.stripe_stride is not None else "auto"
    return (
        f"xor-parity-micro-eval|matched-hash{hash_bits}|loss{loss_rate:g}|"
        f"lengths{lengths}|tpp{tokens_per_packet}|seed{seed}|"
        f"stripe{xor_config.data_packets_per_stripe}|stride{stride}|"
        f"source-{source_layout.mode}-chunk{source_chunk}|"
        f"wire-{wire_interleaving.mode}-span{wire_interleaving.span}|"
        f"channel-{channel_config.mode}"
    )


def _strategy(hash_bits: int) -> str:
    return f"Classical_XORParity_MatchedHash{hash_bits}"


def _manifest(
    *,
    run_id: str,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    hash_bits: int,
    vocab_size: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    xor_config: XorParityConfig,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "runner": "xor_parity_synthetic_micro_eval",
        "model_label": XOR_PARITY_MODEL_LABEL,
        "model_kind": "classical_xor_parity",
        "baseline_family": XOR_PARITY_BASELINE_FAMILY,
        "not_a_research_claim": True,
        "not_a_research_claim_warning": MICRO_EVAL_WARNING,
        "micro_eval": True,
        "strategy": _strategy(hash_bits),
        "config": {
            "sample_lengths": list(sample_lengths),
            "loss_rate": loss_rate,
            "seed": seed,
            "tokens_per_packet": tokens_per_packet,
            "hash_bits": hash_bits,
            "vocab_size": vocab_size,
            "protection_mode": XOR_PARITY_SCHEME,
            "oracle_hash_metadata": False,
            "source_layout": source_layout.to_dict(),
            "wire_interleaving": wire_interleaving.to_dict(),
            "channel": channel_config.to_dict(),
            "xor_parity": xor_config.to_dict(),
        },
        "artifacts": {
            "manifest": "run_manifest.json",
            "results": "results.csv",
            "events": "events.jsonl",
        },
    }


def _lt_run_id(
    *,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    hash_bits: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    lt_config: LTFountainConfig,
) -> str:
    lengths = "-".join(str(length) for length in sample_lengths)
    source_chunk = source_layout.chunk_size if source_layout.chunk_size is not None else "default"
    coverage = "coverage" if lt_config.coverage_aware else "random"
    return (
        f"lt-fountain-micro-eval|matched-hash{hash_bits}|loss{loss_rate:g}|"
        f"lengths{lengths}|tpp{tokens_per_packet}|seed{seed}|"
        f"repair{lt_config.repair_rate:g}|ltseed{lt_config.random_seed}|{coverage}|"
        f"source-{source_layout.mode}-chunk{source_chunk}|"
        f"wire-{wire_interleaving.mode}-span{wire_interleaving.span}|"
        f"channel-{channel_config.mode}"
    )


def _lt_strategy(hash_bits: int, coverage_aware: bool) -> str:
    suffix = "_CoverageAware" if coverage_aware else ""
    return f"Classical_LTFountain{suffix}_MatchedHash{hash_bits}"


def _lt_manifest(
    *,
    run_id: str,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    hash_bits: int,
    vocab_size: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    lt_config: LTFountainConfig,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "runner": "lt_fountain_synthetic_micro_eval",
        "model_label": LT_FOUNTAIN_MODEL_LABEL,
        "model_kind": "classical_lt_fountain",
        "baseline_family": LT_FOUNTAIN_BASELINE_FAMILY,
        "not_a_research_claim": True,
        "not_a_research_claim_warning": MICRO_EVAL_WARNING,
        "micro_eval": True,
        "strategy": _lt_strategy(hash_bits, lt_config.coverage_aware),
        "config": {
            "sample_lengths": list(sample_lengths),
            "loss_rate": loss_rate,
            "seed": seed,
            "tokens_per_packet": tokens_per_packet,
            "hash_bits": hash_bits,
            "vocab_size": vocab_size,
            "protection_mode": LT_FOUNTAIN_SCHEME,
            "oracle_hash_metadata": False,
            "source_layout": source_layout.to_dict(),
            "wire_interleaving": wire_interleaving.to_dict(),
            "channel": channel_config.to_dict(),
            "lt_fountain": lt_config.to_dict(),
        },
        "artifacts": {
            "manifest": "run_manifest.json",
            "results": "results.csv",
            "events": "events.jsonl",
        },
    }


def _stream_run_id(
    *,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    hash_bits: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    stream_config: StreamingWindowConfig,
) -> str:
    lengths = "-".join(str(length) for length in sample_lengths)
    source_chunk = source_layout.chunk_size if source_layout.chunk_size is not None else "default"
    return (
        f"streaming-window-micro-eval|matched-hash{hash_bits}|loss{loss_rate:g}|"
        f"lengths{lengths}|tpp{tokens_per_packet}|seed{seed}|"
        f"window{stream_config.window_size}|stride{stream_config.window_stride}|"
        f"source-{source_layout.mode}-chunk{source_chunk}|"
        f"wire-{wire_interleaving.mode}-span{wire_interleaving.span}|"
        f"channel-{channel_config.mode}"
    )


def _stream_strategy(hash_bits: int) -> str:
    return f"Classical_StreamingWindow_MatchedHash{hash_bits}"


def _stream_manifest(
    *,
    run_id: str,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    hash_bits: int,
    vocab_size: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    stream_config: StreamingWindowConfig,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "runner": "streaming_window_synthetic_micro_eval",
        "model_label": STREAMING_WINDOW_MODEL_LABEL,
        "model_kind": "classical_streaming_window",
        "baseline_family": STREAMING_WINDOW_BASELINE_FAMILY,
        "not_a_research_claim": True,
        "not_a_research_claim_warning": MICRO_EVAL_WARNING,
        "micro_eval": True,
        "strategy": _stream_strategy(hash_bits),
        "config": {
            "sample_lengths": list(sample_lengths),
            "loss_rate": loss_rate,
            "seed": seed,
            "tokens_per_packet": tokens_per_packet,
            "hash_bits": hash_bits,
            "vocab_size": vocab_size,
            "protection_mode": STREAMING_WINDOW_SCHEME,
            "oracle_hash_metadata": False,
            "source_layout": source_layout.to_dict(),
            "wire_interleaving": wire_interleaving.to_dict(),
            "channel": channel_config.to_dict(),
            "streaming_window": stream_config.to_dict(),
        },
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
    sample_id: str,
    case_seed: int,
    loss_rate: float,
    tokens_per_packet: int,
    hash_bits: int,
    source_token_count: int,
    plan: ReconstructionPlan,
    metrics: TokenMetrics,
    received_packet_count: int,
    dropped_packet_count: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    encoded,
    model_label: str,
    strategy: str,
    baseline_family: str,
    protection_mode: str,
) -> dict[str, Any]:
    overhead = encoded.overhead
    return {
        "run_id": run_id,
        "case_id": case_id,
        "sample_id": sample_id,
        "model_label": model_label,
        "strategy": strategy,
        "baseline_family": baseline_family,
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
        "source_token_count": source_token_count,
        "known_count": plan.known_count,
        "missing_count": plan.missing_count,
        "hash_guided_count": plan.hash_guided_count,
        "unguided_count": plan.unguided_count,
        "received_packet_count": received_packet_count,
        "dropped_packet_count": dropped_packet_count,
        "source_packet_count": encoded.source_packet_count,
        "extra_packet_count": encoded.extra_packet_count,
        "repair_packet_count": overhead.repair_packet_count,
        "repair_token_budget": overhead.repair_token_budget,
        "target_overhead_ratio": overhead.target_overhead_ratio,
        "actual_repair_token_overhead_ratio": overhead.actual_repair_token_overhead_ratio,
        "hash_profile_source": "not_used",
        "decode_latency_sec": 0.0,
        "decoder_steps": 0,
        "model_forward_calls": 0,
        **metrics.to_dict(),
    }
