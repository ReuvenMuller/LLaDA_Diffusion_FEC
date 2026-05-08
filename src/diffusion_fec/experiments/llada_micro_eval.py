"""Opt-in real LLaDA synthetic micro-eval runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from diffusion_fec.channels.packet_loss import (
    CHANNEL_RANDOM_IID,
    PacketLossChannelConfig,
)
from diffusion_fec.coding.hash_profiles import (
    DEFAULT_HASH_MAP_MODE,
    HASH_ALGORITHM,
    HASH_PROFILE_FORMAT_VERSION,
    TOKEN_STRING_SOURCE,
    hash_map_filename,
    load_hash_profile,
    load_hash_profile_metadata,
)
from diffusion_fec.coding.packetizer import SourceLayoutConfig, WireInterleavingConfig
from diffusion_fec.coding.protection import LOOKBACK_1_SCHEME
from diffusion_fec.experiments.llada_smoke import (
    RealLLaDASmokeUnavailable,
    _import_torch,
    _load_model,
    _load_tokenizer_config,
    _run_tiny_forward,
)
from diffusion_fec.experiments.logging import write_run_artifacts
from diffusion_fec.experiments.micro_eval import (
    MICRO_EVAL_MODEL_HASH,
    MICRO_EVAL_MODEL_ONLY,
    MICRO_EVAL_WARNING,
)
from diffusion_fec.experiments.smoke import SmokeRecoveryCase, run_smoke_recovery_case
from diffusion_fec.models.llada import LLADA_1_5_MODEL_ID, LLaDAAdapter
from diffusion_fec.types import TokenSample


REAL_LLADA_MICRO_EVAL_HASH_STRATEGY = "LLaDA_MicroEval_LoadedHashLookback1_NoPrompt"
REAL_LLADA_MICRO_EVAL_MODEL_ONLY_STRATEGY = "LLaDA_MicroEval_ModelOnly_NoPrompt"
DEFAULT_REAL_LLADA_MICRO_EVAL_SAMPLE_LENGTHS = (8,)
DEFAULT_REAL_LLADA_MICRO_EVAL_STEPS = 2


class RealLLaDAMicroEvalUnavailable(RuntimeError):
    """Raised when the opt-in real LLaDA micro-eval cannot run."""


def run_real_llada_micro_eval(
    *,
    output_dir: str | Path,
    model_id: str = LLADA_1_5_MODEL_ID,
    sample_lengths: Sequence[int] = DEFAULT_REAL_LLADA_MICRO_EVAL_SAMPLE_LENGTHS,
    loss_rate: float = 0.5,
    seed: int = 1,
    tokens_per_packet: int = 1,
    mode: str = MICRO_EVAL_MODEL_HASH,
    hash_bits: int = 4,
    steps: int = DEFAULT_REAL_LLADA_MICRO_EVAL_STEPS,
    local_files_only: bool = False,
    allow_cpu: bool = False,
    hash_profile_dir: str | Path | None = None,
    hash_map_mode: str = DEFAULT_HASH_MAP_MODE,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
    channel_config: PacketLossChannelConfig | None = None,
    torch_module: Any | None = None,
    tokenizer_adapter: LLaDAAdapter | None = None,
    model_adapter: LLaDAAdapter | None = None,
) -> dict[str, Any]:
    """Run a tiny synthetic real-LLaDA micro-eval and write artifacts.

    Real hash modes require an already-built tokenizer-specific hash profile.
    This runner intentionally does not live-build hash maps.
    """

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
        mode=mode,
        hash_bits=hash_bits,
        steps=steps,
    )

    torch = torch_module or _safe_import_torch()
    tokenizer = tokenizer_adapter or _safe_load_tokenizer_config(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    tokenizer_stage = _tokenizer_stage(tokenizer)
    protection_mode = LOOKBACK_1_SCHEME if mode == MICRO_EVAL_MODEL_HASH else "none"
    token_hash_map = None
    hash_profile_info = _unused_hash_profile_info(hash_bits=hash_bits, hash_map_mode=hash_map_mode)
    if mode == MICRO_EVAL_MODEL_HASH:
        token_hash_map, hash_profile_info = _load_required_hash_profile(
            profile_dir=hash_profile_dir,
            hash_bits=hash_bits,
            hash_map_mode=hash_map_mode,
            tokenizer=tokenizer,
            model_id=model_id,
        )

    if not torch.cuda.is_available() and not allow_cpu:
        raise RealLLaDAMicroEvalUnavailable(
            "CUDA is not available. Real LLaDA micro-eval is disabled before "
            "model weight loading; use the GPU server or pass --allow-cpu-real-llada "
            "only for an explicit CPU attempt."
        )

    adapter = model_adapter or _safe_load_model(
        model_id=model_id,
        local_files_only=local_files_only,
        use_cuda=torch.cuda.is_available(),
        torch_module=torch,
    )
    forward_shape = _safe_run_tiny_forward(adapter)
    config = adapter.decoding_config(steps=steps, block_length=tokens_per_packet)
    run_id = _run_id(
        model_id=model_id,
        sample_lengths=sample_lengths,
        mode=mode,
        hash_bits=hash_bits,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        steps=steps,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
    )
    manifest = _manifest(
        run_id=run_id,
        model_id=model_id,
        sample_lengths=sample_lengths,
        mode=mode,
        protection_mode=protection_mode,
        hash_bits=hash_bits,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        steps=steps,
        local_files_only=local_files_only,
        allow_cpu=allow_cpu,
        tokenizer_stage=tokenizer_stage,
        forward_shape=forward_shape,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        hash_profile_info=hash_profile_info,
    )

    result_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for case_index, sample_length in enumerate(sample_lengths):
        sample = _synthetic_llada_sample(
            adapter=adapter,
            sample_index=case_index,
            token_count=sample_length,
        )
        case_seed = seed + case_index
        case_channel_config = replace(channel_config, seed=case_seed)
        case = run_smoke_recovery_case(
            sample=sample,
            model=adapter,
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
                model_id=model_id,
                seed=case_seed,
                loss_rate=loss_rate,
                tokens_per_packet=tokens_per_packet,
                mode=mode,
                protection_mode=protection_mode,
                hash_bits=hash_bits,
                source_layout=source_layout,
                wire_interleaving=wire_interleaving,
                channel_config=case_channel_config,
                hash_profile_info=hash_profile_info,
            )
        )
        events.append(
            {
                "event_type": "real_llada_micro_eval_case",
                "run_id": run_id,
                "case_id": case_id,
                "model_label": model_id,
                "strategy": _strategy(mode),
                "case": case.to_dict(),
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
    mode: str,
    hash_bits: int,
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
    if steps <= 0:
        raise ValueError("steps must be positive")


def _safe_import_torch():
    try:
        return _import_torch()
    except RealLLaDASmokeUnavailable as exc:
        raise RealLLaDAMicroEvalUnavailable(str(exc)) from exc


def _safe_load_tokenizer_config(*, model_id: str, local_files_only: bool) -> LLaDAAdapter:
    try:
        return _load_tokenizer_config(
            model_id=model_id,
            local_files_only=local_files_only,
        )
    except RealLLaDASmokeUnavailable as exc:
        raise RealLLaDAMicroEvalUnavailable(str(exc)) from exc


def _safe_load_model(
    *,
    model_id: str,
    local_files_only: bool,
    use_cuda: bool,
    torch_module,
) -> LLaDAAdapter:
    try:
        return _load_model(
            model_id=model_id,
            local_files_only=local_files_only,
            use_cuda=use_cuda,
            torch_module=torch_module,
        )
    except RealLLaDASmokeUnavailable as exc:
        raise RealLLaDAMicroEvalUnavailable(str(exc)) from exc


def _safe_run_tiny_forward(adapter: LLaDAAdapter) -> tuple[int, int, int]:
    try:
        return _run_tiny_forward(adapter)
    except RealLLaDASmokeUnavailable as exc:
        raise RealLLaDAMicroEvalUnavailable(str(exc)) from exc


def _load_required_hash_profile(
    *,
    profile_dir: str | Path | None,
    hash_bits: int,
    hash_map_mode: str,
    tokenizer: LLaDAAdapter,
    model_id: str,
):
    if profile_dir is None:
        raise RealLLaDAMicroEvalUnavailable(
            "hash_profile_dir is required for real LLaDA model_hash micro-eval. "
            "Build profiles ahead of time and pass --hash-profile-dir."
        )
    profile_path = Path(profile_dir)
    try:
        token_hash_map = load_hash_profile(
            profile_dir=profile_path,
            hash_bits=hash_bits,
            map_mode=hash_map_mode,
        )
        metadata = load_hash_profile_metadata(profile_path)
    except (FileNotFoundError, ValueError) as exc:
        raise RealLLaDAMicroEvalUnavailable(
            f"Required loaded hash profile is unavailable for hash_bits={hash_bits}: {exc}"
        ) from exc

    if token_hash_map.vocab_size != tokenizer.vocab_size:
        raise RealLLaDAMicroEvalUnavailable(
            "Loaded hash profile vocab_size does not match the LLaDA tokenizer/config "
            f"({token_hash_map.vocab_size} != {tokenizer.vocab_size})."
        )
    for key in ("model_id", "tokenizer_name"):
        value = metadata.get(key)
        if value is not None and value != model_id:
            raise RealLLaDAMicroEvalUnavailable(
                f"Loaded hash profile {key}={value!r} does not match model_id={model_id!r}."
            )

    return token_hash_map, _hash_profile_info(
        profile_dir=profile_path,
        metadata=metadata,
        hash_bits=hash_bits,
        hash_map_mode=hash_map_mode,
    )


def _hash_profile_info(
    *,
    profile_dir: Path,
    metadata: dict[str, Any],
    hash_bits: int,
    hash_map_mode: str,
) -> dict[str, Any]:
    return {
        "source": "loaded_profile",
        "profile_dir": str(profile_dir),
        "profile_name": metadata.get("profile_name"),
        "map_mode": hash_map_mode,
        "hash_bits": hash_bits,
        "file": hash_map_filename(hash_bits, hash_map_mode),
        "format_version": metadata.get("format_version", HASH_PROFILE_FORMAT_VERSION),
        "algorithm": HASH_ALGORITHM,
        "token_string_source": metadata.get("token_string_source", TOKEN_STRING_SOURCE),
        "model_id": metadata.get("model_id"),
        "tokenizer_name": metadata.get("tokenizer_name"),
        "vocab_size": metadata.get("vocab_size"),
    }


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


def _tokenizer_stage(adapter: LLaDAAdapter) -> dict[str, Any]:
    return {
        "mask_token_id": adapter.mask_token_id,
        "eos_token_id": adapter.eos_token_id,
        "pad_token_id": adapter.pad_token_id,
        "vocab_size": adapter.vocab_size,
        "max_sequence_length": adapter.max_sequence_length,
    }


def _synthetic_llada_sample(
    *,
    adapter: LLaDAAdapter,
    sample_index: int,
    token_count: int,
) -> TokenSample:
    special_ids = {
        adapter.mask_token_id,
        *(token_id for token_id in (adapter.eos_token_id, adapter.pad_token_id) if token_id is not None),
    }
    token_ids: list[int] = []
    candidate = 3 + sample_index * token_count
    while len(token_ids) < token_count:
        token_id = candidate % adapter.vocab_size
        candidate += 1
        if token_id in special_ids:
            continue
        token_ids.append(token_id)
    return TokenSample(
        sample_id=f"real-llada-synthetic-len{token_count:04d}-case{sample_index:04d}",
        text=adapter.decode(token_ids, skip_special_tokens=False),
        token_ids=tuple(token_ids),
        tokenizer_name=adapter.model_id,
    )


def _run_id(
    *,
    model_id: str,
    sample_lengths: Sequence[int],
    mode: str,
    hash_bits: int,
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    steps: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
) -> str:
    model_label = model_id.replace("/", "-")
    lengths = "-".join(str(length) for length in sample_lengths)
    source_chunk = source_layout.chunk_size if source_layout.chunk_size is not None else "default"
    return (
        f"real-llada-micro-eval|{model_label}|{mode}|hash{hash_bits}|"
        f"loss{loss_rate:g}|lengths{lengths}|tpp{tokens_per_packet}|steps{steps}|"
        f"seed{seed}|source-{source_layout.mode}-chunk{source_chunk}|"
        f"wire-{wire_interleaving.mode}-span{wire_interleaving.span}"
    )


def _strategy(mode: str) -> str:
    if mode == MICRO_EVAL_MODEL_HASH:
        return REAL_LLADA_MICRO_EVAL_HASH_STRATEGY
    return REAL_LLADA_MICRO_EVAL_MODEL_ONLY_STRATEGY


def _manifest(
    *,
    run_id: str,
    model_id: str,
    sample_lengths: Sequence[int],
    mode: str,
    protection_mode: str,
    hash_bits: int,
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    steps: int,
    local_files_only: bool,
    allow_cpu: bool,
    tokenizer_stage: dict[str, Any],
    forward_shape: tuple[int, int, int],
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    hash_profile_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "runner": "real_llada_synthetic_micro_eval",
        "model_label": model_id,
        "model_kind": "real_llada_huggingface",
        "not_a_research_claim": True,
        "not_a_research_claim_warning": MICRO_EVAL_WARNING,
        "micro_eval": True,
        "opt_in_required": True,
        "strategy": _strategy(mode),
        "preflight": {
            "tokenizer_config_loaded": True,
            "tiny_forward_shape": list(forward_shape),
            "tiny_forward_calls": 1,
        },
        "tokenizer_config": tokenizer_stage,
        "config": {
            "sample_lengths": list(sample_lengths),
            "loss_rate": loss_rate,
            "seed": seed,
            "tokens_per_packet": tokens_per_packet,
            "mode": mode,
            "protection_mode": protection_mode,
            "oracle_hash_metadata": False,
            "hash_bits": hash_bits,
            "steps": steps,
            "local_files_only": local_files_only,
            "allow_cpu": allow_cpu,
            "source_layout": source_layout.to_dict(),
            "wire_interleaving": wire_interleaving.to_dict(),
            "channel": channel_config.to_dict(),
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
    model_id: str,
    seed: int,
    loss_rate: float,
    tokens_per_packet: int,
    mode: str,
    protection_mode: str,
    hash_bits: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    hash_profile_info: dict[str, Any],
) -> dict[str, Any]:
    plan = case.reconstruction_plan
    diagnostics = case.decoding_result.diagnostics
    return {
        "run_id": run_id,
        "case_id": case_id,
        "sample_id": case.sample.sample_id,
        "model_label": model_id,
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
        "seed": seed,
        "tokens_per_packet": tokens_per_packet,
        "source_token_count": len(case.sample.token_ids),
        "known_count": plan.known_count,
        "missing_count": plan.missing_count,
        "hash_guided_count": plan.hash_guided_count,
        "unguided_count": plan.unguided_count,
        "received_packet_count": len(case.loss_result.received),
        "dropped_packet_count": len(case.loss_result.dropped),
        "hash_profile_source": hash_profile_info.get("source"),
        "decode_latency_sec": case.decoding_result.decode_latency_sec,
        "decoder_steps": case.decoding_result.steps,
        "model_forward_calls": diagnostics.get("model_forward_calls"),
        **case.metrics.to_dict(),
    }
