"""Hybrid LLaDA/hash/XOR validation runner."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from diffusion_fec.baselines.overhead import (
    metadata_token_equivalent_overhead_ratio,
    token_bit_width_for_vocab,
    token_equivalent_overhead,
)
from diffusion_fec.baselines.xor_equations import (
    ParityCandidateFilter,
    XorAuditResult,
    XorPeelResult,
    audit_xor_equations,
    equations_from_parity_packets,
    known_tokens_from_data_packets,
    peel_xor_equations,
)
from diffusion_fec.baselines.xor_parity import (
    XOR_PARITY_SCHEME,
    XorParityConfig,
    XorParityEncoded,
    encode_xor_parity,
)
from diffusion_fec.channels.packet_loss import (
    CHANNEL_RANDOM_IID,
    PacketLossChannelConfig,
    apply_packet_loss_channel,
)
from diffusion_fec.channels.random_loss import RandomLossResult
from diffusion_fec.coding.hash_profiles import DEFAULT_HASH_MAP_MODE, load_or_build_hash_profile
from diffusion_fec.coding.packetizer import (
    SOURCE_PACKET_INDEX_METADATA_KEY,
    SourceLayoutConfig,
    WireInterleavingConfig,
)
from diffusion_fec.coding.protection import (
    LOOKBACK_1_SCHEME,
    LOOKBACK_HASH_METADATA_KEY,
    attach_lookback_hashes,
    extract_received_hash_metadata,
)
from diffusion_fec.coding.token_hash import TokenHashMap
from diffusion_fec.decoding.llada_diffusion import (
    EDITABLE_UPDATE_COMMIT_ONCE,
    HASH_CONSTRAINT_ALWAYS,
    DiffusionDecodingConfig,
    decode_masked_diffusion,
)
from diffusion_fec.experiments.llada_micro_eval import (
    RealLLaDAMicroEvalUnavailable,
    _hash_profile_info,
    _load_llada_dataset_samples,
    _load_pretokenized_llada_samples,
    _load_required_hash_profile,
    _safe_import_torch,
    _safe_load_model,
    _safe_load_tokenizer_config,
    _safe_run_tiny_forward,
    _tokenizer_stage,
)
from diffusion_fec.experiments.logging import start_run_timer, write_run_artifacts
from diffusion_fec.experiments.micro_eval import (
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_MASK_TOKEN_ID,
    DEFAULT_MICRO_EVAL_HASH_BITS,
    DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    DEFAULT_MICRO_EVAL_STEPS,
    DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    DEFAULT_MICRO_EVAL_VOCAB_SIZE,
    DEFAULT_PAD_TOKEN_ID,
    MICRO_EVAL_MODEL_LABEL,
    MICRO_EVAL_WARNING,
    FakeDeterministicMicroEvalModel,
    synthetic_sample,
)
from diffusion_fec.metrics.token_metrics import (
    ChannelLostPositionMetrics,
    channel_lost_source_positions,
    compute_channel_lost_position_metrics,
    compute_token_metrics,
    TokenMetrics,
)
from diffusion_fec.models.llada import LLADA_1_5_MODEL_ID, LLaDAAdapter
from diffusion_fec.types import (
    DecodingResult,
    Packet,
    ReconstructionEntry,
    ReconstructionPlan,
    STATE_KNOWN,
    STATE_MISSING,
    STATE_UNGUIDED,
    TokenSample,
)


HYBRID_MODE_PRE_PEEL_ONLY = "pre_peel_only"
HYBRID_MODE_PARITY_FILTER = "parity_filter"
HYBRID_MODE_ITERATIVE_PEEL = "iterative_peel"
VALID_HYBRID_MODES = frozenset(
    {
        HYBRID_MODE_PRE_PEEL_ONLY,
        HYBRID_MODE_PARITY_FILTER,
        HYBRID_MODE_ITERATIVE_PEEL,
    }
)
HYBRID_PROTECTION_MODE = "lookback_1+xor_parity"
FAKE_HYBRID_MODEL_LABEL = "FakeDeterministicHybridXorHashModel"
REAL_HYBRID_MODEL_KIND = "real_llada_huggingface_hybrid_xor_hash"
FAKE_HYBRID_MODEL_KIND = "fake_deterministic_hybrid_xor_hash_model"
DEFAULT_XOR_OVERHEAD_BITS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class HybridRecoveryCase:
    """Artifacts for one hybrid recovery case."""

    sample: TokenSample
    encoded: XorParityEncoded
    transmitted_packets: tuple[Packet, ...]
    loss_result: RandomLossResult
    hash_metadata: dict[int, int]
    initial_peel: XorPeelResult
    final_audit: XorAuditResult
    parity_filter_diagnostics: dict[str, Any]
    channel_config: PacketLossChannelConfig
    hybrid_mode: str
    reconstruction_plan: ReconstructionPlan
    decoding_result: DecodingResult
    mask_token_id: int

    @property
    def metrics(self) -> TokenMetrics:
        return compute_token_metrics(
            original_tokens=self.sample.token_ids,
            reconstructed_tokens=self.decoding_result.reconstructed_tokens,
            reconstruction_plan=self.reconstruction_plan,
            mask_token_id=self.mask_token_id,
        )

    @property
    def channel_lost_positions(self) -> tuple[int, ...]:
        return channel_lost_source_positions(self.loss_result.dropped)

    @property
    def channel_lost_metrics(self) -> ChannelLostPositionMetrics:
        return compute_channel_lost_position_metrics(
            original_tokens=self.sample.token_ids,
            reconstructed_tokens=self.decoding_result.reconstructed_tokens,
            channel_lost_positions=self.channel_lost_positions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "encoded": self.encoded.to_dict(),
            "transmitted_packets": [packet.to_dict() for packet in self.transmitted_packets],
            "loss_result": self.loss_result.to_dict(),
            "hash_metadata": dict(self.hash_metadata),
            "initial_peel": self.initial_peel.to_dict(),
            "final_audit": self.final_audit.to_dict(),
            "parity_filter_diagnostics": dict(self.parity_filter_diagnostics),
            "channel_config": self.channel_config.to_dict(),
            "hybrid_mode": self.hybrid_mode,
            "reconstruction_plan": self.reconstruction_plan.to_dict(),
            "decoding_result": self.decoding_result.to_dict(),
            "channel_lost_positions": list(self.channel_lost_positions),
            "channel_lost_metrics": self.channel_lost_metrics.to_dict(),
            "metrics": {**self.metrics.to_dict(), **self.channel_lost_metrics.to_dict()},
        }


class IterativeXorPeelHook:
    """Post-commit hook that promotes newly XOR-solvable tokens to fixed."""

    def __init__(
        self,
        *,
        equations,
        hash_metadata: Mapping[int, int],
        token_hash_map: TokenHashMap,
        mask_token_id: int,
        vocab_size: int,
        banned_token_ids: Sequence[int],
    ) -> None:
        self.equations = tuple(equations)
        self.hash_metadata = {int(position): int(value) for position, value in hash_metadata.items()}
        self.token_hash_map = token_hash_map
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)
        self.banned_token_ids = frozenset(int(token_id) for token_id in banned_token_ids)
        self.call_count = 0
        self.per_step_recovered_count: list[int] = []
        self.recovered_tokens: dict[int, int] = {}
        self.conflicts = {}

    def __call__(
        self,
        *,
        input_ids,
        step: int,
        prompt_length: int,
        committed_positions,
        fixed_token_ids,
        plan: ReconstructionPlan,
        config: DiffusionDecodingConfig,
    ) -> dict[str, Any]:
        self.call_count += 1
        known_tokens = _known_tokens_from_decoder_state(
            input_ids=input_ids,
            plan=plan,
            prompt_length=prompt_length,
            mask_token_id=self.mask_token_id,
        )
        peel = peel_xor_equations(
            equations=self.equations,
            known_tokens=known_tokens,
            hash_metadata=self.hash_metadata,
            token_hash_map=self.token_hash_map,
            vocab_size=self.vocab_size,
            banned_token_ids=self.banned_token_ids,
        )
        self._record_conflicts(peel)
        newly_fixed = {
            int(position): int(token_id)
            for position, token_id in peel.recovered_tokens.items()
            if int(position) not in self.recovered_tokens
            and input_ids[prompt_length + int(position)] == self.mask_token_id
        }
        self.recovered_tokens.update(newly_fixed)
        self.per_step_recovered_count.append(len(newly_fixed))
        return {"fixed_tokens": newly_fixed}

    def diagnostics(self) -> dict[str, Any]:
        conflicts = tuple(self.conflicts.values())
        return {
            "iterative_peel_enabled": True,
            "iterative_peel_passes": self.call_count,
            "iterative_peel_recovered_count": len(self.recovered_tokens),
            "iterative_peel_recovered_positions": tuple(sorted(self.recovered_tokens)),
            "iterative_peel_recovered_count_by_step": tuple(self.per_step_recovered_count),
            "iterative_peel_conflict_count": len(conflicts),
            "iterative_peel_hash_conflict_count": sum(
                conflict.reason in {
                    "parity_hash_conflict",
                    "hash_metadata_without_token_hash_map",
                }
                for conflict in conflicts
            ),
            "iterative_peel_special_token_conflict_count": sum(
                conflict.reason == "solved_token_is_banned"
                for conflict in conflicts
            ),
            "iterative_peel_vocab_conflict_count": sum(
                conflict.reason in {
                    "solved_token_negative",
                    "solved_token_outside_vocab",
                    "solved_token_outside_hash_vocab",
                }
                for conflict in conflicts
            ),
            "iterative_peel_conflicts": [conflict.to_dict() for conflict in conflicts],
        }

    def _record_conflicts(self, peel: XorPeelResult) -> None:
        for conflict in peel.conflicts:
            key = (
                conflict.equation_id,
                conflict.position,
                conflict.solved_token_id,
                conflict.reason,
            )
            self.conflicts[key] = conflict


def run_hybrid_xor_hash_micro_eval(
    *,
    output_dir: str | Path,
    sample_lengths: Sequence[int] = DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    samples: Sequence[TokenSample] | None = None,
    dataset_info: dict[str, Any] | None = None,
    loss_rate: float = 0.5,
    seed: int = 0,
    tokens_per_packet: int = DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    hash_bits: int = DEFAULT_MICRO_EVAL_HASH_BITS,
    xor_overhead_bits_per_token: float = DEFAULT_XOR_OVERHEAD_BITS_PER_TOKEN,
    vocab_size: int = DEFAULT_MICRO_EVAL_VOCAB_SIZE,
    steps: int = DEFAULT_MICRO_EVAL_STEPS,
    editable_update_mode: str = EDITABLE_UPDATE_COMMIT_ONCE,
    hash_constraint_schedule: str = HASH_CONSTRAINT_ALWAYS,
    hybrid_mode: str = HYBRID_MODE_PARITY_FILTER,
    parity_filter_fallback: bool = True,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
    channel_config: PacketLossChannelConfig | None = None,
    hash_profile_dir: str | Path | None = None,
    build_hash_profile: bool = False,
    hash_map_mode: str = DEFAULT_HASH_MAP_MODE,
    hash_profile_name: str = "fake_hybrid_xor_hash_v1",
) -> dict[str, Any]:
    """Run fake/model-free hybrid validation and write artifacts."""

    token_hash_map, hash_profile_info = load_or_build_hash_profile(
        profile_dir=hash_profile_dir,
        profile_name=hash_profile_name,
        vocab_size=vocab_size,
        hash_bits=hash_bits,
        decode_token=lambda token_id: f"fake-token-{token_id}",
        excluded_token_ids={DEFAULT_MASK_TOKEN_ID, DEFAULT_EOS_TOKEN_ID, DEFAULT_PAD_TOKEN_ID},
        salt="fake-hybrid-xor-hash",
        map_mode=hash_map_mode,
        model_id=FAKE_HYBRID_MODEL_LABEL,
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
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
    )
    samples = None if samples is None else tuple(samples)
    sample_lengths = _resolve_sample_lengths(
        sample_lengths=sample_lengths,
        samples=samples,
        vocab_size=vocab_size,
    )
    return _run_hybrid_cases(
        output_dir=output_dir,
        runner="hybrid_xor_hash_synthetic_micro_eval",
        model_label=FAKE_HYBRID_MODEL_LABEL,
        model_kind=FAKE_HYBRID_MODEL_KIND,
        sample_lengths=sample_lengths,
        samples=samples,
        dataset_info=dataset_info,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        xor_overhead_bits_per_token=xor_overhead_bits_per_token,
        vocab_size=vocab_size,
        steps=steps,
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
        hybrid_mode=hybrid_mode,
        parity_filter_fallback=parity_filter_fallback,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        token_hash_map=token_hash_map,
        hash_profile_info=hash_profile_info,
        config=config,
        model_factory=lambda sample: FakeDeterministicMicroEvalModel(
            target_tokens=sample.token_ids,
            vocab_size=vocab_size,
        ),
        normalize_decode_latency=True,
        real_llada=False,
        preflight=None,
    )


def run_real_llada_hybrid_xor_hash_micro_eval(
    *,
    output_dir: str | Path,
    model_id: str = LLADA_1_5_MODEL_ID,
    sample_lengths: Sequence[int] = (8,),
    loss_rate: float = 0.5,
    seed: int = 0,
    tokens_per_packet: int = DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    hash_bits: int = DEFAULT_MICRO_EVAL_HASH_BITS,
    xor_overhead_bits_per_token: float = DEFAULT_XOR_OVERHEAD_BITS_PER_TOKEN,
    steps: int = 8,
    editable_update_mode: str = EDITABLE_UPDATE_COMMIT_ONCE,
    hash_constraint_schedule: str = HASH_CONSTRAINT_ALWAYS,
    hybrid_mode: str = HYBRID_MODE_PARITY_FILTER,
    parity_filter_fallback: bool = True,
    local_files_only: bool = False,
    allow_cpu: bool = False,
    hash_profile_dir: str | Path | None = None,
    hash_map_mode: str = DEFAULT_HASH_MAP_MODE,
    dataset_path: str | Path | None = None,
    dataset_label: str | None = None,
    dataset_sample_count: int | None = None,
    dataset_seed: int = 0,
    dataset_min_tokens: int = 1,
    dataset_max_tokens: int | None = None,
    tokenized_samples_path: str | Path | None = None,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
    channel_config: PacketLossChannelConfig | None = None,
    torch_module: Any | None = None,
    tokenizer_adapter: LLaDAAdapter | None = None,
    model_adapter: LLaDAAdapter | None = None,
) -> dict[str, Any]:
    """Run opt-in real LLaDA hybrid validation and write artifacts."""

    torch = torch_module or _safe_import_torch()
    tokenizer = tokenizer_adapter or _safe_load_tokenizer_config(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    if dataset_path is not None and tokenized_samples_path is not None:
        raise RealLLaDAMicroEvalUnavailable(
            "Pass either dataset_path or tokenized_samples_path, not both. "
            "Frozen comparisons should prefer tokenized_samples_path."
        )
    samples = None
    dataset_info = None
    resolved_sample_lengths = tuple(sample_lengths)
    if tokenized_samples_path is not None:
        samples, dataset_info = _load_pretokenized_llada_samples(
            tokenized_samples_path=tokenized_samples_path,
            tokenizer=tokenizer,
            model_id=model_id,
        )
        resolved_sample_lengths = tuple(len(sample.token_ids) for sample in samples)
    elif dataset_path is not None:
        samples, dataset_info = _load_llada_dataset_samples(
            dataset_path=dataset_path,
            dataset_label=dataset_label,
            dataset_sample_count=dataset_sample_count,
            dataset_seed=dataset_seed,
            dataset_min_tokens=dataset_min_tokens,
            dataset_max_tokens=dataset_max_tokens,
            tokenizer=tokenizer,
        )
        resolved_sample_lengths = tuple(len(sample.token_ids) for sample in samples)

    token_hash_map, hash_profile_info = _load_required_hash_profile(
        profile_dir=hash_profile_dir,
        hash_bits=hash_bits,
        hash_map_mode=hash_map_mode,
        tokenizer=tokenizer,
        model_id=model_id,
    )
    if not torch.cuda.is_available() and not allow_cpu:
        raise RealLLaDAMicroEvalUnavailable(
            "CUDA is not available. Real LLaDA hybrid validation is disabled before "
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
    config = adapter.decoding_config(
        steps=steps,
        block_length=max(tokens_per_packet, 1),
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
    )
    return _run_hybrid_cases(
        output_dir=output_dir,
        runner="real_llada_hybrid_xor_hash_micro_eval",
        model_label=model_id,
        model_kind=REAL_HYBRID_MODEL_KIND,
        sample_lengths=resolved_sample_lengths,
        samples=samples,
        dataset_info=dataset_info,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        xor_overhead_bits_per_token=xor_overhead_bits_per_token,
        vocab_size=adapter.vocab_size,
        steps=steps,
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
        hybrid_mode=hybrid_mode,
        parity_filter_fallback=parity_filter_fallback,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        token_hash_map=token_hash_map,
        hash_profile_info=hash_profile_info,
        config=config,
        model_factory=lambda sample: adapter,
        normalize_decode_latency=False,
        real_llada=True,
        preflight={
            "tokenizer_config_loaded": True,
            "tiny_forward_shape": list(forward_shape),
            "tiny_forward_calls": 1,
            "tokenizer_config": _tokenizer_stage(tokenizer),
            "local_files_only": local_files_only,
            "allow_cpu": allow_cpu,
        },
    )


def run_hybrid_recovery_case(
    *,
    sample: TokenSample,
    model: object,
    config: DiffusionDecodingConfig,
    tokens_per_packet: int,
    token_hash_map: TokenHashMap,
    xor_config: XorParityConfig,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    hybrid_mode: str,
    parity_filter_fallback: bool,
) -> HybridRecoveryCase:
    """Run one hybrid packet-loss and LLaDA recovery case."""

    _validate_hybrid_mode(hybrid_mode)
    if hybrid_mode == HYBRID_MODE_ITERATIVE_PEEL:
        _validate_iterative_peel_decoder_config(config)
    encoded = encode_xor_parity(
        sample,
        tokens_per_packet=tokens_per_packet,
        config=xor_config,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
    )
    protected_source_packets = attach_lookback_hashes(encoded.source_packets, token_hash_map)
    transmitted_packets = _merge_protected_source_and_parity_packets(
        encoded=encoded,
        protected_source_packets=protected_source_packets,
    )
    loss_result = apply_packet_loss_channel(
        transmitted_packets,
        config=channel_config,
    )
    known_tokens = known_tokens_from_data_packets(
        loss_result.received,
        total_tokens=len(sample.token_ids),
    )
    hash_metadata = _filter_known_position_hash_metadata(
        extract_received_hash_metadata(loss_result.received),
        known_tokens=known_tokens,
    )
    received_equations = equations_from_parity_packets(loss_result.received)
    initial_peel = peel_xor_equations(
        equations=received_equations,
        known_tokens=known_tokens,
        hash_metadata=hash_metadata,
        token_hash_map=token_hash_map,
        vocab_size=config.vocab_size,
        banned_token_ids=config.special_token_ids,
    )
    hash_metadata = _filter_known_position_hash_metadata(
        hash_metadata,
        known_tokens=initial_peel.known_tokens,
    )
    plan = _build_plan_from_known_and_hash(
        total_tokens=len(sample.token_ids),
        known_tokens=initial_peel.known_tokens,
        hash_metadata=hash_metadata,
    )
    parity_filter = None
    if hybrid_mode in {HYBRID_MODE_PARITY_FILTER, HYBRID_MODE_ITERATIVE_PEEL}:
        parity_filter = ParityCandidateFilter(
            equations=received_equations,
            known_tokens=initial_peel.known_tokens,
            mask_token_id=config.mask_token_id,
            fallback_on_empty=parity_filter_fallback,
        )
    post_commit_hook = None
    if hybrid_mode == HYBRID_MODE_ITERATIVE_PEEL:
        post_commit_hook = IterativeXorPeelHook(
            equations=received_equations,
            hash_metadata=hash_metadata,
            token_hash_map=token_hash_map,
            mask_token_id=config.mask_token_id,
            vocab_size=config.vocab_size,
            banned_token_ids=config.special_token_ids,
        )
    decoding_result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=config,
        token_hash_map=token_hash_map,
        candidate_filter=parity_filter,
        post_commit_hook=post_commit_hook,
    )
    parity_filter_diagnostics = (
        parity_filter.diagnostics() if parity_filter is not None else _empty_filter_diagnostics()
    )
    final_audit = audit_xor_equations(
        equations=received_equations,
        token_by_position={
            position: token_id
            for position, token_id in enumerate(decoding_result.reconstructed_tokens)
        },
    )
    decoding_result = _decoding_result_with_hybrid_diagnostics(
        decoding_result=decoding_result,
        hybrid_mode=hybrid_mode,
        initial_peel=initial_peel,
        final_audit=final_audit,
        parity_filter_diagnostics=parity_filter_diagnostics,
        iterative_peel_diagnostics=(
            post_commit_hook.diagnostics()
            if post_commit_hook is not None
            else _empty_iterative_peel_diagnostics()
        ),
        parity_equation_count=len(equations_from_parity_packets(encoded.parity_packets)),
        parity_received_equation_count=len(received_equations),
    )
    return HybridRecoveryCase(
        sample=sample,
        encoded=encoded,
        transmitted_packets=transmitted_packets,
        loss_result=loss_result,
        hash_metadata=hash_metadata,
        initial_peel=initial_peel,
        final_audit=final_audit,
        parity_filter_diagnostics=parity_filter_diagnostics,
        channel_config=channel_config,
        hybrid_mode=hybrid_mode,
        reconstruction_plan=plan,
        decoding_result=decoding_result,
        mask_token_id=config.mask_token_id,
    )


def _run_hybrid_cases(
    *,
    output_dir: str | Path,
    runner: str,
    model_label: str,
    model_kind: str,
    sample_lengths: Sequence[int],
    samples: Sequence[TokenSample] | None,
    dataset_info: dict[str, Any] | None,
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    hash_bits: int,
    xor_overhead_bits_per_token: float,
    vocab_size: int,
    steps: int,
    editable_update_mode: str,
    hash_constraint_schedule: str,
    hybrid_mode: str,
    parity_filter_fallback: bool,
    source_layout: SourceLayoutConfig | None,
    wire_interleaving: WireInterleavingConfig | None,
    channel_config: PacketLossChannelConfig | None,
    token_hash_map: TokenHashMap,
    hash_profile_info: dict[str, Any],
    config: DiffusionDecodingConfig,
    model_factory: Callable[[TokenSample], object],
    normalize_decode_latency: bool,
    real_llada: bool,
    preflight: dict[str, Any] | None,
) -> dict[str, Any]:
    run_timer = start_run_timer()
    source_layout = source_layout or SourceLayoutConfig()
    wire_interleaving = wire_interleaving or WireInterleavingConfig()
    channel_config = channel_config or PacketLossChannelConfig(
        mode=CHANNEL_RANDOM_IID,
        loss_rate=loss_rate,
        seed=seed,
    )
    samples = None if samples is None else tuple(samples)
    sample_lengths = _resolve_sample_lengths(
        sample_lengths=sample_lengths,
        samples=samples,
        vocab_size=vocab_size,
    )
    _validate_common_config(
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        xor_overhead_bits_per_token=xor_overhead_bits_per_token,
        vocab_size=vocab_size,
        steps=steps,
        hybrid_mode=hybrid_mode,
    )
    xor_config = XorParityConfig(
        data_packets_per_stripe=4,
        target_overhead_ratio=xor_overhead_bits_per_token / token_bit_width_for_vocab(vocab_size),
        vocab_size=vocab_size,
    )
    run_id = _run_id(
        runner=runner,
        model_label=model_label,
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        xor_overhead_bits_per_token=xor_overhead_bits_per_token,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        hybrid_mode=hybrid_mode,
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
    )
    manifest = _manifest(
        run_id=run_id,
        runner=runner,
        model_label=model_label,
        model_kind=model_kind,
        sample_lengths=sample_lengths,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        xor_overhead_bits_per_token=xor_overhead_bits_per_token,
        vocab_size=vocab_size,
        steps=steps,
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
        hybrid_mode=hybrid_mode,
        parity_filter_fallback=parity_filter_fallback,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        xor_config=xor_config,
        hash_profile_info=hash_profile_info,
        dataset_info=dataset_info,
        real_llada=real_llada,
        preflight=preflight,
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
        case_seed = seed + case_index
        case_channel_config = replace(channel_config, seed=case_seed)
        case = run_hybrid_recovery_case(
            sample=sample,
            model=model_factory(sample),
            config=config,
            tokens_per_packet=tokens_per_packet,
            token_hash_map=token_hash_map,
            xor_config=xor_config,
            source_layout=source_layout,
            wire_interleaving=wire_interleaving,
            channel_config=case_channel_config,
            hybrid_mode=hybrid_mode,
            parity_filter_fallback=parity_filter_fallback,
        )
        if normalize_decode_latency:
            case = _normalize_case_latency(case)
        case_id = f"case{case_index:04d}"
        result_rows.append(
            _result_row(
                run_id=run_id,
                case_id=case_id,
                case=case,
                model_label=model_label,
                strategy=_strategy(real_llada=real_llada, hybrid_mode=hybrid_mode),
                seed=case_seed,
                loss_rate=loss_rate,
                tokens_per_packet=tokens_per_packet,
                hash_bits=hash_bits,
                xor_overhead_bits_per_token=xor_overhead_bits_per_token,
                vocab_size=vocab_size,
                source_layout=source_layout,
                wire_interleaving=wire_interleaving,
                channel_config=case_channel_config,
                hash_profile_info=hash_profile_info,
            )
        )
        events.append(
            {
                "event_type": "hybrid_xor_hash_micro_eval_case",
                "run_id": run_id,
                "case_id": case_id,
                "model_label": model_label,
                "strategy": _strategy(real_llada=real_llada, hybrid_mode=hybrid_mode),
                "case": case.to_dict(),
            }
        )

    artifact_data = write_run_artifacts(
        output_dir=output_dir,
        manifest=manifest,
        result_rows=result_rows,
        events=events,
        run_timer=run_timer,
    )
    return {
        "run_id": run_id,
        "manifest": artifact_data["manifest"],
        "result_rows": artifact_data["result_rows"],
        "events": events,
    }


def _result_row(
    *,
    run_id: str,
    case_id: str,
    case: HybridRecoveryCase,
    model_label: str,
    strategy: str,
    seed: int,
    loss_rate: float,
    tokens_per_packet: int,
    hash_bits: int,
    xor_overhead_bits_per_token: float,
    vocab_size: int,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    hash_profile_info: dict[str, Any],
) -> dict[str, Any]:
    overhead = case.encoded.overhead
    hash_overhead = _hash_metadata_overhead(
        packets=case.transmitted_packets,
        total_tokens=len(case.sample.token_ids),
        hash_bits=hash_bits,
        vocab_size=vocab_size,
    )
    diagnostics = case.decoding_result.diagnostics
    return {
        "run_id": run_id,
        "case_id": case_id,
        "sample_id": case.sample.sample_id,
        "model_label": model_label,
        "strategy": strategy,
        "baseline_family": "hybrid_xor_hash",
        "protection_mode": HYBRID_PROTECTION_MODE,
        "oracle_hash_metadata": False,
        "hash_bits": hash_bits,
        "hybrid_mode": case.hybrid_mode,
        "xor_overhead_bits_per_token": xor_overhead_bits_per_token,
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
        "known_count": case.reconstruction_plan.known_count,
        "missing_count": case.reconstruction_plan.missing_count,
        "hash_guided_count": case.reconstruction_plan.hash_guided_count,
        "unguided_count": case.reconstruction_plan.unguided_count,
        "received_packet_count": len(case.loss_result.received),
        "dropped_packet_count": len(case.loss_result.dropped),
        "source_packet_count": case.encoded.source_packet_count,
        "extra_packet_count": case.encoded.extra_packet_count,
        "repair_packet_count": overhead.repair_packet_count,
        "repair_token_budget": overhead.repair_token_budget,
        "target_overhead_ratio": overhead.target_overhead_ratio,
        "xor_target_overhead_ratio": overhead.target_overhead_ratio,
        "actual_repair_token_overhead_ratio": overhead.actual_repair_token_overhead_ratio,
        "token_bit_width": hash_overhead["token_bit_width"],
        "hash_metadata_count": hash_overhead["hash_metadata_count"],
        "hash_metadata_bit_count": hash_overhead["hash_metadata_bit_count"],
        "hash_metadata_token_equivalent": hash_overhead["hash_metadata_token_equivalent"],
        "hash_metadata_token_equivalent_overhead_ratio": hash_overhead[
            "hash_metadata_token_equivalent_overhead_ratio"
        ],
        "total_overhead_ratio": (
            hash_overhead["hash_metadata_token_equivalent_overhead_ratio"]
            + overhead.actual_repair_token_overhead_ratio
        ),
        "hash_profile_source": hash_profile_info.get("source"),
        "editable_update_mode": diagnostics.get("editable_update_mode"),
        "hash_constraint_schedule": diagnostics.get("hash_constraint_schedule"),
        "decode_latency_sec": case.decoding_result.decode_latency_sec,
        "decoder_steps": case.decoding_result.steps,
        "model_forward_calls": diagnostics.get("model_forward_calls"),
        "model_proposal_calls": diagnostics.get("model_proposal_calls"),
        "decoder_proposal_mode": diagnostics.get("decoder_proposal_mode"),
        "proposal_interface_used": diagnostics.get("proposal_interface_used"),
        "parity_equation_count": len(equations_from_parity_packets(case.encoded.parity_packets)),
        "parity_received_equation_count": len(equations_from_parity_packets(case.loss_result.received)),
        "parity_peel_iterations": case.initial_peel.peel_iteration_count,
        "parity_peel_recovered_count": case.initial_peel.recovered_count,
        "parity_hash_conflict_count": case.initial_peel.conflict_count,
        "iterative_peel_enabled": diagnostics.get("iterative_peel_enabled", False),
        "iterative_peel_passes": diagnostics.get("iterative_peel_passes", 0),
        "iterative_peel_recovered_count": diagnostics.get("iterative_peel_recovered_count", 0),
        "iterative_peel_hash_conflict_count": diagnostics.get(
            "iterative_peel_hash_conflict_count",
            0,
        ),
        "iterative_peel_special_token_conflict_count": diagnostics.get(
            "iterative_peel_special_token_conflict_count",
            0,
        ),
        "iterative_peel_vocab_conflict_count": diagnostics.get(
            "iterative_peel_vocab_conflict_count",
            0,
        ),
        "iterative_peel_conflict_count": diagnostics.get("iterative_peel_conflict_count", 0),
        "iterative_peel_recovered_positions": _csv_sequence(
            diagnostics.get("iterative_peel_recovered_positions", ())
        ),
        "iterative_peel_recovered_count_by_step": _csv_sequence(
            diagnostics.get("iterative_peel_recovered_count_by_step", ())
        ),
        "parity_candidate_rejections": case.parity_filter_diagnostics.get(
            "parity_candidate_rejections",
            0,
        ),
        "parity_filter_fallback_count": case.parity_filter_diagnostics.get(
            "parity_filter_fallback_count",
            0,
        ),
        "parity_equations_satisfied": case.final_audit.satisfied_count,
        "parity_equations_violated": case.final_audit.violated_count,
        **case.metrics.to_dict(),
        **case.channel_lost_metrics.to_dict(),
    }


def _manifest(
    *,
    run_id: str,
    runner: str,
    model_label: str,
    model_kind: str,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    hash_bits: int,
    xor_overhead_bits_per_token: float,
    vocab_size: int,
    steps: int,
    editable_update_mode: str,
    hash_constraint_schedule: str,
    hybrid_mode: str,
    parity_filter_fallback: bool,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    xor_config: XorParityConfig,
    hash_profile_info: dict[str, Any],
    dataset_info: dict[str, Any] | None,
    real_llada: bool,
    preflight: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "runner": runner,
        "model_label": model_label,
        "model_kind": model_kind,
        "baseline_family": "hybrid_xor_hash",
        "not_a_research_claim": True,
        "not_a_research_claim_warning": MICRO_EVAL_WARNING,
        "micro_eval": True,
        "opt_in_required": real_llada,
        "strategy": _strategy(real_llada=real_llada, hybrid_mode=hybrid_mode),
        "config": {
            "sample_lengths": list(sample_lengths),
            "loss_rate": loss_rate,
            "seed": seed,
            "tokens_per_packet": tokens_per_packet,
            "hash_bits": hash_bits,
            "xor_overhead_bits_per_token": xor_overhead_bits_per_token,
            "vocab_size": vocab_size,
            "steps": steps,
            "editable_update_mode": editable_update_mode,
            "hash_constraint_schedule": hash_constraint_schedule,
            "hybrid_mode": hybrid_mode,
            "parity_filter_fallback": parity_filter_fallback,
            "protection_mode": HYBRID_PROTECTION_MODE,
            "oracle_hash_metadata": False,
            "source_layout": source_layout.to_dict(),
            "wire_interleaving": wire_interleaving.to_dict(),
            "channel": channel_config.to_dict(),
            "xor_parity": xor_config.to_dict(),
            "sample_generation": _sample_generation_info(dataset_info),
        },
        "hash_profile": hash_profile_info,
        "artifacts": {
            "manifest": "run_manifest.json",
            "results": "results.csv",
            "events": "events.jsonl",
        },
    }
    if preflight is not None:
        manifest["preflight"] = dict(preflight)
    return manifest


def _run_id(
    *,
    runner: str,
    model_label: str,
    sample_lengths: Sequence[int],
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    hash_bits: int,
    xor_overhead_bits_per_token: float,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    hybrid_mode: str,
    editable_update_mode: str,
    hash_constraint_schedule: str,
) -> str:
    lengths = "-".join(str(length) for length in sample_lengths)
    source_chunk = source_layout.chunk_size if source_layout.chunk_size is not None else "default"
    model = model_label.replace("/", "-")
    return (
        f"{runner}|{model}|hash{hash_bits}|xorbits{xor_overhead_bits_per_token:g}|"
        f"{hybrid_mode}|loss{loss_rate:g}|lengths{lengths}|tpp{tokens_per_packet}|"
        f"seed{seed}|source-{source_layout.mode}-chunk{source_chunk}|"
        f"wire-{wire_interleaving.mode}-span{wire_interleaving.span}|"
        f"channel-{channel_config.mode}|update-{editable_update_mode}|"
        f"hash-schedule-{hash_constraint_schedule}"
    )


def _strategy(*, real_llada: bool, hybrid_mode: str) -> str:
    prefix = "RealLLaDA" if real_llada else "FakeHybrid"
    return f"{prefix}_HashXOR_{hybrid_mode}"


def _resolve_sample_lengths(
    *,
    sample_lengths: Sequence[int],
    samples: Sequence[TokenSample] | None,
    vocab_size: int,
) -> tuple[int, ...]:
    if samples is None:
        return tuple(sample_lengths)
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
    return tuple(len(sample.token_ids) for sample in samples)


def _validate_common_config(
    *,
    sample_lengths: Sequence[int],
    loss_rate: float,
    tokens_per_packet: int,
    hash_bits: int,
    xor_overhead_bits_per_token: float,
    vocab_size: int,
    steps: int,
    hybrid_mode: str,
) -> None:
    if not sample_lengths:
        raise ValueError("sample_lengths must be non-empty")
    for sample_length in sample_lengths:
        if sample_length <= 0:
            raise ValueError("sample_lengths must contain positive values")
    if loss_rate < 0.0 or loss_rate > 1.0:
        raise ValueError("loss_rate must be between 0.0 and 1.0")
    if tokens_per_packet <= 0:
        raise ValueError("tokens_per_packet must be positive")
    if hash_bits not in {4, 8, 16}:
        raise ValueError("hash_bits must be 4, 8, or 16")
    if xor_overhead_bits_per_token < 0:
        raise ValueError("xor_overhead_bits_per_token must be non-negative")
    if vocab_size <= 16:
        raise ValueError("vocab_size must be greater than 16")
    if steps <= 0:
        raise ValueError("steps must be positive")
    _validate_hybrid_mode(hybrid_mode)


def _validate_hybrid_mode(hybrid_mode: str) -> None:
    if hybrid_mode not in VALID_HYBRID_MODES:
        modes = ", ".join(sorted(VALID_HYBRID_MODES))
        raise ValueError(f"hybrid_mode must be one of: {modes}")


def _validate_iterative_peel_decoder_config(config: DiffusionDecodingConfig) -> None:
    if config.editable_update_mode != EDITABLE_UPDATE_COMMIT_ONCE:
        raise ValueError("hybrid_mode='iterative_peel' requires editable_update_mode='commit_once'")
    if config.hash_constraint_schedule != HASH_CONSTRAINT_ALWAYS:
        raise ValueError("hybrid_mode='iterative_peel' requires hash_constraint_schedule='always'")


def _known_tokens_from_decoder_state(
    *,
    input_ids: Sequence[int],
    plan: ReconstructionPlan,
    prompt_length: int,
    mask_token_id: int,
) -> dict[int, int]:
    return {
        entry.position: int(input_ids[prompt_length + entry.position])
        for entry in plan.entries
        if int(input_ids[prompt_length + entry.position]) != mask_token_id
    }


def _merge_protected_source_and_parity_packets(
    *,
    encoded: XorParityEncoded,
    protected_source_packets: Sequence[Packet],
) -> tuple[Packet, ...]:
    protected_by_index = {
        int(packet.metadata[SOURCE_PACKET_INDEX_METADATA_KEY]): packet
        for packet in protected_source_packets
    }
    packets: list[Packet] = []
    for packet in encoded.packets:
        if packet.kind == "data":
            packets.append(protected_by_index[int(packet.metadata[SOURCE_PACKET_INDEX_METADATA_KEY])])
        else:
            packets.append(packet)
    return tuple(packets)


def _filter_known_position_hash_metadata(
    hash_metadata: dict[int, int],
    *,
    known_tokens: Mapping[int, int],
) -> dict[int, int]:
    return {
        int(position): int(hash_value)
        for position, hash_value in hash_metadata.items()
        if int(position) not in known_tokens
    }


def _build_plan_from_known_and_hash(
    *,
    total_tokens: int,
    known_tokens: Mapping[int, int],
    hash_metadata: Mapping[int, int],
) -> ReconstructionPlan:
    entries: list[ReconstructionEntry] = []
    for position in range(total_tokens):
        if position in known_tokens:
            entries.append(
                ReconstructionEntry(
                    position=position,
                    state=STATE_KNOWN,
                    token_id=int(known_tokens[position]),
                    fixed=True,
                )
            )
        elif position in hash_metadata:
            entries.append(
                ReconstructionEntry(
                    position=position,
                    state=STATE_MISSING,
                    hash_value=int(hash_metadata[position]),
                    fixed=False,
                )
            )
        else:
            entries.append(
                ReconstructionEntry(
                    position=position,
                    state=STATE_UNGUIDED,
                    fixed=False,
                )
            )
    return ReconstructionPlan(entries=tuple(entries), total_tokens=total_tokens)


def _hash_metadata_overhead(
    *,
    packets: Sequence[Packet],
    total_tokens: int,
    hash_bits: int,
    vocab_size: int,
) -> dict[str, Any]:
    metadata_count = sum(_packet_hash_metadata_count(packet) for packet in packets)
    metadata_bits = metadata_count * hash_bits
    return {
        "token_bit_width": token_bit_width_for_vocab(vocab_size),
        "hash_metadata_count": metadata_count,
        "hash_metadata_bit_count": metadata_bits,
        "hash_metadata_token_equivalent": token_equivalent_overhead(
            metadata_bits=metadata_bits,
            vocab_size=vocab_size,
        ),
        "hash_metadata_token_equivalent_overhead_ratio": metadata_token_equivalent_overhead_ratio(
            metadata_bits=metadata_bits,
            total_tokens=total_tokens,
            vocab_size=vocab_size,
        ),
    }


def _packet_hash_metadata_count(packet: Packet) -> int:
    raw_metadata = packet.metadata.get(LOOKBACK_HASH_METADATA_KEY)
    if raw_metadata is None:
        return 0
    if isinstance(raw_metadata, dict):
        return len(raw_metadata.get("hashes", ()))
    return len(tuple(getattr(raw_metadata, "hashes", ())))


def _decoding_result_with_hybrid_diagnostics(
    *,
    decoding_result: DecodingResult,
    hybrid_mode: str,
    initial_peel: XorPeelResult,
    final_audit: XorAuditResult,
    parity_filter_diagnostics: dict[str, Any],
    iterative_peel_diagnostics: dict[str, Any],
    parity_equation_count: int,
    parity_received_equation_count: int,
) -> DecodingResult:
    diagnostics = dict(decoding_result.diagnostics)
    diagnostics.update(
        {
            "hybrid_mode": hybrid_mode,
            "parity_equation_count": parity_equation_count,
            "parity_received_equation_count": parity_received_equation_count,
            "parity_peel_iterations": initial_peel.peel_iteration_count,
            "parity_peel_recovered_count": initial_peel.recovered_count,
            "parity_hash_conflict_count": initial_peel.conflict_count,
            "parity_equations_satisfied": final_audit.satisfied_count,
            "parity_equations_violated": final_audit.violated_count,
            **parity_filter_diagnostics,
            **iterative_peel_diagnostics,
        }
    )
    return DecodingResult(
        reconstructed_text=decoding_result.reconstructed_text,
        reconstructed_tokens=decoding_result.reconstructed_tokens,
        decode_latency_sec=decoding_result.decode_latency_sec,
        steps=decoding_result.steps,
        fixed_token_count=decoding_result.fixed_token_count,
        editable_token_count=decoding_result.editable_token_count,
        hash_guided_token_count=decoding_result.hash_guided_token_count,
        confidence_stats=decoding_result.confidence_stats,
        step_summaries=decoding_result.step_summaries,
        diagnostics=diagnostics,
    )


def _normalize_case_latency(case: HybridRecoveryCase) -> HybridRecoveryCase:
    decoding_result = DecodingResult(
        reconstructed_text=case.decoding_result.reconstructed_text,
        reconstructed_tokens=case.decoding_result.reconstructed_tokens,
        decode_latency_sec=0.0,
        steps=case.decoding_result.steps,
        fixed_token_count=case.decoding_result.fixed_token_count,
        editable_token_count=case.decoding_result.editable_token_count,
        hash_guided_token_count=case.decoding_result.hash_guided_token_count,
        confidence_stats=case.decoding_result.confidence_stats,
        step_summaries=case.decoding_result.step_summaries,
        diagnostics=case.decoding_result.diagnostics,
    )
    return replace(case, decoding_result=decoding_result)


def _empty_filter_diagnostics() -> dict[str, Any]:
    return {
        "parity_candidate_filter_calls": 0,
        "parity_candidate_rejections": 0,
        "parity_filter_fallback_count": 0,
        "parity_filter_fallback_enabled": False,
    }


def _empty_iterative_peel_diagnostics() -> dict[str, Any]:
    return {
        "iterative_peel_enabled": False,
        "iterative_peel_passes": 0,
        "iterative_peel_recovered_count": 0,
        "iterative_peel_recovered_positions": (),
        "iterative_peel_recovered_count_by_step": (),
        "iterative_peel_conflict_count": 0,
        "iterative_peel_hash_conflict_count": 0,
        "iterative_peel_special_token_conflict_count": 0,
        "iterative_peel_vocab_conflict_count": 0,
        "iterative_peel_conflicts": (),
    }


def _csv_sequence(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(list(value), separators=(",", ":"))


def _sample_generation_info(dataset_info: dict[str, Any] | None) -> dict[str, Any]:
    if dataset_info:
        return {
            "type": "provided_token_samples",
            "dataset": dict(dataset_info),
        }
    return {
        "type": "deterministic_synthetic_token_ids",
    }
