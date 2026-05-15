"""Hybrid LLaDA/hash/XOR validation runner."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from diffusion_fec.baselines.overhead import (
    metadata_token_equivalent_overhead_ratio,
    token_bit_width_for_vocab,
    token_equivalent_overhead,
)
from diffusion_fec.baselines.xor_equations import (
    ParityCandidateFilter,
    XorAuditResult,
    XorPeelConflict,
    XorPeelResult,
    audit_xor_equations,
    equations_from_parity_packets,
    equations_from_sparse_fountain_packets,
    known_tokens_from_data_packets,
    peel_xor_equations,
    solve_xor_equations,
)
from diffusion_fec.baselines.sparse_fountain_xor import (
    SPARSE_FOUNTAIN_XOR_SCHEME,
    SparseFountainXorConfig,
    SparseFountainXorEncoded,
    encode_sparse_fountain_xor,
    parse_degree_distribution,
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
    resolve_packet_loss_channel_config,
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
from diffusion_fec.metrics.loss_metrics import compute_packet_loss_diagnostics
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
HYBRID_MODE_ITERATIVE_ROLLBACK = "iterative_rollback"
VALID_HYBRID_MODES = frozenset(
    {
        HYBRID_MODE_PRE_PEEL_ONLY,
        HYBRID_MODE_PARITY_FILTER,
        HYBRID_MODE_ITERATIVE_PEEL,
        HYBRID_MODE_ITERATIVE_ROLLBACK,
    }
)
HYBRID_PROTECTION_MODE = "lookback_1+xor_parity"
FAKE_HYBRID_MODEL_LABEL = "FakeDeterministicHybridXorHashModel"
REAL_HYBRID_MODEL_KIND = "real_llada_huggingface_hybrid_xor_hash"
FAKE_HYBRID_MODEL_KIND = "fake_deterministic_hybrid_xor_hash_model"
DEFAULT_XOR_OVERHEAD_BITS_PER_TOKEN = 4.0
XOR_CODE_STRIPE = "stripe"
XOR_CODE_SPARSE_FOUNTAIN = "sparse_fountain"
VALID_XOR_CODES = frozenset({XOR_CODE_STRIPE, XOR_CODE_SPARSE_FOUNTAIN})
DEFAULT_SPARSE_XOR_MAX_COMPONENT_UNKNOWNS = 8
DEFAULT_ROLLBACK_EXTRA_STEPS = 4
DEFAULT_ROLLBACK_MAX_TOTAL_STEPS = 16
DEFAULT_ROLLBACK_MAX_PER_POSITION = 3
DEFAULT_ROLLBACK_STOP_AFTER_NO_PROGRESS = 2


@dataclass(frozen=True)
class HybridRecoveryCase:
    """Artifacts for one hybrid recovery case."""

    sample: TokenSample
    encoded: XorParityEncoded | SparseFountainXorEncoded
    transmitted_packets: tuple[Packet, ...]
    loss_result: RandomLossResult
    hash_metadata: dict[int, int]
    initial_peel: XorPeelResult
    final_audit: XorAuditResult
    parity_filter_diagnostics: dict[str, Any]
    channel_config: PacketLossChannelConfig
    hybrid_mode: str
    xor_code: str
    sparse_diagnostics: dict[str, Any]
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

    @property
    def loss_diagnostics(self) -> dict[str, Any]:
        return compute_packet_loss_diagnostics(
            loss_result=self.loss_result,
            source_token_count=len(self.sample.token_ids),
            channel_lost_position_count=self.channel_lost_metrics.channel_lost_position_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "encoded": self.encoded.to_dict(),
            "transmitted_packets": [packet.to_dict() for packet in self.transmitted_packets],
            "loss_result": self.loss_result.to_dict(),
            "loss_diagnostics": self.loss_diagnostics,
            "hash_metadata": dict(self.hash_metadata),
            "initial_peel": self.initial_peel.to_dict(),
            "final_audit": self.final_audit.to_dict(),
            "parity_filter_diagnostics": dict(self.parity_filter_diagnostics),
            "channel_config": self.channel_config.to_dict(),
            "hybrid_mode": self.hybrid_mode,
            "xor_code": self.xor_code,
            "sparse_diagnostics": dict(self.sparse_diagnostics),
            "reconstruction_plan": self.reconstruction_plan.to_dict(),
            "decoding_result": self.decoding_result.to_dict(),
            "channel_lost_positions": list(self.channel_lost_positions),
            "channel_lost_metrics": self.channel_lost_metrics.to_dict(),
            "metrics": {**self.metrics.to_dict(), **self.channel_lost_metrics.to_dict()},
        }


@dataclass(frozen=True)
class _HybridTokenProvenance:
    source: str
    token_id: int
    step: int | None = None
    dependency_positions: tuple[int, ...] = ()
    equation_ids: tuple[str, ...] = ()
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "token_id": self.token_id,
            "step": self.step,
            "dependency_positions": list(self.dependency_positions),
            "equation_ids": list(self.equation_ids),
            "confidence": self.confidence,
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
        enable_linear_solve: bool = False,
        max_component_unknowns: int = DEFAULT_SPARSE_XOR_MAX_COMPONENT_UNKNOWNS,
    ) -> None:
        self.equations = tuple(equations)
        self.hash_metadata = {int(position): int(value) for position, value in hash_metadata.items()}
        self.token_hash_map = token_hash_map
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)
        self.banned_token_ids = frozenset(int(token_id) for token_id in banned_token_ids)
        self.enable_linear_solve = bool(enable_linear_solve)
        self.max_component_unknowns = int(max_component_unknowns)
        self.call_count = 0
        self.per_step_recovered_count: list[int] = []
        self.recovered_tokens: dict[int, int] = {}
        self.conflicts = {}
        self.linear_solver_diagnostics = _empty_linear_solver_diagnostics(
            enabled=self.enable_linear_solve
        )
        self.last_step_diagnostics: dict[str, Any] = {}

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
        solve_start = perf_counter()
        peel = solve_xor_equations(
            equations=self.equations,
            known_tokens=known_tokens,
            hash_metadata=self.hash_metadata,
            token_hash_map=self.token_hash_map,
            vocab_size=self.vocab_size,
            banned_token_ids=self.banned_token_ids,
            enable_linear_solve=self.enable_linear_solve,
            max_component_unknowns=self.max_component_unknowns,
        )
        solve_time_sec = perf_counter() - solve_start
        self._record_conflicts(peel)
        self._accumulate_linear_diagnostics(peel.linear_solver_diagnostics)
        newly_fixed = {
            int(position): int(token_id)
            for position, token_id in peel.recovered_tokens.items()
            if int(position) not in self.recovered_tokens
            and input_ids[prompt_length + int(position)] == self.mask_token_id
        }
        self.recovered_tokens.update(newly_fixed)
        self.per_step_recovered_count.append(len(newly_fixed))
        self.last_step_diagnostics = _hook_step_diagnostics(
            equations=self.equations,
            peel=peel,
            newly_fixed_count=len(newly_fixed),
            solve_time_sec=solve_time_sec,
            rollback_count=0,
            rollback_time_sec=0.0,
        )
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
            "iterative_linear_solver_enabled": self.enable_linear_solve,
            "iterative_linear_solver_components_seen": self.linear_solver_diagnostics[
                "linear_solver_components_seen"
            ],
            "iterative_linear_solver_components_solved": self.linear_solver_diagnostics[
                "linear_solver_components_solved"
            ],
            "iterative_linear_solver_tokens_recovered": self.linear_solver_diagnostics[
                "linear_solver_tokens_recovered"
            ],
            "iterative_linear_solver_rank_deficient_count": self.linear_solver_diagnostics[
                "linear_solver_rank_deficient_count"
            ],
            "iterative_linear_solver_validation_conflict_count": self.linear_solver_diagnostics[
                "linear_solver_validation_conflict_count"
            ],
            "iterative_linear_solver_too_large_count": self.linear_solver_diagnostics[
                "linear_solver_too_large_count"
            ],
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

    def _accumulate_linear_diagnostics(self, diagnostics: Mapping[str, Any]) -> None:
        for key in self.linear_solver_diagnostics:
            if key == "linear_solver_enabled":
                self.linear_solver_diagnostics[key] = self.enable_linear_solve
            elif key.endswith("_time_sec"):
                self.linear_solver_diagnostics[key] += float(diagnostics.get(key, 0.0))
            else:
                self.linear_solver_diagnostics[key] += int(diagnostics.get(key, 0))


class IterativeRollbackXorHook:
    """Sparse hybrid hook that remasks model commits when parity/hash conflicts identify them."""

    rollback_enabled = True

    def __init__(
        self,
        *,
        equations,
        hash_metadata: Mapping[int, int],
        token_hash_map: TokenHashMap,
        mask_token_id: int,
        vocab_size: int,
        banned_token_ids: Sequence[int],
        enable_linear_solve: bool = True,
        max_component_unknowns: int = DEFAULT_SPARSE_XOR_MAX_COMPONENT_UNKNOWNS,
        rollback_extra_steps: int = DEFAULT_ROLLBACK_EXTRA_STEPS,
        rollback_max_total_steps: int = DEFAULT_ROLLBACK_MAX_TOTAL_STEPS,
        rollback_max_per_position: int = DEFAULT_ROLLBACK_MAX_PER_POSITION,
        rollback_stop_after_no_progress: int = DEFAULT_ROLLBACK_STOP_AFTER_NO_PROGRESS,
        rollback_continue_until_stable: bool = False,
        rollback_require_zero_masks: bool = False,
        rollback_require_final_parity_clean: bool = False,
    ) -> None:
        self.equations = tuple(equations)
        self.hash_metadata = {int(position): int(value) for position, value in hash_metadata.items()}
        self.token_hash_map = token_hash_map
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)
        self.banned_token_ids = frozenset(int(token_id) for token_id in banned_token_ids)
        self.enable_linear_solve = bool(enable_linear_solve)
        self.max_component_unknowns = int(max_component_unknowns)
        self.rollback_extra_steps = int(rollback_extra_steps)
        self.rollback_max_total_steps = int(rollback_max_total_steps)
        self.rollback_max_per_position = int(rollback_max_per_position)
        self.rollback_stop_after_no_progress = int(rollback_stop_after_no_progress)
        self.rollback_continue_until_stable = bool(rollback_continue_until_stable)
        self.rollback_require_zero_masks = bool(rollback_require_zero_masks)
        self.rollback_require_final_parity_clean = bool(rollback_require_final_parity_clean)
        if self.rollback_extra_steps < 0:
            raise ValueError("rollback_extra_steps must be non-negative")
        if self.rollback_max_total_steps <= 0:
            raise ValueError("rollback_max_total_steps must be positive")
        if self.rollback_max_per_position <= 0:
            raise ValueError("rollback_max_per_position must be positive")
        if self.rollback_stop_after_no_progress < 0:
            raise ValueError("rollback_stop_after_no_progress must be non-negative")

        self.call_count = 0
        self.per_step_recovered_count: list[int] = []
        self.recovered_tokens: dict[int, int] = {}
        self.conflicts: dict[tuple[Any, ...], XorPeelConflict] = {}
        self.linear_solver_diagnostics = _empty_linear_solver_diagnostics(
            enabled=self.enable_linear_solve
        )
        self.provenance: dict[int, _HybridTokenProvenance] = {}
        self.rollback_counts: dict[int, int] = {}
        self.rollback_positions: set[int] = set()
        self.rollback_events: list[dict[str, Any]] = []
        self.rollback_bans: dict[int, set[int]] = {}
        self.single_suspect_count = 0
        self.multi_suspect_count = 0
        self.max_per_position_hits = 0
        self.provenance_invalidated_count = 0
        self.decode_extra_steps_used = 0
        self.decode_total_steps_used = 0
        self.decode_base_steps = 0
        self.decode_stopped_reason = ""
        self.decode_final_zero_masks = False
        self.decode_final_parity_clean: bool | None = None
        self.decode_remaining_masks_after_budget = 0
        self.decode_no_progress_stop = False
        self.rollback_time_sec = 0.0
        self.last_step_diagnostics: dict[str, Any] = {}

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
        self._ensure_fixed_root_provenance(
            input_ids=input_ids,
            plan=plan,
            prompt_length=prompt_length,
        )
        self._record_model_commits(
            input_ids=input_ids,
            prompt_length=prompt_length,
            committed_positions=committed_positions,
            step=step,
        )
        known_tokens = _known_tokens_from_decoder_state(
            input_ids=input_ids,
            plan=plan,
            prompt_length=prompt_length,
            mask_token_id=self.mask_token_id,
        )
        solve_start = perf_counter()
        peel = solve_xor_equations(
            equations=self.equations,
            known_tokens=known_tokens,
            hash_metadata=self.hash_metadata,
            token_hash_map=self.token_hash_map,
            vocab_size=self.vocab_size,
            banned_token_ids=self.banned_token_ids,
            enable_linear_solve=self.enable_linear_solve,
            max_component_unknowns=self.max_component_unknowns,
        )
        solve_time_sec = perf_counter() - solve_start
        self._record_conflicts(peel)
        self._accumulate_linear_diagnostics(peel.linear_solver_diagnostics)

        conflict_candidates = list(peel.conflicts)
        conflict_candidates.extend(
            self._audit_conflicts(token_by_position=known_tokens)
        )
        rollback_start = perf_counter()
        rollback_positions, position_bans = self._rollback_requests(
            conflicts=conflict_candidates,
            input_ids=input_ids,
            prompt_length=prompt_length,
            step=step,
        )
        invalidated = self._invalidate_dependent_parity_tokens(rollback_positions)
        rollback_positions = tuple(sorted(set(rollback_positions) | set(invalidated)))
        for position in rollback_positions:
            self.provenance.pop(position, None)
            self.recovered_tokens.pop(position, None)
        step_rollback_time = perf_counter() - rollback_start
        self.rollback_time_sec += step_rollback_time
        newly_fixed = self._accepted_new_fixed_tokens(
            peel=peel,
            input_ids=input_ids,
            prompt_length=prompt_length,
            rollback_positions=set(rollback_positions),
        )
        self.recovered_tokens.update(newly_fixed)
        self.per_step_recovered_count.append(len(newly_fixed))
        self.last_step_diagnostics = _hook_step_diagnostics(
            equations=self.equations,
            peel=peel,
            newly_fixed_count=len(newly_fixed),
            solve_time_sec=solve_time_sec,
            rollback_count=len(rollback_positions),
            rollback_time_sec=step_rollback_time,
        )
        return {
            "fixed_tokens": newly_fixed,
            "rollback_positions": rollback_positions,
            "position_banned_tokens": position_bans,
        }

    def record_decode_outcome(
        self,
        *,
        extra_steps_used: int,
        remaining_masked_count: int,
        no_progress_stop: bool,
        total_steps_used: int | None = None,
        base_steps: int | None = None,
        stopped_reason: str | None = None,
        final_zero_masks: bool | None = None,
        final_parity_clean: bool | None = None,
    ) -> None:
        self.decode_extra_steps_used = int(extra_steps_used)
        self.decode_total_steps_used = int(
            total_steps_used if total_steps_used is not None else extra_steps_used
        )
        self.decode_base_steps = int(base_steps if base_steps is not None else 0)
        self.decode_stopped_reason = str(stopped_reason or "")
        self.decode_final_zero_masks = (
            bool(final_zero_masks) if final_zero_masks is not None else False
        )
        self.decode_final_parity_clean = final_parity_clean
        self.decode_remaining_masks_after_budget = int(remaining_masked_count)
        self.decode_no_progress_stop = bool(no_progress_stop)

    def is_final_parity_clean(
        self,
        *,
        input_ids,
        plan: ReconstructionPlan,
        prompt_length: int,
        config: DiffusionDecodingConfig,
    ) -> bool:
        known_tokens = _known_tokens_from_decoder_state(
            input_ids=input_ids,
            plan=plan,
            prompt_length=prompt_length,
            mask_token_id=config.mask_token_id,
        )
        if len(known_tokens) < plan.total_tokens:
            return False
        audit = audit_xor_equations(
            equations=self.equations,
            token_by_position=known_tokens,
        )
        return audit.violated_count == 0

    def diagnostics(self) -> dict[str, Any]:
        conflicts = tuple(self.conflicts.values())
        return {
            **_iterative_peel_style_diagnostics(
                enabled=True,
                call_count=self.call_count,
                recovered_tokens=self.recovered_tokens,
                per_step_recovered_count=self.per_step_recovered_count,
                conflicts=conflicts,
                enable_linear_solve=self.enable_linear_solve,
                linear_solver_diagnostics=self.linear_solver_diagnostics,
            ),
            "rollback_enabled": True,
            "rollback_event_count": len(self.rollback_events),
            "rollback_conflict_equation_count": len(
                {event["equation_id"] for event in self.rollback_events}
            ),
            "rollback_positions_count": len(self.rollback_positions),
            "rollback_positions": tuple(sorted(self.rollback_positions)),
            "rollback_single_suspect_count": self.single_suspect_count,
            "rollback_multi_suspect_count": self.multi_suspect_count,
            "rollback_banned_token_count": sum(len(tokens) for tokens in self.rollback_bans.values()),
            "rollback_banned_tokens_by_position": {
                position: tuple(sorted(tokens))
                for position, tokens in sorted(self.rollback_bans.items())
            },
            "rollback_max_per_position_hits": self.max_per_position_hits,
            "rollback_adaptive_enabled": (
                self.rollback_continue_until_stable
                or self.rollback_require_zero_masks
                or self.rollback_require_final_parity_clean
            ),
            "rollback_total_steps_used": self.decode_total_steps_used,
            "rollback_base_steps": self.decode_base_steps,
            "rollback_extra_steps_used": self.decode_extra_steps_used,
            "rollback_stopped_reason": self.decode_stopped_reason,
            "rollback_final_zero_masks": self.decode_final_zero_masks,
            "rollback_final_parity_clean": self.decode_final_parity_clean,
            "rollback_remaining_masks_after_budget": self.decode_remaining_masks_after_budget,
            "rollback_provenance_invalidated_count": self.provenance_invalidated_count,
            "rollback_no_progress_stop": self.decode_no_progress_stop,
            "rollback_time_sec": self.rollback_time_sec,
            "rollback_events": tuple(self.rollback_events),
        }

    def _ensure_fixed_root_provenance(
        self,
        *,
        input_ids,
        plan: ReconstructionPlan,
        prompt_length: int,
    ) -> None:
        for entry in plan.entries:
            token_id = int(input_ids[prompt_length + entry.position])
            if entry.fixed and token_id != self.mask_token_id and entry.position not in self.provenance:
                self.provenance[entry.position] = _HybridTokenProvenance(
                    source="received_or_initial_parity",
                    token_id=token_id,
                    step=None,
                )

    def _record_model_commits(
        self,
        *,
        input_ids,
        prompt_length: int,
        committed_positions,
        step: int,
    ) -> None:
        for raw_position in committed_positions:
            position = int(raw_position)
            token_id = int(input_ids[prompt_length + position])
            if token_id == self.mask_token_id:
                continue
            current = self.provenance.get(position)
            if current is not None and current.source != "model_commit":
                continue
            self.provenance[position] = _HybridTokenProvenance(
                source="model_commit",
                token_id=token_id,
                step=step,
                confidence=None,
            )

    def _accepted_new_fixed_tokens(
        self,
        *,
        peel: XorPeelResult,
        input_ids,
        prompt_length: int,
        rollback_positions: set[int],
    ) -> dict[int, int]:
        newly_fixed: dict[int, int] = {}
        for position, token_id in peel.recovered_tokens.items():
            position = int(position)
            if position in self.recovered_tokens or position in rollback_positions:
                continue
            if int(input_ids[prompt_length + position]) != self.mask_token_id:
                continue
            provenance = peel.recovery_provenance.get(position)
            if provenance is None:
                continue
            dependencies = tuple(int(dep) for dep in provenance.dependency_positions)
            if any(dep in rollback_positions for dep in dependencies):
                continue
            newly_fixed[position] = int(token_id)
            self.provenance[position] = _HybridTokenProvenance(
                source="parity_solve",
                token_id=int(token_id),
                step=None,
                dependency_positions=dependencies,
                equation_ids=tuple(provenance.equation_ids),
            )
        return newly_fixed

    def _audit_conflicts(self, *, token_by_position: Mapping[int, int]) -> tuple[XorPeelConflict, ...]:
        audit = audit_xor_equations(equations=self.equations, token_by_position=token_by_position)
        conflicts: list[XorPeelConflict] = []
        for violation in audit.violations:
            positions = tuple(int(position) for position in violation["positions"])
            conflicts.append(
                XorPeelConflict(
                    equation_id=str(violation["equation_id"]),
                    position=None,
                    solved_token_id=None,
                    reason="parity_equation_violation",
                    equation_positions=positions,
                    dependency_positions=positions,
                )
            )
        if conflicts:
            self._record_conflicts_from_sequence(conflicts)
        return tuple(conflicts)

    def _rollback_requests(
        self,
        *,
        conflicts: Sequence[XorPeelConflict],
        input_ids,
        prompt_length: int,
        step: int,
    ) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
        rollback_positions: set[int] = set()
        position_bans: dict[int, set[int]] = {}
        for conflict in conflicts:
            suspect_positions = conflict.dependency_positions or conflict.equation_positions
            soft_suspects = sorted(self._soft_model_dependencies(suspect_positions))
            if not soft_suspects:
                continue
            if len(soft_suspects) == 1:
                self.single_suspect_count += 1
                position = soft_suspects[0]
                if self._can_rollback(position):
                    rollback_positions.add(position)
                    token_id = int(input_ids[prompt_length + position])
                    if token_id != self.mask_token_id:
                        position_bans.setdefault(position, set()).add(token_id)
                        self.rollback_bans.setdefault(position, set()).add(token_id)
                self._record_rollback_event(
                    conflict=conflict,
                    suspects=(position,),
                    step=step,
                    rollback_kind="single_suspect",
                )
            else:
                self.multi_suspect_count += 1
                accepted = tuple(position for position in soft_suspects if self._can_rollback(position))
                rollback_positions.update(accepted)
                self._record_rollback_event(
                    conflict=conflict,
                    suspects=accepted,
                    step=step,
                    rollback_kind="multi_suspect",
                )
        for position in rollback_positions:
            self.rollback_counts[position] = self.rollback_counts.get(position, 0) + 1
            self.rollback_positions.add(position)
        return (
            tuple(sorted(rollback_positions)),
            {position: tuple(sorted(tokens)) for position, tokens in sorted(position_bans.items())},
        )

    def _soft_model_dependencies(self, positions: Sequence[int]) -> set[int]:
        soft: set[int] = set()
        seen: set[int] = set()

        def visit(position: int) -> None:
            if position in seen:
                return
            seen.add(position)
            provenance = self.provenance.get(position)
            if provenance is None:
                return
            if provenance.source == "model_commit":
                soft.add(position)
                return
            if provenance.source == "parity_solve":
                for dependency in provenance.dependency_positions:
                    visit(dependency)

        for position in positions:
            visit(int(position))
        return soft

    def _can_rollback(self, position: int) -> bool:
        if self.rollback_counts.get(position, 0) >= self.rollback_max_per_position:
            self.max_per_position_hits += 1
            return False
        return True

    def _invalidate_dependent_parity_tokens(self, rollback_positions: Sequence[int]) -> tuple[int, ...]:
        invalidated: set[int] = set()
        queue = list(int(position) for position in rollback_positions)
        while queue:
            position = queue.pop()
            for candidate, provenance in list(self.provenance.items()):
                if candidate in invalidated or provenance.source != "parity_solve":
                    continue
                if position in provenance.dependency_positions:
                    invalidated.add(candidate)
                    queue.append(candidate)
        for position in invalidated:
            self.provenance.pop(position, None)
            self.recovered_tokens.pop(position, None)
        self.provenance_invalidated_count += len(invalidated)
        self.rollback_positions.update(invalidated)
        return tuple(sorted(invalidated))

    def _record_rollback_event(
        self,
        *,
        conflict: XorPeelConflict,
        suspects: Sequence[int],
        step: int,
        rollback_kind: str,
    ) -> None:
        self.rollback_events.append(
            {
                "step": step,
                "equation_id": conflict.equation_id,
                "reason": conflict.reason,
                "rollback_kind": rollback_kind,
                "suspect_positions": list(suspects),
                "conflict_position": conflict.position,
                "equation_positions": list(conflict.equation_positions),
                "dependency_positions": list(conflict.dependency_positions),
            }
        )

    def _record_conflicts(self, peel: XorPeelResult) -> None:
        self._record_conflicts_from_sequence(peel.conflicts)

    def _record_conflicts_from_sequence(self, conflicts: Sequence[XorPeelConflict]) -> None:
        for conflict in conflicts:
            key = (
                conflict.equation_id,
                conflict.position,
                conflict.solved_token_id,
                conflict.reason,
                conflict.dependency_positions,
            )
            self.conflicts[key] = conflict

    def _accumulate_linear_diagnostics(self, diagnostics: Mapping[str, Any]) -> None:
        for key in self.linear_solver_diagnostics:
            if key == "linear_solver_enabled":
                self.linear_solver_diagnostics[key] = self.enable_linear_solve
            elif key.endswith("_time_sec"):
                self.linear_solver_diagnostics[key] += float(diagnostics.get(key, 0.0))
            else:
                self.linear_solver_diagnostics[key] += int(diagnostics.get(key, 0))


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
    xor_code: str = XOR_CODE_STRIPE,
    sparse_xor_seed: int = 7,
    sparse_xor_coverage: bool = True,
    sparse_xor_degree_distribution: str | Sequence[tuple[int, float]] = "2:0.5,3:0.35,4:0.15",
    sparse_xor_max_coverage_degree: int = 8,
    sparse_xor_max_component_unknowns: int = DEFAULT_SPARSE_XOR_MAX_COMPONENT_UNKNOWNS,
    sparse_xor_enable_linear_solve: bool = True,
    rollback_extra_steps: int = DEFAULT_ROLLBACK_EXTRA_STEPS,
    rollback_max_total_steps: int = DEFAULT_ROLLBACK_MAX_TOTAL_STEPS,
    rollback_max_per_position: int = DEFAULT_ROLLBACK_MAX_PER_POSITION,
    rollback_stop_after_no_progress: int = DEFAULT_ROLLBACK_STOP_AFTER_NO_PROGRESS,
    rollback_continue_until_stable: bool = False,
    rollback_require_zero_masks: bool = False,
    rollback_require_final_parity_clean: bool = False,
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
        xor_code=xor_code,
        sparse_xor_seed=sparse_xor_seed,
        sparse_xor_coverage=sparse_xor_coverage,
        sparse_xor_degree_distribution=sparse_xor_degree_distribution,
        sparse_xor_max_coverage_degree=sparse_xor_max_coverage_degree,
        sparse_xor_max_component_unknowns=sparse_xor_max_component_unknowns,
        sparse_xor_enable_linear_solve=sparse_xor_enable_linear_solve,
        rollback_extra_steps=rollback_extra_steps,
        rollback_max_total_steps=rollback_max_total_steps,
        rollback_max_per_position=rollback_max_per_position,
        rollback_stop_after_no_progress=rollback_stop_after_no_progress,
        rollback_continue_until_stable=rollback_continue_until_stable,
        rollback_require_zero_masks=rollback_require_zero_masks,
        rollback_require_final_parity_clean=rollback_require_final_parity_clean,
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
    xor_code: str = XOR_CODE_STRIPE,
    sparse_xor_seed: int = 7,
    sparse_xor_coverage: bool = True,
    sparse_xor_degree_distribution: str | Sequence[tuple[int, float]] = "2:0.5,3:0.35,4:0.15",
    sparse_xor_max_coverage_degree: int = 8,
    sparse_xor_max_component_unknowns: int = DEFAULT_SPARSE_XOR_MAX_COMPONENT_UNKNOWNS,
    sparse_xor_enable_linear_solve: bool = True,
    rollback_extra_steps: int = DEFAULT_ROLLBACK_EXTRA_STEPS,
    rollback_max_total_steps: int = DEFAULT_ROLLBACK_MAX_TOTAL_STEPS,
    rollback_max_per_position: int = DEFAULT_ROLLBACK_MAX_PER_POSITION,
    rollback_stop_after_no_progress: int = DEFAULT_ROLLBACK_STOP_AFTER_NO_PROGRESS,
    rollback_continue_until_stable: bool = False,
    rollback_require_zero_masks: bool = False,
    rollback_require_final_parity_clean: bool = False,
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
        xor_code=xor_code,
        sparse_xor_seed=sparse_xor_seed,
        sparse_xor_coverage=sparse_xor_coverage,
        sparse_xor_degree_distribution=sparse_xor_degree_distribution,
        sparse_xor_max_coverage_degree=sparse_xor_max_coverage_degree,
        sparse_xor_max_component_unknowns=sparse_xor_max_component_unknowns,
        sparse_xor_enable_linear_solve=sparse_xor_enable_linear_solve,
        rollback_extra_steps=rollback_extra_steps,
        rollback_max_total_steps=rollback_max_total_steps,
        rollback_max_per_position=rollback_max_per_position,
        rollback_stop_after_no_progress=rollback_stop_after_no_progress,
        rollback_continue_until_stable=rollback_continue_until_stable,
        rollback_require_zero_masks=rollback_require_zero_masks,
        rollback_require_final_parity_clean=rollback_require_final_parity_clean,
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
    xor_code: str = XOR_CODE_STRIPE,
    sparse_config: SparseFountainXorConfig | None = None,
    sparse_linear_solve_enabled: bool = True,
    sparse_max_component_unknowns: int = DEFAULT_SPARSE_XOR_MAX_COMPONENT_UNKNOWNS,
    rollback_extra_steps: int = DEFAULT_ROLLBACK_EXTRA_STEPS,
    rollback_max_total_steps: int = DEFAULT_ROLLBACK_MAX_TOTAL_STEPS,
    rollback_max_per_position: int = DEFAULT_ROLLBACK_MAX_PER_POSITION,
    rollback_stop_after_no_progress: int = DEFAULT_ROLLBACK_STOP_AFTER_NO_PROGRESS,
    rollback_continue_until_stable: bool = False,
    rollback_require_zero_masks: bool = False,
    rollback_require_final_parity_clean: bool = False,
) -> HybridRecoveryCase:
    """Run one hybrid packet-loss and LLaDA recovery case."""

    _validate_hybrid_mode(hybrid_mode)
    _validate_xor_code(xor_code)
    if hybrid_mode in {HYBRID_MODE_ITERATIVE_PEEL, HYBRID_MODE_ITERATIVE_ROLLBACK}:
        _validate_iterative_peel_decoder_config(config)
    if hybrid_mode == HYBRID_MODE_ITERATIVE_ROLLBACK:
        _validate_iterative_rollback_config(xor_code=xor_code)
    encoded = _encode_repair(
        sample=sample,
        tokens_per_packet=tokens_per_packet,
        xor_code=xor_code,
        xor_config=xor_config,
        sparse_config=sparse_config,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
    )
    protected_source_packets = attach_lookback_hashes(encoded.source_packets, token_hash_map)
    transmitted_packets = _merge_protected_source_and_parity_packets(
        encoded=encoded,
        protected_source_packets=protected_source_packets,
    )
    channel_config = resolve_packet_loss_channel_config(
        transmitted_packets,
        config=channel_config,
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
    received_equations = _received_equations(
        packets=loss_result.received,
        total_tokens=len(sample.token_ids),
        xor_code=xor_code,
        sparse_config=sparse_config,
    )
    initial_peel = solve_xor_equations(
        equations=received_equations,
        known_tokens=known_tokens,
        hash_metadata=hash_metadata,
        token_hash_map=token_hash_map,
        vocab_size=config.vocab_size,
        banned_token_ids=config.special_token_ids,
        enable_linear_solve=_linear_solve_enabled_for_code(
            xor_code=xor_code,
            sparse_linear_solve_enabled=sparse_linear_solve_enabled,
        ),
        max_component_unknowns=sparse_max_component_unknowns,
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
    if hybrid_mode in {
        HYBRID_MODE_PARITY_FILTER,
        HYBRID_MODE_ITERATIVE_PEEL,
        HYBRID_MODE_ITERATIVE_ROLLBACK,
    }:
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
            enable_linear_solve=_linear_solve_enabled_for_code(
                xor_code=xor_code,
                sparse_linear_solve_enabled=sparse_linear_solve_enabled,
            ),
            max_component_unknowns=sparse_max_component_unknowns,
        )
    elif hybrid_mode == HYBRID_MODE_ITERATIVE_ROLLBACK:
        post_commit_hook = IterativeRollbackXorHook(
            equations=received_equations,
            hash_metadata=hash_metadata,
            token_hash_map=token_hash_map,
            mask_token_id=config.mask_token_id,
            vocab_size=config.vocab_size,
            banned_token_ids=config.special_token_ids,
            enable_linear_solve=_linear_solve_enabled_for_code(
                xor_code=xor_code,
                sparse_linear_solve_enabled=sparse_linear_solve_enabled,
            ),
            max_component_unknowns=sparse_max_component_unknowns,
            rollback_extra_steps=rollback_extra_steps,
            rollback_max_total_steps=rollback_max_total_steps,
            rollback_max_per_position=rollback_max_per_position,
            rollback_stop_after_no_progress=rollback_stop_after_no_progress,
            rollback_continue_until_stable=rollback_continue_until_stable,
            rollback_require_zero_masks=rollback_require_zero_masks,
            rollback_require_final_parity_clean=rollback_require_final_parity_clean,
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
        linear_solver_diagnostics=initial_peel.linear_solver_diagnostics,
        sparse_diagnostics=_sparse_diagnostics(encoded),
        xor_code=xor_code,
        parity_equation_count=len(
            _received_equations(
                packets=encoded.parity_packets,
                total_tokens=len(sample.token_ids),
                xor_code=xor_code,
                sparse_config=sparse_config,
            )
        ),
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
        xor_code=xor_code,
        sparse_diagnostics=_sparse_diagnostics(encoded),
        reconstruction_plan=plan,
        decoding_result=decoding_result,
        mask_token_id=config.mask_token_id,
    )


def _encode_repair(
    *,
    sample: TokenSample,
    tokens_per_packet: int,
    xor_code: str,
    xor_config: XorParityConfig,
    sparse_config: SparseFountainXorConfig | None,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
) -> XorParityEncoded | SparseFountainXorEncoded:
    if xor_code == XOR_CODE_STRIPE:
        return encode_xor_parity(
            sample,
            tokens_per_packet=tokens_per_packet,
            config=xor_config,
            source_layout=source_layout,
            wire_interleaving=wire_interleaving,
        )
    if xor_code == XOR_CODE_SPARSE_FOUNTAIN:
        if sparse_config is None:
            raise ValueError("sparse_config is required when xor_code='sparse_fountain'")
        return encode_sparse_fountain_xor(
            sample,
            tokens_per_packet=tokens_per_packet,
            config=sparse_config,
            source_layout=source_layout,
            wire_interleaving=wire_interleaving,
        )
    _validate_xor_code(xor_code)
    raise AssertionError("unreachable")


def _received_equations(
    *,
    packets: Sequence[Packet],
    total_tokens: int,
    xor_code: str,
    sparse_config: SparseFountainXorConfig | None,
):
    if xor_code == XOR_CODE_STRIPE:
        return equations_from_parity_packets(packets)
    if xor_code == XOR_CODE_SPARSE_FOUNTAIN:
        if sparse_config is None:
            raise ValueError("sparse_config is required when xor_code='sparse_fountain'")
        return equations_from_sparse_fountain_packets(
            packets,
            total_tokens=total_tokens,
            config=sparse_config,
        )
    _validate_xor_code(xor_code)
    raise AssertionError("unreachable")


def _linear_solve_enabled_for_code(
    *,
    xor_code: str,
    sparse_linear_solve_enabled: bool,
) -> bool:
    return xor_code == XOR_CODE_SPARSE_FOUNTAIN and sparse_linear_solve_enabled


def _sparse_diagnostics(encoded: XorParityEncoded | SparseFountainXorEncoded) -> dict[str, Any]:
    if isinstance(encoded, SparseFountainXorEncoded):
        return encoded.diagnostics.to_dict()
    return _empty_sparse_diagnostics()


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
    xor_code: str,
    sparse_xor_seed: int,
    sparse_xor_coverage: bool,
    sparse_xor_degree_distribution: str | Sequence[tuple[int, float]],
    sparse_xor_max_coverage_degree: int,
    sparse_xor_max_component_unknowns: int,
    sparse_xor_enable_linear_solve: bool,
    rollback_extra_steps: int,
    rollback_max_total_steps: int,
    rollback_max_per_position: int,
    rollback_stop_after_no_progress: int,
    rollback_continue_until_stable: bool,
    rollback_require_zero_masks: bool,
    rollback_require_final_parity_clean: bool,
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
        xor_code=xor_code,
        sparse_xor_max_component_unknowns=sparse_xor_max_component_unknowns,
        rollback_extra_steps=rollback_extra_steps,
        rollback_max_total_steps=rollback_max_total_steps,
        rollback_max_per_position=rollback_max_per_position,
        rollback_stop_after_no_progress=rollback_stop_after_no_progress,
        rollback_continue_until_stable=rollback_continue_until_stable,
        rollback_require_zero_masks=rollback_require_zero_masks,
        rollback_require_final_parity_clean=rollback_require_final_parity_clean,
    )
    xor_config = XorParityConfig(
        data_packets_per_stripe=4,
        target_overhead_ratio=xor_overhead_bits_per_token / token_bit_width_for_vocab(vocab_size),
        vocab_size=vocab_size,
    )
    sparse_config = SparseFountainXorConfig(
        xor_overhead_bits_per_token=xor_overhead_bits_per_token,
        vocab_size=vocab_size,
        random_seed=sparse_xor_seed,
        coverage_enabled=sparse_xor_coverage,
        degree_distribution=parse_degree_distribution(sparse_xor_degree_distribution),
        max_coverage_degree=sparse_xor_max_coverage_degree,
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
        xor_code=xor_code,
        sparse_xor_seed=sparse_xor_seed,
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
        rollback_extra_steps=rollback_extra_steps,
        rollback_max_total_steps=rollback_max_total_steps,
        rollback_max_per_position=rollback_max_per_position,
        rollback_continue_until_stable=rollback_continue_until_stable,
        rollback_require_zero_masks=rollback_require_zero_masks,
        rollback_require_final_parity_clean=rollback_require_final_parity_clean,
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
        xor_code=xor_code,
        sparse_xor_max_component_unknowns=sparse_xor_max_component_unknowns,
        sparse_xor_enable_linear_solve=sparse_xor_enable_linear_solve,
        rollback_extra_steps=rollback_extra_steps,
        rollback_max_total_steps=rollback_max_total_steps,
        rollback_max_per_position=rollback_max_per_position,
        rollback_stop_after_no_progress=rollback_stop_after_no_progress,
        rollback_continue_until_stable=rollback_continue_until_stable,
        rollback_require_zero_masks=rollback_require_zero_masks,
        rollback_require_final_parity_clean=rollback_require_final_parity_clean,
        source_layout=source_layout,
        wire_interleaving=wire_interleaving,
        channel_config=channel_config,
        xor_config=xor_config,
        sparse_config=sparse_config,
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
            xor_code=xor_code,
            sparse_config=sparse_config,
            sparse_linear_solve_enabled=sparse_xor_enable_linear_solve,
            sparse_max_component_unknowns=sparse_xor_max_component_unknowns,
            rollback_extra_steps=rollback_extra_steps,
            rollback_max_total_steps=rollback_max_total_steps,
            rollback_max_per_position=rollback_max_per_position,
            rollback_stop_after_no_progress=rollback_stop_after_no_progress,
            rollback_continue_until_stable=rollback_continue_until_stable,
            rollback_require_zero_masks=rollback_require_zero_masks,
            rollback_require_final_parity_clean=rollback_require_final_parity_clean,
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
                strategy=_strategy(real_llada=real_llada, hybrid_mode=hybrid_mode, xor_code=xor_code),
                seed=case_seed,
                loss_rate=loss_rate,
                tokens_per_packet=tokens_per_packet,
                hash_bits=hash_bits,
                xor_overhead_bits_per_token=xor_overhead_bits_per_token,
                vocab_size=vocab_size,
                source_layout=source_layout,
                wire_interleaving=wire_interleaving,
                channel_config=case.channel_config,
                hash_profile_info=hash_profile_info,
            )
        )
        events.append(
            {
                "event_type": "hybrid_xor_hash_micro_eval_case",
                "run_id": run_id,
                "case_id": case_id,
                "model_label": model_label,
                "strategy": _strategy(real_llada=real_llada, hybrid_mode=hybrid_mode, xor_code=xor_code),
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
        "xor_code": case.xor_code,
        "xor_overhead_bits_per_token": xor_overhead_bits_per_token,
        "source_layout": source_layout.mode,
        "source_chunk_size": source_layout.chunk_size,
        "wire_interleaving": wire_interleaving.mode,
        "wire_interleaving_span": wire_interleaving.span,
        "channel_mode": channel_config.mode,
        "burst_start_wire_id": channel_config.burst_start_wire_id,
        "burst_length": channel_config.burst_length,
        "requested_burst_loss_rate": channel_config.burst_loss_rate,
        "resolved_burst_length": channel_config.resolved_burst_length,
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
        **case.loss_diagnostics,
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
        "total_decode_time_sec": diagnostics.get(
            "total_decode_time_sec",
            case.decoding_result.decode_latency_sec,
        ),
        "model_forward_time_sec": diagnostics.get("model_forward_time_sec", 0.0),
        "candidate_construction_time_sec": diagnostics.get(
            "candidate_construction_time_sec",
            0.0,
        ),
        "parity_candidate_filter_time_sec": diagnostics.get(
            "parity_candidate_filter_time_sec",
            0.0,
        ),
        "xor_peel_time_sec": diagnostics.get("xor_peel_time_sec", 0.0),
        "linear_solver_time_sec": diagnostics.get("linear_solver_time_sec", 0.0),
        "post_commit_hook_time_sec": diagnostics.get("post_commit_hook_time_sec", 0.0),
        "rollback_time_sec": diagnostics.get("rollback_time_sec", 0.0),
        "decoder_steps": case.decoding_result.steps,
        "model_forward_calls": diagnostics.get("model_forward_calls"),
        "model_proposal_calls": diagnostics.get("model_proposal_calls"),
        "decoder_proposal_mode": diagnostics.get("decoder_proposal_mode"),
        "proposal_interface_used": diagnostics.get("proposal_interface_used"),
        "mean_candidate_count": diagnostics.get("mean_candidate_count", 0.0),
        "max_candidate_count": diagnostics.get("max_candidate_count", 0),
        "parity_equation_count": diagnostics.get("parity_equation_count", 0),
        "parity_received_equation_count": diagnostics.get("parity_received_equation_count", 0),
        "parity_peel_iterations": case.initial_peel.peel_iteration_count,
        "parity_peel_recovered_count": case.initial_peel.recovered_count,
        "parity_hash_conflict_count": case.initial_peel.conflict_count,
        **_linear_solver_result_fields(case.initial_peel.linear_solver_diagnostics),
        **_sparse_result_fields(case.sparse_diagnostics),
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
        "iterative_linear_solver_enabled": diagnostics.get(
            "iterative_linear_solver_enabled",
            False,
        ),
        "iterative_linear_solver_components_seen": diagnostics.get(
            "iterative_linear_solver_components_seen",
            0,
        ),
        "iterative_linear_solver_components_solved": diagnostics.get(
            "iterative_linear_solver_components_solved",
            0,
        ),
        "iterative_linear_solver_tokens_recovered": diagnostics.get(
            "iterative_linear_solver_tokens_recovered",
            0,
        ),
        "iterative_linear_solver_rank_deficient_count": diagnostics.get(
            "iterative_linear_solver_rank_deficient_count",
            0,
        ),
        "iterative_linear_solver_validation_conflict_count": diagnostics.get(
            "iterative_linear_solver_validation_conflict_count",
            0,
        ),
        "iterative_linear_solver_too_large_count": diagnostics.get(
            "iterative_linear_solver_too_large_count",
            0,
        ),
        "iterative_peel_recovered_positions": _csv_sequence(
            diagnostics.get("iterative_peel_recovered_positions", ())
        ),
        "iterative_peel_recovered_count_by_step": _csv_sequence(
            diagnostics.get("iterative_peel_recovered_count_by_step", ())
        ),
        "rollback_enabled": diagnostics.get("rollback_enabled", False),
        "rollback_event_count": diagnostics.get("rollback_event_count", 0),
        "rollback_conflict_equation_count": diagnostics.get(
            "rollback_conflict_equation_count",
            0,
        ),
        "rollback_positions_count": diagnostics.get("rollback_positions_count", 0),
        "rollback_positions": _csv_sequence(diagnostics.get("rollback_positions", ())),
        "rollback_single_suspect_count": diagnostics.get("rollback_single_suspect_count", 0),
        "rollback_multi_suspect_count": diagnostics.get("rollback_multi_suspect_count", 0),
        "rollback_banned_token_count": diagnostics.get("rollback_banned_token_count", 0),
        "rollback_banned_tokens_by_position": json.dumps(
            diagnostics.get("rollback_banned_tokens_by_position", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "rollback_max_per_position_hits": diagnostics.get("rollback_max_per_position_hits", 0),
        "rollback_adaptive_enabled": diagnostics.get("rollback_adaptive_enabled", False),
        "rollback_total_steps_used": diagnostics.get("rollback_total_steps_used", 0),
        "rollback_base_steps": diagnostics.get("rollback_base_steps", 0),
        "rollback_extra_steps_used": diagnostics.get("rollback_extra_steps_used", 0),
        "rollback_stopped_reason": diagnostics.get("rollback_stopped_reason", ""),
        "rollback_final_zero_masks": diagnostics.get("rollback_final_zero_masks", False),
        "rollback_final_parity_clean": diagnostics.get("rollback_final_parity_clean", ""),
        "rollback_remaining_masks_after_budget": diagnostics.get(
            "rollback_remaining_masks_after_budget",
            0,
        ),
        "rollback_provenance_invalidated_count": diagnostics.get(
            "rollback_provenance_invalidated_count",
            0,
        ),
        "rollback_no_progress_stop": diagnostics.get("rollback_no_progress_stop", False),
        "parity_candidate_rejections": case.parity_filter_diagnostics.get(
            "parity_candidate_rejections",
            0,
        ),
        "parity_filter_fallback_count": case.parity_filter_diagnostics.get(
            "parity_filter_fallback_count",
            0,
        ),
        "parity_filter_required_token_checks": case.parity_filter_diagnostics.get(
            "parity_filter_required_token_checks",
            0,
        ),
        "parity_filter_full_scan_count": case.parity_filter_diagnostics.get(
            "parity_filter_full_scan_count",
            0,
        ),
        "parity_filter_candidate_membership_checks": case.parity_filter_diagnostics.get(
            "parity_filter_candidate_membership_checks",
            0,
        ),
        "parity_filter_time_sec": case.parity_filter_diagnostics.get(
            "parity_filter_time_sec",
            0.0,
        ),
        "parity_filter_mean_input_candidate_count": case.parity_filter_diagnostics.get(
            "parity_filter_mean_input_candidate_count",
            0.0,
        ),
        "parity_filter_max_input_candidate_count": case.parity_filter_diagnostics.get(
            "parity_filter_max_input_candidate_count",
            0,
        ),
        "parity_filter_mean_output_candidate_count": case.parity_filter_diagnostics.get(
            "parity_filter_mean_output_candidate_count",
            0.0,
        ),
        "parity_filter_max_output_candidate_count": case.parity_filter_diagnostics.get(
            "parity_filter_max_output_candidate_count",
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
    xor_code: str,
    sparse_xor_max_component_unknowns: int,
    sparse_xor_enable_linear_solve: bool,
    rollback_extra_steps: int,
    rollback_max_total_steps: int,
    rollback_max_per_position: int,
    rollback_stop_after_no_progress: int,
    rollback_continue_until_stable: bool,
    rollback_require_zero_masks: bool,
    rollback_require_final_parity_clean: bool,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    xor_config: XorParityConfig,
    sparse_config: SparseFountainXorConfig,
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
        "strategy": _strategy(real_llada=real_llada, hybrid_mode=hybrid_mode, xor_code=xor_code),
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
            "xor_code": xor_code,
            "parity_filter_fallback": parity_filter_fallback,
            "sparse_xor_max_component_unknowns": sparse_xor_max_component_unknowns,
            "sparse_xor_enable_linear_solve": sparse_xor_enable_linear_solve,
            "rollback_extra_steps": rollback_extra_steps,
            "rollback_max_total_steps": rollback_max_total_steps,
            "rollback_max_per_position": rollback_max_per_position,
            "rollback_stop_after_no_progress": rollback_stop_after_no_progress,
            "rollback_continue_until_stable": rollback_continue_until_stable,
            "rollback_require_zero_masks": rollback_require_zero_masks,
            "rollback_require_final_parity_clean": rollback_require_final_parity_clean,
            "protection_mode": HYBRID_PROTECTION_MODE,
            "oracle_hash_metadata": False,
            "source_layout": source_layout.to_dict(),
            "wire_interleaving": wire_interleaving.to_dict(),
            "channel": channel_config.to_dict(),
            "xor_parity": xor_config.to_dict(),
            "sparse_fountain_xor": sparse_config.to_dict(),
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
    xor_code: str,
    sparse_xor_seed: int,
    editable_update_mode: str,
    hash_constraint_schedule: str,
    rollback_extra_steps: int,
    rollback_max_total_steps: int,
    rollback_max_per_position: int,
    rollback_continue_until_stable: bool,
    rollback_require_zero_masks: bool,
    rollback_require_final_parity_clean: bool,
) -> str:
    lengths = "-".join(str(length) for length in sample_lengths)
    source_chunk = source_layout.chunk_size if source_layout.chunk_size is not None else "default"
    model = model_label.replace("/", "-")
    return (
        f"{runner}|{model}|hash{hash_bits}|xorbits{xor_overhead_bits_per_token:g}|"
        f"{xor_code}|{hybrid_mode}|loss{loss_rate:g}|lengths{lengths}|tpp{tokens_per_packet}|"
        f"seed{seed}|source-{source_layout.mode}-chunk{source_chunk}|"
        f"wire-{wire_interleaving.mode}-span{wire_interleaving.span}|"
        f"channel-{channel_config.mode}|update-{editable_update_mode}|"
        f"hash-schedule-{hash_constraint_schedule}|sparse-seed{sparse_xor_seed}|"
        f"rollback-extra{rollback_extra_steps}-max{rollback_max_total_steps}-perpos{rollback_max_per_position}|"
        f"rollback-adaptive{int(rollback_continue_until_stable)}-zero{int(rollback_require_zero_masks)}-"
        f"parityclean{int(rollback_require_final_parity_clean)}"
    )


def _strategy(*, real_llada: bool, hybrid_mode: str, xor_code: str = XOR_CODE_STRIPE) -> str:
    prefix = "RealLLaDA" if real_llada else "FakeHybrid"
    suffix = "" if xor_code == XOR_CODE_STRIPE else "_SparseFountain"
    return f"{prefix}_HashXOR{suffix}_{hybrid_mode}"


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
    xor_code: str = XOR_CODE_STRIPE,
    sparse_xor_max_component_unknowns: int = DEFAULT_SPARSE_XOR_MAX_COMPONENT_UNKNOWNS,
    rollback_extra_steps: int = DEFAULT_ROLLBACK_EXTRA_STEPS,
    rollback_max_total_steps: int = DEFAULT_ROLLBACK_MAX_TOTAL_STEPS,
    rollback_max_per_position: int = DEFAULT_ROLLBACK_MAX_PER_POSITION,
    rollback_stop_after_no_progress: int = DEFAULT_ROLLBACK_STOP_AFTER_NO_PROGRESS,
    rollback_continue_until_stable: bool = False,
    rollback_require_zero_masks: bool = False,
    rollback_require_final_parity_clean: bool = False,
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
    _validate_xor_code(xor_code)
    if sparse_xor_max_component_unknowns <= 0:
        raise ValueError("sparse_xor_max_component_unknowns must be positive")
    if rollback_extra_steps < 0:
        raise ValueError("rollback_extra_steps must be non-negative")
    if rollback_max_total_steps <= 0:
        raise ValueError("rollback_max_total_steps must be positive")
    if rollback_max_per_position <= 0:
        raise ValueError("rollback_max_per_position must be positive")
    if rollback_stop_after_no_progress < 0:
        raise ValueError("rollback_stop_after_no_progress must be non-negative")
    if rollback_require_zero_masks and not rollback_continue_until_stable:
        raise ValueError("rollback_require_zero_masks requires rollback_continue_until_stable")
    if rollback_require_final_parity_clean and not rollback_continue_until_stable:
        raise ValueError("rollback_require_final_parity_clean requires rollback_continue_until_stable")


def _validate_hybrid_mode(hybrid_mode: str) -> None:
    if hybrid_mode not in VALID_HYBRID_MODES:
        modes = ", ".join(sorted(VALID_HYBRID_MODES))
        raise ValueError(f"hybrid_mode must be one of: {modes}")


def _validate_xor_code(xor_code: str) -> None:
    if xor_code not in VALID_XOR_CODES:
        codes = ", ".join(sorted(VALID_XOR_CODES))
        raise ValueError(f"xor_code must be one of: {codes}")


def _validate_iterative_peel_decoder_config(config: DiffusionDecodingConfig) -> None:
    if config.editable_update_mode != EDITABLE_UPDATE_COMMIT_ONCE:
        raise ValueError("hybrid_mode='iterative_peel' requires editable_update_mode='commit_once'")
    if config.hash_constraint_schedule != HASH_CONSTRAINT_ALWAYS:
        raise ValueError("hybrid_mode='iterative_peel' requires hash_constraint_schedule='always'")


def _validate_iterative_rollback_config(*, xor_code: str) -> None:
    if xor_code != XOR_CODE_SPARSE_FOUNTAIN:
        raise ValueError("hybrid_mode='iterative_rollback' requires xor_code='sparse_fountain'")


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
    encoded: XorParityEncoded | SparseFountainXorEncoded,
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
    linear_solver_diagnostics: dict[str, Any],
    sparse_diagnostics: dict[str, Any],
    xor_code: str,
    parity_equation_count: int,
    parity_received_equation_count: int,
) -> DecodingResult:
    diagnostics = dict(decoding_result.diagnostics)
    initial_xor_peel_time = float(linear_solver_diagnostics.get("xor_peel_time_sec", 0.0) or 0.0)
    initial_linear_solver_time = float(
        linear_solver_diagnostics.get("linear_solver_time_sec", 0.0) or 0.0
    )
    iterative_xor_peel_time = float(
        iterative_peel_diagnostics.get("iterative_xor_peel_time_sec", 0.0) or 0.0
    )
    iterative_linear_solver_time = float(
        iterative_peel_diagnostics.get("iterative_linear_solver_time_sec", 0.0) or 0.0
    )
    diagnostics.update(
        {
            "hybrid_mode": hybrid_mode,
            "xor_code": xor_code,
            "parity_equation_count": parity_equation_count,
            "parity_received_equation_count": parity_received_equation_count,
            "parity_peel_iterations": initial_peel.peel_iteration_count,
            "parity_peel_recovered_count": initial_peel.recovered_count,
            "parity_hash_conflict_count": initial_peel.conflict_count,
            "parity_equations_satisfied": final_audit.satisfied_count,
            "parity_equations_violated": final_audit.violated_count,
            **_linear_solver_result_fields(linear_solver_diagnostics),
            **_sparse_result_fields(sparse_diagnostics),
            **parity_filter_diagnostics,
            **iterative_peel_diagnostics,
            "xor_peel_time_sec": initial_xor_peel_time + iterative_xor_peel_time,
            "linear_solver_time_sec": initial_linear_solver_time + iterative_linear_solver_time,
        }
    )
    if diagnostics.get("rollback_enabled"):
        diagnostics["rollback_final_parity_clean"] = final_audit.violated_count == 0
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
    diagnostics = _zero_timing_diagnostics(case.decoding_result.diagnostics)
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
        diagnostics=diagnostics,
    )
    return replace(
        case,
        decoding_result=decoding_result,
        parity_filter_diagnostics=_zero_timing_diagnostics(case.parity_filter_diagnostics),
        initial_peel=_normalize_peel_timing(case.initial_peel),
    )


def _normalize_peel_timing(result: XorPeelResult) -> XorPeelResult:
    return replace(
        result,
        linear_solver_diagnostics=_zero_timing_diagnostics(result.linear_solver_diagnostics),
    )


def _zero_timing_diagnostics(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    for key in list(normalized):
        if key.endswith("_time_sec") or key in {"decode_latency_sec", "total_decode_time_sec"}:
            normalized[key] = 0.0
    candidate_filter_diagnostics = normalized.get("candidate_filter_diagnostics")
    if isinstance(candidate_filter_diagnostics, Mapping):
        normalized["candidate_filter_diagnostics"] = _zero_timing_diagnostics(
            candidate_filter_diagnostics
        )
    post_commit_hook_diagnostics = normalized.get("post_commit_hook_diagnostics")
    if isinstance(post_commit_hook_diagnostics, Mapping):
        normalized["post_commit_hook_diagnostics"] = _zero_timing_diagnostics(
            post_commit_hook_diagnostics
        )
    step_diagnostics = normalized.get("step_diagnostics")
    if isinstance(step_diagnostics, Sequence) and not isinstance(step_diagnostics, (str, bytes)):
        normalized["step_diagnostics"] = tuple(
            _zero_timing_diagnostics(item) if isinstance(item, Mapping) else item
            for item in step_diagnostics
        )
    return normalized


def _empty_filter_diagnostics() -> dict[str, Any]:
    return {
        "parity_candidate_filter_calls": 0,
        "parity_candidate_rejections": 0,
        "parity_filter_fallback_count": 0,
        "parity_filter_fallback_enabled": False,
        "parity_filter_required_token_checks": 0,
        "parity_filter_full_scan_count": 0,
        "parity_filter_candidate_membership_checks": 0,
        "parity_filter_time_sec": 0.0,
        "parity_filter_mean_input_candidate_count": 0.0,
        "parity_filter_max_input_candidate_count": 0,
        "parity_filter_mean_output_candidate_count": 0.0,
        "parity_filter_max_output_candidate_count": 0,
    }


def _iterative_peel_style_diagnostics(
    *,
    enabled: bool,
    call_count: int,
    recovered_tokens: Mapping[int, int],
    per_step_recovered_count: Sequence[int],
    conflicts: Sequence[XorPeelConflict],
    enable_linear_solve: bool,
    linear_solver_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "iterative_peel_enabled": enabled,
        "iterative_peel_passes": call_count,
        "iterative_peel_recovered_count": len(recovered_tokens),
        "iterative_peel_recovered_positions": tuple(sorted(recovered_tokens)),
        "iterative_peel_recovered_count_by_step": tuple(per_step_recovered_count),
        "iterative_peel_conflict_count": len(conflicts),
        "iterative_linear_solver_enabled": enable_linear_solve,
        "iterative_linear_solver_components_seen": linear_solver_diagnostics[
            "linear_solver_components_seen"
        ],
        "iterative_linear_solver_components_solved": linear_solver_diagnostics[
            "linear_solver_components_solved"
        ],
        "iterative_linear_solver_tokens_recovered": linear_solver_diagnostics[
            "linear_solver_tokens_recovered"
        ],
        "iterative_linear_solver_rank_deficient_count": linear_solver_diagnostics[
            "linear_solver_rank_deficient_count"
        ],
        "iterative_linear_solver_validation_conflict_count": linear_solver_diagnostics[
            "linear_solver_validation_conflict_count"
        ],
        "iterative_linear_solver_too_large_count": linear_solver_diagnostics[
            "linear_solver_too_large_count"
        ],
        "iterative_xor_peel_time_sec": linear_solver_diagnostics.get(
            "xor_peel_time_sec",
            0.0,
        ),
        "iterative_linear_solver_time_sec": linear_solver_diagnostics.get(
            "linear_solver_time_sec",
            0.0,
        ),
        "iterative_total_xor_solve_time_sec": linear_solver_diagnostics.get(
            "total_xor_solve_time_sec",
            0.0,
        ),
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


def _hook_step_diagnostics(
    *,
    equations: Sequence[Any],
    peel: XorPeelResult,
    newly_fixed_count: int,
    solve_time_sec: float,
    rollback_count: int,
    rollback_time_sec: float,
) -> dict[str, Any]:
    linear = peel.linear_solver_diagnostics
    return {
        "active_parity_equation_count": len(equations),
        "parity_peeled_count": newly_fixed_count,
        "linear_solver_components_seen": linear.get("linear_solver_components_seen", 0),
        "linear_solver_components_solved": linear.get("linear_solver_components_solved", 0),
        "linear_solver_rank_deficient_count": linear.get(
            "linear_solver_rank_deficient_count",
            0,
        ),
        "linear_solver_too_large_count": linear.get("linear_solver_too_large_count", 0),
        "linear_solver_tokens_recovered": linear.get("linear_solver_tokens_recovered", 0),
        "xor_peel_time_sec": linear.get("xor_peel_time_sec", 0.0),
        "linear_solver_time_sec": linear.get("linear_solver_time_sec", 0.0),
        "total_xor_solve_time_sec": solve_time_sec,
        "rollback_count": rollback_count,
        "rollback_time_sec": rollback_time_sec,
    }


def _empty_linear_solver_diagnostics(*, enabled: bool = False) -> dict[str, Any]:
    return {
        "linear_solver_enabled": enabled,
        "linear_solver_components_seen": 0,
        "linear_solver_components_solved": 0,
        "linear_solver_tokens_recovered": 0,
        "linear_solver_rank_deficient_count": 0,
        "linear_solver_validation_conflict_count": 0,
        "linear_solver_too_large_count": 0,
        "xor_peel_time_sec": 0.0,
        "linear_solver_time_sec": 0.0,
        "total_xor_solve_time_sec": 0.0,
    }


def _linear_solver_result_fields(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    data = {**_empty_linear_solver_diagnostics(), **dict(diagnostics or {})}
    return {
        "linear_solver_enabled": data["linear_solver_enabled"],
        "linear_solver_components_seen": data["linear_solver_components_seen"],
        "linear_solver_components_solved": data["linear_solver_components_solved"],
        "linear_solver_tokens_recovered": data["linear_solver_tokens_recovered"],
        "linear_solver_rank_deficient_count": data["linear_solver_rank_deficient_count"],
        "linear_solver_validation_conflict_count": data["linear_solver_validation_conflict_count"],
        "linear_solver_too_large_count": data["linear_solver_too_large_count"],
        "initial_xor_peel_time_sec": data["xor_peel_time_sec"],
        "initial_linear_solver_time_sec": data["linear_solver_time_sec"],
    }


def _empty_sparse_diagnostics() -> dict[str, Any]:
    return {
        "equation_count": 0,
        "budget_exhausted": False,
        "coverage_enabled": False,
        "coverage_possible": False,
        "coverage_pass_degree": 0,
        "coverage_zero_count": 0,
        "coverage_min": 0,
        "coverage_mean": 0.0,
        "actual_mean_degree": 0.0,
        "degree_histogram": {},
    }


def _sparse_result_fields(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    data = {**_empty_sparse_diagnostics(), **dict(diagnostics or {})}
    return {
        "sparse_equation_count": data["equation_count"],
        "sparse_budget_exhausted": data["budget_exhausted"],
        "sparse_coverage_enabled": data["coverage_enabled"],
        "sparse_coverage_possible": data["coverage_possible"],
        "sparse_coverage_pass_degree": data["coverage_pass_degree"],
        "sparse_coverage_zero_count": data["coverage_zero_count"],
        "sparse_coverage_min": data["coverage_min"],
        "sparse_coverage_mean": data["coverage_mean"],
        "sparse_actual_mean_degree": data["actual_mean_degree"],
        "sparse_degree_histogram": json.dumps(
            data["degree_histogram"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _empty_iterative_peel_diagnostics() -> dict[str, Any]:
    return {
        "iterative_peel_enabled": False,
        "iterative_peel_passes": 0,
        "iterative_peel_recovered_count": 0,
        "iterative_peel_recovered_positions": (),
        "iterative_peel_recovered_count_by_step": (),
        "iterative_peel_conflict_count": 0,
        "iterative_linear_solver_enabled": False,
        "iterative_linear_solver_components_seen": 0,
        "iterative_linear_solver_components_solved": 0,
        "iterative_linear_solver_tokens_recovered": 0,
        "iterative_linear_solver_rank_deficient_count": 0,
        "iterative_linear_solver_validation_conflict_count": 0,
        "iterative_linear_solver_too_large_count": 0,
        "iterative_xor_peel_time_sec": 0.0,
        "iterative_linear_solver_time_sec": 0.0,
        "iterative_total_xor_solve_time_sec": 0.0,
        "iterative_peel_hash_conflict_count": 0,
        "iterative_peel_special_token_conflict_count": 0,
        "iterative_peel_vocab_conflict_count": 0,
        "iterative_peel_conflicts": (),
        "rollback_enabled": False,
        "rollback_event_count": 0,
        "rollback_conflict_equation_count": 0,
        "rollback_positions_count": 0,
        "rollback_positions": (),
        "rollback_single_suspect_count": 0,
        "rollback_multi_suspect_count": 0,
        "rollback_banned_token_count": 0,
        "rollback_banned_tokens_by_position": {},
        "rollback_max_per_position_hits": 0,
        "rollback_adaptive_enabled": False,
        "rollback_total_steps_used": 0,
        "rollback_base_steps": 0,
        "rollback_extra_steps_used": 0,
        "rollback_stopped_reason": "",
        "rollback_final_zero_masks": False,
        "rollback_final_parity_clean": "",
        "rollback_remaining_masks_after_budget": 0,
        "rollback_provenance_invalidated_count": 0,
        "rollback_no_progress_stop": False,
        "rollback_time_sec": 0.0,
        "rollback_events": (),
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
