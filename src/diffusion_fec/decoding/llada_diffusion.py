"""Constrained masked-diffusion decoding loop.

The implementation is intentionally torch-free for the first fake-model slice,
but the model contract mirrors the future LLaDA adapter:

``model.forward(input_ids, attention_mask=None).logits``

where ``input_ids`` is batch-first and logits are shaped as
``[batch, sequence, vocab]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil, exp
from time import perf_counter
from typing import Any

from diffusion_fec.coding.token_hash import TokenHashMap
from diffusion_fec.decoding.constraints import build_constraint_masks
from diffusion_fec.decoding.diagnostics import (
    editable_confidence_stat,
    fixed_confidence_stat,
    step_summary,
)
from diffusion_fec.types import (
    ConfidenceStat,
    DecodingResult,
    ReconstructionEntry,
    ReconstructionPlan,
    STATE_KNOWN,
    STATE_MISSING,
)


EDITABLE_UPDATE_COMMIT_ONCE = "commit_once"
EDITABLE_UPDATE_RESAMPLE_EACH_STEP = "resample_each_step"
VALID_EDITABLE_UPDATE_MODES = frozenset(
    {
        EDITABLE_UPDATE_COMMIT_ONCE,
        EDITABLE_UPDATE_RESAMPLE_EACH_STEP,
    }
)

HASH_CONSTRAINT_ALWAYS = "always"
HASH_CONSTRAINT_FINAL_ONLY = "final_only"
HASH_CONSTRAINT_LATE_HALF = "late_half"
VALID_HASH_CONSTRAINT_SCHEDULES = frozenset(
    {
        HASH_CONSTRAINT_ALWAYS,
        HASH_CONSTRAINT_FINAL_ONLY,
        HASH_CONSTRAINT_LATE_HALF,
    }
)


@dataclass(frozen=True)
class DiffusionDecodingConfig:
    """Configuration for deterministic constrained masked diffusion."""

    mask_token_id: int
    vocab_size: int
    steps: int = 128
    block_length: int = 32
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    banned_token_ids: tuple[int, ...] = field(default_factory=tuple)
    fallback_on_empty_hash_bucket: bool = True
    editable_update_mode: str = EDITABLE_UPDATE_COMMIT_ONCE
    hash_constraint_schedule: str = HASH_CONSTRAINT_ALWAYS

    def __post_init__(self) -> None:
        if not isinstance(self.vocab_size, int):
            raise TypeError("vocab_size must be an int")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not isinstance(self.steps, int):
            raise TypeError("steps must be an int")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if not isinstance(self.block_length, int):
            raise TypeError("block_length must be an int")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")
        object.__setattr__(self, "banned_token_ids", tuple(self.banned_token_ids))

        for field_name, token_id in (
            ("mask_token_id", self.mask_token_id),
            ("eos_token_id", self.eos_token_id),
            ("pad_token_id", self.pad_token_id),
        ):
            if token_id is not None:
                self._validate_token_id(token_id, field_name)
        for token_id in self.banned_token_ids:
            self._validate_token_id(token_id, "banned_token_ids")
        if self.editable_update_mode not in VALID_EDITABLE_UPDATE_MODES:
            modes = ", ".join(sorted(VALID_EDITABLE_UPDATE_MODES))
            raise ValueError(f"editable_update_mode must be one of: {modes}")
        if self.hash_constraint_schedule not in VALID_HASH_CONSTRAINT_SCHEDULES:
            schedules = ", ".join(sorted(VALID_HASH_CONSTRAINT_SCHEDULES))
            raise ValueError(f"hash_constraint_schedule must be one of: {schedules}")
        if (
            self.editable_update_mode == EDITABLE_UPDATE_COMMIT_ONCE
            and self.hash_constraint_schedule != HASH_CONSTRAINT_ALWAYS
        ):
            raise ValueError(
                "commit_once decoding only supports hash_constraint_schedule='always'; "
                "use editable_update_mode='resample_each_step' for relaxed schedules"
            )

    @property
    def special_token_ids(self) -> frozenset[int]:
        ids = set(self.banned_token_ids)
        ids.add(self.mask_token_id)
        if self.eos_token_id is not None:
            ids.add(self.eos_token_id)
        if self.pad_token_id is not None:
            ids.add(self.pad_token_id)
        return frozenset(ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_token_id": self.mask_token_id,
            "vocab_size": self.vocab_size,
            "steps": self.steps,
            "block_length": self.block_length,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "banned_token_ids": list(self.banned_token_ids),
            "fallback_on_empty_hash_bucket": self.fallback_on_empty_hash_bucket,
            "editable_update_mode": self.editable_update_mode,
            "hash_constraint_schedule": self.hash_constraint_schedule,
        }

    def _validate_token_id(self, token_id: int, field_name: str) -> None:
        if not isinstance(token_id, int):
            raise TypeError(f"{field_name} must be an int")
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"{field_name} must be in range [0, {self.vocab_size})")


@dataclass(frozen=True)
class _Proposal:
    full_position: int
    target_position: int
    token_id: int
    top1_probability: float
    top2_probability: float
    candidate_count: int
    used_fallback: bool


@dataclass(frozen=True)
class _ProposalChoice:
    token_id: int
    top1_probability: float = 1.0
    top2_probability: float = 0.0


@dataclass(frozen=True)
class _PostCommitHookApplication:
    fixed_tokens: dict[int, int] = field(default_factory=dict)
    rollback_positions: tuple[int, ...] = ()
    position_banned_tokens: dict[int, tuple[int, ...]] = field(default_factory=dict)


def should_enforce_hash_constraint(step: int, total_steps: int, schedule: str) -> bool:
    """Return whether hash-guided positions should be bucket-constrained."""

    if not isinstance(step, int):
        raise TypeError("step must be an int")
    if not isinstance(total_steps, int):
        raise TypeError("total_steps must be an int")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0 or step >= total_steps:
        raise ValueError("step must be in range [0, total_steps)")
    if schedule == HASH_CONSTRAINT_ALWAYS:
        return True
    if schedule == HASH_CONSTRAINT_FINAL_ONLY:
        return step == total_steps - 1
    if schedule == HASH_CONSTRAINT_LATE_HALF:
        return step >= total_steps // 2
    schedules = ", ".join(sorted(VALID_HASH_CONSTRAINT_SCHEDULES))
    raise ValueError(f"hash_constraint_schedule must be one of: {schedules}")


def decode_masked_diffusion(
    *,
    model: object,
    plan: ReconstructionPlan,
    config: DiffusionDecodingConfig,
    token_hash_map: TokenHashMap | None = None,
    prompt_token_ids: Sequence[int] | None = None,
    candidate_filter: object | None = None,
    post_commit_hook: object | None = None,
) -> DecodingResult:
    """Decode erased positions with hard fixed-token and hash constraints."""

    prompt_tokens = tuple(prompt_token_ids or ())
    for token_id in prompt_tokens:
        config._validate_token_id(token_id, "prompt_token_ids")

    if plan.hash_guided_count and token_hash_map is None:
        raise ValueError("token_hash_map is required for hash-guided reconstruction plans")
    if token_hash_map is not None and token_hash_map.vocab_size != config.vocab_size:
        raise ValueError("token_hash_map vocab_size must match decoding config vocab_size")
    post_commit_hook_available = callable(post_commit_hook)
    if post_commit_hook is not None and not post_commit_hook_available:
        raise TypeError("post_commit_hook must be callable when set")
    if (
        post_commit_hook_available
        and config.editable_update_mode != EDITABLE_UPDATE_COMMIT_ONCE
    ):
        raise ValueError("post_commit_hook is only supported for commit_once decoding")

    start_time = perf_counter()
    masks = build_constraint_masks(plan)
    prompt_length = len(prompt_tokens)
    x = _build_initial_input_ids(plan, config, prompt_tokens)
    attention_mask = [1] * len(x)
    fixed_token_ids = _build_fixed_token_ids(plan, prompt_tokens, prompt_length)
    hook_fixed_token_positions: set[int] = set()
    position_banned_tokens: dict[int, set[int]] = {}
    non_banned_token_ids = tuple(
        token_id for token_id in range(config.vocab_size)
        if token_id not in config.special_token_ids
    )
    if not non_banned_token_ids:
        raise ValueError("at least one non-banned token ID is required")

    stats_by_position: dict[int, ConfidenceStat] = {
        entry.position: fixed_confidence_stat(entry)
        for entry in plan.entries
        if entry.state == STATE_KNOWN
    }
    restored_positions: set[int] = set()
    fallback_positions: set[int] = set()
    fallback_reasons: dict[int, str] = {}
    step_summaries = []
    propose_token = getattr(model, "propose_token", None)
    proposal_interface_available = callable(propose_token)
    proposal_calls = 0
    candidate_filter_available = callable(candidate_filter)
    candidate_filter_calls = 0
    candidate_filter_rejections = 0

    if masks.editable_count == 0:
        reconstructed_tokens = tuple(x[prompt_length:])
        no_edit_latency = perf_counter() - start_time
        return DecodingResult(
            reconstructed_text=_decode_tokens(model, reconstructed_tokens),
            reconstructed_tokens=reconstructed_tokens,
            decode_latency_sec=no_edit_latency,
            steps=0,
            fixed_token_count=masks.fixed_count,
            editable_token_count=0,
            hash_guided_token_count=0,
            confidence_stats=tuple(stats_by_position[position] for position in range(plan.total_tokens)),
            step_summaries=(),
            diagnostics={
                "config": config.to_dict(),
                "prompt_token_count": prompt_length,
                "model_forward_calls": 0,
                "model_proposal_calls": 0,
                "decoder_proposal_mode": (
                    "model_propose_token" if proposal_interface_available else "logits"
                ),
                "proposal_interface_available": proposal_interface_available,
                "proposal_interface_used": False,
                "candidate_filter_available": candidate_filter_available,
                "candidate_filter_used": False,
                "candidate_filter_calls": 0,
                "candidate_filter_rejections": 0,
                "candidate_filter_diagnostics": _candidate_filter_diagnostics(candidate_filter),
                "mean_candidate_count": 0.0,
                "max_candidate_count": 0,
                "model_forward_time_sec": 0.0,
                "candidate_construction_time_sec": 0.0,
                "parity_candidate_filter_time_sec": 0.0,
                "xor_peel_time_sec": 0.0,
                "linear_solver_time_sec": 0.0,
                "post_commit_hook_time_sec": 0.0,
                "rollback_time_sec": 0.0,
                "total_decode_time_sec": no_edit_latency,
                "step_diagnostics": (),
                "post_commit_hook_available": post_commit_hook_available,
                "post_commit_hook_used": False,
                "post_commit_hook_calls": 0,
                "post_commit_fixed_positions_per_step": [],
                "post_commit_rollback_positions_per_step": [],
                "post_commit_hook_diagnostics": _post_commit_hook_diagnostics(post_commit_hook),
                "position_banned_token_count": 0,
                "position_banned_tokens_by_position": {},
                "editable_update_mode": config.editable_update_mode,
                "hash_constraint_schedule": config.hash_constraint_schedule,
                "hash_enforced_steps": [],
                "hash_relaxed_steps": [],
                "updated_editable_positions_per_step": [],
                "hash_bucket_empty_positions": [],
                "fallback_positions": [],
                "fallback_reasons": {},
                "restored_fixed_positions": [],
                "restoration_count": 0,
            },
        )

    forward_calls = 0
    hash_enforced_steps: list[int] = []
    hash_relaxed_steps: list[int] = []
    updated_editable_positions_per_step: list[int] = []
    post_commit_hook_calls = 0
    post_commit_fixed_positions_per_step: list[int] = []
    post_commit_rollback_positions_per_step: list[int] = []
    step_diagnostics: list[dict[str, Any]] = []
    candidate_count_sum = 0
    candidate_count_seen = 0
    candidate_count_max = 0
    candidate_construction_time_sec = 0.0
    model_forward_time_sec = 0.0
    post_commit_hook_time_sec = 0.0
    rollback_enabled = bool(getattr(post_commit_hook, "rollback_enabled", False))
    rollback_extra_steps = _non_negative_int_attr(
        post_commit_hook,
        "rollback_extra_steps",
        default=0,
    )
    rollback_max_total_steps = _positive_int_attr(
        post_commit_hook,
        "rollback_max_total_steps",
        default=config.steps + rollback_extra_steps,
    )
    rollback_total_steps = config.steps
    if rollback_enabled:
        rollback_total_steps = min(config.steps + rollback_extra_steps, rollback_max_total_steps)
        if rollback_total_steps < config.steps:
            rollback_total_steps = config.steps
    rollback_stop_after_no_progress = _non_negative_int_attr(
        post_commit_hook,
        "rollback_stop_after_no_progress",
        default=0,
    )
    rollback_no_progress_stop = False

    if config.editable_update_mode == EDITABLE_UPDATE_COMMIT_ONCE:
        no_progress_streak = 0
        for step in range(rollback_total_steps):
            editable_target_positions = _still_masked_target_positions(
                x,
                plan,
                prompt_length,
                config.mask_token_id,
            )
            if not editable_target_positions:
                break

            remaining_masks_before = len(editable_target_positions)
            step_filter_calls_before = candidate_filter_calls
            step_filter_rejections_before = int(
                _candidate_filter_diagnostics(candidate_filter).get(
                    "parity_candidate_rejections",
                    0,
                )
                or 0
            )
            hash_guided_editable_count, unguided_editable_count = _editable_state_counts(
                plan=plan,
                target_positions=editable_target_positions,
            )
            hash_enforced_steps.append(step)
            remaining_steps = max(1, rollback_total_steps - step)
            if step < config.steps:
                remaining_steps = config.steps - step
            transfer_count = max(1, ceil(len(editable_target_positions) / remaining_steps))

            proposal_start = perf_counter()
            if proposal_interface_available:
                proposals = []
                for target_position in editable_target_positions:
                    proposals.append(
                        _proposal_from_model_hook(
                            propose_token=propose_token,
                            entry=plan.entries[target_position],
                            full_position=prompt_length + target_position,
                            input_ids=tuple(x),
                            step=step,
                            config=config,
                            token_hash_map=token_hash_map,
                            non_banned_token_ids=non_banned_token_ids,
                            enforce_hash_constraint=True,
                            position_banned_tokens=position_banned_tokens,
                            candidate_filter=candidate_filter,
                        )
                    )
                    proposal_calls += 1
                    if candidate_filter_available:
                        candidate_filter_calls += 1
            else:
                forward_start = perf_counter()
                logits = _extract_logits(
                    model.forward([list(x)], attention_mask=[list(attention_mask)]),
                    sequence_length=len(x),
                    vocab_size=config.vocab_size,
                )
                model_forward_time_sec += perf_counter() - forward_start
                forward_calls += 1
                proposal_start = perf_counter()
                proposals = [
                    _proposal_for_position(
                        entry=plan.entries[target_position],
                        full_position=prompt_length + target_position,
                        position_logits=logits[prompt_length + target_position],
                        config=config,
                        token_hash_map=token_hash_map,
                        non_banned_token_ids=non_banned_token_ids,
                        enforce_hash_constraint=True,
                        input_ids=tuple(x),
                        step=step,
                        position_banned_tokens=position_banned_tokens,
                        candidate_filter=candidate_filter,
                    )
                    for target_position in editable_target_positions
                ]
                if candidate_filter_available:
                    candidate_filter_calls += len(editable_target_positions)
            candidate_construction_time_sec += perf_counter() - proposal_start
            step_candidate_counts = [proposal.candidate_count for proposal in proposals]
            if step_candidate_counts:
                candidate_count_sum += sum(step_candidate_counts)
                candidate_count_seen += len(step_candidate_counts)
                candidate_count_max = max(candidate_count_max, max(step_candidate_counts))
            proposals.sort(
                key=lambda proposal: (
                    -proposal.top1_probability,
                    proposal.target_position,
                )
            )
            committed = proposals[:transfer_count]

            committed_stats: list[ConfidenceStat] = []
            step_fallback_positions: list[int] = []
            for proposal in committed:
                x[proposal.full_position] = proposal.token_id
                entry = plan.entries[proposal.target_position]
                stat = editable_confidence_stat(
                    entry,
                    selected_token_id=proposal.token_id,
                    top1_probability=proposal.top1_probability,
                    top2_probability=proposal.top2_probability,
                    candidate_count=proposal.candidate_count,
                    commit_step=step,
                )
                stats_by_position[proposal.target_position] = stat
                committed_stats.append(stat)
                if proposal.used_fallback:
                    fallback_positions.add(proposal.target_position)
                    step_fallback_positions.append(proposal.target_position)
                    fallback_reasons[proposal.target_position] = "hash_bucket_empty_after_bans"

            post_commit_stats: list[ConfidenceStat] = []
            hook_result = _PostCommitHookApplication()
            if post_commit_hook_available:
                post_commit_hook_calls += 1
                hook_start = perf_counter()
                hook_result = _apply_post_commit_hook(
                    post_commit_hook=post_commit_hook,
                    x=x,
                    plan=plan,
                    config=config,
                    prompt_length=prompt_length,
                    fixed_token_ids=fixed_token_ids,
                    hook_fixed_token_positions=hook_fixed_token_positions,
                    position_banned_tokens=position_banned_tokens,
                    step=step,
                    committed_positions=tuple(
                        proposal.target_position
                        for proposal in committed
                    ),
                )
                post_commit_hook_time_sec += perf_counter() - hook_start
                post_commit_fixed_positions_per_step.append(len(hook_result.fixed_tokens))
                post_commit_rollback_positions_per_step.append(len(hook_result.rollback_positions))
                for target_position in hook_result.rollback_positions:
                    stats_by_position.pop(target_position, None)
                for target_position, token_id in hook_result.fixed_tokens.items():
                    entry = plan.entries[target_position]
                    stat = editable_confidence_stat(
                        entry,
                        selected_token_id=token_id,
                        top1_probability=1.0,
                        top2_probability=0.0,
                        candidate_count=1,
                        commit_step=step,
                    )
                    stats_by_position[target_position] = stat
                    post_commit_stats.append(stat)
            else:
                post_commit_fixed_positions_per_step.append(0)
                post_commit_rollback_positions_per_step.append(0)

            updated_editable_positions_per_step.append(
                len(committed) + len(post_commit_stats)
            )
            step_restored = _restore_fixed_tokens(x, fixed_token_ids)
            restored_positions.update(position - prompt_length for position in step_restored if position >= prompt_length)
            still_masked_count = len(
                _still_masked_target_positions(x, plan, prompt_length, config.mask_token_id)
            )
            step_filter_diagnostics = _candidate_filter_diagnostics(candidate_filter)
            step_filter_rejections_after = int(
                step_filter_diagnostics.get("parity_candidate_rejections", 0) or 0
            )
            step_hook_diagnostics = _post_commit_hook_last_step_diagnostics(post_commit_hook)
            step_diagnostics.append(
                {
                    "step": step,
                    "remaining_masks_before": remaining_masks_before,
                    "remaining_masks_after": still_masked_count,
                    "model_committed_count": len(committed),
                    "parity_peeled_count": len(hook_result.fixed_tokens),
                    "rollback_count": len(hook_result.rollback_positions),
                    "candidate_filter_calls": candidate_filter_calls - step_filter_calls_before,
                    "candidate_filter_rejections": (
                        step_filter_rejections_after - step_filter_rejections_before
                    ),
                    "mean_candidate_count": (
                        sum(step_candidate_counts) / len(step_candidate_counts)
                        if step_candidate_counts
                        else 0.0
                    ),
                    "max_candidate_count": max(step_candidate_counts) if step_candidate_counts else 0,
                    "hash_guided_editable_count": hash_guided_editable_count,
                    "unguided_editable_count": unguided_editable_count,
                    "active_parity_equation_count": step_hook_diagnostics.get(
                        "active_parity_equation_count",
                        step_filter_diagnostics.get("parity_candidate_filter_equation_count", 0),
                    ),
                    "linear_solver_components_seen": step_hook_diagnostics.get(
                        "linear_solver_components_seen",
                        0,
                    ),
                    "linear_solver_components_solved": step_hook_diagnostics.get(
                        "linear_solver_components_solved",
                        0,
                    ),
                    "linear_solver_rank_deficient_count": step_hook_diagnostics.get(
                        "linear_solver_rank_deficient_count",
                        0,
                    ),
                    "linear_solver_too_large_count": step_hook_diagnostics.get(
                        "linear_solver_too_large_count",
                        0,
                    ),
                    "final_masks": still_masked_count,
                }
            )
            step_summaries.append(
                step_summary(
                    step=step,
                    still_masked_count=still_masked_count,
                    committed_stats=[*committed_stats, *post_commit_stats],
                    fallback_positions=step_fallback_positions,
                    restored_fixed_positions=[
                        position - prompt_length for position in step_restored if position >= prompt_length
                    ],
                )
            )
            if rollback_enabled and step >= config.steps:
                if still_masked_count >= len(editable_target_positions):
                    no_progress_streak += 1
                else:
                    no_progress_streak = 0
                if (
                    rollback_stop_after_no_progress
                    and no_progress_streak >= rollback_stop_after_no_progress
                ):
                    rollback_no_progress_stop = True
                    break
    else:
        editable_target_positions = _still_masked_target_positions(
            x,
            plan,
            prompt_length,
            config.mask_token_id,
        )
        for step in range(config.steps):
            enforce_hash = should_enforce_hash_constraint(
                step,
                config.steps,
                config.hash_constraint_schedule,
            )
            if enforce_hash:
                hash_enforced_steps.append(step)
            else:
                hash_relaxed_steps.append(step)

            remaining_masks_before = len(editable_target_positions)
            step_filter_calls_before = candidate_filter_calls
            step_filter_rejections_before = int(
                _candidate_filter_diagnostics(candidate_filter).get(
                    "parity_candidate_rejections",
                    0,
                )
                or 0
            )
            hash_guided_editable_count, unguided_editable_count = _editable_state_counts(
                plan=plan,
                target_positions=editable_target_positions,
            )
            proposal_start = perf_counter()
            if proposal_interface_available:
                proposals = []
                for target_position in editable_target_positions:
                    proposals.append(
                        _proposal_from_model_hook(
                            propose_token=propose_token,
                            entry=plan.entries[target_position],
                            full_position=prompt_length + target_position,
                            input_ids=tuple(x),
                            step=step,
                            config=config,
                            token_hash_map=token_hash_map,
                            non_banned_token_ids=non_banned_token_ids,
                            enforce_hash_constraint=enforce_hash,
                            position_banned_tokens=position_banned_tokens,
                            candidate_filter=candidate_filter,
                        )
                    )
                    proposal_calls += 1
                    if candidate_filter_available:
                        candidate_filter_calls += 1
            else:
                forward_start = perf_counter()
                logits = _extract_logits(
                    model.forward([list(x)], attention_mask=[list(attention_mask)]),
                    sequence_length=len(x),
                    vocab_size=config.vocab_size,
                )
                model_forward_time_sec += perf_counter() - forward_start
                forward_calls += 1
                proposal_start = perf_counter()
                proposals = [
                    _proposal_for_position(
                        entry=plan.entries[target_position],
                        full_position=prompt_length + target_position,
                        position_logits=logits[prompt_length + target_position],
                        config=config,
                        token_hash_map=token_hash_map,
                        non_banned_token_ids=non_banned_token_ids,
                        enforce_hash_constraint=enforce_hash,
                        input_ids=tuple(x),
                        step=step,
                        position_banned_tokens=position_banned_tokens,
                        candidate_filter=candidate_filter,
                    )
                    for target_position in editable_target_positions
                ]
                if candidate_filter_available:
                    candidate_filter_calls += len(editable_target_positions)
            candidate_construction_time_sec += perf_counter() - proposal_start
            step_candidate_counts = [proposal.candidate_count for proposal in proposals]
            if step_candidate_counts:
                candidate_count_sum += sum(step_candidate_counts)
                candidate_count_seen += len(step_candidate_counts)
                candidate_count_max = max(candidate_count_max, max(step_candidate_counts))

            committed_stats = []
            step_fallback_positions = []
            for proposal in proposals:
                x[proposal.full_position] = proposal.token_id
                entry = plan.entries[proposal.target_position]
                stat = editable_confidence_stat(
                    entry,
                    selected_token_id=proposal.token_id,
                    top1_probability=proposal.top1_probability,
                    top2_probability=proposal.top2_probability,
                    candidate_count=proposal.candidate_count,
                    commit_step=step,
                )
                stats_by_position[proposal.target_position] = stat
                committed_stats.append(stat)
                if proposal.used_fallback:
                    fallback_positions.add(proposal.target_position)
                    step_fallback_positions.append(proposal.target_position)
                    fallback_reasons[proposal.target_position] = "hash_bucket_empty_after_bans"

            updated_editable_positions_per_step.append(len(proposals))
            post_commit_fixed_positions_per_step.append(0)
            post_commit_rollback_positions_per_step.append(0)
            step_restored = _restore_fixed_tokens(x, fixed_token_ids)
            restored_positions.update(position - prompt_length for position in step_restored if position >= prompt_length)
            still_masked_count = len(
                _still_masked_target_positions(x, plan, prompt_length, config.mask_token_id)
            )
            step_filter_diagnostics = _candidate_filter_diagnostics(candidate_filter)
            step_filter_rejections_after = int(
                step_filter_diagnostics.get("parity_candidate_rejections", 0) or 0
            )
            step_diagnostics.append(
                {
                    "step": step,
                    "remaining_masks_before": remaining_masks_before,
                    "remaining_masks_after": still_masked_count,
                    "model_committed_count": len(proposals),
                    "parity_peeled_count": 0,
                    "rollback_count": 0,
                    "candidate_filter_calls": candidate_filter_calls - step_filter_calls_before,
                    "candidate_filter_rejections": (
                        step_filter_rejections_after - step_filter_rejections_before
                    ),
                    "mean_candidate_count": (
                        sum(step_candidate_counts) / len(step_candidate_counts)
                        if step_candidate_counts
                        else 0.0
                    ),
                    "max_candidate_count": max(step_candidate_counts) if step_candidate_counts else 0,
                    "hash_guided_editable_count": hash_guided_editable_count,
                    "unguided_editable_count": unguided_editable_count,
                    "active_parity_equation_count": step_filter_diagnostics.get(
                        "parity_candidate_filter_equation_count",
                        0,
                    ),
                    "linear_solver_components_seen": 0,
                    "linear_solver_components_solved": 0,
                    "linear_solver_rank_deficient_count": 0,
                    "linear_solver_too_large_count": 0,
                    "final_masks": still_masked_count,
                }
            )
            step_summaries.append(
                step_summary(
                    step=step,
                    still_masked_count=still_masked_count,
                    committed_stats=committed_stats,
                    fallback_positions=step_fallback_positions,
                    restored_fixed_positions=[
                        position - prompt_length for position in step_restored if position >= prompt_length
                    ],
                )
            )

    remaining_masked = _still_masked_target_positions(
        x,
        plan,
        prompt_length,
        config.mask_token_id,
    )
    if remaining_masked:
        if not rollback_enabled:
            raise RuntimeError(f"decoder left masked target positions unfilled: {remaining_masked}")
    _validate_fixed_tokens_preserved(x, fixed_token_ids)
    if (
        config.editable_update_mode == EDITABLE_UPDATE_RESAMPLE_EACH_STEP
        and should_enforce_hash_constraint(
            config.steps - 1,
            config.steps,
            config.hash_constraint_schedule,
        )
    ):
        _validate_final_hash_constraints(
            x=x,
            plan=plan,
            prompt_length=prompt_length,
            config=config,
            token_hash_map=token_hash_map,
            fallback_positions=fallback_positions,
        )

    reconstructed_tokens = tuple(x[prompt_length:])
    for position in range(plan.total_tokens):
        if position not in stats_by_position:
            entry = plan.entries[position]
            stats_by_position[position] = ConfidenceStat(
                position=entry.position,
                state=entry.state,
                selected_token_id=x[prompt_length + position],
                top1_probability=0.0,
                top2_probability=0.0,
                margin=0.0,
                candidate_count=0,
                commit_step=None,
                was_fixed=False,
                hash_value=entry.hash_value,
            )
    _record_post_commit_hook_decode_outcome(
        post_commit_hook,
        extra_steps_used=max(0, len(step_summaries) - config.steps),
        remaining_masked_count=len(remaining_masked),
        no_progress_stop=rollback_no_progress_stop,
    )
    ordered_stats = tuple(stats_by_position[position] for position in range(plan.total_tokens))
    candidate_filter_diagnostics = _candidate_filter_diagnostics(candidate_filter)
    post_commit_hook_diagnostics = _post_commit_hook_diagnostics(post_commit_hook)
    total_decode_time_sec = perf_counter() - start_time
    candidate_filter_rejections = int(
        candidate_filter_diagnostics.get(
            "parity_candidate_rejections",
            candidate_filter_rejections,
        )
        or 0
    )
    return DecodingResult(
        reconstructed_text=_decode_tokens(model, reconstructed_tokens),
        reconstructed_tokens=reconstructed_tokens,
        decode_latency_sec=total_decode_time_sec,
        steps=len(step_summaries),
        fixed_token_count=masks.fixed_count,
        editable_token_count=masks.editable_count,
        hash_guided_token_count=masks.hash_guided_count,
        confidence_stats=ordered_stats,
        step_summaries=tuple(step_summaries),
        diagnostics={
            "config": config.to_dict(),
            "prompt_token_count": prompt_length,
            "model_forward_calls": forward_calls,
            "model_proposal_calls": proposal_calls,
            "decoder_proposal_mode": (
                "model_propose_token" if proposal_interface_available else "logits"
            ),
            "proposal_interface_available": proposal_interface_available,
            "proposal_interface_used": proposal_calls > 0,
            "candidate_filter_available": candidate_filter_available,
            "candidate_filter_used": candidate_filter_calls > 0,
            "candidate_filter_calls": candidate_filter_calls,
            "candidate_filter_rejections": candidate_filter_rejections,
            "candidate_filter_diagnostics": candidate_filter_diagnostics,
            "mean_candidate_count": (
                candidate_count_sum / candidate_count_seen if candidate_count_seen else 0.0
            ),
            "max_candidate_count": candidate_count_max,
            "model_forward_time_sec": model_forward_time_sec,
            "candidate_construction_time_sec": candidate_construction_time_sec,
            "parity_candidate_filter_time_sec": float(
                candidate_filter_diagnostics.get("parity_filter_time_sec", 0.0) or 0.0
            ),
            "xor_peel_time_sec": float(
                post_commit_hook_diagnostics.get("iterative_xor_peel_time_sec", 0.0)
                or 0.0
            ),
            "linear_solver_time_sec": float(
                post_commit_hook_diagnostics.get("iterative_linear_solver_time_sec", 0.0)
                or 0.0
            ),
            "post_commit_hook_time_sec": post_commit_hook_time_sec,
            "rollback_time_sec": float(
                post_commit_hook_diagnostics.get("rollback_time_sec", 0.0) or 0.0
            ),
            "total_decode_time_sec": total_decode_time_sec,
            "step_diagnostics": tuple(step_diagnostics),
            "post_commit_hook_available": post_commit_hook_available,
            "post_commit_hook_used": post_commit_hook_calls > 0,
            "post_commit_hook_calls": post_commit_hook_calls,
            "post_commit_fixed_positions_per_step": tuple(post_commit_fixed_positions_per_step),
            "post_commit_rollback_positions_per_step": tuple(post_commit_rollback_positions_per_step),
            "post_commit_hook_diagnostics": post_commit_hook_diagnostics,
            "position_banned_token_count": sum(len(tokens) for tokens in position_banned_tokens.values()),
            "position_banned_tokens_by_position": {
                position: tuple(sorted(tokens))
                for position, tokens in sorted(position_banned_tokens.items())
            },
            "editable_update_mode": config.editable_update_mode,
            "hash_constraint_schedule": config.hash_constraint_schedule,
            "hash_enforced_steps": tuple(hash_enforced_steps),
            "hash_relaxed_steps": tuple(hash_relaxed_steps),
            "updated_editable_positions_per_step": tuple(updated_editable_positions_per_step),
            "hash_bucket_empty_positions": sorted(fallback_positions),
            "fallback_positions": sorted(fallback_positions),
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "restored_fixed_positions": sorted(restored_positions),
            "restoration_count": len(restored_positions),
        },
    )


def _build_initial_input_ids(
    plan: ReconstructionPlan,
    config: DiffusionDecodingConfig,
    prompt_tokens: tuple[int, ...],
) -> list[int]:
    x = list(prompt_tokens)
    for entry in plan.entries:
        if entry.fixed:
            if entry.token_id is None:
                raise ValueError("fixed entries require token_id")
            x.append(entry.token_id)
        else:
            x.append(config.mask_token_id)
    return x


def _build_fixed_token_ids(
    plan: ReconstructionPlan,
    prompt_tokens: tuple[int, ...],
    prompt_length: int,
) -> dict[int, int]:
    fixed_token_ids = {position: token_id for position, token_id in enumerate(prompt_tokens)}
    for entry in plan.entries:
        if entry.fixed:
            if entry.token_id is None:
                raise ValueError("fixed entries require token_id")
            fixed_token_ids[prompt_length + entry.position] = entry.token_id
    return fixed_token_ids


def _still_masked_target_positions(
    x: Sequence[int],
    plan: ReconstructionPlan,
    prompt_length: int,
    mask_token_id: int,
) -> list[int]:
    return [
        entry.position
        for entry in plan.entries
        if not entry.fixed and x[prompt_length + entry.position] == mask_token_id
    ]


def _editable_state_counts(
    *,
    plan: ReconstructionPlan,
    target_positions: Sequence[int],
) -> tuple[int, int]:
    hash_guided = 0
    unguided = 0
    for target_position in target_positions:
        entry = plan.entries[int(target_position)]
        if entry.state == STATE_MISSING:
            hash_guided += 1
        else:
            unguided += 1
    return hash_guided, unguided


def _extract_logits(
    output: object,
    *,
    sequence_length: int,
    vocab_size: int,
) -> Sequence[Sequence[float]]:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None:
        raise TypeError("model.forward output must expose logits")
    if len(logits) != 1:
        raise ValueError("decoder currently expects batch size 1 logits")
    batch_logits = logits[0]
    if len(batch_logits) != sequence_length:
        raise ValueError("logits sequence length does not match input_ids")
    for position, position_logits in enumerate(batch_logits):
        if len(position_logits) != vocab_size:
            raise ValueError(f"logits at position {position} do not match vocab_size")
    return batch_logits


def _validate_fixed_tokens_preserved(x: Sequence[int], fixed_token_ids: dict[int, int]) -> None:
    changed = [
        position
        for position, token_id in fixed_token_ids.items()
        if x[position] != token_id
    ]
    if changed:
        raise RuntimeError(f"decoder changed fixed token positions: {changed}")


def _non_negative_int_attr(obj: object | None, name: str, *, default: int) -> int:
    value = int(getattr(obj, name, default))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_int_attr(obj: object | None, name: str, *, default: int) -> int:
    value = int(getattr(obj, name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_final_hash_constraints(
    *,
    x: Sequence[int],
    plan: ReconstructionPlan,
    prompt_length: int,
    config: DiffusionDecodingConfig,
    token_hash_map: TokenHashMap | None,
    fallback_positions: set[int],
) -> None:
    if plan.hash_guided_count and token_hash_map is None:
        raise ValueError("token_hash_map is required for hash-guided reconstruction plans")
    if token_hash_map is None:
        return
    violations = []
    for entry in plan.entries:
        if entry.state != STATE_MISSING or entry.position in fallback_positions:
            continue
        token_id = x[prompt_length + entry.position]
        if token_id in config.special_token_ids:
            violations.append(entry.position)
            continue
        if entry.hash_value is None or token_hash_map.bucket_for_token(token_id) != entry.hash_value:
            violations.append(entry.position)
    if violations:
        raise RuntimeError(f"decoder produced non-hash-legal final positions: {violations}")


def _proposal_for_position(
    *,
    entry: ReconstructionEntry,
    full_position: int,
    position_logits: Sequence[float],
    config: DiffusionDecodingConfig,
    token_hash_map: TokenHashMap | None,
    non_banned_token_ids: tuple[int, ...],
    enforce_hash_constraint: bool,
    input_ids: tuple[int, ...],
    step: int,
    position_banned_tokens: dict[int, set[int]],
    candidate_filter: object | None,
) -> _Proposal:
    candidates, used_fallback = _candidate_token_ids(
        entry=entry,
        config=config,
        token_hash_map=token_hash_map,
        non_banned_token_ids=non_banned_token_ids,
        enforce_hash_constraint=enforce_hash_constraint,
    )
    candidates = _apply_position_bans(
        entry=entry,
        candidate_token_ids=candidates,
        position_banned_tokens=position_banned_tokens,
    )
    candidates = _apply_candidate_filter(
        candidate_filter=candidate_filter,
        entry=entry,
        full_position=full_position,
        candidate_token_ids=candidates,
        input_ids=input_ids,
        step=step,
    )
    token_id, top1_probability, top2_probability = _select_argmax_with_confidence(
        position_logits,
        candidates,
    )
    return _Proposal(
        full_position=full_position,
        target_position=entry.position,
        token_id=token_id,
        top1_probability=top1_probability,
        top2_probability=top2_probability,
        candidate_count=len(candidates),
        used_fallback=used_fallback,
    )


def _proposal_from_model_hook(
    *,
    propose_token,
    entry: ReconstructionEntry,
    full_position: int,
    input_ids: tuple[int, ...],
    step: int,
    config: DiffusionDecodingConfig,
    token_hash_map: TokenHashMap | None,
    non_banned_token_ids: tuple[int, ...],
    enforce_hash_constraint: bool,
    position_banned_tokens: dict[int, set[int]],
    candidate_filter: object | None,
) -> _Proposal:
    candidates, used_fallback = _candidate_token_ids(
        entry=entry,
        config=config,
        token_hash_map=token_hash_map,
        non_banned_token_ids=non_banned_token_ids,
        enforce_hash_constraint=enforce_hash_constraint,
    )
    candidates = _apply_position_bans(
        entry=entry,
        candidate_token_ids=candidates,
        position_banned_tokens=position_banned_tokens,
    )
    candidates = _apply_candidate_filter(
        candidate_filter=candidate_filter,
        entry=entry,
        full_position=full_position,
        candidate_token_ids=candidates,
        input_ids=input_ids,
        step=step,
    )
    choice = _normalize_proposal_choice(
        propose_token(
            position=entry.position,
            full_position=full_position,
            candidate_token_ids=candidates,
            input_ids=input_ids,
            step=step,
        )
    )
    if choice.token_id not in candidates:
        raise ValueError("propose_token must return a token_id from candidate_token_ids")
    return _Proposal(
        full_position=full_position,
        target_position=entry.position,
        token_id=choice.token_id,
        top1_probability=choice.top1_probability,
        top2_probability=choice.top2_probability,
        candidate_count=len(candidates),
        used_fallback=used_fallback,
    )


def _normalize_proposal_choice(value: Any) -> _ProposalChoice:
    if isinstance(value, int):
        return _ProposalChoice(token_id=value)
    if isinstance(value, dict):
        token_id = value.get("token_id", value.get("selected_token_id"))
        if token_id is None:
            raise ValueError("propose_token dict result requires token_id")
        return _ProposalChoice(
            token_id=int(token_id),
            top1_probability=float(value.get("top1_probability", 1.0)),
            top2_probability=float(value.get("top2_probability", 0.0)),
        )
    token_id = getattr(value, "token_id", None)
    if token_id is None:
        raise TypeError("propose_token must return an int, dict, or object with token_id")
    return _ProposalChoice(
        token_id=int(token_id),
        top1_probability=float(getattr(value, "top1_probability", 1.0)),
        top2_probability=float(getattr(value, "top2_probability", 0.0)),
    )


def _apply_candidate_filter(
    *,
    candidate_filter: object | None,
    entry: ReconstructionEntry,
    full_position: int,
    candidate_token_ids: tuple[int, ...],
    input_ids: tuple[int, ...],
    step: int,
) -> tuple[int, ...]:
    if candidate_filter is None:
        return candidate_token_ids
    if not callable(candidate_filter):
        raise TypeError("candidate_filter must be callable when set")

    result = candidate_filter(
        entry=entry,
        candidate_token_ids=candidate_token_ids,
        input_ids=input_ids,
        step=step,
        full_position=full_position,
    )
    if isinstance(result, dict):
        result = result.get("candidate_token_ids", result.get("candidates"))
    if result is None:
        raise ValueError("candidate_filter must return candidate_token_ids")
    filtered = tuple(int(token_id) for token_id in result)
    original = set(candidate_token_ids)
    outside = [token_id for token_id in filtered if token_id not in original]
    if outside:
        raise ValueError("candidate_filter must return a subset of candidate_token_ids")
    if not filtered:
        raise RuntimeError("candidate_filter returned no candidate token IDs")
    return filtered


def _apply_position_bans(
    *,
    entry: ReconstructionEntry,
    candidate_token_ids: tuple[int, ...],
    position_banned_tokens: dict[int, set[int]],
) -> tuple[int, ...]:
    banned = position_banned_tokens.get(entry.position)
    if not banned:
        return candidate_token_ids
    filtered = tuple(token_id for token_id in candidate_token_ids if token_id not in banned)
    if not filtered:
        raise RuntimeError(
            f"position-specific bans removed all candidates for position {entry.position}"
        )
    return filtered


def _candidate_filter_diagnostics(candidate_filter: object | None) -> dict[str, Any]:
    if candidate_filter is None:
        return {}
    diagnostics = getattr(candidate_filter, "diagnostics", None)
    if callable(diagnostics):
        value = diagnostics()
        if not isinstance(value, dict):
            raise TypeError("candidate_filter.diagnostics() must return a dict")
        return dict(value)
    value = getattr(candidate_filter, "diagnostics", None)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _post_commit_hook_diagnostics(post_commit_hook: object | None) -> dict[str, Any]:
    if post_commit_hook is None:
        return {}
    diagnostics = getattr(post_commit_hook, "diagnostics", None)
    if callable(diagnostics):
        value = diagnostics()
        if not isinstance(value, dict):
            raise TypeError("post_commit_hook.diagnostics() must return a dict")
        return dict(value)
    value = getattr(post_commit_hook, "diagnostics", None)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _post_commit_hook_last_step_diagnostics(post_commit_hook: object | None) -> dict[str, Any]:
    if post_commit_hook is None:
        return {}
    value = getattr(post_commit_hook, "last_step_diagnostics", None)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _record_post_commit_hook_decode_outcome(
    post_commit_hook: object | None,
    *,
    extra_steps_used: int,
    remaining_masked_count: int,
    no_progress_stop: bool,
) -> None:
    record = getattr(post_commit_hook, "record_decode_outcome", None)
    if callable(record):
        record(
            extra_steps_used=extra_steps_used,
            remaining_masked_count=remaining_masked_count,
            no_progress_stop=no_progress_stop,
        )


def _apply_post_commit_hook(
    *,
    post_commit_hook: object,
    x: list[int],
    plan: ReconstructionPlan,
    config: DiffusionDecodingConfig,
    prompt_length: int,
    fixed_token_ids: dict[int, int],
    hook_fixed_token_positions: set[int],
    position_banned_tokens: dict[int, set[int]],
    step: int,
    committed_positions: tuple[int, ...],
) -> _PostCommitHookApplication:
    if not callable(post_commit_hook):
        raise TypeError("post_commit_hook must be callable")
    result = post_commit_hook(
        input_ids=tuple(x),
        step=step,
        prompt_length=prompt_length,
        committed_positions=committed_positions,
        fixed_token_ids=dict(fixed_token_ids),
        plan=plan,
        config=config,
    )
    if result is None:
        return _PostCommitHookApplication()
    rollback_positions: tuple[int, ...] = ()
    raw_position_bans: dict[Any, Any] = {}
    fixed_tokens = result
    if isinstance(result, dict) and (
        "fixed_tokens" in result
        or "recovered_tokens" in result
        or "rollback_positions" in result
        or "position_banned_tokens" in result
        or "position_bans" in result
    ):
        fixed_tokens = result.get("fixed_tokens", result.get("recovered_tokens"))
        rollback_positions = tuple(int(position) for position in result.get("rollback_positions", ()))
        raw_position_bans = result.get("position_banned_tokens", result.get("position_bans", {})) or {}
    if fixed_tokens is None:
        fixed_tokens = {}
    if not isinstance(fixed_tokens, dict):
        fixed_tokens = dict(fixed_tokens)

    applied_rollbacks: list[int] = []
    for target_position in rollback_positions:
        if target_position < 0 or target_position >= plan.total_tokens:
            raise ValueError("post_commit_hook returned a rollback position outside the plan")
        full_position = prompt_length + target_position
        existing_fixed = fixed_token_ids.get(full_position)
        if existing_fixed is not None and full_position not in hook_fixed_token_positions:
            raise RuntimeError("post_commit_hook attempted to rollback a fixed token")
        if full_position in hook_fixed_token_positions:
            fixed_token_ids.pop(full_position, None)
            hook_fixed_token_positions.discard(full_position)
        if x[full_position] != config.mask_token_id:
            x[full_position] = config.mask_token_id
        applied_rollbacks.append(target_position)

    applied_bans: dict[int, tuple[int, ...]] = {}
    for raw_position, raw_token_ids in raw_position_bans.items():
        target_position = int(raw_position)
        if target_position < 0 or target_position >= plan.total_tokens:
            raise ValueError("post_commit_hook returned a ban position outside the plan")
        full_position = prompt_length + target_position
        existing_fixed = fixed_token_ids.get(full_position)
        if existing_fixed is not None and full_position not in hook_fixed_token_positions:
            raise RuntimeError("post_commit_hook attempted to ban a fixed token")
        if isinstance(raw_token_ids, int):
            token_ids = (raw_token_ids,)
        else:
            token_ids = tuple(raw_token_ids)
        normalized: list[int] = []
        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            config._validate_token_id(token_id, "post_commit_hook banned token_id")
            position_banned_tokens.setdefault(target_position, set()).add(token_id)
            normalized.append(token_id)
        if normalized:
            applied_bans[target_position] = tuple(sorted(set(normalized)))

    applied: dict[int, int] = {}
    for raw_position, raw_token_id in fixed_tokens.items():
        target_position = int(raw_position)
        token_id = int(raw_token_id)
        if target_position < 0 or target_position >= plan.total_tokens:
            raise ValueError("post_commit_hook returned a target position outside the plan")
        full_position = prompt_length + target_position
        existing_fixed = fixed_token_ids.get(full_position)
        if existing_fixed is not None and existing_fixed != token_id:
            raise RuntimeError("post_commit_hook attempted to change a fixed token")
        if x[full_position] != config.mask_token_id:
            if x[full_position] == token_id:
                fixed_token_ids[full_position] = token_id
                hook_fixed_token_positions.add(full_position)
                continue
            raise RuntimeError("post_commit_hook attempted to overwrite a committed token")
        config._validate_token_id(token_id, "post_commit_hook token_id")
        if token_id in config.special_token_ids:
            raise RuntimeError("post_commit_hook returned a banned or special token")
        x[full_position] = token_id
        fixed_token_ids[full_position] = token_id
        hook_fixed_token_positions.add(full_position)
        applied[target_position] = token_id
    return _PostCommitHookApplication(
        fixed_tokens=applied,
        rollback_positions=tuple(sorted(set(applied_rollbacks))),
        position_banned_tokens=applied_bans,
    )


def _candidate_token_ids(
    *,
    entry: ReconstructionEntry,
    config: DiffusionDecodingConfig,
    token_hash_map: TokenHashMap | None,
    non_banned_token_ids: tuple[int, ...],
    enforce_hash_constraint: bool,
) -> tuple[tuple[int, ...], bool]:
    if entry.state != STATE_MISSING:
        return non_banned_token_ids, False
    if not enforce_hash_constraint:
        return non_banned_token_ids, False
    if token_hash_map is None:
        raise ValueError("token_hash_map is required for hash-guided entries")
    if entry.hash_value is None:
        raise ValueError("hash-guided entries require hash_value")

    token_hash_map.validate_bucket_id(entry.hash_value)
    candidates = tuple(
        token_id for token_id in token_hash_map.candidate_token_ids(entry.hash_value)
        if token_id not in config.special_token_ids
    )
    if candidates:
        return candidates, False
    if not config.fallback_on_empty_hash_bucket:
        raise RuntimeError(f"hash bucket for position {entry.position} has no allowed tokens")
    return non_banned_token_ids, True


def _select_argmax_with_confidence(
    position_logits: Sequence[float],
    candidate_token_ids: tuple[int, ...],
) -> tuple[int, float, float]:
    if not candidate_token_ids:
        raise RuntimeError("cannot select from an empty candidate set")

    tensor_result = _select_argmax_with_confidence_tensor(
        position_logits,
        candidate_token_ids,
    )
    if tensor_result is not None:
        return tensor_result

    best_token_id = candidate_token_ids[0]
    best_logit = float(position_logits[best_token_id])
    second_logit: float | None = None
    for token_id in candidate_token_ids[1:]:
        logit = float(position_logits[token_id])
        if logit > best_logit or (logit == best_logit and token_id < best_token_id):
            second_logit = best_logit
            best_token_id = token_id
            best_logit = logit
        elif second_logit is None or logit > second_logit:
            second_logit = logit

    max_logit = max(float(position_logits[token_id]) for token_id in candidate_token_ids)
    exp_values = [
        exp(float(position_logits[token_id]) - max_logit)
        for token_id in candidate_token_ids
    ]
    denominator = sum(exp_values)
    best_probability = exp(best_logit - max_logit) / denominator
    if second_logit is None:
        second_probability = 0.0
    else:
        second_probability = exp(second_logit - max_logit) / denominator
    return best_token_id, best_probability, second_probability


def _select_argmax_with_confidence_tensor(
    position_logits: Sequence[float],
    candidate_token_ids: tuple[int, ...],
) -> tuple[int, float, float] | None:
    """Use vectorized tensor selection for real full-vocabulary logits."""

    detach = getattr(position_logits, "detach", None)
    index_select = getattr(position_logits, "index_select", None)
    if not callable(detach) or not callable(index_select):
        return None

    try:
        import torch
    except ImportError:
        return None

    logits_tensor = detach()
    if getattr(logits_tensor, "ndim", None) != 1:
        return None

    candidate_tensor = torch.tensor(
        candidate_token_ids,
        dtype=torch.long,
        device=logits_tensor.device,
    )
    candidate_logits = logits_tensor.index_select(0, candidate_tensor).float()
    if candidate_logits.numel() != len(candidate_token_ids):
        return None

    max_logit = candidate_logits.max()
    best_local_index = int(
        torch.nonzero(candidate_logits == max_logit, as_tuple=False)[0].item()
    )
    best_token_id = int(candidate_tensor[best_local_index].item())

    exp_values = torch.exp(candidate_logits - max_logit)
    denominator = float(exp_values.sum().item())
    best_probability = float(exp_values[best_local_index].item()) / denominator

    if candidate_logits.numel() == 1:
        second_probability = 0.0
    else:
        second_logits = candidate_logits.clone()
        second_logits[best_local_index] = -float("inf")
        second_logit = second_logits.max()
        second_probability = float(torch.exp(second_logit - max_logit).item()) / denominator

    return best_token_id, best_probability, second_probability


def _restore_fixed_tokens(x: list[int], fixed_token_ids: dict[int, int]) -> list[int]:
    restored_positions: list[int] = []
    for position, token_id in fixed_token_ids.items():
        if x[position] != token_id:
            x[position] = token_id
            restored_positions.append(position)
    return restored_positions


def _decode_tokens(model: object, token_ids: tuple[int, ...]) -> str:
    decode = getattr(model, "decode", None)
    if callable(decode):
        return decode(list(token_ids), skip_special_tokens=False)
    return " ".join(str(token_id) for token_id in token_ids)
