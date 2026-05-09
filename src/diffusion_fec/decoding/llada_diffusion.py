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
) -> DecodingResult:
    """Decode erased positions with hard fixed-token and hash constraints."""

    prompt_tokens = tuple(prompt_token_ids or ())
    for token_id in prompt_tokens:
        config._validate_token_id(token_id, "prompt_token_ids")

    if plan.hash_guided_count and token_hash_map is None:
        raise ValueError("token_hash_map is required for hash-guided reconstruction plans")
    if token_hash_map is not None and token_hash_map.vocab_size != config.vocab_size:
        raise ValueError("token_hash_map vocab_size must match decoding config vocab_size")

    start_time = perf_counter()
    masks = build_constraint_masks(plan)
    prompt_length = len(prompt_tokens)
    x = _build_initial_input_ids(plan, config, prompt_tokens)
    attention_mask = [1] * len(x)
    fixed_token_ids = _build_fixed_token_ids(plan, prompt_tokens, prompt_length)
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

    if masks.editable_count == 0:
        reconstructed_tokens = tuple(x[prompt_length:])
        return DecodingResult(
            reconstructed_text=_decode_tokens(model, reconstructed_tokens),
            reconstructed_tokens=reconstructed_tokens,
            decode_latency_sec=perf_counter() - start_time,
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

    if config.editable_update_mode == EDITABLE_UPDATE_COMMIT_ONCE:
        for step in range(config.steps):
            editable_target_positions = _still_masked_target_positions(
                x,
                plan,
                prompt_length,
                config.mask_token_id,
            )
            if not editable_target_positions:
                break

            hash_enforced_steps.append(step)
            remaining_steps = config.steps - step
            transfer_count = max(1, ceil(len(editable_target_positions) / remaining_steps))

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
                        )
                    )
                    proposal_calls += 1
            else:
                logits = _extract_logits(
                    model.forward([list(x)], attention_mask=[list(attention_mask)]),
                    sequence_length=len(x),
                    vocab_size=config.vocab_size,
                )
                forward_calls += 1
                proposals = [
                    _proposal_for_position(
                        entry=plan.entries[target_position],
                        full_position=prompt_length + target_position,
                        position_logits=logits[prompt_length + target_position],
                        config=config,
                        token_hash_map=token_hash_map,
                        non_banned_token_ids=non_banned_token_ids,
                        enforce_hash_constraint=True,
                    )
                    for target_position in editable_target_positions
                ]
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

            updated_editable_positions_per_step.append(len(committed))
            step_restored = _restore_fixed_tokens(x, fixed_token_ids)
            restored_positions.update(position - prompt_length for position in step_restored if position >= prompt_length)
            still_masked_count = len(
                _still_masked_target_positions(x, plan, prompt_length, config.mask_token_id)
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
                        )
                    )
                    proposal_calls += 1
            else:
                logits = _extract_logits(
                    model.forward([list(x)], attention_mask=[list(attention_mask)]),
                    sequence_length=len(x),
                    vocab_size=config.vocab_size,
                )
                forward_calls += 1
                proposals = [
                    _proposal_for_position(
                        entry=plan.entries[target_position],
                        full_position=prompt_length + target_position,
                        position_logits=logits[prompt_length + target_position],
                        config=config,
                        token_hash_map=token_hash_map,
                        non_banned_token_ids=non_banned_token_ids,
                        enforce_hash_constraint=enforce_hash,
                    )
                    for target_position in editable_target_positions
                ]

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
            step_restored = _restore_fixed_tokens(x, fixed_token_ids)
            restored_positions.update(position - prompt_length for position in step_restored if position >= prompt_length)
            still_masked_count = len(
                _still_masked_target_positions(x, plan, prompt_length, config.mask_token_id)
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
    ordered_stats = tuple(stats_by_position[position] for position in range(plan.total_tokens))
    return DecodingResult(
        reconstructed_text=_decode_tokens(model, reconstructed_tokens),
        reconstructed_tokens=reconstructed_tokens,
        decode_latency_sec=perf_counter() - start_time,
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
) -> _Proposal:
    candidates, used_fallback = _candidate_token_ids(
        entry=entry,
        config=config,
        token_hash_map=token_hash_map,
        non_banned_token_ids=non_banned_token_ids,
        enforce_hash_constraint=enforce_hash_constraint,
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
) -> _Proposal:
    candidates, used_fallback = _candidate_token_ids(
        entry=entry,
        config=config,
        token_hash_map=token_hash_map,
        non_banned_token_ids=non_banned_token_ids,
        enforce_hash_constraint=enforce_hash_constraint,
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
