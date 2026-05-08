"""Diagnostics helpers for constrained masked-diffusion decoding."""

from __future__ import annotations

from collections.abc import Iterable

from diffusion_fec.types import (
    ConfidenceStat,
    ReconstructionEntry,
    STATE_MISSING,
    STATE_UNGUIDED,
    StepSummary,
)


def fixed_confidence_stat(
    entry: ReconstructionEntry,
    *,
    was_restored: bool = False,
) -> ConfidenceStat:
    """Build a deterministic confidence stat for a fixed known token."""

    return ConfidenceStat(
        position=entry.position,
        state=entry.state,
        selected_token_id=entry.token_id,
        top1_probability=1.0,
        top2_probability=0.0,
        margin=1.0,
        candidate_count=1,
        commit_step=None,
        was_fixed=True,
        was_restored=was_restored,
        hash_value=entry.hash_value,
    )


def editable_confidence_stat(
    entry: ReconstructionEntry,
    *,
    selected_token_id: int,
    top1_probability: float,
    top2_probability: float,
    candidate_count: int,
    commit_step: int,
) -> ConfidenceStat:
    """Build a confidence stat for an editable position when it is committed."""

    return ConfidenceStat(
        position=entry.position,
        state=entry.state,
        selected_token_id=selected_token_id,
        top1_probability=top1_probability,
        top2_probability=top2_probability,
        margin=top1_probability - top2_probability,
        candidate_count=candidate_count,
        commit_step=commit_step,
        was_fixed=False,
        was_restored=False,
        hash_value=entry.hash_value,
    )


def step_summary(
    *,
    step: int,
    still_masked_count: int,
    committed_stats: Iterable[ConfidenceStat],
    fallback_positions: Iterable[int] = (),
    restored_fixed_positions: Iterable[int] = (),
) -> StepSummary:
    """Summarize one denoising step from the committed position stats."""

    committed = tuple(committed_stats)
    if committed:
        average_confidence = sum(
            stat.top1_probability or 0.0 for stat in committed
        ) / len(committed)
    else:
        average_confidence = None

    return StepSummary(
        step=step,
        still_masked_count=still_masked_count,
        committed_count=len(committed),
        average_confidence=average_confidence,
        hash_guided_committed_count=sum(stat.state == STATE_MISSING for stat in committed),
        unguided_committed_count=sum(stat.state == STATE_UNGUIDED for stat in committed),
        diagnostics={
            "fallback_positions": sorted(fallback_positions),
            "restored_fixed_positions": sorted(restored_fixed_positions),
        },
    )
