"""Token-level reconstruction metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from diffusion_fec.types import ReconstructionPlan, STATE_MISSING, STATE_UNGUIDED


@dataclass(frozen=True)
class TokenMetrics:
    """Compact token-level metrics for one reconstructed sequence."""

    exact_match: bool
    token_edit_distance: int
    normalized_token_edit_distance: float
    lost_position_recovery_rate: float
    lost_position_count: int
    lost_position_recovered_count: int
    known_position_preserved: bool
    known_position_count: int
    remaining_mask_token_count: int
    original_token_count: int
    reconstructed_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "token_edit_distance": self.token_edit_distance,
            "normalized_token_edit_distance": self.normalized_token_edit_distance,
            "lost_position_recovery_rate": self.lost_position_recovery_rate,
            "lost_position_count": self.lost_position_count,
            "lost_position_recovered_count": self.lost_position_recovered_count,
            "known_position_preserved": self.known_position_preserved,
            "known_position_count": self.known_position_count,
            "remaining_mask_token_count": self.remaining_mask_token_count,
            "original_token_count": self.original_token_count,
            "reconstructed_token_count": self.reconstructed_token_count,
        }


def compute_token_metrics(
    *,
    original_tokens: Sequence[int],
    reconstructed_tokens: Sequence[int],
    reconstruction_plan: ReconstructionPlan | None = None,
    mask_token_id: int | None = None,
) -> TokenMetrics:
    """Compute token-level recovery metrics for one output sequence."""

    original = tuple(original_tokens)
    reconstructed = tuple(reconstructed_tokens)
    edit_distance = token_edit_distance(original, reconstructed)
    denominator = max(len(original), len(reconstructed), 1)

    known_positions = _known_positions(reconstruction_plan)
    lost_positions = _lost_positions(reconstruction_plan)
    known_preserved = all(
        position < len(reconstructed) and reconstructed[position] == token_id
        for position, token_id in known_positions.items()
    )
    lost_recovered = sum(
        position < len(original)
        and position < len(reconstructed)
        and original[position] == reconstructed[position]
        for position in lost_positions
    )
    if lost_positions:
        lost_recovery_rate = lost_recovered / len(lost_positions)
    else:
        lost_recovery_rate = 1.0

    remaining_mask_count = (
        0
        if mask_token_id is None
        else sum(token_id == mask_token_id for token_id in reconstructed)
    )

    return TokenMetrics(
        exact_match=original == reconstructed,
        token_edit_distance=edit_distance,
        normalized_token_edit_distance=edit_distance / denominator,
        lost_position_recovery_rate=lost_recovery_rate,
        lost_position_count=len(lost_positions),
        lost_position_recovered_count=lost_recovered,
        known_position_preserved=known_preserved,
        known_position_count=len(known_positions),
        remaining_mask_token_count=remaining_mask_count,
        original_token_count=len(original),
        reconstructed_token_count=len(reconstructed),
    )


def token_edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Compute Levenshtein edit distance over token IDs."""

    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _known_positions(plan: ReconstructionPlan | None) -> dict[int, int]:
    if plan is None:
        return {}
    return {
        entry.position: entry.token_id
        for entry in plan.entries
        if entry.fixed and entry.token_id is not None
    }


def _lost_positions(plan: ReconstructionPlan | None) -> tuple[int, ...]:
    if plan is None:
        return ()
    return tuple(
        entry.position
        for entry in plan.entries
        if entry.state in {STATE_MISSING, STATE_UNGUIDED}
    )
