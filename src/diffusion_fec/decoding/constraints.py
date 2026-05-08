"""Model-free reconstruction constraints derived from a reconstruction plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diffusion_fec.coding.token_hash import TokenHashMap
from diffusion_fec.types import (
    ReconstructionPlan,
    STATE_KNOWN,
    STATE_MISSING,
    STATE_UNGUIDED,
)


@dataclass(frozen=True)
class ConstraintMasks:
    """Boolean masks over target positions for later tensor construction."""

    known_mask: tuple[bool, ...]
    editable_mask: tuple[bool, ...]
    hash_guided_mask: tuple[bool, ...]
    unguided_mask: tuple[bool, ...]
    fixed_mask: tuple[bool, ...]
    target_mask: tuple[bool, ...]
    fixed_token_ids: tuple[int | None, ...]
    hash_values: tuple[int | None, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.known_mask),
            len(self.editable_mask),
            len(self.hash_guided_mask),
            len(self.unguided_mask),
            len(self.fixed_mask),
            len(self.target_mask),
            len(self.fixed_token_ids),
            len(self.hash_values),
        }
        if len(lengths) != 1:
            raise ValueError("all constraint masks must have the same length")

        for index in range(len(self.target_mask)):
            if self.known_mask[index] != self.fixed_mask[index]:
                raise ValueError("known_mask and fixed_mask must match in this slice")
            if self.editable_mask[index] == self.fixed_mask[index]:
                raise ValueError("each target position must be fixed or editable")
            if self.hash_guided_mask[index] and not self.editable_mask[index]:
                raise ValueError("hash-guided positions must be editable")
            if self.unguided_mask[index] and not self.editable_mask[index]:
                raise ValueError("unguided positions must be editable")
            if self.hash_guided_mask[index] and self.unguided_mask[index]:
                raise ValueError("positions cannot be both hash-guided and unguided")

    @property
    def total_tokens(self) -> int:
        return len(self.target_mask)

    @property
    def fixed_count(self) -> int:
        return sum(self.fixed_mask)

    @property
    def editable_count(self) -> int:
        return sum(self.editable_mask)

    @property
    def hash_guided_count(self) -> int:
        return sum(self.hash_guided_mask)

    @property
    def unguided_count(self) -> int:
        return sum(self.unguided_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "known_mask": list(self.known_mask),
            "editable_mask": list(self.editable_mask),
            "hash_guided_mask": list(self.hash_guided_mask),
            "unguided_mask": list(self.unguided_mask),
            "fixed_mask": list(self.fixed_mask),
            "target_mask": list(self.target_mask),
            "fixed_token_ids": list(self.fixed_token_ids),
            "hash_values": list(self.hash_values),
            "fixed_count": self.fixed_count,
            "editable_count": self.editable_count,
            "hash_guided_count": self.hash_guided_count,
            "unguided_count": self.unguided_count,
        }


@dataclass(frozen=True)
class HashConstraintView:
    """Per-position allowed token IDs for hash-guided missing positions."""

    position_to_allowed_token_ids: dict[int, tuple[int, ...]]
    position_to_allowed_mask: dict[int, tuple[bool, ...]]
    empty_bucket_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_to_allowed_token_ids",
            {
                int(position): tuple(token_ids)
                for position, token_ids in self.position_to_allowed_token_ids.items()
            },
        )
        object.__setattr__(
            self,
            "position_to_allowed_mask",
            {
                int(position): tuple(mask)
                for position, mask in self.position_to_allowed_mask.items()
            },
        )
        object.__setattr__(self, "empty_bucket_positions", tuple(self.empty_bucket_positions))

    def allowed_token_ids(self, position: int) -> tuple[int, ...]:
        return self.position_to_allowed_token_ids[position]

    def allowed_mask(self, position: int) -> tuple[bool, ...]:
        return self.position_to_allowed_mask[position]

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_to_allowed_token_ids": {
                position: list(token_ids)
                for position, token_ids in self.position_to_allowed_token_ids.items()
            },
            "position_to_allowed_mask": {
                position: list(mask)
                for position, mask in self.position_to_allowed_mask.items()
            },
            "empty_bucket_positions": list(self.empty_bucket_positions),
        }


def build_constraint_masks(plan: ReconstructionPlan) -> ConstraintMasks:
    """Build target-position masks from a reconstruction plan."""

    known_mask: list[bool] = []
    editable_mask: list[bool] = []
    hash_guided_mask: list[bool] = []
    unguided_mask: list[bool] = []
    fixed_mask: list[bool] = []
    target_mask: list[bool] = []
    fixed_token_ids: list[int | None] = []
    hash_values: list[int | None] = []

    for entry in plan.entries:
        is_known = entry.state == STATE_KNOWN
        is_hash_guided = entry.state == STATE_MISSING
        is_unguided = entry.state == STATE_UNGUIDED
        is_editable = is_hash_guided or is_unguided

        known_mask.append(is_known)
        editable_mask.append(is_editable)
        hash_guided_mask.append(is_hash_guided)
        unguided_mask.append(is_unguided)
        fixed_mask.append(entry.fixed)
        target_mask.append(True)
        fixed_token_ids.append(entry.token_id if entry.fixed else None)
        hash_values.append(entry.hash_value)

    return ConstraintMasks(
        known_mask=tuple(known_mask),
        editable_mask=tuple(editable_mask),
        hash_guided_mask=tuple(hash_guided_mask),
        unguided_mask=tuple(unguided_mask),
        fixed_mask=tuple(fixed_mask),
        target_mask=tuple(target_mask),
        fixed_token_ids=tuple(fixed_token_ids),
        hash_values=tuple(hash_values),
    )


def build_hash_constraint_view(
    plan: ReconstructionPlan,
    token_hash_map: TokenHashMap,
    *,
    include_masks: bool = True,
) -> HashConstraintView:
    """Resolve hash-guided plan positions into allowed token IDs."""

    position_to_allowed_token_ids: dict[int, tuple[int, ...]] = {}
    position_to_allowed_mask: dict[int, tuple[bool, ...]] = {}
    empty_bucket_positions: list[int] = []

    for entry in plan.entries:
        if entry.state != STATE_MISSING:
            continue
        if entry.hash_value is None:
            raise ValueError("hash-guided missing entries require hash_value")

        token_hash_map.validate_bucket_id(entry.hash_value)
        candidates = token_hash_map.candidate_token_ids(entry.hash_value)
        position_to_allowed_token_ids[entry.position] = candidates
        if not candidates:
            empty_bucket_positions.append(entry.position)
        if include_masks:
            position_to_allowed_mask[entry.position] = token_hash_map.allowed_mask(
                entry.hash_value
            )

    return HashConstraintView(
        position_to_allowed_token_ids=position_to_allowed_token_ids,
        position_to_allowed_mask=position_to_allowed_mask,
        empty_bucket_positions=tuple(empty_bucket_positions),
    )
