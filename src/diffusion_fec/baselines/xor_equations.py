"""Reusable XOR parity equations for classical and hybrid recovery."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from diffusion_fec.baselines.xor_parity import (
    XOR_PARITY_METADATA_KEY,
    XOR_PARITY_SCHEME,
)
from diffusion_fec.coding.token_hash import TokenHashMap
from diffusion_fec.types import Packet


@dataclass(frozen=True)
class XorTokenEquation:
    """One token-level XOR equation derived from a received parity packet."""

    equation_id: str
    parity_packet_wire_id: int
    stripe_id: int
    parity_offset: int
    positions: tuple[int, ...]
    parity_value: int

    def __post_init__(self) -> None:
        if not self.equation_id:
            raise ValueError("equation_id must be non-empty")
        if self.parity_packet_wire_id < 0:
            raise ValueError("parity_packet_wire_id must be non-negative")
        if self.stripe_id < 0:
            raise ValueError("stripe_id must be non-negative")
        if self.parity_offset < 0:
            raise ValueError("parity_offset must be non-negative")
        object.__setattr__(self, "positions", tuple(int(position) for position in self.positions))
        if not self.positions:
            raise ValueError("positions must be non-empty")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must not contain duplicates")
        for position in self.positions:
            if position < 0:
                raise ValueError("positions must be non-negative")
        if self.parity_value < 0:
            raise ValueError("parity_value must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "parity_packet_wire_id": self.parity_packet_wire_id,
            "stripe_id": self.stripe_id,
            "parity_offset": self.parity_offset,
            "positions": list(self.positions),
            "parity_value": self.parity_value,
        }


@dataclass(frozen=True)
class XorPeelConflict:
    """A parity solution that could not be accepted safely."""

    equation_id: str
    position: int | None
    solved_token_id: int | None
    reason: str
    expected_hash_value: int | None = None
    solved_hash_value: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "position": self.position,
            "solved_token_id": self.solved_token_id,
            "reason": self.reason,
            "expected_hash_value": self.expected_hash_value,
            "solved_hash_value": self.solved_hash_value,
        }


@dataclass(frozen=True)
class XorPeelResult:
    """Result of iterative XOR peeling."""

    known_tokens: dict[int, int]
    recovered_tokens: dict[int, int]
    conflicts: tuple[XorPeelConflict, ...] = field(default_factory=tuple)
    peel_iteration_count: int = 0

    @property
    def recovered_count(self) -> int:
        return len(self.recovered_tokens)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "known_tokens": dict(sorted(self.known_tokens.items())),
            "recovered_tokens": dict(sorted(self.recovered_tokens.items())),
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "peel_iteration_count": self.peel_iteration_count,
            "recovered_count": self.recovered_count,
            "conflict_count": self.conflict_count,
        }


@dataclass(frozen=True)
class XorAuditResult:
    """Final satisfaction audit for a set of XOR equations."""

    satisfied_count: int
    violated_count: int
    unresolved_count: int
    violations: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "satisfied_count": self.satisfied_count,
            "violated_count": self.violated_count,
            "unresolved_count": self.unresolved_count,
            "violations": [dict(item) for item in self.violations],
        }


def equations_from_parity_packets(packets: Sequence[Packet]) -> tuple[XorTokenEquation, ...]:
    """Extract token-level equations from XOR parity packets."""

    equations: list[XorTokenEquation] = []
    for packet in sorted(packets, key=lambda item: item.wire_id):
        if packet.kind != "parity":
            continue
        metadata = _parity_metadata(packet)
        stripe_id = int(metadata["stripe_id"])
        member_lengths = [int(value) for value in metadata["stripe_member_lengths"]]
        member_positions = [
            [int(position) for position in positions]
            for positions in metadata["stripe_member_token_positions"]
        ]
        for offset, parity_value in enumerate(packet.token_ids):
            positions = tuple(
                positions[offset]
                for positions, member_length in zip(member_positions, member_lengths)
                if offset < member_length
            )
            if not positions:
                continue
            equations.append(
                XorTokenEquation(
                    equation_id=f"wire{packet.wire_id}:stripe{stripe_id}:offset{offset}",
                    parity_packet_wire_id=packet.wire_id,
                    stripe_id=stripe_id,
                    parity_offset=offset,
                    positions=positions,
                    parity_value=int(parity_value),
                )
            )
    return tuple(equations)


def known_tokens_from_data_packets(
    packets: Sequence[Packet],
    *,
    total_tokens: int,
) -> dict[int, int]:
    """Build a position -> token map from received data packets."""

    known_tokens: dict[int, int] = {}
    for packet in sorted(packets, key=lambda item: item.wire_id):
        if packet.kind != "data":
            continue
        for token_id, position in zip(packet.token_ids, packet.token_positions):
            if position >= total_tokens:
                raise ValueError(f"packet position {position} is outside total_tokens")
            existing = known_tokens.get(position)
            if existing is not None and existing != token_id:
                raise ValueError(
                    f"conflicting token IDs for received position {position}: "
                    f"{existing} != {token_id}"
                )
            known_tokens[position] = token_id
    return known_tokens


def peel_xor_equations(
    *,
    equations: Sequence[XorTokenEquation],
    known_tokens: Mapping[int, int],
    hash_metadata: Mapping[int, int] | None = None,
    token_hash_map: TokenHashMap | None = None,
    vocab_size: int | None = None,
    banned_token_ids: Collection[int] | None = None,
) -> XorPeelResult:
    """Iteratively solve equations with exactly one unknown token position."""

    known = {int(position): int(token_id) for position, token_id in known_tokens.items()}
    recovered: dict[int, int] = {}
    conflicts: list[XorPeelConflict] = []
    hash_metadata = dict(hash_metadata or {})
    banned = {int(token_id) for token_id in (banned_token_ids or ())}
    iteration_count = 0

    while True:
        progressed = False
        for equation in equations:
            unknown_positions: list[int] = []
            accumulator = equation.parity_value
            for position in equation.positions:
                if position in known:
                    accumulator ^= known[position]
                else:
                    unknown_positions.append(position)

            if len(unknown_positions) != 1:
                continue

            position = unknown_positions[0]
            solved_token_id = accumulator
            conflict = _legality_conflict(
                equation=equation,
                position=position,
                solved_token_id=solved_token_id,
                vocab_size=vocab_size,
                banned_token_ids=banned,
            )
            if conflict is not None:
                if conflict not in conflicts:
                    conflicts.append(conflict)
                continue
            conflict = _hash_conflict(
                equation=equation,
                position=position,
                solved_token_id=solved_token_id,
                hash_metadata=hash_metadata,
                token_hash_map=token_hash_map,
            )
            if conflict is not None:
                if conflict not in conflicts:
                    conflicts.append(conflict)
                continue

            known[position] = solved_token_id
            recovered[position] = solved_token_id
            progressed = True

        if not progressed:
            break
        iteration_count += 1

    return XorPeelResult(
        known_tokens=known,
        recovered_tokens=recovered,
        conflicts=tuple(conflicts),
        peel_iteration_count=iteration_count,
    )


def audit_xor_equations(
    *,
    equations: Sequence[XorTokenEquation],
    token_by_position: Mapping[int, int],
) -> XorAuditResult:
    """Count satisfied, violated, and unresolved XOR equations."""

    satisfied = 0
    violated = 0
    unresolved = 0
    violations: list[dict[str, Any]] = []
    known = {int(position): int(token_id) for position, token_id in token_by_position.items()}
    for equation in equations:
        if any(position not in known for position in equation.positions):
            unresolved += 1
            continue
        value = 0
        for position in equation.positions:
            value ^= known[position]
        if value == equation.parity_value:
            satisfied += 1
        else:
            violated += 1
            violations.append(
                {
                    "equation_id": equation.equation_id,
                    "expected_parity_value": equation.parity_value,
                    "actual_parity_value": value,
                    "positions": list(equation.positions),
                }
            )
    return XorAuditResult(
        satisfied_count=satisfied,
        violated_count=violated,
        unresolved_count=unresolved,
        violations=tuple(violations),
    )


class ParityCandidateFilter:
    """Local parity filter for candidate token IDs during diffusion decoding."""

    def __init__(
        self,
        *,
        equations: Sequence[XorTokenEquation],
        known_tokens: Mapping[int, int],
        mask_token_id: int,
        prompt_length: int = 0,
        fallback_on_empty: bool = True,
    ) -> None:
        self.equations = tuple(equations)
        self.known_tokens = {
            int(position): int(token_id)
            for position, token_id in known_tokens.items()
        }
        self.mask_token_id = int(mask_token_id)
        self.prompt_length = int(prompt_length)
        self.fallback_on_empty = bool(fallback_on_empty)
        self._equations_by_position: dict[int, list[XorTokenEquation]] = defaultdict(list)
        for equation in self.equations:
            for position in equation.positions:
                self._equations_by_position[position].append(equation)
        self.call_count = 0
        self.rejected_count = 0
        self.fallback_count = 0

    def __call__(
        self,
        *,
        entry,
        candidate_token_ids: Sequence[int],
        input_ids: Sequence[int],
        step: int,
        full_position: int,
    ) -> tuple[int, ...]:
        self.call_count += 1
        candidates = tuple(int(token_id) for token_id in candidate_token_ids)
        if not candidates:
            return candidates
        relevant = self._equations_by_position.get(int(entry.position), ())
        if not relevant:
            return candidates

        kept = tuple(
            token_id
            for token_id in candidates
            if self._candidate_satisfies_relevant_equations(
                position=int(entry.position),
                candidate_token_id=token_id,
                input_ids=input_ids,
                equations=relevant,
            )
        )
        self.rejected_count += len(candidates) - len(kept)
        if kept:
            return kept
        if self.fallback_on_empty:
            self.fallback_count += 1
            return candidates
        return kept

    def diagnostics(self) -> dict[str, Any]:
        return {
            "parity_candidate_filter_calls": self.call_count,
            "parity_candidate_rejections": self.rejected_count,
            "parity_filter_fallback_count": self.fallback_count,
            "parity_filter_fallback_enabled": self.fallback_on_empty,
        }

    def _candidate_satisfies_relevant_equations(
        self,
        *,
        position: int,
        candidate_token_id: int,
        input_ids: Sequence[int],
        equations: Sequence[XorTokenEquation],
    ) -> bool:
        for equation in equations:
            required_token_id = self._required_token_if_determined(
                equation=equation,
                position=position,
                input_ids=input_ids,
            )
            if required_token_id is not None and candidate_token_id != required_token_id:
                return False
        return True

    def _required_token_if_determined(
        self,
        *,
        equation: XorTokenEquation,
        position: int,
        input_ids: Sequence[int],
    ) -> int | None:
        accumulator = equation.parity_value
        for other_position in equation.positions:
            if other_position == position:
                continue
            token_id = self._current_token_for_position(
                position=other_position,
                input_ids=input_ids,
            )
            if token_id is None:
                return None
            accumulator ^= token_id
        return accumulator

    def _current_token_for_position(
        self,
        *,
        position: int,
        input_ids: Sequence[int],
    ) -> int | None:
        if position in self.known_tokens:
            return self.known_tokens[position]
        full_position = self.prompt_length + position
        if full_position < 0 or full_position >= len(input_ids):
            return None
        token_id = int(input_ids[full_position])
        if token_id == self.mask_token_id:
            return None
        return token_id


def _hash_conflict(
    *,
    equation: XorTokenEquation,
    position: int,
    solved_token_id: int,
    hash_metadata: Mapping[int, int],
    token_hash_map: TokenHashMap | None,
) -> XorPeelConflict | None:
    expected_hash = hash_metadata.get(position)
    if expected_hash is None:
        return None
    if token_hash_map is None:
        return XorPeelConflict(
            equation_id=equation.equation_id,
            position=position,
            solved_token_id=solved_token_id,
            reason="hash_metadata_without_token_hash_map",
            expected_hash_value=expected_hash,
        )
    try:
        solved_hash = token_hash_map.bucket_for_token(solved_token_id)
    except ValueError:
        return XorPeelConflict(
            equation_id=equation.equation_id,
            position=position,
            solved_token_id=solved_token_id,
            reason="solved_token_outside_hash_vocab",
            expected_hash_value=expected_hash,
        )
    if solved_hash != expected_hash:
        return XorPeelConflict(
            equation_id=equation.equation_id,
            position=position,
            solved_token_id=solved_token_id,
            reason="parity_hash_conflict",
            expected_hash_value=expected_hash,
            solved_hash_value=solved_hash,
        )
    return None


def _legality_conflict(
    *,
    equation: XorTokenEquation,
    position: int,
    solved_token_id: int,
    vocab_size: int | None,
    banned_token_ids: Collection[int],
) -> XorPeelConflict | None:
    if solved_token_id < 0:
        return XorPeelConflict(
            equation_id=equation.equation_id,
            position=position,
            solved_token_id=solved_token_id,
            reason="solved_token_negative",
        )
    if vocab_size is not None and solved_token_id >= vocab_size:
        return XorPeelConflict(
            equation_id=equation.equation_id,
            position=position,
            solved_token_id=solved_token_id,
            reason="solved_token_outside_vocab",
        )
    if solved_token_id in banned_token_ids:
        return XorPeelConflict(
            equation_id=equation.equation_id,
            position=position,
            solved_token_id=solved_token_id,
            reason="solved_token_is_banned",
        )
    return None


def _parity_metadata(packet: Packet) -> dict[str, Any]:
    metadata = packet.metadata.get(XOR_PARITY_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError("parity packet is missing xor parity metadata")
    if metadata.get("scheme") != XOR_PARITY_SCHEME:
        raise ValueError("parity packet metadata has unsupported scheme")
    return metadata
