"""Aggregate experiment result CSV files."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


DEFAULT_GROUP_BY = (
    "strategy",
    "protection_mode",
    "channel_mode",
    "hybrid_mode",
    "editable_update_mode",
    "hash_constraint_schedule",
)
NUMERIC_MEAN_FIELDS = (
    "token_edit_distance",
    "normalized_token_edit_distance",
    "lost_position_recovery_rate",
    "channel_lost_position_recovery_rate",
    "channel_lost_position_count",
    "channel_lost_position_recovered_count",
    "known_position_count",
    "remaining_mask_token_count",
    "decode_latency_sec",
    "run_wall_time_sec",
    "model_forward_calls",
    "hash_metadata_count",
    "hash_metadata_bit_count",
    "hash_metadata_token_equivalent",
    "hash_metadata_token_equivalent_overhead_ratio",
    "actual_repair_token_overhead_ratio",
    "total_overhead_ratio",
    "parity_equation_count",
    "parity_received_equation_count",
    "parity_peel_iterations",
    "parity_peel_recovered_count",
    "parity_hash_conflict_count",
    "iterative_peel_passes",
    "iterative_peel_recovered_count",
    "iterative_peel_hash_conflict_count",
    "iterative_peel_special_token_conflict_count",
    "iterative_peel_vocab_conflict_count",
    "iterative_peel_conflict_count",
    "parity_candidate_rejections",
    "parity_filter_fallback_count",
    "parity_equations_satisfied",
    "parity_equations_violated",
)
BOOLEAN_RATE_FIELDS = (
    "exact_match",
    "known_position_preserved",
)


def load_result_rows(paths: Iterable[str | Path]) -> tuple[dict[str, str], ...]:
    """Load rows from one or more `results.csv` files."""

    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return tuple(rows)


def aggregate_result_rows(
    rows: Iterable[dict[str, Any]],
    *,
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
) -> tuple[dict[str, Any], ...]:
    """Aggregate result rows by strategy/config keys."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in group_by)
        groups[key].append(dict(row))

    aggregate_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            field: value
            for field, value in zip(group_by, key)
        }
        aggregate["case_count"] = len(group_rows)
        for field in NUMERIC_MEAN_FIELDS:
            values = [_coerce_float(row.get(field)) for row in group_rows]
            values = [value for value in values if value is not None]
            aggregate[f"mean_{field}"] = _mean(values)
        for field in BOOLEAN_RATE_FIELDS:
            values = [_coerce_bool(row.get(field)) for row in group_rows]
            values = [value for value in values if value is not None]
            aggregate[f"{field}_rate"] = _mean([1.0 if value else 0.0 for value in values])
        aggregate_rows.append(aggregate)
    return tuple(aggregate_rows)


def write_aggregate_csv(
    *,
    output_path: str | Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Write aggregate rows to CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in {"True", "true", "1", 1}:
        return True
    if value in {"False", "false", "0", 0}:
        return False
    return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
