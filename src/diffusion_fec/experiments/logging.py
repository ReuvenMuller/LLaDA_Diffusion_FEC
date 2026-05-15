"""Artifact writing utilities for local experiment runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


RUN_TIMING_FIELDS = (
    "run_started_at",
    "run_finished_at",
    "run_wall_time_sec",
)

RESULTS_FIELDNAMES = [
    "run_id",
    "run_started_at",
    "run_finished_at",
    "run_wall_time_sec",
    "case_id",
    "sample_id",
    "model_label",
    "strategy",
    "baseline_family",
    "protection_mode",
    "oracle_hash_metadata",
    "hash_bits",
    "hybrid_mode",
    "xor_code",
    "xor_overhead_bits_per_token",
    "source_layout",
    "source_chunk_size",
    "wire_interleaving",
    "wire_interleaving_span",
    "channel_mode",
    "burst_start_wire_id",
    "burst_length",
    "requested_burst_loss_rate",
    "resolved_burst_length",
    "ge_good_loss_rate",
    "ge_bad_loss_rate",
    "ge_good_to_bad_rate",
    "ge_bad_to_good_rate",
    "ge_initial_state",
    "loss_rate",
    "seed",
    "tokens_per_packet",
    "source_token_count",
    "known_count",
    "missing_count",
    "hash_guided_count",
    "unguided_count",
    "total_transmitted_packet_count",
    "received_packet_count",
    "dropped_packet_count",
    "actual_wire_packet_loss_rate",
    "source_packet_count",
    "dropped_data_packet_count",
    "actual_data_packet_loss_rate",
    "extra_packet_count",
    "repair_packet_count",
    "dropped_repair_packet_count",
    "actual_repair_packet_loss_rate",
    "repair_token_budget",
    "target_overhead_ratio",
    "xor_target_overhead_ratio",
    "actual_repair_token_overhead_ratio",
    "token_bit_width",
    "hash_metadata_count",
    "hash_metadata_bit_count",
    "hash_metadata_token_equivalent",
    "hash_metadata_token_equivalent_overhead_ratio",
    "total_overhead_ratio",
    "hash_profile_source",
    "editable_update_mode",
    "hash_constraint_schedule",
    "decode_latency_sec",
    "total_decode_time_sec",
    "model_forward_time_sec",
    "candidate_construction_time_sec",
    "parity_candidate_filter_time_sec",
    "xor_peel_time_sec",
    "linear_solver_time_sec",
    "post_commit_hook_time_sec",
    "rollback_time_sec",
    "decoder_steps",
    "model_forward_calls",
    "model_proposal_calls",
    "decoder_proposal_mode",
    "proposal_interface_used",
    "mean_candidate_count",
    "max_candidate_count",
    "parity_equation_count",
    "parity_received_equation_count",
    "parity_peel_iterations",
    "parity_peel_recovered_count",
    "parity_hash_conflict_count",
    "linear_solver_enabled",
    "linear_solver_components_seen",
    "linear_solver_components_solved",
    "linear_solver_tokens_recovered",
    "linear_solver_rank_deficient_count",
    "linear_solver_validation_conflict_count",
    "linear_solver_too_large_count",
    "sparse_equation_count",
    "sparse_budget_exhausted",
    "sparse_coverage_enabled",
    "sparse_coverage_possible",
    "sparse_coverage_pass_degree",
    "sparse_coverage_zero_count",
    "sparse_coverage_min",
    "sparse_coverage_mean",
    "sparse_actual_mean_degree",
    "sparse_degree_histogram",
    "iterative_peel_enabled",
    "iterative_peel_passes",
    "iterative_peel_recovered_count",
    "iterative_peel_hash_conflict_count",
    "iterative_peel_special_token_conflict_count",
    "iterative_peel_vocab_conflict_count",
    "iterative_peel_conflict_count",
    "iterative_linear_solver_enabled",
    "iterative_linear_solver_components_seen",
    "iterative_linear_solver_components_solved",
    "iterative_linear_solver_tokens_recovered",
    "iterative_linear_solver_rank_deficient_count",
    "iterative_linear_solver_validation_conflict_count",
    "iterative_linear_solver_too_large_count",
    "iterative_peel_recovered_positions",
    "iterative_peel_recovered_count_by_step",
    "rollback_enabled",
    "rollback_event_count",
    "rollback_conflict_equation_count",
    "rollback_positions_count",
    "rollback_positions",
    "rollback_single_suspect_count",
    "rollback_multi_suspect_count",
    "rollback_banned_token_count",
    "rollback_banned_tokens_by_position",
    "rollback_max_per_position_hits",
    "rollback_extra_steps_used",
    "rollback_remaining_masks_after_budget",
    "rollback_provenance_invalidated_count",
    "rollback_no_progress_stop",
    "parity_candidate_rejections",
    "parity_filter_fallback_count",
    "parity_filter_required_token_checks",
    "parity_filter_full_scan_count",
    "parity_filter_candidate_membership_checks",
    "parity_filter_time_sec",
    "parity_filter_mean_input_candidate_count",
    "parity_filter_max_input_candidate_count",
    "parity_filter_mean_output_candidate_count",
    "parity_filter_max_output_candidate_count",
    "parity_equations_satisfied",
    "parity_equations_violated",
    "exact_match",
    "token_edit_distance",
    "normalized_token_edit_distance",
    "lost_position_recovery_rate",
    "lost_position_count",
    "lost_position_recovered_count",
    "channel_lost_position_recovery_rate",
    "channel_lost_position_count",
    "channel_lost_position_recovered_count",
    "actual_source_token_loss_rate",
    "known_position_preserved",
    "known_position_count",
    "remaining_mask_token_count",
    "original_token_count",
    "reconstructed_token_count",
]


@dataclass(frozen=True)
class RunTiming:
    """Completed wall-clock timing fields for one artifact-writing run."""

    run_started_at: str
    run_finished_at: str
    run_wall_time_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_started_at": self.run_started_at,
            "run_finished_at": self.run_finished_at,
            "run_wall_time_sec": self.run_wall_time_sec,
        }


@dataclass(frozen=True)
class RunTimer:
    """Monotonic timer paired with UTC timestamps for run manifests."""

    run_started_at: str
    start_perf_counter: float

    def finish(self) -> RunTiming:
        finished_at = _utc_timestamp()
        wall_time = max(0.0, perf_counter() - self.start_perf_counter)
        return RunTiming(
            run_started_at=self.run_started_at,
            run_finished_at=finished_at,
            run_wall_time_sec=wall_time,
        )


def start_run_timer() -> RunTimer:
    """Start a timer for one run-level artifact bundle."""

    return RunTimer(
        run_started_at=_utc_timestamp(),
        start_perf_counter=perf_counter(),
    )


def write_run_artifacts(
    *,
    output_dir: str | Path,
    manifest: dict[str, Any],
    result_rows: Sequence[dict[str, Any]],
    events: Iterable[dict[str, Any]],
    run_timer: RunTimer | None = None,
    timing: RunTiming | None = None,
) -> dict[str, Any]:
    """Write manifest, CSV results, and JSONL events with run timing fields."""

    if timing is None:
        timing = (run_timer or start_run_timer()).finish()
    timing_dict = timing.to_dict()
    timed_manifest = {**manifest, **timing_dict}
    timed_rows = [
        {**dict(row), **timing_dict}
        for row in result_rows
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "run_manifest.json", timed_manifest)
    _write_results_csv(output_path / "results.csv", timed_rows)
    _write_jsonl(output_path / "events.jsonl", events)
    return {
        "manifest": timed_manifest,
        "result_rows": timed_rows,
        "timing": timing_dict,
    }


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_results_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULTS_FIELDNAMES})


def _write_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
