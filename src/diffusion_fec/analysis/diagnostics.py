"""Focused diagnostics for sparse hybrid validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


CASE_SUMMARY_FIELDS = (
    "run_id",
    "case_id",
    "sample_id",
    "strategy",
    "channel_mode",
    "hybrid_mode",
    "xor_code",
    "source_token_count",
    "channel_lost_position_count",
    "channel_lost_position_recovered_count",
    "channel_lost_position_recovery_rate",
    "token_edit_distance",
    "normalized_token_edit_distance",
    "exact_match",
    "remaining_mask_token_count",
    "decode_latency_sec",
    "total_decode_time_sec",
    "model_forward_time_sec",
    "candidate_construction_time_sec",
    "parity_candidate_filter_time_sec",
    "xor_peel_time_sec",
    "linear_solver_time_sec",
    "post_commit_hook_time_sec",
    "rollback_time_sec",
    "model_forward_calls",
    "decoder_steps",
    "mean_candidate_count",
    "max_candidate_count",
    "parity_filter_required_token_checks",
    "parity_filter_full_scan_count",
    "parity_filter_candidate_membership_checks",
    "parity_filter_time_sec",
    "rollback_event_count",
    "rollback_positions_count",
    "rollback_remaining_masks_after_budget",
    "timing_breakdown_available",
)

STEP_FIELDS = (
    "run_id",
    "case_id",
    "sample_id",
    "step",
    "remaining_masks_before",
    "remaining_masks_after",
    "model_committed_count",
    "parity_peeled_count",
    "rollback_count",
    "candidate_filter_calls",
    "candidate_filter_rejections",
    "mean_candidate_count",
    "max_candidate_count",
    "hash_guided_editable_count",
    "unguided_editable_count",
    "active_parity_equation_count",
    "linear_solver_components_seen",
    "linear_solver_components_solved",
    "linear_solver_rank_deficient_count",
    "linear_solver_too_large_count",
    "final_masks",
)

ERROR_FIELDS = (
    "run_id",
    "case_id",
    "sample_id",
    "position",
    "original_token_id",
    "reconstructed_token_id",
    "hash_metadata_available",
    "final_token_hash_legal",
    "sparse_parity_coverage_count",
    "surviving_parity_equation_count",
    "was_model_committed",
    "commit_step",
    "commit_confidence",
    "was_parity_solved",
    "was_rolled_back",
    "was_token_banned",
    "involved_in_final_violated_parity_equation",
    "rollback_no_progress_stop",
    "rollback_remaining_masks_after_budget",
    "linear_solver_rank_deficient_count",
    "linear_solver_too_large_count",
    "unguided_due_to_missing_hash_metadata",
)


def build_diagnostic_artifacts(
    *,
    run_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write compact diagnostic artifacts for existing event JSONL files."""

    root = Path(run_root)
    output = Path(output_dir) if output_dir is not None else root / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    event_paths = tuple(sorted(root.rglob("events.jsonl")))

    case_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []

    for event in _read_events(event_paths):
        case = event.get("case", event)
        run_id = str(event.get("run_id", ""))
        case_id = str(event.get("case_id", ""))
        sample = case.get("sample", {})
        sample_id = str(sample.get("sample_id", ""))
        diagnostics = case.get("decoding_result", {}).get("diagnostics", {})

        case_rows.append(_case_summary_row(event=event, case=case, diagnostics=diagnostics))
        step_rows.extend(
            _step_rows(
                run_id=run_id,
                case_id=case_id,
                sample_id=sample_id,
                case=case,
                diagnostics=diagnostics,
            )
        )
        case_error_rows = _error_rows(
            run_id=run_id,
            case_id=case_id,
            sample_id=sample_id,
            case=case,
            diagnostics=diagnostics,
        )
        error_rows.extend(case_error_rows)
        if case_error_rows:
            failure_examples.append(
                {
                    "run_id": run_id,
                    "case_id": case_id,
                    "sample_id": sample_id,
                    "error_count": len(case_error_rows),
                    "text": str(sample.get("text", "")),
                    "positions": [row["position"] for row in case_error_rows[:12]],
                }
            )

    _write_csv(output / "case_summary.csv", CASE_SUMMARY_FIELDS, case_rows)
    _write_csv(output / "step_timing.csv", STEP_FIELDS, step_rows)
    _write_csv(output / "error_taxonomy.csv", ERROR_FIELDS, error_rows)
    _write_failure_examples(output / "failure_examples.md", failure_examples)
    summary = _summary_text(
        event_count=sum(1 for _ in _read_events(event_paths)),
        case_rows=case_rows,
        step_rows=step_rows,
        error_rows=error_rows,
        output=output,
    )
    (output / "diagnostic_summary.md").write_text(summary, encoding="utf-8")

    return {
        "run_root": str(root),
        "output_dir": str(output),
        "event_paths": [str(path) for path in event_paths],
        "case_count": len(case_rows),
        "step_row_count": len(step_rows),
        "error_row_count": len(error_rows),
        "timing_breakdown_available": any(
            _as_bool(row.get("timing_breakdown_available")) for row in case_rows
        ),
    }


def _read_events(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _case_summary_row(
    *,
    event: Mapping[str, Any],
    case: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = case.get("metrics", {})
    channel_metrics = case.get("channel_lost_metrics", {})
    sample = case.get("sample", {})
    config = case.get("channel_config", {})
    timing_available = "model_forward_time_sec" in diagnostics
    return {
        "run_id": event.get("run_id", ""),
        "case_id": event.get("case_id", ""),
        "sample_id": sample.get("sample_id", ""),
        "strategy": event.get("strategy", ""),
        "channel_mode": config.get("mode", ""),
        "hybrid_mode": case.get("hybrid_mode", ""),
        "xor_code": case.get("xor_code", ""),
        "source_token_count": len(sample.get("token_ids", ())),
        "channel_lost_position_count": channel_metrics.get("channel_lost_position_count", ""),
        "channel_lost_position_recovered_count": channel_metrics.get(
            "channel_lost_position_recovered_count",
            "",
        ),
        "channel_lost_position_recovery_rate": channel_metrics.get(
            "channel_lost_position_recovery_rate",
            "",
        ),
        "token_edit_distance": metrics.get("token_edit_distance", ""),
        "normalized_token_edit_distance": metrics.get("normalized_token_edit_distance", ""),
        "exact_match": metrics.get("exact_match", ""),
        "remaining_mask_token_count": metrics.get("remaining_mask_token_count", ""),
        "decode_latency_sec": case.get("decoding_result", {}).get("decode_latency_sec", ""),
        "total_decode_time_sec": diagnostics.get("total_decode_time_sec", ""),
        "model_forward_time_sec": diagnostics.get("model_forward_time_sec", ""),
        "candidate_construction_time_sec": diagnostics.get(
            "candidate_construction_time_sec",
            "",
        ),
        "parity_candidate_filter_time_sec": diagnostics.get(
            "parity_candidate_filter_time_sec",
            "",
        ),
        "xor_peel_time_sec": diagnostics.get("xor_peel_time_sec", ""),
        "linear_solver_time_sec": diagnostics.get("linear_solver_time_sec", ""),
        "post_commit_hook_time_sec": diagnostics.get("post_commit_hook_time_sec", ""),
        "rollback_time_sec": diagnostics.get("rollback_time_sec", ""),
        "model_forward_calls": diagnostics.get("model_forward_calls", ""),
        "decoder_steps": case.get("decoding_result", {}).get("steps", ""),
        "mean_candidate_count": diagnostics.get("mean_candidate_count", ""),
        "max_candidate_count": diagnostics.get("max_candidate_count", ""),
        "parity_filter_required_token_checks": diagnostics.get(
            "parity_filter_required_token_checks",
            "",
        ),
        "parity_filter_full_scan_count": diagnostics.get("parity_filter_full_scan_count", ""),
        "parity_filter_candidate_membership_checks": diagnostics.get(
            "parity_filter_candidate_membership_checks",
            "",
        ),
        "parity_filter_time_sec": diagnostics.get("parity_filter_time_sec", ""),
        "rollback_event_count": diagnostics.get("rollback_event_count", ""),
        "rollback_positions_count": diagnostics.get("rollback_positions_count", ""),
        "rollback_remaining_masks_after_budget": diagnostics.get(
            "rollback_remaining_masks_after_budget",
            "",
        ),
        "timing_breakdown_available": timing_available,
    }


def _step_rows(
    *,
    run_id: str,
    case_id: str,
    sample_id: str,
    case: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for item in diagnostics.get("step_diagnostics", ()) or ():
        row = {"run_id": run_id, "case_id": case_id, "sample_id": sample_id}
        row.update({field: item.get(field, "") for field in STEP_FIELDS if field not in row})
        rows.append(row)
    if rows:
        return rows
    for index, summary in enumerate(case.get("decoding_result", {}).get("step_summaries", ()) or ()):
        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "sample_id": sample_id,
                "step": summary.get("step", index),
                "remaining_masks_after": summary.get("still_masked_count", ""),
                "final_masks": summary.get("still_masked_count", ""),
            }
        )
    return rows


def _error_rows(
    *,
    run_id: str,
    case_id: str,
    sample_id: str,
    case: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sample_tokens = tuple(int(token) for token in case.get("sample", {}).get("token_ids", ()))
    reconstructed = tuple(
        int(token) for token in case.get("decoding_result", {}).get("reconstructed_tokens", ())
    )
    hash_metadata = _int_key_map(case.get("hash_metadata", {}))
    plan_entries = {
        int(entry.get("position")): entry
        for entry in case.get("reconstruction_plan", {}).get("entries", ())
        if isinstance(entry, dict) and entry.get("position") is not None
    }
    stats_by_position = {
        int(stat.get("position")): stat
        for stat in case.get("decoding_result", {}).get("confidence_stats", ())
        if isinstance(stat, dict) and stat.get("position") is not None
    }
    initial_recovered = set(_int_key_map(case.get("initial_peel", {}).get("recovered_tokens", {})))
    iterative_recovered = set(int(position) for position in diagnostics.get("iterative_peel_recovered_positions", ()) or ())
    rollback_positions = set(int(position) for position in diagnostics.get("rollback_positions", ()) or ())
    rollback_bans = _int_key_map(diagnostics.get("rollback_banned_tokens_by_position", {}))
    all_coverage = _coverage_counts(case.get("encoded", {}).get("equation_specs", ()))
    surviving_coverage = _surviving_sparse_coverage(case.get("loss_result", {}).get("received", ()))
    violated_positions = _violated_positions(case.get("final_audit", {}).get("violations", ()))

    rows: list[dict[str, Any]] = []
    for position in case.get("channel_lost_positions", ()) or ():
        position = int(position)
        if position >= len(sample_tokens) or position >= len(reconstructed):
            continue
        if sample_tokens[position] == reconstructed[position]:
            continue
        stat = stats_by_position.get(position, {})
        plan_entry = plan_entries.get(position, {})
        was_parity_solved = position in initial_recovered or position in iterative_recovered
        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "sample_id": sample_id,
                "position": position,
                "original_token_id": sample_tokens[position],
                "reconstructed_token_id": reconstructed[position],
                "hash_metadata_available": position in hash_metadata,
                "final_token_hash_legal": "",
                "sparse_parity_coverage_count": all_coverage.get(position, 0),
                "surviving_parity_equation_count": surviving_coverage.get(position, 0),
                "was_model_committed": (
                    stat.get("commit_step") is not None
                    and not stat.get("was_fixed", False)
                    and not was_parity_solved
                ),
                "commit_step": stat.get("commit_step", ""),
                "commit_confidence": stat.get("top1_probability", ""),
                "was_parity_solved": was_parity_solved,
                "was_rolled_back": position in rollback_positions,
                "was_token_banned": position in rollback_bans,
                "involved_in_final_violated_parity_equation": position in violated_positions,
                "rollback_no_progress_stop": diagnostics.get("rollback_no_progress_stop", ""),
                "rollback_remaining_masks_after_budget": diagnostics.get(
                    "rollback_remaining_masks_after_budget",
                    "",
                ),
                "linear_solver_rank_deficient_count": diagnostics.get(
                    "iterative_linear_solver_rank_deficient_count",
                    diagnostics.get("linear_solver_rank_deficient_count", ""),
                ),
                "linear_solver_too_large_count": diagnostics.get(
                    "iterative_linear_solver_too_large_count",
                    diagnostics.get("linear_solver_too_large_count", ""),
                ),
                "unguided_due_to_missing_hash_metadata": plan_entry.get("state") == "unguided",
            }
        )
    return rows


def _int_key_map(value: Any) -> dict[int, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key, item in value.items():
        try:
            result[int(key)] = item
        except (TypeError, ValueError):
            continue
    return result


def _coverage_counts(equation_specs: Sequence[Any]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for spec in equation_specs:
        if not isinstance(spec, Mapping):
            continue
        for position in spec.get("positions", ()) or ():
            counts[int(position)] += 1
    return dict(counts)


def _surviving_sparse_coverage(packets: Sequence[Any]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for packet in packets:
        if not isinstance(packet, Mapping):
            continue
        metadata = packet.get("metadata", {}).get("sparse_fountain_xor", {})
        if not isinstance(metadata, Mapping):
            continue
        for position in metadata.get("positions", ()) or ():
            counts[int(position)] += 1
    return dict(counts)


def _violated_positions(violations: Sequence[Any]) -> set[int]:
    positions: set[int] = set()
    for violation in violations:
        if not isinstance(violation, Mapping):
            continue
        positions.update(int(position) for position in violation.get("positions", ()) or ())
    return positions


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_failure_examples(path: Path, examples: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Failure Examples",
        "",
        "These examples are diagnostic validation artifacts, not research claims.",
        "",
    ]
    if not examples:
        lines.append("No wrong channel-lost positions were found.")
    for example in examples[:20]:
        text = str(example.get("text", "")).replace("\n", " ")
        if len(text) > 400:
            text = text[:397] + "..."
        lines.extend(
            [
                f"## {example.get('sample_id', '')}",
                "",
                f"- run: `{example.get('run_id', '')}`",
                f"- case: `{example.get('case_id', '')}`",
                f"- wrong channel-lost positions: {example.get('error_count', 0)}",
                f"- first positions: `{example.get('positions', [])}`",
                "",
                text,
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_text(
    *,
    event_count: int,
    case_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    error_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> str:
    timing_available = sum(_as_bool(row.get("timing_breakdown_available")) for row in case_rows)
    iid_rows = [row for row in case_rows if row.get("channel_mode") == "random_iid"]
    burst_rows = [row for row in case_rows if row.get("channel_mode") == "burst"]
    lines = [
        "# Sparse Hybrid Diagnostic Summary",
        "",
        "This report is engineering validation only, not a final research result.",
        "",
        f"- events read: {event_count}",
        f"- cases summarized: {len(case_rows)}",
        f"- step rows written: {len(step_rows)}",
        f"- wrong channel-lost token rows: {len(error_rows)}",
        f"- cases with timing breakdown: {timing_available}",
        f"- IID cases: {len(iid_rows)}",
        f"- burst cases: {len(burst_rows)}",
        "",
        "Artifacts:",
        f"- `{output / 'case_summary.csv'}`",
        f"- `{output / 'step_timing.csv'}`",
        f"- `{output / 'error_taxonomy.csv'}`",
        f"- `{output / 'failure_examples.md'}`",
        "",
    ]
    if timing_available != len(case_rows):
        lines.append(
            "Timing breakdown is missing for at least one case; rerun focused IID/burst "
            "rollback cells with the instrumented code to split model/filter/XOR/rollback time."
        )
    return "\n".join(lines) + "\n"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write sparse hybrid diagnostic artifacts.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    manifest = build_diagnostic_artifacts(run_root=args.run_root, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
