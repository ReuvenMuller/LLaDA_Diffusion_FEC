"""Artifact writing utilities for local experiment runs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


RESULTS_FIELDNAMES = [
    "run_id",
    "case_id",
    "sample_id",
    "model_label",
    "strategy",
    "baseline_family",
    "protection_mode",
    "oracle_hash_metadata",
    "hash_bits",
    "source_layout",
    "source_chunk_size",
    "wire_interleaving",
    "wire_interleaving_span",
    "channel_mode",
    "burst_start_wire_id",
    "burst_length",
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
    "received_packet_count",
    "dropped_packet_count",
    "source_packet_count",
    "extra_packet_count",
    "repair_packet_count",
    "repair_token_budget",
    "target_overhead_ratio",
    "actual_repair_token_overhead_ratio",
    "hash_profile_source",
    "decode_latency_sec",
    "decoder_steps",
    "model_forward_calls",
    "exact_match",
    "token_edit_distance",
    "normalized_token_edit_distance",
    "lost_position_recovery_rate",
    "lost_position_count",
    "lost_position_recovered_count",
    "known_position_preserved",
    "known_position_count",
    "remaining_mask_token_count",
    "original_token_count",
    "reconstructed_token_count",
]


def write_run_artifacts(
    *,
    output_dir: str | Path,
    manifest: dict[str, Any],
    result_rows: Sequence[dict[str, Any]],
    events: Iterable[dict[str, Any]],
) -> None:
    """Write manifest, CSV results, and JSONL events."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "run_manifest.json", manifest)
    _write_results_csv(output_path / "results.csv", result_rows)
    _write_jsonl(output_path / "events.jsonl", events)


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
