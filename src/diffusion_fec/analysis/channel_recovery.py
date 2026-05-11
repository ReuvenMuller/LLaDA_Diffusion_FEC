"""Recompute channel-loss recovery metrics from existing event artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from diffusion_fec.analysis.aggregate import aggregate_result_rows, write_aggregate_csv
from diffusion_fec.analysis.reporting import (
    discover_event_jsonls,
    discover_result_csvs,
    write_summary_markdown,
)
from diffusion_fec.experiments.logging import RESULTS_FIELDNAMES
from diffusion_fec.metrics.token_metrics import (
    channel_lost_source_positions,
    compute_channel_lost_position_metrics,
)


CHANNEL_METRIC_FIELDS = (
    "channel_lost_position_recovery_rate",
    "channel_lost_position_count",
    "channel_lost_position_recovered_count",
)


def recompute_channel_lost_metrics_for_run(
    *,
    run_root: str | Path,
    output_dir: str | Path | None = None,
    patch_results: bool = False,
) -> dict[str, Any]:
    """Recompute channel-loss metrics from events and write reanalysis artifacts."""

    root = Path(run_root)
    output = Path(output_dir) if output_dir is not None else root / "channel_reanalysis"
    output.mkdir(parents=True, exist_ok=True)
    event_paths = discover_event_jsonls(root)
    result_paths = discover_result_csvs(root)

    metric_rows: list[dict[str, Any]] = []
    metrics_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for event_path in event_paths:
        for event in _read_jsonl(event_path):
            record = _channel_metric_record(event=event, event_path=event_path)
            if record is None:
                continue
            key = (str(record["run_id"]), str(record["case_id"]))
            metrics_by_key[key] = {
                field: record[field]
                for field in CHANNEL_METRIC_FIELDS
            }
            metric_rows.append(record)

    corrected_rows: list[dict[str, Any]] = []
    patched_row_count = 0
    for result_path in result_paths:
        rows, fieldnames = _read_csv(result_path)
        patched_rows = []
        for row in rows:
            key = (str(row.get("run_id", "")), str(row.get("case_id", "")))
            patched = dict(row)
            if key in metrics_by_key:
                patched.update(metrics_by_key[key])
                patched_row_count += 1
            patched_rows.append(patched)
            corrected_rows.append({**patched, "source_results_path": str(result_path)})
        if patch_results and rows:
            _write_csv(
                result_path,
                rows=patched_rows,
                fieldnames=_patched_fieldnames(fieldnames),
            )

    _write_csv(
        output / "channel_metrics.csv",
        rows=metric_rows,
        fieldnames=_fieldnames(metric_rows),
    )
    _write_csv(
        output / "corrected_results.csv",
        rows=corrected_rows,
        fieldnames=_fieldnames(corrected_rows),
    )
    aggregate_rows = aggregate_result_rows(corrected_rows)
    write_aggregate_csv(output_path=output / "aggregate.csv", rows=aggregate_rows)
    write_summary_markdown(output_path=output / "summary.md", rows=aggregate_rows)

    manifest = {
        "artifact_kind": "channel_lost_metric_reanalysis",
        "run_root": str(root),
        "event_paths": [str(path) for path in event_paths],
        "result_paths": [str(path) for path in result_paths],
        "output_dir": str(output),
        "event_metric_row_count": len(metric_rows),
        "result_row_count": len(corrected_rows),
        "patched_result_row_count": patched_row_count,
        "patch_results": patch_results,
        "outputs": {
            "channel_metrics_csv": str(output / "channel_metrics.csv"),
            "corrected_results_csv": str(output / "corrected_results.csv"),
            "aggregate_csv": str(output / "aggregate.csv"),
            "summary_markdown": str(output / "summary.md"),
        },
    }
    (output / "channel_reanalysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute channel-loss recovery metrics from event artifacts."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--patch-results",
        action="store_true",
        help="Rewrite discovered results.csv files with recomputed channel-loss fields.",
    )
    args = parser.parse_args(argv)

    recompute_channel_lost_metrics_for_run(
        run_root=args.run_root,
        output_dir=args.output_dir,
        patch_results=args.patch_results,
    )
    return 0


def _channel_metric_record(
    *,
    event: dict[str, Any],
    event_path: Path,
) -> dict[str, Any] | None:
    sample = _event_sample(event)
    loss_result = _event_loss_result(event)
    reconstructed_tokens = _event_reconstructed_tokens(event)
    if not sample or not loss_result or reconstructed_tokens is None:
        return None
    token_ids = sample.get("token_ids")
    dropped = loss_result.get("dropped")
    if not isinstance(token_ids, list) or not isinstance(dropped, list):
        return None

    lost_positions = channel_lost_source_positions(dropped)
    metrics = compute_channel_lost_position_metrics(
        original_tokens=token_ids,
        reconstructed_tokens=reconstructed_tokens,
        channel_lost_positions=lost_positions,
    )
    return {
        "run_id": event.get("run_id", ""),
        "case_id": event.get("case_id", ""),
        "event_type": event.get("event_type", ""),
        "strategy": event.get("strategy", ""),
        "model_label": event.get("model_label", ""),
        "sample_id": sample.get("sample_id", ""),
        "event_path": str(event_path),
        "channel_lost_positions": json.dumps(list(lost_positions), separators=(",", ":")),
        **metrics.to_dict(),
    }


def _event_sample(event: dict[str, Any]) -> dict[str, Any]:
    if isinstance(event.get("sample"), dict):
        return dict(event["sample"])
    case = event.get("case")
    if isinstance(case, dict) and isinstance(case.get("sample"), dict):
        return dict(case["sample"])
    return {}


def _event_loss_result(event: dict[str, Any]) -> dict[str, Any]:
    if isinstance(event.get("loss_result"), dict):
        return dict(event["loss_result"])
    case = event.get("case")
    if isinstance(case, dict) and isinstance(case.get("loss_result"), dict):
        return dict(case["loss_result"])
    return {}


def _event_reconstructed_tokens(event: dict[str, Any]) -> list[int] | None:
    if isinstance(event.get("reconstructed_tokens"), list):
        return [int(token_id) for token_id in event["reconstructed_tokens"]]
    case = event.get("case")
    if not isinstance(case, dict):
        return None
    decoding_result = case.get("decoding_result")
    if isinstance(decoding_result, dict) and isinstance(decoding_result.get("reconstructed_tokens"), list):
        return [int(token_id) for token_id in decoding_result["reconstructed_tokens"]]
    return None


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _patched_fieldnames(existing_fieldnames: list[str]) -> list[str]:
    preferred = [
        field
        for field in RESULTS_FIELDNAMES
        if field in existing_fieldnames or field in CHANNEL_METRIC_FIELDS
    ]
    extras = [
        field
        for field in existing_fieldnames
        if field not in preferred
    ]
    return [*preferred, *extras]


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for field in RESULTS_FIELDNAMES:
        if any(field in row for row in rows):
            fieldnames.append(field)
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames


if __name__ == "__main__":
    raise SystemExit(main())
