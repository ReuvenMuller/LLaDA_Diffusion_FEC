import csv
import json
from pathlib import Path
from typing import Any


RUN_TIMING_FIELDS = {
    "run_started_at",
    "run_finished_at",
    "run_wall_time_sec",
}


def assert_manifest_has_run_timing(manifest: dict[str, Any]) -> None:
    assert manifest["run_started_at"]
    assert manifest["run_finished_at"]
    assert float(manifest["run_wall_time_sec"]) >= 0.0


def assert_row_has_run_timing(row: dict[str, Any]) -> None:
    assert row["run_started_at"]
    assert row["run_finished_at"]
    assert float(row["run_wall_time_sec"]) >= 0.0


def strip_run_timing_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in RUN_TIMING_FIELDS
        }
        for row in rows
    ]


def normalized_artifact_text(path: Path) -> str:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(_strip_run_timing(data), sort_keys=True)
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key not in RUN_TIMING_FIELDS
                }
                for row in csv.DictReader(handle)
            ]
        return json.dumps(rows, sort_keys=True)
    return path.read_text(encoding="utf-8")


def _strip_run_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_run_timing(child)
            for key, child in value.items()
            if key not in RUN_TIMING_FIELDS
        }
    if isinstance(value, list):
        return [_strip_run_timing(child) for child in value]
    return value
