"""Build deterministic analysis artifacts from experiment run outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from html import escape
from pathlib import Path
from typing import Any

from diffusion_fec.analysis.aggregate import aggregate_result_rows, load_result_rows, write_aggregate_csv


DEFAULT_REPORT_GROUP_BY = (
    "strategy",
    "protection_mode",
    "channel_mode",
    "source_layout",
    "wire_interleaving",
    "loss_rate",
    "hash_bits",
)
SUMMARY_FIELDS = (
    "strategy",
    "protection_mode",
    "channel_mode",
    "source_layout",
    "wire_interleaving",
    "loss_rate",
    "hash_bits",
    "case_count",
    "exact_match_rate",
    "mean_lost_position_recovery_rate",
    "mean_token_edit_distance",
    "known_position_preserved_rate",
    "mean_decode_latency_sec",
    "mean_run_wall_time_sec",
    "mean_model_forward_calls",
    "mean_hash_metadata_token_equivalent_overhead_ratio",
    "mean_actual_repair_token_overhead_ratio",
    "mean_total_overhead_ratio",
)
PLOT_SPECS = (
    ("exact_match_rate.svg", "Exact Match Rate", "exact_match_rate"),
    (
        "lost_position_recovery_rate.svg",
        "Mean Lost-Position Recovery Rate",
        "mean_lost_position_recovery_rate",
    ),
    ("decode_latency_sec.svg", "Mean Decode Latency Sec", "mean_decode_latency_sec"),
    ("total_overhead_ratio.svg", "Mean Total Overhead Ratio", "mean_total_overhead_ratio"),
    (
        "repair_overhead_ratio.svg",
        "Mean Repair Token Overhead Ratio",
        "mean_actual_repair_token_overhead_ratio",
    ),
)


def discover_result_csvs(run_root: str | Path) -> tuple[Path, ...]:
    """Find run result CSVs below a run root."""

    root = Path(run_root)
    return tuple(sorted(path for path in root.rglob("results.csv") if path.is_file()))


def discover_event_jsonls(run_root: str | Path) -> tuple[Path, ...]:
    """Find event JSONL files below a run root."""

    root = Path(run_root)
    return tuple(sorted(path for path in root.rglob("events.jsonl") if path.is_file()))


def build_analysis_artifacts(
    *,
    run_root: str | Path,
    output_dir: str | Path | None = None,
    result_paths: Iterable[str | Path] | None = None,
    event_paths: Iterable[str | Path] | None = None,
    group_by: Sequence[str] = DEFAULT_REPORT_GROUP_BY,
    max_failure_examples: int = 20,
) -> dict[str, Any]:
    """Aggregate runs, write a report bundle, and return report metadata."""

    root = Path(run_root)
    output = Path(output_dir) if output_dir is not None else root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    results = tuple(Path(path) for path in result_paths) if result_paths is not None else discover_result_csvs(root)
    events = tuple(Path(path) for path in event_paths) if event_paths is not None else discover_event_jsonls(root)

    rows = load_result_rows(results)
    aggregate_rows = aggregate_result_rows(rows, group_by=group_by)
    aggregate_path = output / "aggregate.csv"
    summary_path = output / "summary.md"
    failures_path = output / "failure_examples.jsonl"

    write_aggregate_csv(output_path=aggregate_path, rows=aggregate_rows)
    write_summary_markdown(output_path=summary_path, rows=aggregate_rows)
    plot_paths = []
    for filename, title, field in PLOT_SPECS:
        path = output / filename
        write_metric_bar_svg(output_path=path, rows=aggregate_rows, title=title, metric_field=field)
        plot_paths.append(path)
    failure_examples = extract_failure_examples(
        event_paths=events,
        output_path=failures_path,
        max_examples=max_failure_examples,
    )

    manifest = {
        "artifact_kind": "analysis_report",
        "run_root": str(root),
        "group_by": list(group_by),
        "result_paths": [str(path) for path in results],
        "event_paths": [str(path) for path in events],
        "outputs": {
            "aggregate_csv": str(aggregate_path),
            "summary_markdown": str(summary_path),
            "failure_examples_jsonl": str(failures_path),
            "plots": [str(path) for path in plot_paths],
        },
        "input_result_row_count": len(rows),
        "aggregate_row_count": len(aggregate_rows),
        "failure_example_count": len(failure_examples),
        "not_a_research_claim_warning": (
            "Analysis artifacts summarize whatever inputs were provided. Smoke and "
            "micro-eval inputs are engineering validation only, not research claims."
        ),
    }
    _write_json(output / "analysis_manifest.json", manifest)
    return manifest


def write_summary_markdown(
    *,
    output_path: str | Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Write a compact markdown summary table."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Diffusion FEC Analysis Summary",
        "",
        "These artifacts summarize the selected run outputs. Smoke and micro-eval rows are engineering validation only, not research claims.",
        "",
    ]
    if not rows:
        lines.append("No result rows were available.")
    else:
        fields = [field for field in SUMMARY_FIELDS if any(field in row for row in rows)]
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            values = [_format_markdown_value(row.get(field)) for field in fields]
            lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_bar_svg(
    *,
    output_path: str | Path,
    rows: Sequence[dict[str, Any]],
    title: str,
    metric_field: str,
    label_field: str = "strategy",
) -> None:
    """Write a simple deterministic SVG bar chart for one aggregate metric."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [
        (
            str(row.get(label_field, f"row-{index}")),
            _coerce_float(row.get(metric_field)),
        )
        for index, row in enumerate(rows)
    ]
    values = [(label, value) for label, value in values if value is not None]
    width = 900
    height = max(220, 92 + (36 * max(len(values), 1)))
    left = 260
    right = 40
    bar_height = 22
    row_gap = 14
    plot_width = width - left - right
    max_value = max((value for _, value in values), default=1.0)
    if max_value <= 0.0:
        max_value = 1.0

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="32" font-family="Arial, sans-serif" font-size="20" font-weight="700">{escape(title)}</text>',
        f'<line x1="{left}" y1="58" x2="{width - right}" y2="58" stroke="#d7dde5"/>',
    ]
    if not values:
        svg.append(
            '<text x="24" y="86" font-family="Arial, sans-serif" font-size="14" fill="#526070">No numeric values available.</text>'
        )
    for index, (label, value) in enumerate(values):
        y = 74 + (index * (bar_height + row_gap))
        bar_width = 0 if value is None else int((value / max_value) * plot_width)
        color = _palette_color(index)
        svg.extend(
            [
                f'<text x="24" y="{y + 16}" font-family="Arial, sans-serif" font-size="12" fill="#233142">{escape(_shorten(label, 34))}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="{bar_height}" rx="3" fill="{color}"/>',
                f'<text x="{left + bar_width + 8}" y="{y + 16}" font-family="Arial, sans-serif" font-size="12" fill="#233142">{escape(_format_number(value))}</text>',
            ]
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def extract_failure_examples(
    *,
    event_paths: Iterable[str | Path],
    output_path: str | Path,
    max_examples: int = 20,
) -> tuple[dict[str, Any], ...]:
    """Extract compact failure examples from event JSONL files."""

    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    examples: list[dict[str, Any]] = []
    for path in event_paths:
        for event in _read_jsonl(Path(path)):
            metrics = _event_metrics(event)
            exact_match = _coerce_bool(metrics.get("exact_match"))
            remaining_masks = _coerce_float(metrics.get("remaining_mask_token_count"))
            if exact_match is True and (remaining_masks is None or remaining_masks == 0):
                continue
            examples.append(_failure_example(event, metrics))
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")
    return tuple(examples)


def _failure_example(event: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    sample = _event_sample(event)
    reconstruction_plan = _event_reconstruction_plan(event)
    return {
        "run_id": event.get("run_id"),
        "case_id": event.get("case_id"),
        "event_type": event.get("event_type"),
        "strategy": event.get("strategy"),
        "model_label": event.get("model_label"),
        "sample_id": sample.get("sample_id"),
        "exact_match": metrics.get("exact_match"),
        "token_edit_distance": metrics.get("token_edit_distance"),
        "lost_position_recovery_rate": metrics.get("lost_position_recovery_rate"),
        "remaining_mask_token_count": metrics.get("remaining_mask_token_count"),
        "known_count": reconstruction_plan.get("known_count"),
        "hash_guided_count": reconstruction_plan.get("hash_guided_count"),
        "unguided_count": reconstruction_plan.get("unguided_count"),
        "dropped_wire_ids": _event_dropped_wire_ids(event),
        "original_tokens": sample.get("token_ids"),
        "reconstructed_tokens": _event_reconstructed_tokens(event),
    }


def _event_metrics(event: dict[str, Any]) -> dict[str, Any]:
    if isinstance(event.get("metrics"), dict):
        return dict(event["metrics"])
    case = event.get("case")
    if isinstance(case, dict) and isinstance(case.get("metrics"), dict):
        return dict(case["metrics"])
    return {}


def _event_sample(event: dict[str, Any]) -> dict[str, Any]:
    if isinstance(event.get("sample"), dict):
        return dict(event["sample"])
    case = event.get("case")
    if isinstance(case, dict) and isinstance(case.get("sample"), dict):
        return dict(case["sample"])
    return {}


def _event_reconstruction_plan(event: dict[str, Any]) -> dict[str, Any]:
    if isinstance(event.get("reconstruction_plan"), dict):
        return dict(event["reconstruction_plan"])
    case = event.get("case")
    if isinstance(case, dict) and isinstance(case.get("reconstruction_plan"), dict):
        return dict(case["reconstruction_plan"])
    return {}


def _event_reconstructed_tokens(event: dict[str, Any]) -> list[int] | None:
    if isinstance(event.get("reconstructed_tokens"), list):
        return list(event["reconstructed_tokens"])
    case = event.get("case")
    if isinstance(case, dict):
        decoding_result = case.get("decoding_result")
        if isinstance(decoding_result, dict) and isinstance(decoding_result.get("reconstructed_tokens"), list):
            return list(decoding_result["reconstructed_tokens"])
    return None


def _event_dropped_wire_ids(event: dict[str, Any]) -> list[int]:
    loss_result = event.get("loss_result")
    if not isinstance(loss_result, dict):
        case = event.get("case")
        if isinstance(case, dict):
            loss_result = case.get("loss_result")
    if not isinstance(loss_result, dict):
        return []
    dropped = loss_result.get("dropped")
    if not isinstance(dropped, list):
        return []
    wire_ids = []
    for packet in dropped:
        if isinstance(packet, dict) and isinstance(packet.get("wire_id"), int):
            wire_ids.append(packet["wire_id"])
    return wire_ids


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return _format_number(value)
    return escape(str(value))


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.4g}"


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


def _shorten(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def _palette_color(index: int) -> str:
    palette = (
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#17becf",
        "#7f7f7f",
        "#bcbd22",
        "#8c564b",
    )
    return palette[index % len(palette)]
