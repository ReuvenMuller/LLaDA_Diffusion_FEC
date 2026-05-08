"""Result aggregation helpers."""

from diffusion_fec.analysis.aggregate import (
    aggregate_result_rows,
    load_result_rows,
    write_aggregate_csv,
)
from diffusion_fec.analysis.reporting import (
    DEFAULT_REPORT_GROUP_BY,
    build_analysis_artifacts,
    discover_event_jsonls,
    discover_result_csvs,
    extract_failure_examples,
    write_metric_bar_svg,
    write_summary_markdown,
)

__all__ = [
    "DEFAULT_REPORT_GROUP_BY",
    "aggregate_result_rows",
    "build_analysis_artifacts",
    "discover_event_jsonls",
    "discover_result_csvs",
    "extract_failure_examples",
    "load_result_rows",
    "write_metric_bar_svg",
    "write_aggregate_csv",
    "write_summary_markdown",
]
