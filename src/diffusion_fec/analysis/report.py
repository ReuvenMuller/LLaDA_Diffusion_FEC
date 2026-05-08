"""Command-line entrypoint for deterministic run analysis artifacts."""

from __future__ import annotations

import argparse

from diffusion_fec.analysis.reporting import DEFAULT_REPORT_GROUP_BY, build_analysis_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Diffusion FEC run artifacts.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--group-by",
        default=",".join(DEFAULT_REPORT_GROUP_BY),
        help="Comma-separated result fields used for aggregate grouping.",
    )
    parser.add_argument("--max-failure-examples", type=int, default=20)
    args = parser.parse_args(argv)

    build_analysis_artifacts(
        run_root=args.run_root,
        output_dir=args.output_dir,
        group_by=_parse_group_by(args.group_by),
        max_failure_examples=args.max_failure_examples,
    )
    return 0


def _parse_group_by(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("group-by must contain at least one field")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
