import csv

from diffusion_fec.analysis.aggregate import (
    aggregate_result_rows,
    load_result_rows,
    write_aggregate_csv,
)


def test_aggregate_result_rows_groups_and_averages_metrics() -> None:
    rows = [
        {
            "strategy": "A",
            "protection_mode": "p",
            "channel_mode": "random_iid",
            "exact_match": "True",
            "known_position_preserved": "True",
            "token_edit_distance": "0",
            "lost_position_recovery_rate": "1.0",
            "channel_lost_position_recovery_rate": "0.75",
            "channel_lost_position_count": "4",
            "channel_lost_position_recovered_count": "3",
            "decode_latency_sec": "2.0",
            "run_wall_time_sec": "10.0",
            "hash_metadata_token_equivalent_overhead_ratio": "0.25",
            "actual_repair_token_overhead_ratio": "0.0",
            "total_overhead_ratio": "0.25",
        },
        {
            "strategy": "A",
            "protection_mode": "p",
            "channel_mode": "random_iid",
            "exact_match": "False",
            "known_position_preserved": "True",
            "token_edit_distance": "4",
            "lost_position_recovery_rate": "0.0",
            "channel_lost_position_recovery_rate": "0.25",
            "channel_lost_position_count": "4",
            "channel_lost_position_recovered_count": "1",
            "decode_latency_sec": "4.0",
            "run_wall_time_sec": "20.0",
            "hash_metadata_token_equivalent_overhead_ratio": "0.25",
            "actual_repair_token_overhead_ratio": "0.0",
            "total_overhead_ratio": "0.25",
        },
    ]

    aggregate = aggregate_result_rows(rows)

    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["case_count"] == 2
    assert row["mean_token_edit_distance"] == 2.0
    assert row["mean_lost_position_recovery_rate"] == 0.5
    assert row["mean_channel_lost_position_recovery_rate"] == 0.5
    assert row["mean_channel_lost_position_count"] == 4.0
    assert row["mean_channel_lost_position_recovered_count"] == 2.0
    assert row["mean_decode_latency_sec"] == 3.0
    assert row["mean_run_wall_time_sec"] == 15.0
    assert row["mean_hash_metadata_token_equivalent_overhead_ratio"] == 0.25
    assert row["mean_total_overhead_ratio"] == 0.25
    assert row["exact_match_rate"] == 0.5
    assert row["known_position_preserved_rate"] == 1.0


def test_load_and_write_aggregate_csv(tmp_path) -> None:
    results_path = tmp_path / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "protection_mode",
                "channel_mode",
                "exact_match",
                "token_edit_distance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "strategy": "A",
                "protection_mode": "p",
                "channel_mode": "burst",
                "exact_match": "True",
                "token_edit_distance": "1",
            }
        )
    rows = load_result_rows([results_path])
    aggregate = aggregate_result_rows(rows)
    output_path = tmp_path / "aggregate.csv"

    write_aggregate_csv(output_path=output_path, rows=aggregate)

    assert output_path.exists()
    assert "exact_match_rate" in output_path.read_text(encoding="utf-8")
