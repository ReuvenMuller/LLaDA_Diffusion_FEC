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
            "total_decode_time_sec": "2.1",
            "model_forward_time_sec": "1.0",
            "candidate_construction_time_sec": "0.5",
            "parity_candidate_filter_time_sec": "0.25",
            "xor_peel_time_sec": "0.1",
            "linear_solver_time_sec": "0.2",
            "post_commit_hook_time_sec": "0.4",
            "rollback_time_sec": "0.05",
            "run_wall_time_sec": "10.0",
            "mean_candidate_count": "12",
            "max_candidate_count": "20",
            "hash_metadata_token_equivalent_overhead_ratio": "0.25",
            "actual_repair_token_overhead_ratio": "0.0",
            "total_overhead_ratio": "0.25",
            "rollback_event_count": "2",
            "rollback_positions_count": "1",
            "rollback_banned_token_count": "1",
            "rollback_provenance_invalidated_count": "0",
            "parity_filter_required_token_checks": "3",
            "parity_filter_full_scan_count": "0",
            "parity_filter_candidate_membership_checks": "2",
            "parity_filter_time_sec": "0.25",
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
            "total_decode_time_sec": "4.1",
            "model_forward_time_sec": "2.0",
            "candidate_construction_time_sec": "1.5",
            "parity_candidate_filter_time_sec": "0.75",
            "xor_peel_time_sec": "0.3",
            "linear_solver_time_sec": "0.4",
            "post_commit_hook_time_sec": "0.6",
            "rollback_time_sec": "0.15",
            "run_wall_time_sec": "20.0",
            "mean_candidate_count": "18",
            "max_candidate_count": "30",
            "hash_metadata_token_equivalent_overhead_ratio": "0.25",
            "actual_repair_token_overhead_ratio": "0.0",
            "total_overhead_ratio": "0.25",
            "rollback_event_count": "4",
            "rollback_positions_count": "3",
            "rollback_banned_token_count": "1",
            "rollback_provenance_invalidated_count": "2",
            "parity_filter_required_token_checks": "5",
            "parity_filter_full_scan_count": "0",
            "parity_filter_candidate_membership_checks": "4",
            "parity_filter_time_sec": "0.75",
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
    assert row["mean_total_decode_time_sec"] == 3.0999999999999996
    assert row["mean_model_forward_time_sec"] == 1.5
    assert row["mean_candidate_construction_time_sec"] == 1.0
    assert row["mean_parity_candidate_filter_time_sec"] == 0.5
    assert row["mean_xor_peel_time_sec"] == 0.2
    assert row["mean_linear_solver_time_sec"] == 0.30000000000000004
    assert row["mean_post_commit_hook_time_sec"] == 0.5
    assert row["mean_rollback_time_sec"] == 0.1
    assert row["mean_run_wall_time_sec"] == 15.0
    assert row["mean_mean_candidate_count"] == 15.0
    assert row["mean_max_candidate_count"] == 25.0
    assert row["mean_hash_metadata_token_equivalent_overhead_ratio"] == 0.25
    assert row["mean_total_overhead_ratio"] == 0.25
    assert row["mean_rollback_event_count"] == 3.0
    assert row["mean_rollback_positions_count"] == 2.0
    assert row["mean_rollback_banned_token_count"] == 1.0
    assert row["mean_rollback_provenance_invalidated_count"] == 1.0
    assert row["mean_parity_filter_required_token_checks"] == 4.0
    assert row["mean_parity_filter_full_scan_count"] == 0.0
    assert row["mean_parity_filter_candidate_membership_checks"] == 3.0
    assert row["mean_parity_filter_time_sec"] == 0.5
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
