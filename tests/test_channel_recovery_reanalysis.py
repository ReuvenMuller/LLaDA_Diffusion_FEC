import csv
import json

from diffusion_fec.analysis.channel_recovery import recompute_channel_lost_metrics_for_run


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_channel_reanalysis_patches_existing_result_rows(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "child"
    run_dir.mkdir(parents=True)
    _write_results_csv(run_dir / "results.csv")
    _write_events_jsonl(run_dir / "events.jsonl")

    manifest = recompute_channel_lost_metrics_for_run(
        run_root=tmp_path / "runs",
        output_dir=tmp_path / "reanalysis",
        patch_results=True,
    )

    patched_row = read_csv(run_dir / "results.csv")[0]
    channel_row = read_csv(tmp_path / "reanalysis" / "channel_metrics.csv")[0]
    corrected_row = read_csv(tmp_path / "reanalysis" / "corrected_results.csv")[0]
    aggregate = read_csv(tmp_path / "reanalysis" / "aggregate.csv")[0]
    manifest_json = read_json(tmp_path / "reanalysis" / "channel_reanalysis_manifest.json")

    assert manifest["patched_result_row_count"] == 1
    assert manifest_json["event_metric_row_count"] == 1
    assert patched_row["channel_lost_position_count"] == "2"
    assert patched_row["channel_lost_position_recovered_count"] == "2"
    assert patched_row["channel_lost_position_recovery_rate"] == "1.0"
    assert channel_row["channel_lost_positions"] == "[0,1]"
    assert corrected_row["channel_lost_position_count"] == "2"
    assert aggregate["mean_channel_lost_position_recovery_rate"] == "1.0"


def test_channel_reanalysis_uses_child_run_directory_to_avoid_key_collisions(tmp_path) -> None:
    first = tmp_path / "runs" / "first"
    second = tmp_path / "runs" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _write_results_csv(first / "results.csv")
    _write_results_csv(second / "results.csv")
    _write_events_jsonl(first / "events.jsonl")
    _write_events_jsonl(
        second / "events.jsonl",
        dropped_positions=[2, 3],
        reconstructed_tokens=[10, 11, 0, 0],
    )

    recompute_channel_lost_metrics_for_run(
        run_root=tmp_path / "runs",
        output_dir=tmp_path / "reanalysis",
        patch_results=True,
    )

    first_row = read_csv(first / "results.csv")[0]
    second_row = read_csv(second / "results.csv")[0]

    assert first_row["channel_lost_position_recovery_rate"] == "1.0"
    assert second_row["channel_lost_position_recovery_rate"] == "0.0"


def _write_results_csv(path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "case_id",
                "strategy",
                "protection_mode",
                "channel_mode",
                "exact_match",
                "lost_position_recovery_rate",
                "token_edit_distance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "run-a",
                "case_id": "case0000",
                "strategy": "Classical_XORParity_MatchedHash4",
                "protection_mode": "xor_parity",
                "channel_mode": "burst",
                "exact_match": "True",
                "lost_position_recovery_rate": "1.0",
                "token_edit_distance": "0",
            }
        )


def _write_events_jsonl(
    path,
    *,
    dropped_positions=None,
    reconstructed_tokens=None,
) -> None:
    dropped_positions = dropped_positions or [0, 1]
    reconstructed_tokens = reconstructed_tokens or [10, 11, 12, 13]
    event = {
        "event_type": "xor_parity_micro_eval_case",
        "run_id": "run-a",
        "case_id": "case0000",
        "strategy": "Classical_XORParity_MatchedHash4",
        "model_label": "ClassicalXORParity",
        "sample": {
            "sample_id": "sample-a",
            "token_ids": [10, 11, 12, 13],
        },
        "loss_result": {
            "dropped": [
                {
                    "source_id": "sample-a",
                    "wire_id": 0,
                    "kind": "data",
                    "token_ids": [10, 11],
                    "token_positions": dropped_positions,
                    "metadata": {},
                },
                {
                    "source_id": "sample-a",
                    "wire_id": 4,
                    "kind": "parity",
                    "token_ids": [99, 98],
                    "token_positions": [0, 1],
                    "metadata": {},
                },
            ]
        },
        "reconstructed_tokens": reconstructed_tokens,
        "metrics": {
            "exact_match": True,
            "lost_position_recovery_rate": 1.0,
        },
    }
    path.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
