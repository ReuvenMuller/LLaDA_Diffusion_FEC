import csv
import json

from diffusion_fec.analysis.report import main as report_main
from diffusion_fec.analysis.reporting import (
    build_analysis_artifacts,
    discover_event_jsonls,
    discover_result_csvs,
    extract_failure_examples,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_build_analysis_artifacts_writes_summary_plots_and_failures(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "case_a"
    run_dir.mkdir(parents=True)
    _write_results_csv(run_dir / "results.csv")
    _write_events_jsonl(run_dir / "events.jsonl")

    manifest = build_analysis_artifacts(run_root=tmp_path / "runs")

    analysis_dir = tmp_path / "runs" / "analysis"
    assert manifest["input_result_row_count"] == 2
    assert manifest["aggregate_row_count"] == 1
    assert manifest["failure_example_count"] == 1
    assert (analysis_dir / "aggregate.csv").exists()
    assert "FakeMicroEval_ModelOnly_NoPrompt" in (analysis_dir / "summary.md").read_text(
        encoding="utf-8"
    )
    assert "<svg" in (analysis_dir / "exact_match_rate.svg").read_text(encoding="utf-8")
    failures = read_jsonl(analysis_dir / "failure_examples.jsonl")
    assert failures[0]["case_id"] == "case0001"
    assert failures[0]["dropped_wire_ids"] == [1]


def test_discovery_and_failure_extraction_helpers(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "case_a"
    run_dir.mkdir(parents=True)
    _write_results_csv(run_dir / "results.csv")
    _write_events_jsonl(run_dir / "events.jsonl")

    assert discover_result_csvs(tmp_path / "runs") == (run_dir / "results.csv",)
    assert discover_event_jsonls(tmp_path / "runs") == (run_dir / "events.jsonl",)

    failures = extract_failure_examples(
        event_paths=[run_dir / "events.jsonl"],
        output_path=tmp_path / "failures.jsonl",
        max_examples=1,
    )

    assert len(failures) == 1
    assert failures[0]["remaining_mask_token_count"] == 1


def test_report_cli_entrypoint(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "case_a"
    output_dir = tmp_path / "report"
    run_dir.mkdir(parents=True)
    _write_results_csv(run_dir / "results.csv")
    _write_events_jsonl(run_dir / "events.jsonl")

    exit_code = report_main(
        [
            "--run-root",
            str(tmp_path / "runs"),
            "--output-dir",
            str(output_dir),
            "--group-by",
            "strategy,protection_mode,channel_mode",
        ]
    )

    assert exit_code == 0
    assert read_json(output_dir / "analysis_manifest.json")["aggregate_row_count"] == 1
    assert (output_dir / "repair_overhead_ratio.svg").exists()


def _write_results_csv(path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "protection_mode",
                "channel_mode",
                "source_layout",
                "wire_interleaving",
                "loss_rate",
                "hash_bits",
                "exact_match",
                "known_position_preserved",
                "token_edit_distance",
                "lost_position_recovery_rate",
                "remaining_mask_token_count",
                "decode_latency_sec",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "strategy": "FakeMicroEval_ModelOnly_NoPrompt",
                "protection_mode": "none",
                "channel_mode": "random_iid",
                "source_layout": "contiguous",
                "wire_interleaving": "none",
                "loss_rate": "0.5",
                "hash_bits": "4",
                "exact_match": "True",
                "known_position_preserved": "True",
                "token_edit_distance": "0",
                "lost_position_recovery_rate": "1.0",
                "remaining_mask_token_count": "0",
                "decode_latency_sec": "0.0",
            }
        )
        writer.writerow(
            {
                "strategy": "FakeMicroEval_ModelOnly_NoPrompt",
                "protection_mode": "none",
                "channel_mode": "random_iid",
                "source_layout": "contiguous",
                "wire_interleaving": "none",
                "loss_rate": "0.5",
                "hash_bits": "4",
                "exact_match": "False",
                "known_position_preserved": "True",
                "token_edit_distance": "1",
                "lost_position_recovery_rate": "0.0",
                "remaining_mask_token_count": "1",
                "decode_latency_sec": "0.0",
            }
        )


def _write_events_jsonl(path) -> None:
    events = [
        {
            "event_type": "micro_eval_case",
            "run_id": "run-a",
            "case_id": "case0000",
            "strategy": "FakeMicroEval_ModelOnly_NoPrompt",
            "model_label": "Fake",
            "case": {
                "sample": {"sample_id": "sample-a", "token_ids": [4, 5]},
                "loss_result": {"dropped": []},
                "reconstruction_plan": {"known_count": 2, "hash_guided_count": 0, "unguided_count": 0},
                "decoding_result": {"reconstructed_tokens": [4, 5]},
                "metrics": {"exact_match": True, "remaining_mask_token_count": 0},
            },
        },
        {
            "event_type": "micro_eval_case",
            "run_id": "run-a",
            "case_id": "case0001",
            "strategy": "FakeMicroEval_ModelOnly_NoPrompt",
            "model_label": "Fake",
            "case": {
                "sample": {"sample_id": "sample-b", "token_ids": [6, 7]},
                "loss_result": {"dropped": [{"wire_id": 1}]},
                "reconstruction_plan": {"known_count": 1, "hash_guided_count": 0, "unguided_count": 1},
                "decoding_result": {"reconstructed_tokens": [6, 0]},
                "metrics": {
                    "exact_match": False,
                    "token_edit_distance": 1,
                    "lost_position_recovery_rate": 0.0,
                    "remaining_mask_token_count": 1,
                },
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
