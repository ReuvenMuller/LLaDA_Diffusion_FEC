import csv
import json

from diffusion_fec.channels.packet_loss import CHANNEL_BURST, CHANNEL_GILBERT_ELLIOTT
from diffusion_fec.experiments.runner import main
from diffusion_fec.experiments.sweep import (
    STATUS_COMPLETED_REPLACED_STALE,
    SWEEP_RUNNER_MODEL_HASH,
    SWEEP_RUNNER_MODEL_ONLY,
    SWEEP_RUNNER_XOR_PARITY,
    build_synthetic_sweep_config,
    run_synthetic_sweep,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_synthetic_sweep_writes_child_runs_and_analysis(tmp_path) -> None:
    config = build_synthetic_sweep_config(
        sample_lengths=(4,),
        loss_rates=(0.5,),
        seed=3,
        tokens_per_packet=2,
        runners=(SWEEP_RUNNER_MODEL_ONLY, SWEEP_RUNNER_MODEL_HASH, SWEEP_RUNNER_XOR_PARITY),
    )

    result = run_synthetic_sweep(output_dir=tmp_path, config=config)

    manifest = read_json(tmp_path / "sweep_manifest.json")
    rows = read_csv(tmp_path / "sweep_runs.csv")

    assert result["manifest"]["run_count"] == 3
    assert manifest["not_a_research_claim"] is True
    assert {row["status"] for row in rows} == {"completed"}
    assert (tmp_path / "runs" / rows[0]["name"] / "run_manifest.json").exists()
    assert (tmp_path / "analysis" / "aggregate.csv").exists()
    assert (tmp_path / "analysis" / "summary.md").exists()
    assert (tmp_path / "analysis" / "failure_examples.jsonl").exists()
    assert (tmp_path / "hash_profiles" / "fake_synthetic_sweep_v1" / "uniform_hash4_map.npy").exists()


def test_synthetic_sweep_skips_completed_runs(tmp_path) -> None:
    config = build_synthetic_sweep_config(
        sample_lengths=(4,),
        runners=(SWEEP_RUNNER_MODEL_ONLY,),
    )

    run_synthetic_sweep(output_dir=tmp_path, config=config)
    second = run_synthetic_sweep(output_dir=tmp_path, config=config)

    assert second["run_rows"][0]["status"] == "skipped_existing"


def test_synthetic_sweep_cli_entrypoint(tmp_path) -> None:
    output_dir = tmp_path / "cli_sweep"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--synthetic-sweep",
            "--sample-lengths",
            "4",
            "--tokens-per-packet",
            "2",
            "--loss-rate",
            "0.5",
            "--sweep-runners",
            f"{SWEEP_RUNNER_MODEL_ONLY},{SWEEP_RUNNER_XOR_PARITY}",
            "--sweep-include-burst",
            "--burst-length",
            "1",
        ]
    )

    rows = read_csv(output_dir / "sweep_runs.csv")

    assert exit_code == 0
    assert len(rows) == 4
    assert CHANNEL_BURST in (output_dir / "sweep_manifest.json").read_text(encoding="utf-8")
    assert (output_dir / "analysis" / "analysis_manifest.json").exists()


def test_synthetic_sweep_cli_allows_gilbert_elliott(tmp_path) -> None:
    output_dir = tmp_path / "ge_sweep"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--synthetic-sweep",
            "--sample-lengths",
            "4",
            "--tokens-per-packet",
            "2",
            "--sweep-runners",
            SWEEP_RUNNER_MODEL_ONLY,
            "--channel",
            CHANNEL_GILBERT_ELLIOTT,
            "--ge-good-loss-rate",
            "0.0",
            "--ge-bad-loss-rate",
            "1.0",
            "--ge-good-to-bad-rate",
            "1.0",
            "--ge-bad-to-good-rate",
            "0.0",
        ]
    )

    rows = read_csv(output_dir / "sweep_runs.csv")
    child_results = read_csv(output_dir / rows[0]["results"])

    assert exit_code == 0
    assert len(rows) == 1
    assert child_results[0]["channel_mode"] == CHANNEL_GILBERT_ELLIOTT


def test_synthetic_sweep_cli_uses_dataset_samples(tmp_path) -> None:
    dataset_path = tmp_path / "genfec_messages.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "wiki_0", "original_message": "alpha beta gamma", "word_count": 3},
                {"id": "wiki_1", "original_message": "delta epsilon zeta", "word_count": 3},
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "dataset_sweep"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--synthetic-sweep",
            "--dataset-file",
            str(dataset_path),
            "--dataset-label",
            "test_wikitext_copy",
            "--dataset-sample-count",
            "2",
            "--dataset-max-tokens",
            "12",
            "--vocab-size",
            "64",
            "--tokens-per-packet",
            "3",
            "--sweep-runners",
            f"{SWEEP_RUNNER_MODEL_ONLY},{SWEEP_RUNNER_XOR_PARITY}",
        ]
    )

    manifest = read_json(output_dir / "sweep_manifest.json")
    rows = read_csv(output_dir / "sweep_runs.csv")
    first_child_rows = read_csv(output_dir / rows[0]["results"])
    first_child_manifest = read_json(output_dir / rows[0]["output_dir"] / "run_manifest.json")

    assert exit_code == 0
    assert manifest["dataset"]["dataset_label"] == "test_wikitext_copy"
    assert manifest["dataset"]["sample_ids"] == ["wiki_0", "wiki_1"]
    assert first_child_rows[0]["sample_id"] == "wiki_0"
    assert int(first_child_rows[0]["source_token_count"]) <= 12
    assert first_child_manifest["config"]["sample_generation"]["type"] == "provided_token_samples"


def test_synthetic_sweep_reruns_stale_dataset_child_when_selection_changes(tmp_path) -> None:
    dataset_path = tmp_path / "genfec_messages.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "wiki_0", "original_message": "alpha beta gamma"},
                {"id": "wiki_1", "original_message": "delta beta gamma"},
                {"id": "wiki_2", "original_message": "zeta beta gamma"},
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "dataset_sweep"

    first_exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--synthetic-sweep",
            "--dataset-file",
            str(dataset_path),
            "--dataset-sample-count",
            "1",
            "--dataset-seed",
            "0",
            "--dataset-max-tokens",
            "12",
            "--vocab-size",
            "64",
            "--sweep-runners",
            SWEEP_RUNNER_MODEL_ONLY,
        ]
    )
    first_rows = read_csv(output_dir / "sweep_runs.csv")
    first_child_rows = read_csv(output_dir / first_rows[0]["results"])

    second_exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--synthetic-sweep",
            "--dataset-file",
            str(dataset_path),
            "--dataset-sample-count",
            "1",
            "--dataset-seed",
            "1",
            "--dataset-max-tokens",
            "12",
            "--vocab-size",
            "64",
            "--sweep-runners",
            SWEEP_RUNNER_MODEL_ONLY,
        ]
    )
    second_manifest = read_json(output_dir / "sweep_manifest.json")
    second_rows = read_csv(output_dir / "sweep_runs.csv")
    second_child_rows = read_csv(output_dir / second_rows[0]["results"])
    second_child_manifest = read_json(output_dir / second_rows[0]["output_dir"] / "run_manifest.json")

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_child_rows[0]["sample_id"] == "wiki_0"
    assert second_rows[0]["status"] == STATUS_COMPLETED_REPLACED_STALE
    assert "dataset" in second_rows[0]["reuse_decision"]
    assert second_manifest["dataset"]["sample_ids"] == ["wiki_1"]
    assert second_manifest["stale_existing_run_count"] == 1
    assert second_child_rows[0]["sample_id"] == "wiki_1"
    assert second_child_manifest["config"]["sample_generation"]["dataset"]["selection_seed"] == 1


def test_synthetic_sweep_reruns_stale_dataset_child_when_token_cap_changes(tmp_path) -> None:
    dataset_path = tmp_path / "genfec_messages.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "wiki_0", "original_message": "alpha beta gamma"},
                {"id": "wiki_1", "original_message": "delta beta gamma"},
                {"id": "wiki_2", "original_message": "zeta beta gamma"},
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "dataset_sweep"

    main(
        [
            "--output-dir",
            str(output_dir),
            "--synthetic-sweep",
            "--dataset-file",
            str(dataset_path),
            "--dataset-sample-count",
            "1",
            "--dataset-seed",
            "0",
            "--dataset-max-tokens",
            "6",
            "--vocab-size",
            "64",
            "--sweep-runners",
            SWEEP_RUNNER_MODEL_ONLY,
        ]
    )
    first_rows = read_csv(output_dir / "sweep_runs.csv")
    first_child_rows = read_csv(output_dir / first_rows[0]["results"])

    main(
        [
            "--output-dir",
            str(output_dir),
            "--synthetic-sweep",
            "--dataset-file",
            str(dataset_path),
            "--dataset-sample-count",
            "1",
            "--dataset-seed",
            "0",
            "--dataset-max-tokens",
            "10",
            "--vocab-size",
            "64",
            "--sweep-runners",
            SWEEP_RUNNER_MODEL_ONLY,
        ]
    )
    second_manifest = read_json(output_dir / "sweep_manifest.json")
    second_rows = read_csv(output_dir / "sweep_runs.csv")
    second_child_rows = read_csv(output_dir / second_rows[0]["results"])

    assert first_child_rows[0]["sample_id"] == "wiki_0"
    assert int(first_child_rows[0]["source_token_count"]) == 6
    assert second_rows[0]["status"] == STATUS_COMPLETED_REPLACED_STALE
    assert "sample_lengths" in second_rows[0]["reuse_decision"]
    assert second_manifest["dataset"]["max_tokens"] == 10
    assert int(second_child_rows[0]["source_token_count"]) == 10
