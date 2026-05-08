import csv
import json

from diffusion_fec.experiments.runner import run_minimal_smoke, main


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_minimal_smoke_runner_writes_expected_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "run"

    run_minimal_smoke(
        output_dir=output_dir,
        sample_count=2,
        loss_rate=0.5,
        seed=0,
        tokens_per_packet=1,
        protection_mode="lookback_1",
        hash_bits=4,
    )

    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "events.jsonl").exists()


def test_manifest_labels_fake_model_and_strategy_without_timestamps_or_paths(tmp_path) -> None:
    output_dir = tmp_path / "run"

    run_minimal_smoke(output_dir=output_dir, sample_count=1, seed=7)
    manifest = read_json(output_dir / "run_manifest.json")

    assert manifest["model_label"] == "FakeDeterministicSmokeModel"
    assert manifest["model_kind"] == "fake_deterministic_smoke_model"
    assert manifest["not_a_research_baseline"] is True
    assert manifest["strategy"] == "FakeSmoke_LookbackHash_NoPrompt"
    assert manifest["config"]["protection_mode"] == "lookback_1"
    assert manifest["config"]["oracle_hash_metadata"] is False
    assert "created_at" not in manifest
    assert str(output_dir) not in json.dumps(manifest)


def test_results_csv_includes_metrics_and_run_metadata(tmp_path) -> None:
    output_dir = tmp_path / "run"

    run_minimal_smoke(output_dir=output_dir, sample_count=2, seed=3)
    rows = read_csv(output_dir / "results.csv")

    assert len(rows) == 2
    first = rows[0]
    assert first["model_label"] == "FakeDeterministicSmokeModel"
    assert first["protection_mode"] == "lookback_1"
    assert first["oracle_hash_metadata"] == "False"
    assert first["exact_match"] == "True"
    assert first["token_edit_distance"] == "0"
    assert first["lost_position_recovery_rate"] == "1.0"
    assert first["known_position_preserved"] == "True"
    assert first["remaining_mask_token_count"] == "0"
    assert first["model_forward_calls"] == "0"
    assert int(first["model_proposal_calls"]) > 0
    assert first["decoder_proposal_mode"] == "model_propose_token"
    assert first["proposal_interface_used"] == "True"


def test_events_jsonl_contains_detailed_case_data_and_normalized_latency(tmp_path) -> None:
    output_dir = tmp_path / "run"

    run_minimal_smoke(output_dir=output_dir, sample_count=1, seed=5)
    events = read_jsonl(output_dir / "events.jsonl")

    assert len(events) == 1
    event = events[0]
    case = event["case"]
    assert event["event_type"] == "smoke_case"
    assert event["model_label"] == "FakeDeterministicSmokeModel"
    assert case["protection_mode"] == "lookback_1"
    assert case["oracle_hash_metadata"] is False
    assert case["sample"]["sample_id"] == "synthetic-0000"
    assert "reconstruction_plan" in case
    assert "decoding_result" in case
    assert "metrics" in case
    assert case["decoding_result"]["decode_latency_sec"] == 0.0


def test_runner_output_is_deterministic_for_same_seed(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    run_minimal_smoke(output_dir=first_dir, sample_count=3, seed=11)
    run_minimal_smoke(output_dir=second_dir, sample_count=3, seed=11)

    for filename in ("run_manifest.json", "results.csv", "events.jsonl"):
        assert (first_dir / filename).read_text(encoding="utf-8") == (
            second_dir / filename
        ).read_text(encoding="utf-8")


def test_runner_cli_entrypoint_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "cli"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--sample-count",
            "1",
            "--seed",
            "9",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "run_manifest.json").exists()
    assert read_json(output_dir / "run_manifest.json")["config"]["seed"] == 9


def test_runner_can_build_and_reuse_hash_profile(tmp_path) -> None:
    profile_dir = tmp_path / "hash_profile"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    run_minimal_smoke(
        output_dir=first_dir,
        sample_count=1,
        seed=4,
        hash_profile_dir=profile_dir,
        build_hash_profile=True,
    )
    run_minimal_smoke(
        output_dir=second_dir,
        sample_count=1,
        seed=4,
        hash_profile_dir=profile_dir,
    )

    first_manifest = read_json(first_dir / "run_manifest.json")
    second_manifest = read_json(second_dir / "run_manifest.json")

    assert first_manifest["hash_profile"]["source"] == "built_profile"
    assert second_manifest["hash_profile"]["source"] == "loaded_profile"
    assert (profile_dir / "uniform_hash4_map.npy").exists()
    assert (profile_dir / "hash_profile_metadata.json").exists()
    assert read_csv(first_dir / "results.csv") == read_csv(second_dir / "results.csv")


def test_runner_cli_can_build_hash_profile(tmp_path) -> None:
    output_dir = tmp_path / "cli"
    profile_dir = tmp_path / "profile"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--sample-count",
            "1",
            "--seed",
            "12",
            "--hash-profile-dir",
            str(profile_dir),
            "--build-hash-profile",
        ]
    )

    assert exit_code == 0
    assert read_json(output_dir / "run_manifest.json")["hash_profile"]["source"] == "built_profile"
    assert (profile_dir / "uniform_hash4_map.npy").exists()
