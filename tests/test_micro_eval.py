import csv
import json

from artifact_helpers import (
    assert_manifest_has_run_timing,
    assert_row_has_run_timing,
    normalized_artifact_text,
    strip_run_timing_from_rows,
)
from diffusion_fec.channels.packet_loss import (
    CHANNEL_BURST,
    PacketLossChannelConfig,
)
from diffusion_fec.coding.packetizer import (
    SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
    WIRE_INTERLEAVING_MATRIX,
    SourceLayoutConfig,
    WireInterleavingConfig,
)
from diffusion_fec.experiments.micro_eval import (
    MICRO_EVAL_MODEL_HASH,
    MICRO_EVAL_MODEL_ONLY,
    run_synthetic_micro_eval,
)
from diffusion_fec.experiments.runner import main


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


def test_model_only_micro_eval_writes_unguided_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "model_only"

    run_synthetic_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        loss_rate=1.0,
        seed=0,
        tokens_per_packet=4,
        mode=MICRO_EVAL_MODEL_ONLY,
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert manifest["runner"] == "synthetic_micro_eval"
    assert manifest["not_a_research_claim"] is True
    assert manifest["config"]["mode"] == MICRO_EVAL_MODEL_ONLY
    assert manifest["config"]["protection_mode"] == "none"
    assert manifest["config"]["editable_update_mode"] == "commit_once"
    assert manifest["config"]["hash_constraint_schedule"] == "always"
    assert manifest["hash_profile"]["source"] == "not_used"
    assert_manifest_has_run_timing(manifest)
    assert rows[0]["protection_mode"] == "none"
    assert rows[0]["hash_guided_count"] == "0"
    assert rows[0]["unguided_count"] == "8"
    assert rows[0]["hash_metadata_count"] == "0"
    assert rows[0]["hash_metadata_bit_count"] == "0"
    assert rows[0]["editable_update_mode"] == "commit_once"
    assert rows[0]["hash_constraint_schedule"] == "always"
    assert float(rows[0]["total_overhead_ratio"]) == 0.0
    assert_row_has_run_timing(rows[0])
    assert events[0]["case"]["hash_metadata"] == {}
    assert events[0]["case"]["oracle_hash_metadata"] is False
    assert events[0]["case"]["decoding_result"]["diagnostics"]["editable_update_mode"] == "commit_once"
    assert events[0]["case"]["decoding_result"]["diagnostics"]["hash_constraint_schedule"] == "always"


def test_model_hash_micro_eval_uses_only_transmitted_lookback_metadata(tmp_path) -> None:
    output_dir = tmp_path / "model_hash"

    run_synthetic_micro_eval(
        output_dir=output_dir,
        sample_lengths=(2,),
        loss_rate=0.5,
        seed=1,
        tokens_per_packet=1,
        mode=MICRO_EVAL_MODEL_HASH,
        hash_bits=4,
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert manifest["config"]["mode"] == MICRO_EVAL_MODEL_HASH
    assert manifest["config"]["protection_mode"] == "lookback_1"
    assert manifest["config"]["oracle_hash_metadata"] is False
    assert manifest["config"]["editable_update_mode"] == "commit_once"
    assert manifest["config"]["hash_constraint_schedule"] == "always"
    assert rows[0]["known_count"] == "1"
    assert rows[0]["hash_guided_count"] == "1"
    assert rows[0]["unguided_count"] == "0"
    assert rows[0]["hash_metadata_count"] == "1"
    assert rows[0]["hash_metadata_bit_count"] == "4"
    assert rows[0]["editable_update_mode"] == "commit_once"
    assert rows[0]["hash_constraint_schedule"] == "always"
    assert float(rows[0]["hash_metadata_token_equivalent_overhead_ratio"]) > 0.0
    assert rows[0]["total_overhead_ratio"] == rows[0]["hash_metadata_token_equivalent_overhead_ratio"]
    assert events[0]["case"]["oracle_hash_metadata"] is False
    assert list(events[0]["case"]["hash_metadata"]) == ["0"]


def test_micro_eval_artifacts_record_resample_mode_and_hash_schedule(tmp_path) -> None:
    output_dir = tmp_path / "resample"

    run_synthetic_micro_eval(
        output_dir=output_dir,
        sample_lengths=(4,),
        loss_rate=1.0,
        seed=0,
        tokens_per_packet=2,
        mode=MICRO_EVAL_MODEL_ONLY,
        editable_update_mode="resample_each_step",
        hash_constraint_schedule="final_only",
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert manifest["config"]["editable_update_mode"] == "resample_each_step"
    assert manifest["config"]["hash_constraint_schedule"] == "final_only"
    assert rows[0]["editable_update_mode"] == "resample_each_step"
    assert rows[0]["hash_constraint_schedule"] == "final_only"
    diagnostics = events[0]["case"]["decoding_result"]["diagnostics"]
    assert diagnostics["editable_update_mode"] == "resample_each_step"
    assert diagnostics["hash_constraint_schedule"] == "final_only"


def test_model_hash_micro_eval_counts_full_transmitted_lookback_overhead(tmp_path) -> None:
    output_dir = tmp_path / "hash_overhead"

    run_synthetic_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        loss_rate=1.0,
        seed=0,
        tokens_per_packet=4,
        mode=MICRO_EVAL_MODEL_HASH,
        hash_bits=4,
        vocab_size=128,
    )

    rows = read_csv(output_dir / "results.csv")

    assert rows[0]["hash_metadata_count"] == "4"
    assert rows[0]["hash_metadata_bit_count"] == "16"
    assert rows[0]["token_bit_width"] == "7"
    assert float(rows[0]["hash_metadata_token_equivalent"]) == 16 / 7
    assert float(rows[0]["total_overhead_ratio"]) == (16 / 7) / 8


def test_micro_eval_output_is_deterministic_for_same_seed(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    run_synthetic_micro_eval(
        output_dir=first_dir,
        sample_lengths=(8, 16),
        seed=11,
        tokens_per_packet=4,
        mode=MICRO_EVAL_MODEL_HASH,
    )
    run_synthetic_micro_eval(
        output_dir=second_dir,
        sample_lengths=(8, 16),
        seed=11,
        tokens_per_packet=4,
        mode=MICRO_EVAL_MODEL_HASH,
    )

    for filename in ("run_manifest.json", "results.csv", "events.jsonl"):
        assert normalized_artifact_text(first_dir / filename) == normalized_artifact_text(
            second_dir / filename
        )


def test_profile_backed_fake_micro_eval_loads_existing_profile_on_second_run(tmp_path) -> None:
    profile_dir = tmp_path / "profile"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    run_synthetic_micro_eval(
        output_dir=first_dir,
        sample_lengths=(8,),
        seed=4,
        hash_profile_dir=profile_dir,
        build_hash_profile=True,
    )
    run_synthetic_micro_eval(
        output_dir=second_dir,
        sample_lengths=(8,),
        seed=4,
        hash_profile_dir=profile_dir,
    )

    first_manifest = read_json(first_dir / "run_manifest.json")
    second_manifest = read_json(second_dir / "run_manifest.json")

    assert first_manifest["hash_profile"]["source"] == "built_profile"
    assert second_manifest["hash_profile"]["source"] == "loaded_profile"
    assert (profile_dir / "uniform_hash4_map.npy").exists()
    assert (profile_dir / "hash_profile_metadata.json").exists()
    first_rows = read_csv(first_dir / "results.csv")
    second_rows = read_csv(second_dir / "results.csv")
    first_rows = strip_run_timing_from_rows(first_rows)
    second_rows = strip_run_timing_from_rows(second_rows)
    first_rows[0].pop("hash_profile_source")
    second_rows[0].pop("hash_profile_source")
    assert first_rows == second_rows


def test_micro_eval_cli_writes_layout_and_wire_config(tmp_path) -> None:
    output_dir = tmp_path / "cli"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--micro-eval",
            "--sample-lengths",
            "8",
            "--tokens-per-packet",
            "4",
            "--source-layout",
            SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
            "--source-chunk-size",
            "1",
            "--wire-interleaving",
            WIRE_INTERLEAVING_MATRIX,
            "--wire-interleaving-span",
            "4",
            "--channel",
            CHANNEL_BURST,
            "--burst-length",
            "2",
        ]
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")

    assert exit_code == 0
    assert manifest["config"]["source_layout"]["mode"] == SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS
    assert manifest["config"]["wire_interleaving"]["mode"] == WIRE_INTERLEAVING_MATRIX
    assert manifest["config"]["channel"]["mode"] == CHANNEL_BURST
    assert rows[0]["source_layout"] == SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS
    assert rows[0]["wire_interleaving"] == WIRE_INTERLEAVING_MATRIX
    assert rows[0]["channel_mode"] == CHANNEL_BURST


def test_micro_eval_burst_channel_records_contiguous_wire_loss_geometry(tmp_path) -> None:
    output_dir = tmp_path / "burst"

    run_synthetic_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=1,
        mode=MICRO_EVAL_MODEL_ONLY,
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=2,
        ),
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")
    dropped_positions = [
        position
        for packet in events[0]["case"]["loss_result"]["dropped"]
        for position in packet["token_positions"]
    ]

    assert manifest["config"]["channel"]["mode"] == CHANNEL_BURST
    assert rows[0]["channel_mode"] == CHANNEL_BURST
    assert rows[0]["burst_length"] == "2"
    assert dropped_positions == [0, 1]


def test_micro_eval_wire_interleaving_changes_burst_loss_geometry(tmp_path) -> None:
    output_dir = tmp_path / "wire_burst"

    run_synthetic_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=1,
        mode=MICRO_EVAL_MODEL_ONLY,
        wire_interleaving=WireInterleavingConfig(
            mode=WIRE_INTERLEAVING_MATRIX,
            span=4,
        ),
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=2,
        ),
    )

    events = read_jsonl(output_dir / "events.jsonl")
    dropped_positions = [
        position
        for packet in events[0]["case"]["loss_result"]["dropped"]
        for position in packet["token_positions"]
    ]

    assert dropped_positions == [0, 4]


def test_micro_eval_source_layout_changes_erased_token_geometry(tmp_path) -> None:
    output_dir = tmp_path / "source_burst"

    run_synthetic_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=4,
        mode=MICRO_EVAL_MODEL_ONLY,
        source_layout=SourceLayoutConfig(
            mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
            chunk_size=1,
        ),
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=1,
        ),
    )

    events = read_jsonl(output_dir / "events.jsonl")
    dropped_positions = [
        position
        for packet in events[0]["case"]["loss_result"]["dropped"]
        for position in packet["token_positions"]
    ]

    assert dropped_positions == [0, 2, 4, 6]
