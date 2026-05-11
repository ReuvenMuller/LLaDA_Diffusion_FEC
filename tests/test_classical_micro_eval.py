import csv
import json

from artifact_helpers import (
    assert_manifest_has_run_timing,
    assert_row_has_run_timing,
    normalized_artifact_text,
)
from diffusion_fec.channels.packet_loss import CHANNEL_BURST, PacketLossChannelConfig
from diffusion_fec.coding.packetizer import (
    SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
    SourceLayoutConfig,
)
from diffusion_fec.experiments.classical_micro_eval import (
    run_lt_fountain_micro_eval,
    run_streaming_window_micro_eval,
    run_xor_parity_micro_eval,
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


def test_xor_parity_micro_eval_writes_artifacts_and_repairs_single_loss(tmp_path) -> None:
    output_dir = tmp_path / "xor"

    run_xor_parity_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=128,
        data_packets_per_stripe=2,
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=1,
        ),
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert manifest["runner"] == "xor_parity_synthetic_micro_eval"
    assert manifest["baseline_family"] == "xor_parity"
    assert manifest["config"]["xor_parity"]["data_packets_per_stripe"] == 2
    assert_manifest_has_run_timing(manifest)
    assert rows[0]["strategy"] == "Classical_XORParity_MatchedHash4"
    assert rows[0]["baseline_family"] == "xor_parity"
    assert rows[0]["known_count"] == "8"
    assert rows[0]["unguided_count"] == "0"
    assert rows[0]["exact_match"] == "True"
    assert rows[0]["lost_position_count"] == "0"
    assert rows[0]["channel_lost_position_count"] == "2"
    assert rows[0]["channel_lost_position_recovered_count"] == "2"
    assert rows[0]["channel_lost_position_recovery_rate"] == "1.0"
    assert rows[0]["repair_packet_count"] == "2"
    assert rows[0]["repair_token_budget"] == "4"
    assert rows[0]["target_overhead_ratio"] == str(4 / 7)
    assert rows[0]["hash_metadata_bit_count"] == "0"
    assert rows[0]["total_overhead_ratio"] == rows[0]["actual_repair_token_overhead_ratio"]
    assert_row_has_run_timing(rows[0])
    assert events[0]["event_type"] == "xor_parity_micro_eval_case"
    assert events[0]["metrics"]["exact_match"] is True
    assert events[0]["channel_lost_positions"] == [0, 1]
    assert events[0]["metrics"]["channel_lost_position_count"] == 2


def test_xor_parity_dropped_repair_packet_does_not_count_as_lost_source(tmp_path) -> None:
    output_dir = tmp_path / "xor_dropped_repair"

    run_xor_parity_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=128,
        data_packets_per_stripe=2,
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=4,
            burst_length=1,
        ),
    )

    row = read_csv(output_dir / "results.csv")[0]
    event = read_jsonl(output_dir / "events.jsonl")[0]

    assert row["dropped_packet_count"] == "1"
    assert row["channel_lost_position_count"] == "0"
    assert row["channel_lost_position_recovery_rate"] == "1.0"
    assert event["channel_lost_positions"] == []


def test_xor_parity_channel_lost_positions_follow_token_interleaving(tmp_path) -> None:
    output_dir = tmp_path / "xor_round_robin"

    run_xor_parity_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=4,
        hash_bits=4,
        vocab_size=128,
        data_packets_per_stripe=2,
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

    event = read_jsonl(output_dir / "events.jsonl")[0]
    row = read_csv(output_dir / "results.csv")[0]

    assert event["channel_lost_positions"] == [0, 2, 4, 6]
    assert row["channel_lost_position_count"] == "4"


def test_xor_parity_micro_eval_leaves_unrepaired_multi_loss_unguided(tmp_path) -> None:
    output_dir = tmp_path / "xor_multi_loss"

    run_xor_parity_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=128,
        data_packets_per_stripe=2,
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=2,
        ),
    )

    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert rows[0]["known_count"] == "4"
    assert rows[0]["unguided_count"] == "4"
    assert rows[0]["exact_match"] == "False"
    assert events[0]["metrics"]["remaining_mask_token_count"] == 4


def test_xor_parity_micro_eval_output_is_deterministic(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    run_xor_parity_micro_eval(
        output_dir=first_dir,
        sample_lengths=(8, 16),
        seed=5,
        tokens_per_packet=2,
        data_packets_per_stripe=2,
    )
    run_xor_parity_micro_eval(
        output_dir=second_dir,
        sample_lengths=(8, 16),
        seed=5,
        tokens_per_packet=2,
        data_packets_per_stripe=2,
    )

    for filename in ("run_manifest.json", "results.csv", "events.jsonl"):
        assert normalized_artifact_text(first_dir / filename) == normalized_artifact_text(
            second_dir / filename
        )


def test_xor_parity_cli_entrypoint_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "cli"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--xor-parity-micro-eval",
            "--sample-lengths",
            "8",
            "--tokens-per-packet",
            "2",
            "--xor-stripe-size",
            "2",
            "--channel",
            CHANNEL_BURST,
            "--burst-length",
            "1",
        ]
    )

    assert exit_code == 0
    assert read_json(output_dir / "run_manifest.json")["baseline_family"] == "xor_parity"
    assert read_csv(output_dir / "results.csv")[0]["protection_mode"] == "xor_parity"


def test_lt_fountain_micro_eval_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "lt"

    run_lt_fountain_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=128,
        repair_rate=1.0,
        lt_random_seed=3,
        coverage_aware=True,
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=1,
        ),
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert manifest["runner"] == "lt_fountain_synthetic_micro_eval"
    assert manifest["baseline_family"] == "lt_fountain"
    assert rows[0]["strategy"] == "Classical_LTFountain_CoverageAware_MatchedHash4"
    assert rows[0]["baseline_family"] == "lt_fountain"
    assert rows[0]["protection_mode"] == "lt_fountain"
    assert rows[0]["repair_packet_count"] == rows[0]["extra_packet_count"]
    assert events[0]["event_type"] == "lt_fountain_micro_eval_case"
    assert "repair_packets" in events[0]["encoded"]


def test_lt_fountain_micro_eval_output_is_deterministic(tmp_path) -> None:
    first_dir = tmp_path / "lt_first"
    second_dir = tmp_path / "lt_second"

    run_lt_fountain_micro_eval(
        output_dir=first_dir,
        sample_lengths=(8, 16),
        seed=5,
        tokens_per_packet=2,
        repair_rate=0.5,
        lt_random_seed=9,
    )
    run_lt_fountain_micro_eval(
        output_dir=second_dir,
        sample_lengths=(8, 16),
        seed=5,
        tokens_per_packet=2,
        repair_rate=0.5,
        lt_random_seed=9,
    )

    for filename in ("run_manifest.json", "results.csv", "events.jsonl"):
        assert normalized_artifact_text(first_dir / filename) == normalized_artifact_text(
            second_dir / filename
        )


def test_lt_fountain_cli_entrypoint_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "lt_cli"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--lt-fountain-micro-eval",
            "--sample-lengths",
            "8",
            "--tokens-per-packet",
            "2",
            "--lt-repair-rate",
            "1.0",
            "--lt-coverage-aware",
            "--channel",
            CHANNEL_BURST,
            "--burst-length",
            "1",
        ]
    )

    assert exit_code == 0
    assert read_json(output_dir / "run_manifest.json")["baseline_family"] == "lt_fountain"
    assert read_csv(output_dir / "results.csv")[0]["protection_mode"] == "lt_fountain"


def test_streaming_window_micro_eval_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "stream"

    run_streaming_window_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=128,
        window_size=2,
        window_stride=1,
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=1,
        ),
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert manifest["runner"] == "streaming_window_synthetic_micro_eval"
    assert manifest["baseline_family"] == "streaming_window"
    assert rows[0]["strategy"] == "Classical_StreamingWindow_MatchedHash4"
    assert rows[0]["baseline_family"] == "streaming_window"
    assert rows[0]["protection_mode"] == "streaming_window"
    assert rows[0]["repair_packet_count"] == rows[0]["extra_packet_count"]
    assert events[0]["event_type"] == "streaming_window_micro_eval_case"
    assert "repair_packets" in events[0]["encoded"]


def test_streaming_window_micro_eval_output_is_deterministic(tmp_path) -> None:
    first_dir = tmp_path / "stream_first"
    second_dir = tmp_path / "stream_second"

    run_streaming_window_micro_eval(
        output_dir=first_dir,
        sample_lengths=(8, 16),
        seed=5,
        tokens_per_packet=2,
        window_size=2,
        window_stride=1,
    )
    run_streaming_window_micro_eval(
        output_dir=second_dir,
        sample_lengths=(8, 16),
        seed=5,
        tokens_per_packet=2,
        window_size=2,
        window_stride=1,
    )

    for filename in ("run_manifest.json", "results.csv", "events.jsonl"):
        assert normalized_artifact_text(first_dir / filename) == normalized_artifact_text(
            second_dir / filename
        )


def test_streaming_window_cli_entrypoint_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "stream_cli"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--streaming-window-micro-eval",
            "--sample-lengths",
            "8",
            "--tokens-per-packet",
            "2",
            "--stream-window-size",
            "2",
            "--stream-window-stride",
            "1",
            "--channel",
            CHANNEL_BURST,
            "--burst-length",
            "1",
        ]
    )

    assert exit_code == 0
    assert read_json(output_dir / "run_manifest.json")["baseline_family"] == "streaming_window"
    assert read_csv(output_dir / "results.csv")[0]["protection_mode"] == "streaming_window"
