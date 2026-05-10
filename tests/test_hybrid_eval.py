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
    WIRE_INTERLEAVING_MATRIX,
    SourceLayoutConfig,
    WireInterleavingConfig,
)
from diffusion_fec.experiments.hybrid_eval import (
    HYBRID_MODE_PARITY_FILTER,
    HYBRID_MODE_PRE_PEEL_ONLY,
    run_hybrid_xor_hash_micro_eval,
)
from diffusion_fec.experiments.runner import main as runner_main


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_hybrid_micro_eval_writes_artifacts_and_overhead_fields(tmp_path) -> None:
    output_dir = tmp_path / "hybrid"

    run_hybrid_xor_hash_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        loss_rate=0.5,
        seed=0,
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=64,
        steps=2,
        hybrid_mode=HYBRID_MODE_PARITY_FILTER,
        xor_overhead_bits_per_token=4.0,
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert_manifest_has_run_timing(manifest)
    assert manifest["runner"] == "hybrid_xor_hash_synthetic_micro_eval"
    assert manifest["config"]["hybrid_mode"] == HYBRID_MODE_PARITY_FILTER
    assert manifest["config"]["protection_mode"] == "lookback_1+xor_parity"
    assert manifest["config"]["oracle_hash_metadata"] is False
    assert rows[0]["hybrid_mode"] == HYBRID_MODE_PARITY_FILTER
    assert rows[0]["xor_overhead_bits_per_token"] == "4.0"
    assert float(rows[0]["hash_metadata_token_equivalent_overhead_ratio"]) > 0.0
    assert float(rows[0]["actual_repair_token_overhead_ratio"]) > 0.0
    assert abs(
        float(rows[0]["total_overhead_ratio"])
        - (
            float(rows[0]["hash_metadata_token_equivalent_overhead_ratio"])
            + float(rows[0]["actual_repair_token_overhead_ratio"])
        )
    ) < 1e-12
    assert "parity_candidate_rejections" in rows[0]
    assert "parity_filter_fallback_count" in rows[0]
    assert_row_has_run_timing(rows[0])
    assert events[0]["event_type"] == "hybrid_xor_hash_micro_eval_case"
    assert events[0]["case"]["hybrid_mode"] == HYBRID_MODE_PARITY_FILTER
    assert events[0]["case"]["decoding_result"]["diagnostics"]["hybrid_mode"] == HYBRID_MODE_PARITY_FILTER


def test_hybrid_pre_peel_mode_disables_candidate_filter(tmp_path) -> None:
    output_dir = tmp_path / "hybrid_pre_peel"

    run_hybrid_xor_hash_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        loss_rate=0.5,
        seed=0,
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=64,
        steps=2,
        hybrid_mode=HYBRID_MODE_PRE_PEEL_ONLY,
    )

    row = read_csv(output_dir / "results.csv")[0]
    event = read_jsonl(output_dir / "events.jsonl")[0]

    assert row["hybrid_mode"] == HYBRID_MODE_PRE_PEEL_ONLY
    assert row["parity_candidate_rejections"] == "0"
    assert event["case"]["parity_filter_diagnostics"]["parity_candidate_filter_calls"] == 0


def test_hybrid_burst_channel_output_is_deterministic(tmp_path) -> None:
    source_layout = SourceLayoutConfig(mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS, chunk_size=1)
    wire_interleaving = WireInterleavingConfig(mode=WIRE_INTERLEAVING_MATRIX, span=4)
    channel_config = PacketLossChannelConfig(
        mode=CHANNEL_BURST,
        loss_rate=0.5,
        seed=0,
        burst_start_wire_id=0,
        burst_length=3,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    for output_dir in (first, second):
        run_hybrid_xor_hash_micro_eval(
            output_dir=output_dir,
            sample_lengths=(12,),
            loss_rate=0.5,
            seed=0,
            tokens_per_packet=3,
            hash_bits=4,
            vocab_size=64,
            steps=2,
            hybrid_mode=HYBRID_MODE_PARITY_FILTER,
            source_layout=source_layout,
            wire_interleaving=wire_interleaving,
            channel_config=channel_config,
        )

    assert normalized_artifact_text(first / "results.csv") == normalized_artifact_text(
        second / "results.csv"
    )
    assert normalized_artifact_text(first / "events.jsonl") == normalized_artifact_text(
        second / "events.jsonl"
    )


def test_hybrid_cli_entrypoint_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "hybrid_cli"

    exit_code = runner_main(
        [
            "--output-dir",
            str(output_dir),
            "--hybrid-xor-hash-micro-eval",
            "--sample-lengths",
            "8",
            "--tokens-per-packet",
            "2",
            "--vocab-size",
            "64",
            "--steps",
            "2",
            "--hybrid-mode",
            "pre_peel_only",
            "--xor-overhead-bits-per-token",
            "4",
        ]
    )

    assert exit_code == 0
    assert read_json(output_dir / "run_manifest.json")["config"]["hybrid_mode"] == "pre_peel_only"
    assert read_csv(output_dir / "results.csv")[0]["protection_mode"] == "lookback_1+xor_parity"
