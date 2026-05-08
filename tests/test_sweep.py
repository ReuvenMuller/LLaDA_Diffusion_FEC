import csv
import json

from diffusion_fec.channels.packet_loss import CHANNEL_BURST
from diffusion_fec.experiments.runner import main
from diffusion_fec.experiments.sweep import (
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
