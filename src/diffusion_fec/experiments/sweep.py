"""Small deterministic sweep orchestration for local research-readiness checks."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from diffusion_fec.analysis.reporting import build_analysis_artifacts
from diffusion_fec.channels.packet_loss import (
    CHANNEL_BURST,
    CHANNEL_GILBERT_ELLIOTT,
    CHANNEL_RANDOM_IID,
    PacketLossChannelConfig,
)
from diffusion_fec.coding.hash_profiles import DEFAULT_HASH_MAP_MODE
from diffusion_fec.coding.packetizer import (
    SOURCE_LAYOUT_CONTIGUOUS,
    SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
    WIRE_INTERLEAVING_MATRIX,
    WIRE_INTERLEAVING_NONE,
    SourceLayoutConfig,
    WireInterleavingConfig,
)
from diffusion_fec.experiments.classical_micro_eval import (
    run_lt_fountain_micro_eval,
    run_streaming_window_micro_eval,
    run_xor_parity_micro_eval,
)
from diffusion_fec.experiments.micro_eval import (
    DEFAULT_MICRO_EVAL_HASH_BITS,
    DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    DEFAULT_MICRO_EVAL_VOCAB_SIZE,
    MICRO_EVAL_MODEL_HASH,
    MICRO_EVAL_MODEL_ONLY,
    MICRO_EVAL_WARNING,
    run_synthetic_micro_eval,
)


SWEEP_RUNNER_MODEL_ONLY = "llada_model_only_fake"
SWEEP_RUNNER_MODEL_HASH = "llada_model_hash_fake"
SWEEP_RUNNER_XOR_PARITY = "xor_parity"
SWEEP_RUNNER_LT_FOUNTAIN = "lt_fountain"
SWEEP_RUNNER_STREAMING_WINDOW = "streaming_window"
DEFAULT_SWEEP_RUNNERS = (
    SWEEP_RUNNER_MODEL_ONLY,
    SWEEP_RUNNER_MODEL_HASH,
    SWEEP_RUNNER_XOR_PARITY,
    SWEEP_RUNNER_LT_FOUNTAIN,
    SWEEP_RUNNER_STREAMING_WINDOW,
)


@dataclass(frozen=True)
class SweepRunSpec:
    """One child run in a synthetic micro-eval sweep."""

    name: str
    runner: str
    sample_lengths: tuple[int, ...] = DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS
    loss_rate: float = 0.5
    seed: int = 0
    tokens_per_packet: int = DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET
    hash_bits: int = DEFAULT_MICRO_EVAL_HASH_BITS
    vocab_size: int = DEFAULT_MICRO_EVAL_VOCAB_SIZE
    source_layout: SourceLayoutConfig = field(default_factory=SourceLayoutConfig)
    wire_interleaving: WireInterleavingConfig = field(default_factory=WireInterleavingConfig)
    channel_config: PacketLossChannelConfig = field(default_factory=PacketLossChannelConfig)
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("sweep run name must be non-empty")
        if self.runner not in set(DEFAULT_SWEEP_RUNNERS):
            raise ValueError(f"unknown sweep runner: {self.runner}")
        object.__setattr__(self, "sample_lengths", tuple(self.sample_lengths))
        object.__setattr__(self, "options", dict(self.options))
        if not self.sample_lengths:
            raise ValueError("sample_lengths must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runner": self.runner,
            "sample_lengths": list(self.sample_lengths),
            "loss_rate": self.loss_rate,
            "seed": self.seed,
            "tokens_per_packet": self.tokens_per_packet,
            "hash_bits": self.hash_bits,
            "vocab_size": self.vocab_size,
            "source_layout": self.source_layout.to_dict(),
            "wire_interleaving": self.wire_interleaving.to_dict(),
            "channel": self.channel_config.to_dict(),
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class SyntheticSweepConfig:
    """Serializable config for a model-free sweep."""

    specs: tuple[SweepRunSpec, ...]
    profile_name: str = "fake_synthetic_sweep_v1"
    hash_map_mode: str = DEFAULT_HASH_MAP_MODE
    skip_completed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", tuple(self.specs))
        if not self.specs:
            raise ValueError("sweep config must contain at least one run spec")
        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("sweep run names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "hash_map_mode": self.hash_map_mode,
            "skip_completed": self.skip_completed,
            "specs": [spec.to_dict() for spec in self.specs],
        }


def build_synthetic_sweep_config(
    *,
    sample_lengths: Sequence[int] = DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    loss_rates: Sequence[float] = (0.5,),
    seed: int = 0,
    tokens_per_packet: int = DEFAULT_MICRO_EVAL_TOKENS_PER_PACKET,
    hash_bits: int = DEFAULT_MICRO_EVAL_HASH_BITS,
    vocab_size: int = DEFAULT_MICRO_EVAL_VOCAB_SIZE,
    runners: Sequence[str] = DEFAULT_SWEEP_RUNNERS,
    source_layouts: Sequence[SourceLayoutConfig] | None = None,
    wire_interleavings: Sequence[WireInterleavingConfig] | None = None,
    channel_modes: Sequence[str] = (CHANNEL_RANDOM_IID,),
    burst_length: int = 2,
    ge_good_loss_rate: float = 0.0,
    ge_bad_loss_rate: float = 1.0,
    ge_good_to_bad_rate: float = 0.05,
    ge_bad_to_good_rate: float = 0.5,
    ge_initial_state: str = "good",
) -> SyntheticSweepConfig:
    """Build a compact model-free sweep covering selected strategies and geometry."""

    source_layouts = tuple(source_layouts or (SourceLayoutConfig(mode=SOURCE_LAYOUT_CONTIGUOUS),))
    wire_interleavings = tuple(wire_interleavings or (WireInterleavingConfig(mode=WIRE_INTERLEAVING_NONE),))
    specs: list[SweepRunSpec] = []
    for runner in runners:
        if runner not in set(DEFAULT_SWEEP_RUNNERS):
            raise ValueError(f"unknown sweep runner: {runner}")
        for loss_rate in loss_rates:
            for source_layout in source_layouts:
                for wire_interleaving in wire_interleavings:
                    for channel_mode in channel_modes:
                        channel_config = _channel_config(
                            channel_mode=channel_mode,
                            loss_rate=loss_rate,
                            seed=seed,
                            burst_length=burst_length,
                            ge_good_loss_rate=ge_good_loss_rate,
                            ge_bad_loss_rate=ge_bad_loss_rate,
                            ge_good_to_bad_rate=ge_good_to_bad_rate,
                            ge_bad_to_good_rate=ge_bad_to_good_rate,
                            ge_initial_state=ge_initial_state,
                        )
                        specs.append(
                            SweepRunSpec(
                                name=_spec_name(
                                    runner=runner,
                                    loss_rate=loss_rate,
                                    source_layout=source_layout,
                                    wire_interleaving=wire_interleaving,
                                    channel_config=channel_config,
                                    hash_bits=hash_bits,
                                ),
                                runner=runner,
                                sample_lengths=tuple(sample_lengths),
                                loss_rate=loss_rate,
                                seed=seed,
                                tokens_per_packet=tokens_per_packet,
                                hash_bits=hash_bits,
                                vocab_size=vocab_size,
                                source_layout=source_layout,
                                wire_interleaving=wire_interleaving,
                                channel_config=channel_config,
                                options=_default_runner_options(runner),
                            )
                        )
    return SyntheticSweepConfig(specs=tuple(specs))


def run_synthetic_sweep(
    *,
    output_dir: str | Path,
    config: SyntheticSweepConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run a model-free synthetic sweep and write sweep plus analysis artifacts."""

    output = Path(output_dir)
    runs_dir = output / "runs"
    analysis_dir = output / "analysis"
    profile_dir = output / "hash_profiles" / config.profile_name
    output.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    for spec in config.specs:
        child_dir = runs_dir / spec.name
        completed = _run_artifacts_complete(child_dir)
        if completed and config.skip_completed and not overwrite:
            status = "skipped_existing"
        else:
            _execute_spec(
                spec=spec,
                output_dir=child_dir,
                profile_dir=profile_dir,
                profile_name=config.profile_name,
                hash_map_mode=config.hash_map_mode,
            )
            status = "completed"
        run_rows.append(
            {
                "name": spec.name,
                "runner": spec.runner,
                "status": status,
                "output_dir": str(child_dir.relative_to(output)),
                "results": str((child_dir / "results.csv").relative_to(output)),
                "events": str((child_dir / "events.jsonl").relative_to(output)),
            }
        )

    analysis_manifest = build_analysis_artifacts(
        run_root=runs_dir,
        output_dir=analysis_dir,
    )
    sweep_manifest = {
        "artifact_kind": "synthetic_sweep",
        "not_a_research_claim": True,
        "not_a_research_claim_warning": MICRO_EVAL_WARNING,
        "config": config.to_dict(),
        "child_runs_dir": "runs",
        "analysis_dir": "analysis",
        "run_count": len(run_rows),
        "completed_run_count": sum(row["status"] == "completed" for row in run_rows),
        "skipped_existing_run_count": sum(row["status"] == "skipped_existing" for row in run_rows),
        "analysis_manifest": "analysis/analysis_manifest.json",
    }
    _write_json(output / "sweep_manifest.json", sweep_manifest)
    _write_sweep_rows(output / "sweep_runs.csv", run_rows)
    return {
        "manifest": sweep_manifest,
        "run_rows": tuple(run_rows),
        "analysis_manifest": analysis_manifest,
    }


def _execute_spec(
    *,
    spec: SweepRunSpec,
    output_dir: Path,
    profile_dir: Path,
    profile_name: str,
    hash_map_mode: str,
) -> None:
    if spec.runner == SWEEP_RUNNER_MODEL_ONLY:
        run_synthetic_micro_eval(
            output_dir=output_dir,
            sample_lengths=spec.sample_lengths,
            loss_rate=spec.loss_rate,
            seed=spec.seed,
            tokens_per_packet=spec.tokens_per_packet,
            mode=MICRO_EVAL_MODEL_ONLY,
            hash_bits=spec.hash_bits,
            vocab_size=spec.vocab_size,
            source_layout=spec.source_layout,
            wire_interleaving=spec.wire_interleaving,
            channel_config=spec.channel_config,
            hash_map_mode=hash_map_mode,
            hash_profile_name=profile_name,
        )
        return
    if spec.runner == SWEEP_RUNNER_MODEL_HASH:
        run_synthetic_micro_eval(
            output_dir=output_dir,
            sample_lengths=spec.sample_lengths,
            loss_rate=spec.loss_rate,
            seed=spec.seed,
            tokens_per_packet=spec.tokens_per_packet,
            mode=MICRO_EVAL_MODEL_HASH,
            hash_bits=spec.hash_bits,
            vocab_size=spec.vocab_size,
            source_layout=spec.source_layout,
            wire_interleaving=spec.wire_interleaving,
            channel_config=spec.channel_config,
            hash_profile_dir=profile_dir,
            build_hash_profile=True,
            hash_map_mode=hash_map_mode,
            hash_profile_name=profile_name,
        )
        return
    if spec.runner == SWEEP_RUNNER_XOR_PARITY:
        run_xor_parity_micro_eval(
            output_dir=output_dir,
            sample_lengths=spec.sample_lengths,
            loss_rate=spec.loss_rate,
            seed=spec.seed,
            tokens_per_packet=spec.tokens_per_packet,
            hash_bits=spec.hash_bits,
            vocab_size=spec.vocab_size,
            source_layout=spec.source_layout,
            wire_interleaving=spec.wire_interleaving,
            channel_config=spec.channel_config,
            data_packets_per_stripe=int(spec.options.get("data_packets_per_stripe", 4)),
            stripe_stride=spec.options.get("stripe_stride"),
        )
        return
    if spec.runner == SWEEP_RUNNER_LT_FOUNTAIN:
        run_lt_fountain_micro_eval(
            output_dir=output_dir,
            sample_lengths=spec.sample_lengths,
            loss_rate=spec.loss_rate,
            seed=spec.seed,
            tokens_per_packet=spec.tokens_per_packet,
            hash_bits=spec.hash_bits,
            vocab_size=spec.vocab_size,
            source_layout=spec.source_layout,
            wire_interleaving=spec.wire_interleaving,
            channel_config=spec.channel_config,
            repair_rate=float(spec.options.get("repair_rate", 0.25)),
            lt_random_seed=int(spec.options.get("lt_random_seed", 7)),
            coverage_aware=bool(spec.options.get("coverage_aware", False)),
        )
        return
    run_streaming_window_micro_eval(
        output_dir=output_dir,
        sample_lengths=spec.sample_lengths,
        loss_rate=spec.loss_rate,
        seed=spec.seed,
        tokens_per_packet=spec.tokens_per_packet,
        hash_bits=spec.hash_bits,
        vocab_size=spec.vocab_size,
        source_layout=spec.source_layout,
        wire_interleaving=spec.wire_interleaving,
        channel_config=spec.channel_config,
        window_size=int(spec.options.get("window_size", 5)),
        window_stride=int(spec.options.get("window_stride", 1)),
    )


def _default_runner_options(runner: str) -> dict[str, Any]:
    if runner == SWEEP_RUNNER_XOR_PARITY:
        return {"data_packets_per_stripe": 4, "stripe_stride": None}
    if runner == SWEEP_RUNNER_LT_FOUNTAIN:
        return {"repair_rate": 0.25, "lt_random_seed": 7, "coverage_aware": True}
    if runner == SWEEP_RUNNER_STREAMING_WINDOW:
        return {"window_size": 5, "window_stride": 1}
    return {}


def _channel_config(
    *,
    channel_mode: str,
    loss_rate: float,
    seed: int,
    burst_length: int,
    ge_good_loss_rate: float,
    ge_bad_loss_rate: float,
    ge_good_to_bad_rate: float,
    ge_bad_to_good_rate: float,
    ge_initial_state: str,
) -> PacketLossChannelConfig:
    if channel_mode == CHANNEL_RANDOM_IID:
        return PacketLossChannelConfig(mode=CHANNEL_RANDOM_IID, loss_rate=loss_rate, seed=seed)
    if channel_mode == CHANNEL_BURST:
        return PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            loss_rate=loss_rate,
            seed=seed,
            burst_start_wire_id=0,
            burst_length=burst_length,
        )
    if channel_mode == CHANNEL_GILBERT_ELLIOTT:
        return PacketLossChannelConfig(
            mode=CHANNEL_GILBERT_ELLIOTT,
            loss_rate=loss_rate,
            seed=seed,
            good_loss_rate=ge_good_loss_rate,
            bad_loss_rate=ge_bad_loss_rate,
            good_to_bad_rate=ge_good_to_bad_rate,
            bad_to_good_rate=ge_bad_to_good_rate,
            initial_state=ge_initial_state,
        )
    raise ValueError("synthetic sweep supports random_iid, burst, and gilbert_elliott channels")


def _spec_name(
    *,
    runner: str,
    loss_rate: float,
    source_layout: SourceLayoutConfig,
    wire_interleaving: WireInterleavingConfig,
    channel_config: PacketLossChannelConfig,
    hash_bits: int,
) -> str:
    source_chunk = source_layout.chunk_size if source_layout.chunk_size is not None else "default"
    return _slug(
        "|".join(
            [
                runner,
                f"hash{hash_bits}",
                f"loss{loss_rate:g}",
                f"source-{source_layout.mode}-chunk{source_chunk}",
                f"wire-{wire_interleaving.mode}-span{wire_interleaving.span}",
                f"channel-{channel_config.mode}",
            ]
        )
    )


def _slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def _run_artifacts_complete(path: Path) -> bool:
    return all((path / filename).exists() for filename in ("run_manifest.json", "results.csv", "events.jsonl"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sweep_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = ["name", "runner", "status", "output_dir", "results", "events"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def default_interleaving_source_layouts() -> tuple[SourceLayoutConfig, ...]:
    """Return the source layouts used for interleaving-sensitive sweeps."""

    return (
        SourceLayoutConfig(mode=SOURCE_LAYOUT_CONTIGUOUS),
        SourceLayoutConfig(mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS, chunk_size=1),
    )


def default_interleaving_wire_orders() -> tuple[WireInterleavingConfig, ...]:
    """Return the wire orders used for interleaving-sensitive sweeps."""

    return (
        WireInterleavingConfig(mode=WIRE_INTERLEAVING_NONE),
        WireInterleavingConfig(mode=WIRE_INTERLEAVING_MATRIX, span=4),
    )
