"""Minimal deterministic smoke runner.

This runner intentionally uses a fake deterministic model. It is an artifact-writing
smoke test for the local pipeline, not a real LLaDA baseline.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from diffusion_fec.channels.gilbert_elliott import GE_STATE_BAD, GE_STATE_GOOD
from diffusion_fec.channels.packet_loss import (
    CHANNEL_BURST,
    CHANNEL_GILBERT_ELLIOTT,
    CHANNEL_RANDOM_IID,
    PacketLossChannelConfig,
)
from diffusion_fec.coding.hash_profiles import DEFAULT_HASH_MAP_MODE, load_or_build_hash_profile
from diffusion_fec.coding.packetizer import (
    SOURCE_LAYOUT_CONTIGUOUS,
    SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
    WIRE_INTERLEAVING_MATRIX,
    WIRE_INTERLEAVING_NONE,
    SourceLayoutConfig,
    WireInterleavingConfig,
)
from diffusion_fec.coding.protection import LOOKBACK_1_SCHEME
from diffusion_fec.decoding.llada_diffusion import DiffusionDecodingConfig
from diffusion_fec.experiments.logging import write_run_artifacts
from diffusion_fec.experiments.micro_eval import (
    DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS,
    MICRO_EVAL_MODEL_HASH,
    MICRO_EVAL_MODEL_ONLY,
    run_synthetic_micro_eval,
)
from diffusion_fec.experiments.smoke import SmokeRecoveryCase, run_smoke_recovery_case
from diffusion_fec.types import TokenSample


FAKE_MODEL_LABEL = "FakeDeterministicSmokeModel"
SMOKE_STRATEGY = "FakeSmoke_LookbackHash_NoPrompt"
DEFAULT_VOCAB_SIZE = 128
DEFAULT_MASK_TOKEN_ID = 0
DEFAULT_EOS_TOKEN_ID = 1
DEFAULT_PAD_TOKEN_ID = 2


@dataclass
class FakeForwardOutput:
    logits: list[list[list[float]]]


class FakeDeterministicSmokeModel:
    """Tiny oracle-like model for pipeline smoke artifacts only."""

    def __init__(self, target_tokens: tuple[int, ...], vocab_size: int):
        self.target_tokens = target_tokens
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        sequence_length = len(input_ids[0])
        logits = [[[0.0 for _ in range(self.vocab_size)] for _ in range(sequence_length)]]
        for position, token_id in enumerate(self.target_tokens):
            if position < sequence_length:
                logits[0][position][token_id] = 100.0 - position
        return FakeForwardOutput(logits=logits)

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)


def run_minimal_smoke(
    *,
    output_dir: str | Path,
    sample_count: int = 2,
    loss_rate: float = 0.5,
    seed: int = 0,
    tokens_per_packet: int = 1,
    protection_mode: str = LOOKBACK_1_SCHEME,
    hash_bits: int = 4,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    steps: int = 4,
    hash_profile_dir: str | Path | None = None,
    build_hash_profile: bool = False,
    hash_map_mode: str = DEFAULT_HASH_MAP_MODE,
    hash_profile_name: str = "fake_smoke_v1",
) -> dict[str, Any]:
    """Run tiny deterministic fake-model smoke cases and write artifacts."""

    _validate_runner_config(
        sample_count=sample_count,
        loss_rate=loss_rate,
        tokens_per_packet=tokens_per_packet,
        vocab_size=vocab_size,
        steps=steps,
    )
    if protection_mode not in {"none", LOOKBACK_1_SCHEME}:
        raise ValueError("protection_mode must be 'none' or 'lookback_1'")

    config = DiffusionDecodingConfig(
        mask_token_id=DEFAULT_MASK_TOKEN_ID,
        eos_token_id=DEFAULT_EOS_TOKEN_ID,
        pad_token_id=DEFAULT_PAD_TOKEN_ID,
        vocab_size=vocab_size,
        steps=steps,
        block_length=max(tokens_per_packet, 1),
    )
    excluded_token_ids = {
        DEFAULT_MASK_TOKEN_ID,
        DEFAULT_EOS_TOKEN_ID,
        DEFAULT_PAD_TOKEN_ID,
    }
    token_hash_map, hash_profile_info = load_or_build_hash_profile(
        profile_dir=hash_profile_dir,
        profile_name=hash_profile_name,
        vocab_size=vocab_size,
        hash_bits=hash_bits,
        decode_token=lambda token_id: f"fake-token-{token_id}",
        excluded_token_ids=excluded_token_ids,
        salt="fake-smoke",
        map_mode=hash_map_mode,
        model_id=FAKE_MODEL_LABEL,
        tokenizer_name="fake-deterministic-tokenizer",
        build_if_missing=build_hash_profile,
    )
    run_id = _run_id(
        sample_count=sample_count,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        protection_mode=protection_mode,
        hash_bits=hash_bits,
    )
    manifest = _manifest(
        run_id=run_id,
        sample_count=sample_count,
        loss_rate=loss_rate,
        seed=seed,
        tokens_per_packet=tokens_per_packet,
        protection_mode=protection_mode,
        hash_bits=hash_bits,
        vocab_size=vocab_size,
        steps=steps,
        hash_profile_info=hash_profile_info,
    )

    result_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for case_index in range(sample_count):
        sample = _synthetic_sample(case_index, vocab_size)
        model = FakeDeterministicSmokeModel(
            target_tokens=sample.token_ids,
            vocab_size=vocab_size,
        )
        case_seed = seed + case_index
        case = run_smoke_recovery_case(
            sample=sample,
            model=model,
            config=config,
            tokens_per_packet=tokens_per_packet,
            loss_rate=loss_rate,
            seed=case_seed,
            token_hash_map=token_hash_map if protection_mode == LOOKBACK_1_SCHEME else None,
            protection_mode=protection_mode,
        )
        case_id = f"case{case_index:04d}"
        result_rows.append(
            _result_row(
                run_id=run_id,
                case_id=case_id,
                case=case,
                sample_index=case_index,
                case_seed=case_seed,
                loss_rate=loss_rate,
                tokens_per_packet=tokens_per_packet,
                protection_mode=protection_mode,
                hash_bits=hash_bits,
            )
        )
        events.append(
            _event(
                run_id=run_id,
                case_id=case_id,
                case=case,
            )
        )

    write_run_artifacts(
        output_dir=output_dir,
        manifest=manifest,
        result_rows=result_rows,
        events=events,
    )
    return {
        "run_id": run_id,
        "manifest": manifest,
        "result_rows": result_rows,
        "events": events,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fake deterministic smoke artifacts.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--build-llada-tokenized-artifact", action="store_true")
    parser.add_argument("--micro-eval", action="store_true")
    parser.add_argument("--xor-parity-micro-eval", action="store_true")
    parser.add_argument("--lt-fountain-micro-eval", action="store_true")
    parser.add_argument("--streaming-window-micro-eval", action="store_true")
    parser.add_argument("--synthetic-sweep", action="store_true")
    parser.add_argument("--real-llada-smoke", action="store_true")
    parser.add_argument("--real-llada-micro-eval", action="store_true")
    parser.add_argument("--llada-model-id", default="GSAI-ML/LLaDA-1.5")
    parser.add_argument("--llada-local-files-only", action="store_true")
    parser.add_argument("--allow-cpu-real-llada", action="store_true")
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--dataset-file")
    parser.add_argument("--dataset-label")
    parser.add_argument("--dataset-sample-count", type=int)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--dataset-min-tokens", type=int, default=1)
    parser.add_argument("--dataset-max-tokens", type=int)
    parser.add_argument("--tokenized-samples-file")
    parser.add_argument("--tokenized-output-file")
    parser.add_argument("--source-dataset-manifest")
    parser.add_argument("--loss-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens-per-packet", type=int, default=1)
    parser.add_argument("--protection-mode", default=LOOKBACK_1_SCHEME, choices=["none", LOOKBACK_1_SCHEME])
    parser.add_argument("--hash-bits", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--hash-profile-dir")
    parser.add_argument("--build-hash-profile", action="store_true")
    parser.add_argument("--hash-map-mode", default=DEFAULT_HASH_MAP_MODE)
    parser.add_argument("--hash-profile-name")
    parser.add_argument(
        "--sample-lengths",
        default=None,
        help="Comma-separated synthetic micro-eval sample lengths.",
    )
    parser.add_argument(
        "--micro-eval-mode",
        default=MICRO_EVAL_MODEL_HASH,
        choices=[MICRO_EVAL_MODEL_ONLY, MICRO_EVAL_MODEL_HASH],
    )
    parser.add_argument(
        "--source-layout",
        default=SOURCE_LAYOUT_CONTIGUOUS,
        choices=[SOURCE_LAYOUT_CONTIGUOUS, SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS],
    )
    parser.add_argument("--source-chunk-size", type=int, default=1)
    parser.add_argument(
        "--wire-interleaving",
        default=WIRE_INTERLEAVING_NONE,
        choices=[WIRE_INTERLEAVING_NONE, WIRE_INTERLEAVING_MATRIX],
    )
    parser.add_argument("--wire-interleaving-span", type=int, default=4)
    parser.add_argument(
        "--channel",
        default=CHANNEL_RANDOM_IID,
        choices=[CHANNEL_RANDOM_IID, CHANNEL_BURST, CHANNEL_GILBERT_ELLIOTT],
    )
    parser.add_argument("--burst-start-wire-id", type=int, default=0)
    parser.add_argument("--burst-length", type=int)
    parser.add_argument("--ge-good-loss-rate", type=float, default=0.0)
    parser.add_argument("--ge-bad-loss-rate", type=float, default=1.0)
    parser.add_argument("--ge-good-to-bad-rate", type=float, default=0.05)
    parser.add_argument("--ge-bad-to-good-rate", type=float, default=0.5)
    parser.add_argument(
        "--ge-initial-state",
        default=GE_STATE_GOOD,
        choices=[GE_STATE_GOOD, GE_STATE_BAD],
    )
    parser.add_argument("--xor-stripe-size", type=int, default=4)
    parser.add_argument("--xor-stripe-stride", type=int)
    parser.add_argument("--lt-repair-rate", type=float, default=0.25)
    parser.add_argument("--lt-random-seed", type=int, default=7)
    parser.add_argument("--lt-coverage-aware", action="store_true")
    parser.add_argument("--stream-window-size", type=int, default=5)
    parser.add_argument("--stream-window-stride", type=int, default=1)
    parser.add_argument(
        "--sweep-loss-rates",
        default=None,
        help="Comma-separated loss rates for --synthetic-sweep.",
    )
    parser.add_argument(
        "--sweep-runners",
        default=None,
        help="Comma-separated synthetic sweep runners. Defaults to the main comparison set.",
    )
    parser.add_argument("--sweep-include-interleaving-variants", action="store_true")
    parser.add_argument("--sweep-include-burst", action="store_true")
    parser.add_argument("--sweep-overwrite", action="store_true")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    vocab_size_was_explicit = "--vocab-size" in raw_argv
    if args.dataset_file and args.tokenized_samples_file and not args.build_llada_tokenized_artifact:
        parser.error("--dataset-file and --tokenized-samples-file are separate run inputs")
    if args.build_llada_tokenized_artifact and args.tokenized_samples_file:
        parser.error("--build-llada-tokenized-artifact writes, rather than consumes, tokenized samples")
    if args.build_llada_tokenized_artifact and not args.dataset_file:
        parser.error("--dataset-file is required with --build-llada-tokenized-artifact")

    dataset_samples = None
    dataset_info = None
    if args.tokenized_samples_file and not args.real_llada_micro_eval and not args.real_llada_smoke:
        dataset_samples, dataset_info = _tokenized_samples_from_args(
            args,
            expected_vocab_size=args.vocab_size if vocab_size_was_explicit else None,
        )
        if not vocab_size_was_explicit and dataset_info.get("vocab_size") is not None:
            args.vocab_size = int(dataset_info["vocab_size"])
    elif (
        args.dataset_file
        and not args.real_llada_micro_eval
        and not args.real_llada_smoke
        and not args.build_llada_tokenized_artifact
    ):
        dataset_samples, dataset_info = _fake_dataset_samples_from_args(args)

    selected_runners = [
        args.build_llada_tokenized_artifact,
        args.micro_eval,
        args.xor_parity_micro_eval,
        args.lt_fountain_micro_eval,
        args.streaming_window_micro_eval,
        args.synthetic_sweep,
        args.real_llada_smoke,
        args.real_llada_micro_eval,
    ]
    if sum(1 for selected in selected_runners if selected) > 1:
        parser.error(
            "--build-llada-tokenized-artifact, --micro-eval, --xor-parity-micro-eval, "
            "--lt-fountain-micro-eval, --streaming-window-micro-eval, --synthetic-sweep, "
            "--real-llada-smoke, and --real-llada-micro-eval are separate runners"
        )

    if args.build_llada_tokenized_artifact:
        from diffusion_fec.experiments.llada_tokenized_artifact import (
            LLaDATokenizedArtifactUnavailable,
            build_llada_tokenized_dataset_artifact,
        )

        output_path = (
            Path(args.tokenized_output_file)
            if args.tokenized_output_file
            else Path(args.output_dir) / "llada_tokenized_samples.json"
        )
        try:
            build_llada_tokenized_dataset_artifact(
                dataset_path=args.dataset_file,
                output_path=output_path,
                model_id=args.llada_model_id,
                dataset_label=args.dataset_label,
                sample_count=args.dataset_sample_count,
                seed=args.dataset_seed,
                min_tokens=args.dataset_min_tokens,
                max_tokens=args.dataset_max_tokens,
                source_dataset_manifest_path=args.source_dataset_manifest,
                local_files_only=args.llada_local_files_only,
            )
        except LLaDATokenizedArtifactUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0

    if args.synthetic_sweep:
        from diffusion_fec.experiments.sweep import (
            DEFAULT_SWEEP_RUNNERS,
            build_synthetic_sweep_config,
            default_interleaving_source_layouts,
            default_interleaving_wire_orders,
            run_synthetic_sweep,
        )

        source_layouts = (
            default_interleaving_source_layouts()
            if args.sweep_include_interleaving_variants
            else (_source_layout_from_args(args),)
        )
        wire_interleavings = (
            default_interleaving_wire_orders()
            if args.sweep_include_interleaving_variants
            else (_wire_interleaving_from_args(args),)
        )
        channel_modes = (
            tuple(dict.fromkeys((CHANNEL_RANDOM_IID, CHANNEL_BURST, args.channel)))
            if args.sweep_include_burst
            else (args.channel,)
        )
        config = build_synthetic_sweep_config(
            sample_lengths=_parse_sample_lengths(
                args.sample_lengths,
                default=(
                    tuple(len(sample.token_ids) for sample in dataset_samples)
                    if dataset_samples is not None
                    else DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS
                ),
            ),
            loss_rates=_parse_float_list(args.sweep_loss_rates, default=(args.loss_rate,)),
            seed=args.seed,
            tokens_per_packet=args.tokens_per_packet,
            hash_bits=args.hash_bits,
            vocab_size=args.vocab_size,
            runners=_parse_string_list(args.sweep_runners, default=DEFAULT_SWEEP_RUNNERS),
            source_layouts=source_layouts,
            wire_interleavings=wire_interleavings,
            channel_modes=channel_modes,
            burst_length=1 if args.burst_length is None else args.burst_length,
            ge_good_loss_rate=args.ge_good_loss_rate,
            ge_bad_loss_rate=args.ge_bad_loss_rate,
            ge_good_to_bad_rate=args.ge_good_to_bad_rate,
            ge_bad_to_good_rate=args.ge_bad_to_good_rate,
            ge_initial_state=args.ge_initial_state,
        )
        run_synthetic_sweep(
            output_dir=args.output_dir,
            config=config,
            samples=dataset_samples,
            dataset_info=dataset_info,
            overwrite=args.sweep_overwrite,
        )
        return 0

    if args.micro_eval:
        run_synthetic_micro_eval(
            output_dir=args.output_dir,
            sample_lengths=_parse_sample_lengths(
                args.sample_lengths,
                default=(
                    tuple(len(sample.token_ids) for sample in dataset_samples)
                    if dataset_samples is not None
                    else DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS
                ),
            ),
            samples=dataset_samples,
            dataset_info=dataset_info,
            loss_rate=args.loss_rate,
            seed=args.seed,
            tokens_per_packet=args.tokens_per_packet,
            mode=args.micro_eval_mode,
            hash_bits=args.hash_bits,
            vocab_size=args.vocab_size,
            steps=args.steps,
            source_layout=_source_layout_from_args(args),
            wire_interleaving=_wire_interleaving_from_args(args),
            channel_config=_channel_config_from_args(args),
            hash_profile_dir=args.hash_profile_dir,
            build_hash_profile=args.build_hash_profile,
            hash_map_mode=args.hash_map_mode,
            hash_profile_name=args.hash_profile_name or "fake_micro_eval_v1",
        )
        return 0

    if args.xor_parity_micro_eval:
        from diffusion_fec.experiments.classical_micro_eval import run_xor_parity_micro_eval

        run_xor_parity_micro_eval(
            output_dir=args.output_dir,
            sample_lengths=_parse_sample_lengths(
                args.sample_lengths,
                default=(
                    tuple(len(sample.token_ids) for sample in dataset_samples)
                    if dataset_samples is not None
                    else DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS
                ),
            ),
            samples=dataset_samples,
            dataset_info=dataset_info,
            loss_rate=args.loss_rate,
            seed=args.seed,
            tokens_per_packet=args.tokens_per_packet,
            hash_bits=args.hash_bits,
            vocab_size=args.vocab_size,
            source_layout=_source_layout_from_args(args),
            wire_interleaving=_wire_interleaving_from_args(args),
            channel_config=_channel_config_from_args(args),
            data_packets_per_stripe=args.xor_stripe_size,
            stripe_stride=args.xor_stripe_stride,
        )
        return 0

    if args.lt_fountain_micro_eval:
        from diffusion_fec.experiments.classical_micro_eval import run_lt_fountain_micro_eval

        run_lt_fountain_micro_eval(
            output_dir=args.output_dir,
            sample_lengths=_parse_sample_lengths(
                args.sample_lengths,
                default=(
                    tuple(len(sample.token_ids) for sample in dataset_samples)
                    if dataset_samples is not None
                    else DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS
                ),
            ),
            samples=dataset_samples,
            dataset_info=dataset_info,
            loss_rate=args.loss_rate,
            seed=args.seed,
            tokens_per_packet=args.tokens_per_packet,
            hash_bits=args.hash_bits,
            vocab_size=args.vocab_size,
            source_layout=_source_layout_from_args(args),
            wire_interleaving=_wire_interleaving_from_args(args),
            channel_config=_channel_config_from_args(args),
            repair_rate=args.lt_repair_rate,
            lt_random_seed=args.lt_random_seed,
            coverage_aware=args.lt_coverage_aware,
        )
        return 0

    if args.streaming_window_micro_eval:
        from diffusion_fec.experiments.classical_micro_eval import run_streaming_window_micro_eval

        run_streaming_window_micro_eval(
            output_dir=args.output_dir,
            sample_lengths=_parse_sample_lengths(
                args.sample_lengths,
                default=(
                    tuple(len(sample.token_ids) for sample in dataset_samples)
                    if dataset_samples is not None
                    else DEFAULT_MICRO_EVAL_SAMPLE_LENGTHS
                ),
            ),
            samples=dataset_samples,
            dataset_info=dataset_info,
            loss_rate=args.loss_rate,
            seed=args.seed,
            tokens_per_packet=args.tokens_per_packet,
            hash_bits=args.hash_bits,
            vocab_size=args.vocab_size,
            source_layout=_source_layout_from_args(args),
            wire_interleaving=_wire_interleaving_from_args(args),
            channel_config=_channel_config_from_args(args),
            window_size=args.stream_window_size,
            window_stride=args.stream_window_stride,
        )
        return 0

    if args.real_llada_micro_eval:
        if args.build_hash_profile:
            parser.error(
                "--build-hash-profile is not allowed for real LLaDA micro-eval; "
                "build profiles ahead of time and pass --hash-profile-dir"
            )
        from diffusion_fec.experiments.llada_micro_eval import (
            DEFAULT_REAL_LLADA_MICRO_EVAL_SAMPLE_LENGTHS,
            RealLLaDAMicroEvalUnavailable,
            run_real_llada_micro_eval,
        )

        try:
            run_real_llada_micro_eval(
                output_dir=args.output_dir,
                model_id=args.llada_model_id,
                sample_lengths=_parse_sample_lengths(
                    args.sample_lengths,
                    default=DEFAULT_REAL_LLADA_MICRO_EVAL_SAMPLE_LENGTHS,
                ),
                loss_rate=args.loss_rate,
                seed=args.seed,
                tokens_per_packet=args.tokens_per_packet,
                mode=args.micro_eval_mode,
                hash_bits=args.hash_bits,
                steps=args.steps,
                local_files_only=args.llada_local_files_only,
                allow_cpu=args.allow_cpu_real_llada,
                hash_profile_dir=args.hash_profile_dir,
                hash_map_mode=args.hash_map_mode,
                dataset_path=args.dataset_file,
                dataset_label=args.dataset_label,
                dataset_sample_count=args.dataset_sample_count,
                dataset_seed=args.dataset_seed,
                dataset_min_tokens=args.dataset_min_tokens,
                dataset_max_tokens=args.dataset_max_tokens,
                tokenized_samples_path=args.tokenized_samples_file,
                source_layout=_source_layout_from_args(args),
                wire_interleaving=_wire_interleaving_from_args(args),
                channel_config=_channel_config_from_args(args),
            )
        except RealLLaDAMicroEvalUnavailable as exc:
            print(f"Real LLaDA micro-eval unavailable: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.real_llada_smoke:
        from diffusion_fec.experiments.llada_smoke import (
            RealLLaDASmokeUnavailable,
            run_real_llada_smoke,
        )

        try:
            run_real_llada_smoke(
                output_dir=args.output_dir,
                model_id=args.llada_model_id,
                seed=args.seed,
                loss_rate=args.loss_rate,
                tokens_per_packet=args.tokens_per_packet,
                hash_bits=args.hash_bits,
                steps=args.steps,
                local_files_only=args.llada_local_files_only,
                allow_cpu=args.allow_cpu_real_llada,
                hash_profile_dir=args.hash_profile_dir,
                build_hash_profile=args.build_hash_profile,
                hash_map_mode=args.hash_map_mode,
                hash_profile_name=args.hash_profile_name,
            )
        except RealLLaDASmokeUnavailable as exc:
            print(f"Real LLaDA smoke unavailable: {exc}", file=sys.stderr)
            return 2
        return 0

    run_minimal_smoke(
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        loss_rate=args.loss_rate,
        seed=args.seed,
        tokens_per_packet=args.tokens_per_packet,
        protection_mode=args.protection_mode,
        hash_bits=args.hash_bits,
        vocab_size=args.vocab_size,
        steps=args.steps,
        hash_profile_dir=args.hash_profile_dir,
        build_hash_profile=args.build_hash_profile,
        hash_map_mode=args.hash_map_mode,
        hash_profile_name=args.hash_profile_name or "fake_smoke_v1",
    )
    return 0


def _parse_sample_lengths(raw: str | None, *, default: Sequence[int]) -> tuple[int, ...]:
    if raw is None:
        return tuple(default)
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("sample lengths must contain at least one value")
    return tuple(values)


def _parse_float_list(raw: str | None, *, default: Sequence[float]) -> tuple[float, ...]:
    if raw is None:
        return tuple(default)
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("float list must contain at least one value")
    return tuple(values)


def _parse_string_list(raw: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(default)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("string list must contain at least one value")
    return values


def _fake_dataset_samples_from_args(args: argparse.Namespace):
    from diffusion_fec.experiments.dataset_samples import (
        FAKE_DATASET_TOKENIZER_NAME,
        fake_tokenize_text,
        load_dataset_token_samples,
    )

    return load_dataset_token_samples(
        dataset_path=args.dataset_file,
        tokenize=lambda text: fake_tokenize_text(text, vocab_size=args.vocab_size),
        tokenizer_name=FAKE_DATASET_TOKENIZER_NAME,
        sample_count=args.dataset_sample_count,
        seed=args.dataset_seed,
        min_tokens=args.dataset_min_tokens,
        max_tokens=args.dataset_max_tokens,
        dataset_label=args.dataset_label,
    )


def _tokenized_samples_from_args(
    args: argparse.Namespace,
    *,
    expected_vocab_size: int | None,
):
    from diffusion_fec.data.tokenized_samples import load_tokenized_sample_artifact

    return load_tokenized_sample_artifact(
        args.tokenized_samples_file,
        expected_vocab_size=expected_vocab_size,
    )


def _source_layout_from_args(args: argparse.Namespace) -> SourceLayoutConfig:
    if args.source_layout == SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS:
        return SourceLayoutConfig(
            mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
            chunk_size=args.source_chunk_size,
        )
    return SourceLayoutConfig(mode=SOURCE_LAYOUT_CONTIGUOUS)


def _wire_interleaving_from_args(args: argparse.Namespace) -> WireInterleavingConfig:
    return WireInterleavingConfig(
        mode=args.wire_interleaving,
        span=args.wire_interleaving_span,
    )


def _channel_config_from_args(args: argparse.Namespace) -> PacketLossChannelConfig:
    if args.channel == CHANNEL_BURST:
        return PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            loss_rate=args.loss_rate,
            seed=args.seed,
            burst_start_wire_id=args.burst_start_wire_id,
            burst_length=1 if args.burst_length is None else args.burst_length,
        )
    if args.channel == CHANNEL_GILBERT_ELLIOTT:
        return PacketLossChannelConfig(
            mode=CHANNEL_GILBERT_ELLIOTT,
            loss_rate=args.loss_rate,
            seed=args.seed,
            good_loss_rate=args.ge_good_loss_rate,
            bad_loss_rate=args.ge_bad_loss_rate,
            good_to_bad_rate=args.ge_good_to_bad_rate,
            bad_to_good_rate=args.ge_bad_to_good_rate,
            initial_state=args.ge_initial_state,
        )
    return PacketLossChannelConfig(
        mode=CHANNEL_RANDOM_IID,
        loss_rate=args.loss_rate,
        seed=args.seed,
    )


def _validate_runner_config(
    *,
    sample_count: int,
    loss_rate: float,
    tokens_per_packet: int,
    vocab_size: int,
    steps: int,
) -> None:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if loss_rate < 0.0 or loss_rate > 1.0:
        raise ValueError("loss_rate must be between 0.0 and 1.0")
    if tokens_per_packet <= 0:
        raise ValueError("tokens_per_packet must be positive")
    if vocab_size <= 16:
        raise ValueError("vocab_size must be greater than 16 for synthetic samples")
    if steps <= 0:
        raise ValueError("steps must be positive")


def _synthetic_sample(sample_index: int, vocab_size: int) -> TokenSample:
    token_count = 6
    start = 5 + (sample_index * token_count)
    token_ids = tuple(3 + ((start + offset - 3) % (vocab_size - 3)) for offset in range(token_count))
    return TokenSample(
        sample_id=f"synthetic-{sample_index:04d}",
        text=" ".join(f"token{token_id}" for token_id in token_ids),
        token_ids=token_ids,
        tokenizer_name="fake-deterministic-tokenizer",
    )


def _run_id(
    *,
    sample_count: int,
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    protection_mode: str,
    hash_bits: int,
) -> str:
    return (
        f"fake-smoke|{protection_mode}|hash{hash_bits}|loss{loss_rate:g}|"
        f"samples{sample_count}|tpp{tokens_per_packet}|seed{seed}"
    )


def _manifest(
    *,
    run_id: str,
    sample_count: int,
    loss_rate: float,
    seed: int,
    tokens_per_packet: int,
    protection_mode: str,
    hash_bits: int,
    vocab_size: int,
    steps: int,
    hash_profile_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "runner": "minimal_cli_smoke",
        "model_label": FAKE_MODEL_LABEL,
        "model_kind": "fake_deterministic_smoke_model",
        "not_a_research_baseline": True,
        "strategy": SMOKE_STRATEGY if protection_mode == LOOKBACK_1_SCHEME else "FakeSmoke_Unprotected_NoPrompt",
        "config": {
            "sample_count": sample_count,
            "loss_rate": loss_rate,
            "seed": seed,
            "tokens_per_packet": tokens_per_packet,
            "protection_mode": protection_mode,
            "oracle_hash_metadata": False,
            "hash_bits": hash_bits,
            "vocab_size": vocab_size,
            "mask_token_id": DEFAULT_MASK_TOKEN_ID,
            "eos_token_id": DEFAULT_EOS_TOKEN_ID,
            "pad_token_id": DEFAULT_PAD_TOKEN_ID,
            "steps": steps,
        },
        "hash_profile": hash_profile_info,
        "artifacts": {
            "manifest": "run_manifest.json",
            "results": "results.csv",
            "events": "events.jsonl",
        },
    }


def _result_row(
    *,
    run_id: str,
    case_id: str,
    case: SmokeRecoveryCase,
    sample_index: int,
    case_seed: int,
    loss_rate: float,
    tokens_per_packet: int,
    protection_mode: str,
    hash_bits: int,
) -> dict[str, Any]:
    metrics = case.metrics.to_dict()
    plan = case.reconstruction_plan
    return {
        "run_id": run_id,
        "case_id": case_id,
        "sample_id": case.sample.sample_id,
        "model_label": FAKE_MODEL_LABEL,
        "strategy": SMOKE_STRATEGY if protection_mode == LOOKBACK_1_SCHEME else "FakeSmoke_Unprotected_NoPrompt",
        "protection_mode": protection_mode,
        "oracle_hash_metadata": False,
        "hash_bits": hash_bits,
        "loss_rate": loss_rate,
        "seed": case_seed,
        "tokens_per_packet": tokens_per_packet,
        "source_token_count": len(case.sample.token_ids),
        "known_count": plan.known_count,
        "missing_count": plan.missing_count,
        "hash_guided_count": plan.hash_guided_count,
        "unguided_count": plan.unguided_count,
        "received_packet_count": len(case.loss_result.received),
        "dropped_packet_count": len(case.loss_result.dropped),
        **metrics,
    }


def _event(
    *,
    run_id: str,
    case_id: str,
    case: SmokeRecoveryCase,
) -> dict[str, Any]:
    case_data = _normalize_case_for_artifacts(case.to_dict())
    return {
        "event_type": "smoke_case",
        "run_id": run_id,
        "case_id": case_id,
        "model_label": FAKE_MODEL_LABEL,
        "case": case_data,
    }


def _normalize_case_for_artifacts(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    decoding_result = dict(normalized["decoding_result"])
    decoding_result["decode_latency_sec"] = 0.0
    normalized["decoding_result"] = decoding_result
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
