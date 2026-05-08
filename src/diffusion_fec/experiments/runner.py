"""Minimal deterministic smoke runner.

This runner intentionally uses a fake deterministic model. It is an artifact-writing
smoke test for the local pipeline, not a real LLaDA baseline.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diffusion_fec.coding.protection import LOOKBACK_1_SCHEME
from diffusion_fec.coding.token_hash import build_token_hash_map
from diffusion_fec.decoding.llada_diffusion import DiffusionDecodingConfig
from diffusion_fec.experiments.logging import write_run_artifacts
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
    token_hash_map = build_token_hash_map(
        vocab_size=vocab_size,
        hash_bits=hash_bits,
        decode_token=lambda token_id: f"fake-token-{token_id}",
        excluded_token_ids={
            DEFAULT_MASK_TOKEN_ID,
            DEFAULT_EOS_TOKEN_ID,
            DEFAULT_PAD_TOKEN_ID,
        },
        salt="fake-smoke",
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
    parser.add_argument("--real-llada-smoke", action="store_true")
    parser.add_argument("--llada-model-id", default="GSAI-ML/LLaDA-1.5")
    parser.add_argument("--llada-local-files-only", action="store_true")
    parser.add_argument("--allow-cpu-real-llada", action="store_true")
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--loss-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens-per-packet", type=int, default=1)
    parser.add_argument("--protection-mode", default=LOOKBACK_1_SCHEME, choices=["none", LOOKBACK_1_SCHEME])
    parser.add_argument("--hash-bits", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args(argv)

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
    )
    return 0


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
