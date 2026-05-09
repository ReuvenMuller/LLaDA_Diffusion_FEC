"""Opt-in real LLaDA end-to-end smoke run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diffusion_fec.coding.hash_profiles import DEFAULT_HASH_MAP_MODE, load_or_build_hash_profile
from diffusion_fec.coding.protection import LOOKBACK_1_SCHEME
from diffusion_fec.decoding.llada_diffusion import EDITABLE_UPDATE_COMMIT_ONCE, HASH_CONSTRAINT_ALWAYS
from diffusion_fec.experiments.logging import start_run_timer, write_run_artifacts
from diffusion_fec.experiments.smoke import SmokeRecoveryCase, run_smoke_recovery_case
from diffusion_fec.models.llada import LLADA_1_5_MODEL_ID, LLaDAAdapter
from diffusion_fec.types import TokenSample


REAL_LLADA_STRATEGY = "LLaDA_RealSmoke_HashLookback1_NoPrompt"


class RealLLaDASmokeUnavailable(RuntimeError):
    """Raised when the opt-in real LLaDA smoke cannot run in this environment."""


def run_real_llada_smoke(
    *,
    output_dir: str | Path,
    model_id: str = LLADA_1_5_MODEL_ID,
    seed: int = 1,
    loss_rate: float = 0.5,
    tokens_per_packet: int = 1,
    hash_bits: int = 4,
    steps: int = 2,
    editable_update_mode: str = EDITABLE_UPDATE_COMMIT_ONCE,
    hash_constraint_schedule: str = HASH_CONSTRAINT_ALWAYS,
    local_files_only: bool = False,
    allow_cpu: bool = False,
    hash_profile_dir: str | Path | None = None,
    build_hash_profile: bool = False,
    hash_map_mode: str = DEFAULT_HASH_MAP_MODE,
    hash_profile_name: str | None = None,
) -> dict[str, Any]:
    """Load real LLaDA and run one tiny transmitted-protection smoke case."""

    run_timer = start_run_timer()
    torch = _import_torch()
    tokenizer_adapter = _load_tokenizer_config(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    tokenizer_stage = {
        "mask_token_id": tokenizer_adapter.mask_token_id,
        "eos_token_id": tokenizer_adapter.eos_token_id,
        "pad_token_id": tokenizer_adapter.pad_token_id,
        "vocab_size": tokenizer_adapter.vocab_size,
        "max_sequence_length": tokenizer_adapter.max_sequence_length,
    }
    profile_name = hash_profile_name or _default_hash_profile_name(model_id)
    token_hash_map, hash_profile_info = load_or_build_hash_profile(
        profile_dir=hash_profile_dir,
        profile_name=profile_name,
        vocab_size=tokenizer_adapter.vocab_size,
        hash_bits=hash_bits,
        decode_token=tokenizer_adapter,
        excluded_token_ids={
            tokenizer_adapter.mask_token_id,
            *(
                token_id
                for token_id in (tokenizer_adapter.eos_token_id, tokenizer_adapter.pad_token_id)
                if token_id is not None
            ),
        },
        salt=f"{model_id}|real-llada-smoke",
        map_mode=hash_map_mode,
        model_id=model_id,
        tokenizer_name=model_id,
        build_if_missing=build_hash_profile,
    )

    if not torch.cuda.is_available() and not allow_cpu:
        raise RealLLaDASmokeUnavailable(
            "CUDA is not available. Real LLaDA smoke is disabled before model "
            "weight loading; pass --allow-cpu-real-llada to try CPU explicitly."
        )

    adapter = _load_model(
        model_id=model_id,
        local_files_only=local_files_only,
        use_cuda=torch.cuda.is_available(),
        torch_module=torch,
    )
    forward_shape = _run_tiny_forward(adapter)
    sample = _tiny_token_sample(adapter)
    config = adapter.decoding_config(
        steps=steps,
        block_length=tokens_per_packet,
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
    )
    case = run_smoke_recovery_case(
        sample=sample,
        model=adapter,
        config=config,
        tokens_per_packet=tokens_per_packet,
        loss_rate=loss_rate,
        seed=seed,
        token_hash_map=token_hash_map,
        protection_mode=LOOKBACK_1_SCHEME,
    )
    run_id = _run_id(
        model_id=model_id,
        hash_bits=hash_bits,
        loss_rate=loss_rate,
        seed=seed,
        steps=steps,
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
    )
    manifest = _manifest(
        run_id=run_id,
        model_id=model_id,
        seed=seed,
        loss_rate=loss_rate,
        tokens_per_packet=tokens_per_packet,
        hash_bits=hash_bits,
        steps=steps,
        editable_update_mode=editable_update_mode,
        hash_constraint_schedule=hash_constraint_schedule,
        local_files_only=local_files_only,
        allow_cpu=allow_cpu,
        tokenizer_stage=tokenizer_stage,
        forward_shape=forward_shape,
        hash_profile_info=hash_profile_info,
    )
    result_rows = [
        _result_row(
            run_id=run_id,
            case_id="case0000",
            case=case,
            model_id=model_id,
            seed=seed,
            loss_rate=loss_rate,
            tokens_per_packet=tokens_per_packet,
            hash_bits=hash_bits,
        )
    ]
    events = [
        {
            "event_type": "real_llada_smoke_case",
            "run_id": run_id,
            "case_id": "case0000",
            "model_label": model_id,
            "forward_shape": list(forward_shape),
            "case": case.to_dict(),
        }
    ]
    artifact_data = write_run_artifacts(
        output_dir=output_dir,
        manifest=manifest,
        result_rows=result_rows,
        events=events,
        run_timer=run_timer,
    )
    return {
        "run_id": run_id,
        "manifest": artifact_data["manifest"],
        "result_rows": artifact_data["result_rows"],
        "events": events,
    }


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RealLLaDASmokeUnavailable(
            "PyTorch is not installed. Install optional dependencies with "
            "`pip install -e .[hf]` before running real LLaDA smoke."
        ) from exc
    return torch


def _load_tokenizer_config(*, model_id: str, local_files_only: bool) -> LLaDAAdapter:
    try:
        return LLaDAAdapter.from_pretrained(
            model_id,
            load_model=False,
            config_kwargs={"local_files_only": local_files_only},
            tokenizer_kwargs={"local_files_only": local_files_only},
        )
    except Exception as exc:
        raise RealLLaDASmokeUnavailable(
            f"Could not load LLaDA tokenizer/config for {model_id!r}. "
            "If using cache-only mode, confirm the model files are cached locally."
        ) from exc


def _load_model(
    *,
    model_id: str,
    local_files_only: bool,
    use_cuda: bool,
    torch_module,
) -> LLaDAAdapter:
    model_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if use_cuda:
        model_kwargs["torch_dtype"] = torch_module.bfloat16
    try:
        adapter = LLaDAAdapter.from_pretrained(
            model_id,
            load_model=True,
            config_kwargs={"local_files_only": local_files_only},
            tokenizer_kwargs={"local_files_only": local_files_only},
            model_kwargs=model_kwargs,
        )
        if use_cuda and adapter.model is not None:
            adapter.model.to("cuda")
        return adapter
    except Exception as exc:
        cache_hint = (
            " Cache-only mode is enabled; confirm all model weight files are present."
            if local_files_only
            else ""
        )
        raise RealLLaDASmokeUnavailable(
            f"Could not load LLaDA model weights for {model_id!r}: {exc}.{cache_hint}"
        ) from exc


def _run_tiny_forward(adapter: LLaDAAdapter) -> tuple[int, int, int]:
    try:
        output = adapter.forward([[adapter.mask_token_id]], attention_mask=[[1]])
        shape = tuple(int(value) for value in output.logits.shape)
    except Exception as exc:
        raise RealLLaDASmokeUnavailable("Tiny LLaDA forward pass failed.") from exc
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 1 or shape[2] != adapter.vocab_size:
        raise RealLLaDASmokeUnavailable(
            f"Unexpected LLaDA forward logits shape {shape}; expected "
            f"(1, 1, {adapter.vocab_size})."
        )
    return shape


def _tiny_token_sample(adapter: LLaDAAdapter) -> TokenSample:
    candidate_texts = [
        "alpha beta",
        "small smoke",
        "one two three",
    ]
    special_ids = {
        adapter.mask_token_id,
        *(token_id for token_id in (adapter.eos_token_id, adapter.pad_token_id) if token_id is not None),
    }
    for text in candidate_texts:
        token_ids = tuple(
            token_id
            for token_id in adapter.tokenize(text, add_special_tokens=False)
            if token_id not in special_ids
        )
        if len(token_ids) >= 2:
            tiny_tokens = token_ids[:2]
            return TokenSample(
                sample_id="real-llada-smoke-0000",
                text=adapter.decode(tiny_tokens, skip_special_tokens=False),
                token_ids=tiny_tokens,
                tokenizer_name=adapter.model_id,
            )
    raise RealLLaDASmokeUnavailable("Could not build a two-token LLaDA smoke sample.")


def _run_id(
    *,
    model_id: str,
    hash_bits: int,
    loss_rate: float,
    seed: int,
    steps: int,
    editable_update_mode: str,
    hash_constraint_schedule: str,
) -> str:
    model_label = model_id.replace("/", "-")
    return (
        f"real-llada-smoke|{model_label}|hash{hash_bits}|loss{loss_rate:g}|"
        f"steps{steps}|seed{seed}|update-{editable_update_mode}|"
        f"hash-schedule-{hash_constraint_schedule}"
    )


def _default_hash_profile_name(model_id: str) -> str:
    return f"{model_id.replace('/', '-')}-real-smoke-v1"


def _manifest(
    *,
    run_id: str,
    model_id: str,
    seed: int,
    loss_rate: float,
    tokens_per_packet: int,
    hash_bits: int,
    steps: int,
    editable_update_mode: str,
    hash_constraint_schedule: str,
    local_files_only: bool,
    allow_cpu: bool,
    tokenizer_stage: dict[str, Any],
    forward_shape: tuple[int, int, int],
    hash_profile_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "runner": "real_llada_e2e_smoke",
        "model_label": model_id,
        "model_kind": "real_llada_huggingface",
        "not_a_research_baseline": True,
        "opt_in_required": True,
        "strategy": REAL_LLADA_STRATEGY,
        "preflight": {
            "tokenizer_config_loaded": True,
            "tiny_forward_shape": list(forward_shape),
        },
        "tokenizer_config": tokenizer_stage,
        "config": {
            "sample_count": 1,
            "loss_rate": loss_rate,
            "seed": seed,
            "tokens_per_packet": tokens_per_packet,
            "protection_mode": LOOKBACK_1_SCHEME,
            "oracle_hash_metadata": False,
            "hash_bits": hash_bits,
            "steps": steps,
            "editable_update_mode": editable_update_mode,
            "hash_constraint_schedule": hash_constraint_schedule,
            "local_files_only": local_files_only,
            "allow_cpu": allow_cpu,
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
    model_id: str,
    seed: int,
    loss_rate: float,
    tokens_per_packet: int,
    hash_bits: int,
) -> dict[str, Any]:
    plan = case.reconstruction_plan
    diagnostics = case.decoding_result.diagnostics
    return {
        "run_id": run_id,
        "case_id": case_id,
        "sample_id": case.sample.sample_id,
        "model_label": model_id,
        "strategy": REAL_LLADA_STRATEGY,
        "protection_mode": LOOKBACK_1_SCHEME,
        "oracle_hash_metadata": False,
        "hash_bits": hash_bits,
        "loss_rate": loss_rate,
        "seed": seed,
        "tokens_per_packet": tokens_per_packet,
        "source_token_count": len(case.sample.token_ids),
        "known_count": plan.known_count,
        "missing_count": plan.missing_count,
        "hash_guided_count": plan.hash_guided_count,
        "unguided_count": plan.unguided_count,
        "received_packet_count": len(case.loss_result.received),
        "dropped_packet_count": len(case.loss_result.dropped),
        "editable_update_mode": diagnostics.get("editable_update_mode", ""),
        "hash_constraint_schedule": diagnostics.get("hash_constraint_schedule", ""),
        "decode_latency_sec": case.decoding_result.decode_latency_sec,
        "decoder_steps": case.decoding_result.steps,
        "model_forward_calls": diagnostics.get("model_forward_calls", ""),
        "model_proposal_calls": diagnostics.get("model_proposal_calls", ""),
        "decoder_proposal_mode": diagnostics.get("decoder_proposal_mode", ""),
        "proposal_interface_used": diagnostics.get("proposal_interface_used", ""),
        **case.metrics.to_dict(),
    }
