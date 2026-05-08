import json

import pytest

from diffusion_fec.types import (
    ConfidenceStat,
    DecodingResult,
    Packet,
    ReconstructionEntry,
    ReconstructionPlan,
    STATE_KNOWN,
    STATE_MISSING,
    STATE_UNGUIDED,
    StepSummary,
    TokenSample,
)


def test_core_dataclasses_serialize_to_json() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="hello",
        token_ids=[10, 20],
        tokenizer_name="fake-tokenizer",
    )
    packet = Packet(
        source_id=sample.sample_id,
        wire_id=0,
        kind="data",
        token_ids=[10],
        token_positions=[0],
        metadata={"note": "first"},
    )
    stat = ConfidenceStat(
        position=1,
        state=STATE_MISSING,
        selected_token_id=20,
        top1_probability=0.8,
        top2_probability=0.1,
        margin=0.7,
        candidate_count=4,
        commit_step=2,
        hash_value=3,
    )
    summary = StepSummary(
        step=2,
        still_masked_count=0,
        committed_count=1,
        average_confidence=0.8,
        hash_guided_committed_count=1,
    )
    result = DecodingResult(
        reconstructed_text="hello",
        reconstructed_tokens=[10, 20],
        decode_latency_sec=0.01,
        steps=3,
        fixed_token_count=1,
        editable_token_count=1,
        hash_guided_token_count=1,
        confidence_stats=[stat],
        step_summaries=[summary],
        diagnostics={"backend": "fake"},
    )

    for item in (sample, packet, stat, summary, result):
        json.dumps(item.to_dict())


def test_reconstruction_plan_counts_states() -> None:
    plan = ReconstructionPlan(
        entries=[
            ReconstructionEntry(position=0, state=STATE_KNOWN, token_id=10, fixed=True),
            ReconstructionEntry(position=1, state=STATE_MISSING, hash_value=7),
            ReconstructionEntry(position=2, state=STATE_UNGUIDED),
        ],
        total_tokens=3,
    )

    assert plan.known_count == 1
    assert plan.missing_count == 2
    assert plan.hash_guided_count == 1
    assert plan.unguided_count == 1
    assert plan.to_dict()["known_count"] == 1


def test_invalid_entries_fail_clearly() -> None:
    with pytest.raises(ValueError, match="known entries require token_id"):
        ReconstructionEntry(position=0, state=STATE_KNOWN, fixed=True)

    with pytest.raises(ValueError, match="missing entries require hash_value"):
        ReconstructionEntry(position=0, state=STATE_MISSING)

    with pytest.raises(ValueError, match="unguided entries must not carry hash_value"):
        ReconstructionEntry(position=0, state=STATE_UNGUIDED, hash_value=1)


def test_reconstruction_plan_requires_ordered_full_coverage() -> None:
    with pytest.raises(ValueError, match="cover 0..total_tokens-1"):
        ReconstructionPlan(
            entries=[
                ReconstructionEntry(position=1, state=STATE_UNGUIDED),
                ReconstructionEntry(position=0, state=STATE_UNGUIDED),
            ],
            total_tokens=2,
        )
