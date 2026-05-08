from dataclasses import dataclass

import pytest

from diffusion_fec.coding.token_hash import build_token_hash_map
from diffusion_fec.decoding.llada_diffusion import DiffusionDecodingConfig
from diffusion_fec.experiments.smoke import run_smoke_recovery_case
from diffusion_fec.types import STATE_KNOWN, STATE_MISSING, STATE_UNGUIDED, TokenSample


@dataclass
class FakeForwardOutput:
    logits: list[list[list[float]]]


class OracleFakeModel:
    def __init__(self, target_tokens: tuple[int, ...], vocab_size: int):
        self.target_tokens = target_tokens
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        sequence_length = len(input_ids[0])
        logits = [[[0.0 for _ in range(self.vocab_size)] for _ in range(sequence_length)]]
        for position, token_id in enumerate(self.target_tokens):
            logits[0][position][token_id] = 10.0 - position
        return FakeForwardOutput(logits=logits)

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)


def test_smoke_harness_runs_packet_loss_transmitted_lookback_hash_planning_and_decode() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="synthetic",
        token_ids=[5, 6, 7, 8],
        tokenizer_name="fake",
    )
    token_hash = build_token_hash_map(
        vocab_size=16,
        hash_bits=4,
        decode_token=lambda token_id: f"token-{token_id}",
        excluded_token_ids={0, 1, 2},
    )
    config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=16,
        steps=2,
        block_length=4,
    )
    model = OracleFakeModel(target_tokens=sample.token_ids, vocab_size=16)

    case = run_smoke_recovery_case(
        sample=sample,
        model=model,
        config=config,
        tokens_per_packet=1,
        loss_rate=0.5,
        seed=0,
        token_hash_map=token_hash,
        protection_mode="lookback_1",
    )

    assert [packet.wire_id for packet in case.loss_result.received] == [0, 1]
    assert [packet.wire_id for packet in case.loss_result.dropped] == [2, 3]
    assert [entry.state for entry in case.reconstruction_plan.entries] == [
        STATE_KNOWN,
        STATE_KNOWN,
        STATE_UNGUIDED,
        STATE_UNGUIDED,
    ]
    assert case.hash_metadata == {}
    assert case.protection_mode == "lookback_1"
    assert case.oracle_hash_metadata is False
    assert case.decoding_result.reconstructed_tokens == sample.token_ids
    assert case.exact_token_match is True
    assert case.known_positions_preserved is True
    assert case.metrics.exact_match is True
    assert case.metrics.lost_position_recovery_rate == 1.0
    assert case.to_dict()["metrics"]["known_position_preserved"] is True


def test_hash_guided_smoke_requires_explicit_oracle_hash_metadata_flag() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="synthetic",
        token_ids=[5, 6, 7, 8],
        tokenizer_name="fake",
    )
    token_hash = build_token_hash_map(
        vocab_size=16,
        hash_bits=4,
        decode_token=lambda token_id: f"token-{token_id}",
        excluded_token_ids={0, 1, 2},
    )
    config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=16,
        steps=2,
        block_length=4,
    )
    model = OracleFakeModel(target_tokens=sample.token_ids, vocab_size=16)

    with pytest.raises(ValueError, match="oracle_hash_metadata=True"):
        run_smoke_recovery_case(
            sample=sample,
            model=model,
            config=config,
            tokens_per_packet=1,
            loss_rate=0.5,
            seed=0,
            token_hash_map=token_hash,
        )


def test_smoke_harness_can_still_use_explicit_oracle_hash_metadata_mode() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="synthetic",
        token_ids=[5, 6, 7, 8],
        tokenizer_name="fake",
    )
    token_hash = build_token_hash_map(
        vocab_size=16,
        hash_bits=4,
        decode_token=lambda token_id: f"token-{token_id}",
        excluded_token_ids={0, 1, 2},
    )
    config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=16,
        steps=2,
        block_length=4,
    )
    model = OracleFakeModel(target_tokens=sample.token_ids, vocab_size=16)

    case = run_smoke_recovery_case(
        sample=sample,
        model=model,
        config=config,
        tokens_per_packet=1,
        loss_rate=0.5,
        seed=0,
        token_hash_map=token_hash,
        oracle_hash_metadata=True,
    )

    assert case.protection_mode == "none"
    assert case.oracle_hash_metadata is True
    assert case.hash_metadata == {
        2: token_hash.bucket_for_token(7),
        3: token_hash.bucket_for_token(8),
    }


def test_smoke_harness_rejects_oracle_hash_metadata_with_lookback_protection() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="synthetic",
        token_ids=[5, 6, 7, 8],
        tokenizer_name="fake",
    )
    token_hash = build_token_hash_map(
        vocab_size=16,
        hash_bits=4,
        decode_token=lambda token_id: f"token-{token_id}",
        excluded_token_ids={0, 1, 2},
    )
    config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=16,
        steps=2,
        block_length=4,
    )
    model = OracleFakeModel(target_tokens=sample.token_ids, vocab_size=16)

    with pytest.raises(ValueError, match="must not set oracle_hash_metadata"):
        run_smoke_recovery_case(
            sample=sample,
            model=model,
            config=config,
            tokens_per_packet=1,
            loss_rate=0.5,
            seed=0,
            token_hash_map=token_hash,
            protection_mode="lookback_1",
            oracle_hash_metadata=True,
        )


def test_dropped_packet_becomes_hash_guided_when_protecting_packet_survives() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="synthetic",
        token_ids=[5, 6],
        tokenizer_name="fake",
    )
    token_hash = build_token_hash_map(
        vocab_size=16,
        hash_bits=4,
        decode_token=lambda token_id: f"token-{token_id}",
        excluded_token_ids={0, 1, 2},
    )
    config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=16,
        steps=2,
        block_length=2,
    )
    model = OracleFakeModel(target_tokens=sample.token_ids, vocab_size=16)

    case = run_smoke_recovery_case(
        sample=sample,
        model=model,
        config=config,
        tokens_per_packet=1,
        loss_rate=0.5,
        seed=1,
        token_hash_map=token_hash,
        protection_mode="lookback_1",
    )

    assert [packet.wire_id for packet in case.loss_result.dropped] == [0]
    assert [packet.wire_id for packet in case.loss_result.received] == [1]
    assert [entry.state for entry in case.reconstruction_plan.entries] == [
        STATE_MISSING,
        STATE_KNOWN,
    ]
    assert case.hash_metadata == {0: token_hash.bucket_for_token(5)}


def test_dropped_packet_becomes_unguided_when_protecting_packet_drops() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="synthetic",
        token_ids=[5, 6],
        tokenizer_name="fake",
    )
    token_hash = build_token_hash_map(
        vocab_size=16,
        hash_bits=4,
        decode_token=lambda token_id: f"token-{token_id}",
        excluded_token_ids={0, 1, 2},
    )
    config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=16,
        steps=2,
        block_length=2,
    )
    model = OracleFakeModel(target_tokens=sample.token_ids, vocab_size=16)

    case = run_smoke_recovery_case(
        sample=sample,
        model=model,
        config=config,
        tokens_per_packet=1,
        loss_rate=1.0,
        seed=1,
        token_hash_map=token_hash,
        protection_mode="lookback_1",
    )

    assert [packet.wire_id for packet in case.loss_result.dropped] == [0, 1]
    assert case.hash_metadata == {}
    assert [entry.state for entry in case.reconstruction_plan.entries] == [
        STATE_UNGUIDED,
        STATE_UNGUIDED,
    ]


def test_smoke_harness_without_hash_map_builds_unguided_missing_positions() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="synthetic",
        token_ids=[5, 6, 7, 8],
        tokenizer_name="fake",
    )
    config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=16,
        steps=2,
        block_length=4,
    )
    model = OracleFakeModel(target_tokens=sample.token_ids, vocab_size=16)

    case = run_smoke_recovery_case(
        sample=sample,
        model=model,
        config=config,
        tokens_per_packet=1,
        loss_rate=0.5,
        seed=0,
    )

    assert case.hash_metadata == {}
    assert case.oracle_hash_metadata is False
    assert [entry.state for entry in case.reconstruction_plan.entries] == [
        STATE_KNOWN,
        STATE_KNOWN,
        STATE_UNGUIDED,
        STATE_UNGUIDED,
    ]
    assert case.decoding_result.reconstructed_tokens == sample.token_ids
