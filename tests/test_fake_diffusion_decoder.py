from dataclasses import dataclass

import pytest

from diffusion_fec.coding.packetizer import build_reconstruction_plan
from diffusion_fec.coding.token_hash import TokenHashMap, build_token_hash_map
from diffusion_fec.decoding.llada_diffusion import (
    DiffusionDecodingConfig,
    decode_masked_diffusion,
)
from diffusion_fec.types import Packet


@dataclass
class FakeForwardOutput:
    logits: list[list[list[float]]]


class FakeMaskedDiffusionModel:
    def __init__(self, vocab_size: int, preferred_by_position: dict[int, list[tuple[int, float]]]):
        self.vocab_size = vocab_size
        self.preferred_by_position = preferred_by_position
        self.forward_inputs: list[list[list[int]]] = []
        self.attention_masks: list[list[list[int]]] = []

    def forward(self, input_ids, attention_mask=None):
        self.forward_inputs.append([list(row) for row in input_ids])
        self.attention_masks.append([list(row) for row in attention_mask])
        sequence_length = len(input_ids[0])
        logits = [[[0.0 for _ in range(self.vocab_size)] for _ in range(sequence_length)]]
        for position, scored_tokens in self.preferred_by_position.items():
            for token_id, score in scored_tokens:
                logits[0][position][token_id] = score
        return FakeForwardOutput(logits=logits)

    def decode(self, token_ids, skip_special_tokens=False):
        return "|".join(str(token_id) for token_id in token_ids)


class ProposalOnlyModel:
    def __init__(self, target_tokens: tuple[int, ...]):
        self.target_tokens = target_tokens
        self.proposal_calls = []

    def propose_token(
        self,
        *,
        position,
        full_position,
        candidate_token_ids,
        input_ids,
        step,
    ):
        self.proposal_calls.append(
            {
                "position": position,
                "full_position": full_position,
                "candidate_count": len(candidate_token_ids),
                "input_ids": tuple(input_ids),
                "step": step,
            }
        )
        target_token_id = self.target_tokens[position]
        token_id = (
            target_token_id
            if target_token_id in candidate_token_ids
            else candidate_token_ids[0]
        )
        return {"token_id": token_id, "top1_probability": 1.0, "top2_probability": 0.0}

    def forward(self, input_ids, attention_mask=None):
        raise AssertionError("proposal-only model should not materialize logits")

    def decode(self, token_ids, skip_special_tokens=False):
        return "|".join(str(token_id) for token_id in token_ids)


class TorchLogitsModel:
    def __init__(self, vocab_size: int, preferred_token_id: int):
        self.vocab_size = vocab_size
        self.preferred_token_id = preferred_token_id

    def forward(self, input_ids, attention_mask=None):
        torch = pytest.importorskip("torch")
        sequence_length = len(input_ids[0])
        logits = torch.zeros((1, sequence_length, self.vocab_size), dtype=torch.float32)
        logits[0, 0, self.preferred_token_id] = 100.0
        return FakeForwardOutput(logits=logits)

    def decode(self, token_ids, skip_special_tokens=False):
        return "|".join(str(token_id) for token_id in token_ids)


def config() -> DiffusionDecodingConfig:
    return DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=32,
        steps=4,
        block_length=4,
    )


def test_known_tokens_never_change_and_model_contract_is_batch_first() -> None:
    plan = build_reconstruction_plan(
        total_tokens=3,
        received_packets=[
            Packet(
                source_id="sample-1",
                wire_id=0,
                kind="data",
                token_ids=[10, 12],
                token_positions=[0, 2],
            )
        ],
    )
    model = FakeMaskedDiffusionModel(
        vocab_size=32,
        preferred_by_position={
            0: [(29, 100.0)],
            1: [(20, 5.0)],
            2: [(28, 100.0)],
        },
    )

    result = decode_masked_diffusion(model=model, plan=plan, config=config())

    assert result.reconstructed_tokens == (10, 20, 12)
    assert result.confidence_stats[0].was_fixed is True
    assert result.confidence_stats[2].was_fixed is True
    assert model.forward_inputs[0] == [[10, 0, 12]]
    assert model.attention_masks[0] == [[1, 1, 1]]


def test_fixed_suffix_tokens_never_change() -> None:
    plan = build_reconstruction_plan(
        total_tokens=5,
        received_packets=[
            Packet(
                source_id="sample-1",
                wire_id=0,
                kind="data",
                token_ids=[13, 14],
                token_positions=[3, 4],
            )
        ],
    )
    model = FakeMaskedDiffusionModel(
        vocab_size=32,
        preferred_by_position={
            0: [(20, 5.0)],
            1: [(21, 5.0)],
            2: [(22, 5.0)],
            3: [(30, 100.0)],
            4: [(31, 100.0)],
        },
    )

    result = decode_masked_diffusion(model=model, plan=plan, config=config())

    assert result.reconstructed_tokens[-2:] == (13, 14)
    assert result.reconstructed_tokens[:3] == (20, 21, 22)


def test_hash_guided_position_only_commits_matching_bucket_token() -> None:
    token_hash = build_token_hash_map(
        vocab_size=32,
        hash_bits=4,
        decode_token=lambda token_id: f"token-{token_id}",
        excluded_token_ids={0, 1, 2},
    )
    hash_value = token_hash.bucket_for_token(22)
    nonmatching_token = next(
        token_id
        for token_id in range(3, 32)
        if token_hash.bucket_for_token(token_id) != hash_value
    )
    plan = build_reconstruction_plan(
        total_tokens=1,
        received_packets=[],
        hash_metadata={0: hash_value},
    )
    model = FakeMaskedDiffusionModel(
        vocab_size=32,
        preferred_by_position={0: [(nonmatching_token, 100.0), (22, 10.0)]},
    )

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=config(),
        token_hash_map=token_hash,
    )

    assert token_hash.bucket_for_token(result.reconstructed_tokens[0]) == hash_value
    assert result.reconstructed_tokens[0] != nonmatching_token
    assert result.confidence_stats[0].candidate_count == len(
        token_hash.candidate_token_ids(hash_value)
    )


def test_unguided_positions_ban_special_tokens_but_allow_normal_tokens() -> None:
    plan = build_reconstruction_plan(total_tokens=1, received_packets=[])
    model = FakeMaskedDiffusionModel(
        vocab_size=32,
        preferred_by_position={0: [(0, 100.0), (1, 99.0), (2, 98.0), (8, 5.0)]},
    )

    result = decode_masked_diffusion(model=model, plan=plan, config=config())

    assert result.reconstructed_tokens == (8,)


def test_tensor_logits_select_from_large_vocab_without_python_logits_lists() -> None:
    pytest.importorskip("torch")
    vocab_size = 50000
    preferred_token_id = vocab_size - 1
    plan = build_reconstruction_plan(total_tokens=1, received_packets=[])
    model = TorchLogitsModel(
        vocab_size=vocab_size,
        preferred_token_id=preferred_token_id,
    )

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=vocab_size,
            steps=1,
            block_length=1,
        ),
    )

    assert result.reconstructed_tokens == (preferred_token_id,)
    assert result.diagnostics["decoder_proposal_mode"] == "logits"
    assert result.diagnostics["model_forward_calls"] == 1


def test_prompt_tokens_are_fixed_prefix_but_not_returned_as_target_tokens() -> None:
    plan = build_reconstruction_plan(total_tokens=1, received_packets=[])
    model = FakeMaskedDiffusionModel(
        vocab_size=32,
        preferred_by_position={2: [(19, 5.0)]},
    )

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=config(),
        prompt_token_ids=[7, 8],
    )

    assert model.forward_inputs[0] == [[7, 8, 0]]
    assert result.reconstructed_tokens == (19,)
    assert result.diagnostics["prompt_token_count"] == 2


def test_all_editable_positions_fill_and_commit_steps_are_recorded() -> None:
    plan = build_reconstruction_plan(total_tokens=4, received_packets=[])
    fast_config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=32,
        steps=2,
        block_length=4,
    )
    model = FakeMaskedDiffusionModel(
        vocab_size=32,
        preferred_by_position={
            0: [(10, 9.0)],
            1: [(11, 8.0)],
            2: [(12, 7.0)],
            3: [(13, 6.0)],
        },
    )

    result = decode_masked_diffusion(model=model, plan=plan, config=fast_config)

    assert result.reconstructed_tokens == (10, 11, 12, 13)
    assert [summary.committed_count for summary in result.step_summaries] == [2, 2]
    assert [stat.commit_step for stat in result.confidence_stats] == [0, 0, 1, 1]


def test_empty_hash_bucket_falls_back_and_logs_position() -> None:
    token_hash = TokenHashMap(
        hash_bits=4,
        vocab_size=3,
        token_to_bucket=(0, 0, 0),
        bucket_to_token_ids=tuple(() for _ in range(16)),
        excluded_token_ids={0, 1, 2},
    )
    plan = build_reconstruction_plan(
        total_tokens=1,
        received_packets=[],
        hash_metadata={0: 0},
    )
    fallback_config = DiffusionDecodingConfig(
        mask_token_id=0,
        vocab_size=3,
        steps=1,
        block_length=1,
    )
    model = FakeMaskedDiffusionModel(
        vocab_size=3,
        preferred_by_position={0: [(2, 5.0), (1, 3.0)]},
    )

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=fallback_config,
        token_hash_map=token_hash,
    )

    assert result.reconstructed_tokens == (2,)
    assert result.diagnostics["hash_bucket_empty_positions"] == [0]
    assert result.diagnostics["fallback_reasons"] == {0: "hash_bucket_empty_after_bans"}


def test_hash_guided_plan_requires_token_hash_map() -> None:
    plan = build_reconstruction_plan(
        total_tokens=1,
        received_packets=[],
        hash_metadata={0: 0},
    )
    model = FakeMaskedDiffusionModel(vocab_size=32, preferred_by_position={})

    with pytest.raises(ValueError, match="token_hash_map is required"):
        decode_masked_diffusion(model=model, plan=plan, config=config())


def test_proposal_interface_uses_constraints_without_full_logits_for_huge_vocab() -> None:
    huge_config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=126464,
        steps=2,
        block_length=4,
    )
    plan = build_reconstruction_plan(
        total_tokens=3,
        received_packets=[
            Packet(
                source_id="sample-1",
                wire_id=0,
                kind="data",
                token_ids=[1000],
                token_positions=[0],
            )
        ],
    )
    model = ProposalOnlyModel(target_tokens=(99999, 12345, 45678))

    result = decode_masked_diffusion(model=model, plan=plan, config=huge_config)

    assert result.reconstructed_tokens == (1000, 12345, 45678)
    assert result.confidence_stats[0].was_fixed is True
    assert result.diagnostics["model_forward_calls"] == 0
    assert result.diagnostics["model_proposal_calls"] == 3
    assert result.diagnostics["decoder_proposal_mode"] == "model_propose_token"
    assert result.diagnostics["proposal_interface_used"] is True
    assert len(model.proposal_calls) == 3
    assert all(call["candidate_count"] == huge_config.vocab_size - 3 for call in model.proposal_calls)


def test_proposal_interface_cannot_escape_hash_bucket_candidates() -> None:
    vocab_size = 126464
    token_to_bucket = tuple(token_id % 16 for token_id in range(vocab_size))
    bucket_to_token_ids = tuple(
        tuple(
            token_id
            for token_id in range(bucket, vocab_size, 16)
            if token_id not in {0, 1, 2}
        )
        for bucket in range(16)
    )
    token_hash = TokenHashMap(
        hash_bits=4,
        vocab_size=vocab_size,
        token_to_bucket=token_to_bucket,
        bucket_to_token_ids=bucket_to_token_ids,
        excluded_token_ids={0, 1, 2},
    )
    hash_value = 7
    nonmatching_target = 12344
    assert nonmatching_target % 16 != hash_value
    plan = build_reconstruction_plan(
        total_tokens=1,
        received_packets=[],
        hash_metadata={0: hash_value},
    )
    model = ProposalOnlyModel(target_tokens=(nonmatching_target,))

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=vocab_size,
            steps=1,
            block_length=1,
        ),
        token_hash_map=token_hash,
    )

    assert result.reconstructed_tokens[0] in token_hash.candidate_token_ids(hash_value)
    assert result.reconstructed_tokens[0] != nonmatching_target
    assert result.diagnostics["model_forward_calls"] == 0
    assert result.diagnostics["model_proposal_calls"] == 1


def test_proposal_interface_unguided_positions_still_ban_special_tokens() -> None:
    plan = build_reconstruction_plan(total_tokens=1, received_packets=[])
    model = ProposalOnlyModel(target_tokens=(0,))

    result = decode_masked_diffusion(model=model, plan=plan, config=config())

    assert result.reconstructed_tokens == (3,)
    assert result.diagnostics["model_forward_calls"] == 0
    assert result.diagnostics["proposal_interface_used"] is True
