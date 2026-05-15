from dataclasses import dataclass

import pytest

from diffusion_fec.coding.packetizer import build_reconstruction_plan
from diffusion_fec.coding.token_hash import TokenHashMap, build_token_hash_map
from diffusion_fec.decoding.llada_diffusion import (
    DiffusionDecodingConfig,
    decode_masked_diffusion,
    should_enforce_hash_constraint,
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


class StepwiseProposalModel:
    def __init__(self, choices_by_step: dict[tuple[int, int], int], default_token_id: int = 3):
        self.choices_by_step = choices_by_step
        self.default_token_id = default_token_id
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
                "candidate_token_ids": tuple(candidate_token_ids),
                "input_ids": tuple(input_ids),
                "step": step,
            }
        )
        preferred = self.choices_by_step.get((position, step), self.default_token_id)
        token_id = preferred if preferred in candidate_token_ids else candidate_token_ids[0]
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


def modulo_hash_map(vocab_size: int = 64) -> TokenHashMap:
    buckets = [[] for _ in range(16)]
    excluded = {0, 1, 2}
    token_to_bucket = []
    for token_id in range(vocab_size):
        bucket = token_id % 16
        token_to_bucket.append(bucket)
        if token_id not in excluded:
            buckets[bucket].append(token_id)
    return TokenHashMap(
        hash_bits=4,
        vocab_size=vocab_size,
        token_to_bucket=tuple(token_to_bucket),
        bucket_to_token_ids=tuple(tuple(bucket) for bucket in buckets),
        excluded_token_ids=excluded,
    )


def test_decoder_config_defaults_and_schedule_helper() -> None:
    default_config = config()

    assert default_config.editable_update_mode == "commit_once"
    assert default_config.hash_constraint_schedule == "always"
    assert default_config.to_dict()["editable_update_mode"] == "commit_once"
    assert default_config.to_dict()["hash_constraint_schedule"] == "always"
    assert [should_enforce_hash_constraint(step, 4, "always") for step in range(4)] == [
        True,
        True,
        True,
        True,
    ]
    assert [should_enforce_hash_constraint(step, 4, "final_only") for step in range(4)] == [
        False,
        False,
        False,
        True,
    ]
    assert [should_enforce_hash_constraint(step, 4, "late_half") for step in range(4)] == [
        False,
        False,
        True,
        True,
    ]


def test_decoder_config_rejects_invalid_update_mode_and_schedule() -> None:
    with pytest.raises(ValueError, match="editable_update_mode"):
        DiffusionDecodingConfig(mask_token_id=0, vocab_size=8, editable_update_mode="bad")
    with pytest.raises(ValueError, match="hash_constraint_schedule"):
        DiffusionDecodingConfig(mask_token_id=0, vocab_size=8, hash_constraint_schedule="bad")
    with pytest.raises(ValueError, match="commit_once decoding only supports"):
        DiffusionDecodingConfig(
            mask_token_id=0,
            vocab_size=8,
            hash_constraint_schedule="final_only",
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


def test_resample_each_step_always_rewrites_with_hash_bucket_candidates() -> None:
    token_hash = modulo_hash_map()
    hash_value = 5
    plan = build_reconstruction_plan(
        total_tokens=1,
        received_packets=[],
        hash_metadata={0: hash_value},
    )
    model = StepwiseProposalModel(
        choices_by_step={
            (0, 0): 21,
            (0, 1): 37,
            (0, 2): 53,
        }
    )

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=64,
            steps=3,
            block_length=1,
            editable_update_mode="resample_each_step",
            hash_constraint_schedule="always",
        ),
        token_hash_map=token_hash,
    )

    assert result.reconstructed_tokens == (53,)
    assert [call["step"] for call in model.proposal_calls] == [0, 1, 2]
    assert all(
        set(call["candidate_token_ids"]) == set(token_hash.candidate_token_ids(hash_value))
        for call in model.proposal_calls
    )
    assert result.diagnostics["hash_enforced_steps"] == (0, 1, 2)
    assert result.diagnostics["hash_relaxed_steps"] == ()
    assert result.diagnostics["updated_editable_positions_per_step"] == (1, 1, 1)


def test_resample_each_step_final_only_relaxes_then_enforces_hash() -> None:
    token_hash = modulo_hash_map()
    hash_value = 5
    plan = build_reconstruction_plan(
        total_tokens=1,
        received_packets=[],
        hash_metadata={0: hash_value},
    )
    model = StepwiseProposalModel(
        choices_by_step={
            (0, 0): 4,
            (0, 1): 6,
            (0, 2): 21,
        }
    )

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=64,
            steps=3,
            block_length=1,
            editable_update_mode="resample_each_step",
            hash_constraint_schedule="final_only",
        ),
        token_hash_map=token_hash,
    )

    assert result.reconstructed_tokens == (21,)
    assert model.proposal_calls[0]["candidate_token_ids"] == tuple(range(3, 64))
    assert model.proposal_calls[1]["candidate_token_ids"] == tuple(range(3, 64))
    assert set(model.proposal_calls[2]["candidate_token_ids"]) == set(
        token_hash.candidate_token_ids(hash_value)
    )
    assert result.diagnostics["hash_enforced_steps"] == (2,)
    assert result.diagnostics["hash_relaxed_steps"] == (0, 1)


def test_resample_each_step_late_half_enforces_only_late_steps() -> None:
    token_hash = modulo_hash_map()
    hash_value = 5
    plan = build_reconstruction_plan(
        total_tokens=1,
        received_packets=[],
        hash_metadata={0: hash_value},
    )
    model = StepwiseProposalModel(
        choices_by_step={
            (0, 0): 4,
            (0, 1): 6,
            (0, 2): 21,
            (0, 3): 37,
        }
    )

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=64,
            steps=4,
            block_length=1,
            editable_update_mode="resample_each_step",
            hash_constraint_schedule="late_half",
        ),
        token_hash_map=token_hash,
    )

    assert result.reconstructed_tokens == (37,)
    assert result.diagnostics["hash_enforced_steps"] == (2, 3)
    assert result.diagnostics["hash_relaxed_steps"] == (0, 1)
    assert set(model.proposal_calls[2]["candidate_token_ids"]) == set(
        token_hash.candidate_token_ids(hash_value)
    )


def test_resample_each_step_preserves_known_tokens_and_excludes_banned_relaxed_tokens() -> None:
    token_hash = modulo_hash_map()
    hash_value = 5
    plan = build_reconstruction_plan(
        total_tokens=2,
        received_packets=[
            Packet(
                source_id="sample-1",
                wire_id=0,
                kind="data",
                token_ids=[10],
                token_positions=[0],
            )
        ],
        hash_metadata={1: hash_value},
    )
    model = StepwiseProposalModel(
        choices_by_step={
            (1, 0): 0,
            (1, 1): 21,
        }
    )

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=64,
            steps=2,
            block_length=1,
            editable_update_mode="resample_each_step",
            hash_constraint_schedule="final_only",
        ),
        token_hash_map=token_hash,
    )

    assert result.reconstructed_tokens == (10, 21)
    assert model.proposal_calls[0]["candidate_token_ids"] == tuple(range(3, 64))
    assert 0 not in model.proposal_calls[0]["candidate_token_ids"]
    assert 1 not in model.proposal_calls[0]["candidate_token_ids"]
    assert 2 not in model.proposal_calls[0]["candidate_token_ids"]
    assert result.confidence_stats[0].was_fixed is True


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


def test_post_commit_hook_can_fix_new_tokens_after_model_commit() -> None:
    plan = build_reconstruction_plan(total_tokens=2, received_packets=[])
    model = StepwiseProposalModel(choices_by_step={(0, 0): 10, (1, 1): 99})
    hook_calls = []

    def post_commit_hook(
        *,
        input_ids,
        step,
        prompt_length,
        committed_positions,
        fixed_token_ids,
        plan,
        config,
    ):
        hook_calls.append(
            {
                "step": step,
                "input_ids": tuple(input_ids),
                "committed_positions": tuple(committed_positions),
            }
        )
        if step == 0 and input_ids[prompt_length] == 10:
            return {"fixed_tokens": {1: 11}}
        return {"fixed_tokens": {}}

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=128,
            steps=2,
            block_length=2,
        ),
        post_commit_hook=post_commit_hook,
    )

    assert result.reconstructed_tokens == (10, 11)
    assert len(model.proposal_calls) == 2
    assert {call["position"] for call in model.proposal_calls} == {0, 1}
    assert hook_calls[0]["committed_positions"] == (0,)
    assert result.step_summaries[0].committed_count == 2
    assert result.diagnostics["post_commit_hook_used"] is True
    assert result.diagnostics["post_commit_fixed_positions_per_step"] == (1,)
    assert result.diagnostics["updated_editable_positions_per_step"] == (2,)


def test_post_commit_hook_rejects_overwriting_committed_tokens() -> None:
    plan = build_reconstruction_plan(total_tokens=1, received_packets=[])
    model = StepwiseProposalModel(choices_by_step={(0, 0): 10})

    def post_commit_hook(**kwargs):
        return {"fixed_tokens": {0: 11}}

    with pytest.raises(RuntimeError, match="overwrite a committed token"):
        decode_masked_diffusion(
            model=model,
            plan=plan,
            config=DiffusionDecodingConfig(
                mask_token_id=0,
                eos_token_id=1,
                pad_token_id=2,
                vocab_size=128,
                steps=1,
                block_length=1,
            ),
            post_commit_hook=post_commit_hook,
        )


def test_position_specific_ban_excludes_only_that_position_after_rollback() -> None:
    plan = build_reconstruction_plan(total_tokens=2, received_packets=[])
    model = StepwiseProposalModel(
        choices_by_step={
            (0, 0): 10,
            (1, 0): 10,
            (0, 1): 10,
        },
        default_token_id=11,
    )

    class RollbackBanHook:
        rollback_enabled = True
        rollback_extra_steps = 1
        rollback_max_total_steps = 2
        rollback_stop_after_no_progress = 0

        def __call__(self, **kwargs):
            if kwargs["step"] == 0:
                return {
                    "rollback_positions": (0,),
                    "position_banned_tokens": {0: (10,)},
                }
            return {}

        def record_decode_outcome(self, **kwargs):
            pass

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=32,
            steps=1,
            block_length=2,
        ),
        post_commit_hook=RollbackBanHook(),
    )

    assert result.reconstructed_tokens == (3, 10)
    position0_extra_call = [
        call for call in model.proposal_calls
        if call["position"] == 0 and call["step"] == 1
    ][0]
    assert 10 not in position0_extra_call["candidate_token_ids"]
    assert 10 in [
        call for call in model.proposal_calls
        if call["position"] == 1 and call["step"] == 0
    ][0]["candidate_token_ids"]


def test_position_specific_ban_empty_candidates_fails_clearly() -> None:
    plan = build_reconstruction_plan(total_tokens=1, received_packets=[])
    model = StepwiseProposalModel({(0, 0): 3})

    class BanOnlyCandidateHook:
        rollback_enabled = True
        rollback_extra_steps = 1
        rollback_max_total_steps = 2
        rollback_stop_after_no_progress = 0

        def __call__(self, **kwargs):
            if kwargs["step"] == 0:
                return {
                    "rollback_positions": (0,),
                    "position_banned_tokens": {0: (3,)},
                }
            return {}

        def record_decode_outcome(self, **kwargs):
            pass

    with pytest.raises(RuntimeError, match="position-specific bans removed all candidates"):
        decode_masked_diffusion(
            model=model,
            plan=plan,
            config=DiffusionDecodingConfig(
                mask_token_id=0,
                eos_token_id=1,
                pad_token_id=2,
                vocab_size=4,
                steps=1,
                block_length=1,
            ),
            post_commit_hook=BanOnlyCandidateHook(),
        )


def test_adaptive_rollback_continues_past_base_steps_to_refill_masks() -> None:
    plan = build_reconstruction_plan(total_tokens=2, received_packets=[])
    model = StepwiseProposalModel(
        choices_by_step={
            (0, 0): 10,
            (1, 1): 20,
            (0, 2): 11,
        },
        default_token_id=11,
    )

    class LateRollbackHook:
        rollback_enabled = True
        rollback_extra_steps = 0
        rollback_max_total_steps = 4
        rollback_stop_after_no_progress = 0
        rollback_continue_until_stable = True

        def __call__(self, **kwargs):
            if kwargs["step"] == 1:
                return {
                    "rollback_positions": (0,),
                    "position_banned_tokens": {0: (10,)},
                }
            return {}

        def record_decode_outcome(self, **kwargs):
            self.outcome = dict(kwargs)

    hook = LateRollbackHook()
    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=32,
            steps=2,
            block_length=1,
        ),
        post_commit_hook=hook,
    )

    assert result.reconstructed_tokens == (11, 20)
    assert result.steps == 3
    assert result.diagnostics["rollback_adaptive_enabled"] is True
    assert result.diagnostics["rollback_total_steps_used"] == 3
    assert result.diagnostics["rollback_extra_steps_used"] == 1
    assert result.diagnostics["rollback_stopped_reason"] == "stable"
    assert result.diagnostics["rollback_final_zero_masks"] is True
    assert hook.outcome["stopped_reason"] == "stable"


def test_fixed_rollback_budget_can_leave_masks_when_adaptive_disabled() -> None:
    plan = build_reconstruction_plan(total_tokens=2, received_packets=[])
    model = StepwiseProposalModel(
        choices_by_step={
            (0, 0): 10,
            (1, 1): 20,
        },
        default_token_id=11,
    )

    class LateRollbackHook:
        rollback_enabled = True
        rollback_extra_steps = 0
        rollback_max_total_steps = 4
        rollback_stop_after_no_progress = 0

        def __call__(self, **kwargs):
            if kwargs["step"] == 1:
                return {"rollback_positions": (0,)}
            return {}

        def record_decode_outcome(self, **kwargs):
            pass

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=32,
            steps=2,
            block_length=1,
        ),
        post_commit_hook=LateRollbackHook(),
    )

    assert result.reconstructed_tokens == (0, 20)
    assert result.diagnostics["rollback_adaptive_enabled"] is False
    assert result.diagnostics["rollback_stopped_reason"] == "fixed_step_budget"
    assert result.diagnostics["rollback_remaining_masks_after_budget"] == 1


def test_adaptive_rollback_stops_at_max_total_steps() -> None:
    plan = build_reconstruction_plan(total_tokens=1, received_packets=[])
    model = StepwiseProposalModel({(0, 0): 10, (0, 1): 11}, default_token_id=12)

    class AlwaysRollbackHook:
        rollback_enabled = True
        rollback_extra_steps = 0
        rollback_max_total_steps = 2
        rollback_stop_after_no_progress = 0
        rollback_continue_until_stable = True

        def __call__(self, **kwargs):
            return {"rollback_positions": (0,)}

        def record_decode_outcome(self, **kwargs):
            pass

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=32,
            steps=1,
            block_length=1,
        ),
        post_commit_hook=AlwaysRollbackHook(),
    )

    assert result.steps == 2
    assert result.reconstructed_tokens == (0,)
    assert result.diagnostics["rollback_stopped_reason"] == "max_total_steps"
    assert result.diagnostics["rollback_remaining_masks_after_budget"] == 1


def test_adaptive_rollback_no_progress_stop_triggers() -> None:
    plan = build_reconstruction_plan(total_tokens=1, received_packets=[])
    model = StepwiseProposalModel({(0, 0): 10, (0, 1): 11}, default_token_id=12)

    class AlwaysRollbackHook:
        rollback_enabled = True
        rollback_extra_steps = 0
        rollback_max_total_steps = 5
        rollback_stop_after_no_progress = 1
        rollback_continue_until_stable = True

        def __call__(self, **kwargs):
            return {"rollback_positions": (0,)}

        def record_decode_outcome(self, **kwargs):
            pass

    result = decode_masked_diffusion(
        model=model,
        plan=plan,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=32,
            steps=1,
            block_length=1,
        ),
        post_commit_hook=AlwaysRollbackHook(),
    )

    assert result.steps == 2
    assert result.diagnostics["rollback_no_progress_stop"] is True
    assert result.diagnostics["rollback_stopped_reason"] == "no_progress"
    assert result.diagnostics["rollback_remaining_masks_after_budget"] == 1


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
