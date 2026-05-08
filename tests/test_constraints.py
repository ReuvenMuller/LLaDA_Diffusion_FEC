import json

from diffusion_fec.coding.packetizer import build_reconstruction_plan
from diffusion_fec.coding.token_hash import build_token_hash_map
from diffusion_fec.decoding.constraints import (
    build_constraint_masks,
    build_hash_constraint_view,
)
from diffusion_fec.types import Packet, STATE_MISSING


def decode_token(token_id: int) -> str:
    return f"token-{token_id}"


def make_plan():
    received = [
        Packet(
            source_id="sample-1",
            wire_id=0,
            kind="data",
            token_ids=[10, 13],
            token_positions=[0, 3],
        )
    ]
    token_hash = build_token_hash_map(
        vocab_size=32,
        hash_bits=4,
        decode_token=decode_token,
        excluded_token_ids={0, 31},
    )
    hash_value = token_hash.bucket_for_token(22)
    plan = build_reconstruction_plan(
        total_tokens=5,
        received_packets=received,
        hash_metadata={1: hash_value},
    )
    return plan, token_hash, hash_value


def test_constraint_masks_mark_known_positions_fixed_and_erased_positions_editable() -> None:
    plan, _, _ = make_plan()
    masks = build_constraint_masks(plan)

    assert masks.known_mask == (True, False, False, True, False)
    assert masks.fixed_mask == (True, False, False, True, False)
    assert masks.editable_mask == (False, True, True, False, True)
    assert masks.hash_guided_mask == (False, True, False, False, False)
    assert masks.unguided_mask == (False, False, True, False, True)
    assert masks.fixed_token_ids == (10, None, None, 13, None)
    assert masks.fixed_count == 2
    assert masks.editable_count == 3
    assert masks.hash_guided_count == 1
    assert masks.unguided_count == 2


def test_hash_guided_positions_only_allow_matching_bucket_token_ids() -> None:
    plan, token_hash, hash_value = make_plan()
    view = build_hash_constraint_view(plan, token_hash)

    allowed = view.allowed_token_ids(1)
    assert 22 in allowed
    assert all(token_hash.bucket_for_token(token_id) == hash_value for token_id in allowed)
    assert 0 not in allowed
    assert 31 not in allowed
    assert set(view.position_to_allowed_token_ids) == {1}


def test_hash_constraint_view_mask_matches_candidate_ids() -> None:
    plan, token_hash, _ = make_plan()
    view = build_hash_constraint_view(plan, token_hash)
    allowed_mask = view.allowed_mask(1)

    assert len(allowed_mask) == token_hash.vocab_size
    assert {
        token_id for token_id, allowed in enumerate(allowed_mask) if allowed
    } == set(view.allowed_token_ids(1))


def test_hash_constraint_view_can_skip_boolean_masks() -> None:
    plan, token_hash, _ = make_plan()
    view = build_hash_constraint_view(plan, token_hash, include_masks=False)

    assert set(view.position_to_allowed_token_ids) == {1}
    assert view.position_to_allowed_mask == {}


def test_empty_bucket_positions_are_reported_without_decoder_fallback() -> None:
    token_hash = build_token_hash_map(
        vocab_size=1,
        hash_bits=4,
        decode_token=decode_token,
        excluded_token_ids={0},
    )
    plan = build_reconstruction_plan(
        total_tokens=1,
        received_packets=[],
        hash_metadata={0: token_hash.bucket_for_token(0)},
    )
    view = build_hash_constraint_view(plan, token_hash)

    assert plan.entries[0].state == STATE_MISSING
    assert view.allowed_token_ids(0) == ()
    assert view.empty_bucket_positions == (0,)


def test_constraints_serialize_to_json() -> None:
    plan, token_hash, _ = make_plan()
    masks = build_constraint_masks(plan)
    view = build_hash_constraint_view(plan, token_hash)

    json.dumps(masks.to_dict())
    json.dumps(view.to_dict())
