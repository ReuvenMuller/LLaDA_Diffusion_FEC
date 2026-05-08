import json

import pytest

from diffusion_fec.coding.token_hash import (
    TokenHashMap,
    build_token_hash_map,
)


def decode_token(token_id: int) -> str:
    return f"token-{token_id}"


@pytest.mark.parametrize("hash_bits", [4, 8, 16])
def test_every_vocab_token_maps_to_valid_bucket(hash_bits: int) -> None:
    token_hash = build_token_hash_map(
        vocab_size=32,
        hash_bits=hash_bits,
        decode_token=decode_token,
    )

    assert token_hash.bucket_count == 1 << hash_bits
    assert len(token_hash.token_to_bucket) == 32
    assert all(0 <= bucket < token_hash.bucket_count for bucket in token_hash.token_to_bucket)


def test_candidate_lists_match_token_buckets() -> None:
    token_hash = build_token_hash_map(
        vocab_size=48,
        hash_bits=4,
        decode_token=decode_token,
    )

    for token_id in range(token_hash.vocab_size):
        bucket_id = token_hash.bucket_for_token(token_id)
        assert token_id in token_hash.candidate_token_ids(bucket_id)


def test_special_token_exclusion_removes_candidates_but_keeps_bucket_mapping() -> None:
    excluded = {0, 5, 9}
    token_hash = build_token_hash_map(
        vocab_size=16,
        hash_bits=4,
        decode_token=decode_token,
        excluded_token_ids=excluded,
    )

    assert len(token_hash.token_to_bucket) == 16
    for token_id in excluded:
        bucket_id = token_hash.bucket_for_token(token_id)
        assert token_id not in token_hash.candidate_token_ids(bucket_id)


def test_allowed_mask_has_expected_shape_and_candidates() -> None:
    token_hash = build_token_hash_map(
        vocab_size=24,
        hash_bits=4,
        decode_token=decode_token,
        excluded_token_ids={3},
    )
    bucket_id = token_hash.bucket_for_token(4)
    allowed_mask = token_hash.allowed_mask(bucket_id)

    assert len(allowed_mask) == 24
    assert allowed_mask[4] is True
    assert allowed_mask[3] is False
    assert {
        token_id for token_id, allowed in enumerate(allowed_mask) if allowed
    } == set(token_hash.candidate_token_ids(bucket_id))


def test_hash_map_serializes_to_json() -> None:
    token_hash = build_token_hash_map(
        vocab_size=8,
        hash_bits=4,
        decode_token=decode_token,
        excluded_token_ids={1},
        salt="test",
    )

    json.dumps(token_hash.to_dict())


def test_hash_map_rejects_unsupported_hash_bits() -> None:
    with pytest.raises(ValueError, match="hash_bits must be one of"):
        build_token_hash_map(vocab_size=8, hash_bits=12, decode_token=decode_token)


def test_hash_map_validates_manual_bucket_consistency() -> None:
    with pytest.raises(ValueError, match="wrong bucket"):
        TokenHashMap(
            hash_bits=4,
            vocab_size=2,
            token_to_bucket=(0, 1),
            bucket_to_token_ids=((0, 1), (1,)) + tuple(() for _ in range(14)),
        )
