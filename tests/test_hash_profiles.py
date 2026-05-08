import json

import pytest

from diffusion_fec.coding.hash_profiles import (
    HASH_PROFILE_METADATA_FILENAME,
    build_and_save_hash_profile,
    hash_map_filename,
    load_hash_profile,
    load_or_build_hash_profile,
)


def decode_token(token_id: int) -> str:
    return f"token-{token_id}"


def test_build_and_load_hash_profile_round_trips_token_map(tmp_path) -> None:
    profile_dir = tmp_path / "profiles" / "fake"

    built = build_and_save_hash_profile(
        profile_dir=profile_dir,
        profile_name="fake-profile",
        vocab_size=16,
        hash_bits=4,
        decode_token=decode_token,
        excluded_token_ids={0, 1},
        salt="unit-test",
        model_id="fake-model",
        tokenizer_name="fake-tokenizer",
    )
    loaded = load_hash_profile(profile_dir=profile_dir, hash_bits=4)

    assert loaded.token_to_bucket == built.token_to_bucket
    assert loaded.bucket_to_token_ids == built.bucket_to_token_ids
    assert loaded.excluded_token_ids == frozenset({0, 1})
    assert loaded.salt == "unit-test"
    assert (profile_dir / hash_map_filename(4)).exists()

    metadata = json.loads(
        (profile_dir / HASH_PROFILE_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["profile_name"] == "fake-profile"
    assert metadata["model_id"] == "fake-model"
    assert metadata["tokenizer_name"] == "fake-tokenizer"
    assert metadata["maps"]["uniform"]["4"]["file"] == hash_map_filename(4)


def test_hash_profile_save_refuses_to_overwrite_by_default(tmp_path) -> None:
    profile_dir = tmp_path / "profile"

    build_and_save_hash_profile(
        profile_dir=profile_dir,
        profile_name="fake-profile",
        vocab_size=8,
        hash_bits=4,
        decode_token=decode_token,
    )

    with pytest.raises(FileExistsError, match="hash map already exists"):
        build_and_save_hash_profile(
            profile_dir=profile_dir,
            profile_name="fake-profile",
            vocab_size=8,
            hash_bits=4,
            decode_token=decode_token,
        )


def test_load_or_build_hash_profile_requires_explicit_build(tmp_path) -> None:
    profile_dir = tmp_path / "profile"

    with pytest.raises(FileNotFoundError, match="--build-hash-profile"):
        load_or_build_hash_profile(
            profile_dir=profile_dir,
            profile_name="fake-profile",
            vocab_size=8,
            hash_bits=4,
            decode_token=decode_token,
        )

    token_hash, info = load_or_build_hash_profile(
        profile_dir=profile_dir,
        profile_name="fake-profile",
        vocab_size=8,
        hash_bits=4,
        decode_token=decode_token,
        build_if_missing=True,
    )

    assert token_hash.vocab_size == 8
    assert info["source"] == "built_profile"
    assert load_or_build_hash_profile(
        profile_dir=profile_dir,
        profile_name="fake-profile",
        vocab_size=8,
        hash_bits=4,
        decode_token=decode_token,
    )[1]["source"] == "loaded_profile"
