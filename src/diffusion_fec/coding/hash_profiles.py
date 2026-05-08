"""Persisted token-hash profiles for reproducible experiment runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from pathlib import Path
from typing import Any

import numpy as np

from diffusion_fec.coding.token_hash import (
    TokenHashMap,
    build_token_hash_map,
    token_hash_map_from_token_to_bucket,
)


HASH_PROFILE_METADATA_FILENAME = "hash_profile_metadata.json"
HASH_PROFILE_FORMAT_VERSION = 1
DEFAULT_HASH_MAP_MODE = "uniform"
HASH_ALGORITHM = "sha256_tokenizer_token_string_token_id_salt_mod_bucket_count"
TOKEN_STRING_SOURCE = "tokenizer_native_token_string_plus_token_id"


def hash_map_filename(hash_bits: int, map_mode: str = DEFAULT_HASH_MAP_MODE) -> str:
    """Return the stable profile filename for a hash map."""

    _validate_map_mode(map_mode)
    return f"{map_mode}_hash{hash_bits}_map.npy"


def metadata_path(profile_dir: str | Path) -> Path:
    return Path(profile_dir) / HASH_PROFILE_METADATA_FILENAME


def map_path(
    profile_dir: str | Path,
    *,
    hash_bits: int,
    map_mode: str = DEFAULT_HASH_MAP_MODE,
) -> Path:
    return Path(profile_dir) / hash_map_filename(hash_bits, map_mode)


def build_and_save_hash_profile(
    *,
    profile_dir: str | Path,
    profile_name: str,
    vocab_size: int,
    hash_bits: int,
    decode_token: Callable[[int], str | bytes] | object,
    excluded_token_ids: Iterable[int] = (),
    salt: str = "",
    map_mode: str = DEFAULT_HASH_MAP_MODE,
    model_id: str | None = None,
    tokenizer_name: str | None = None,
    overwrite: bool = False,
) -> TokenHashMap:
    """Build a token hash map, save it, and return the in-memory map."""

    token_hash_map = build_token_hash_map(
        vocab_size=vocab_size,
        hash_bits=hash_bits,
        decode_token=decode_token,
        excluded_token_ids=excluded_token_ids,
        salt=salt,
    )
    save_hash_profile(
        profile_dir=profile_dir,
        profile_name=profile_name,
        token_hash_map=token_hash_map,
        map_mode=map_mode,
        model_id=model_id,
        tokenizer_name=tokenizer_name,
        overwrite=overwrite,
    )
    return token_hash_map


def save_hash_profile(
    *,
    profile_dir: str | Path,
    profile_name: str,
    token_hash_map: TokenHashMap,
    map_mode: str = DEFAULT_HASH_MAP_MODE,
    model_id: str | None = None,
    tokenizer_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one token-hash map plus profile metadata."""

    if not profile_name:
        raise ValueError("profile_name is required")
    _validate_map_mode(map_mode)

    profile_path = Path(profile_dir)
    profile_path.mkdir(parents=True, exist_ok=True)
    target_map_path = map_path(
        profile_path,
        hash_bits=token_hash_map.hash_bits,
        map_mode=map_mode,
    )
    if target_map_path.exists() and not overwrite:
        raise FileExistsError(f"hash map already exists: {target_map_path}")

    metadata = _load_metadata_if_exists(profile_path) or _new_metadata(
        profile_name=profile_name,
        token_hash_map=token_hash_map,
        model_id=model_id,
        tokenizer_name=tokenizer_name,
    )
    _validate_metadata_compatible(
        metadata=metadata,
        profile_name=profile_name,
        token_hash_map=token_hash_map,
        model_id=model_id,
        tokenizer_name=tokenizer_name,
    )

    np.save(
        target_map_path,
        np.asarray(token_hash_map.token_to_bucket, dtype=_array_dtype(token_hash_map.hash_bits)),
        allow_pickle=False,
    )
    maps = metadata.setdefault("maps", {})
    mode_maps = maps.setdefault(map_mode, {})
    mode_maps[str(token_hash_map.hash_bits)] = {
        "file": target_map_path.name,
        "hash_bits": token_hash_map.hash_bits,
        "bucket_count": token_hash_map.bucket_count,
        "algorithm": HASH_ALGORITHM,
    }
    _write_metadata(profile_path, metadata)
    return metadata


def load_hash_profile(
    *,
    profile_dir: str | Path,
    hash_bits: int,
    map_mode: str = DEFAULT_HASH_MAP_MODE,
) -> TokenHashMap:
    """Load a saved token hash map from a profile directory."""

    _validate_map_mode(map_mode)
    profile_path = Path(profile_dir)
    metadata = load_hash_profile_metadata(profile_path)
    try:
        map_info = metadata["maps"][map_mode][str(hash_bits)]
    except KeyError as exc:
        raise FileNotFoundError(
            f"hash profile does not contain {map_mode!r} hash_bits={hash_bits}"
        ) from exc

    path = profile_path / map_info["file"]
    if not path.exists():
        raise FileNotFoundError(f"hash map file not found: {path}")

    token_to_bucket = np.load(path, allow_pickle=False)
    if token_to_bucket.ndim != 1:
        raise ValueError("stored hash map must be one-dimensional")

    vocab_size = int(metadata["vocab_size"])
    if len(token_to_bucket) != vocab_size:
        raise ValueError(
            f"hash map length mismatch: expected vocab_size {vocab_size}, "
            f"got {len(token_to_bucket)}"
        )

    return token_hash_map_from_token_to_bucket(
        hash_bits=hash_bits,
        token_to_bucket=tuple(int(bucket_id) for bucket_id in token_to_bucket.tolist()),
        excluded_token_ids=metadata.get("excluded_token_ids", ()),
        salt=str(metadata.get("salt", "")),
        vocab_size=vocab_size,
    )


def load_or_build_hash_profile(
    *,
    profile_dir: str | Path | None,
    profile_name: str,
    vocab_size: int,
    hash_bits: int,
    decode_token: Callable[[int], str | bytes] | object,
    excluded_token_ids: Iterable[int] = (),
    salt: str = "",
    map_mode: str = DEFAULT_HASH_MAP_MODE,
    model_id: str | None = None,
    tokenizer_name: str | None = None,
    build_if_missing: bool = False,
    overwrite: bool = False,
) -> tuple[TokenHashMap, dict[str, Any]]:
    """Resolve a profile-backed map, optionally building it when absent."""

    if profile_dir is None:
        token_hash_map = build_token_hash_map(
            vocab_size=vocab_size,
            hash_bits=hash_bits,
            decode_token=decode_token,
            excluded_token_ids=excluded_token_ids,
            salt=salt,
        )
        return token_hash_map, _hash_profile_info(
            source="live_batch_build",
            profile_dir=None,
            profile_name=profile_name,
            hash_bits=hash_bits,
            map_mode=map_mode,
            file=None,
        )

    profile_path = Path(profile_dir)
    try:
        token_hash_map = load_hash_profile(
            profile_dir=profile_path,
            hash_bits=hash_bits,
            map_mode=map_mode,
        )
        source = "loaded_profile"
    except FileNotFoundError:
        if not build_if_missing:
            raise FileNotFoundError(
                f"hash profile is missing {map_mode!r} hash_bits={hash_bits}; "
                "pass --build-hash-profile to create it"
            )
        token_hash_map = build_and_save_hash_profile(
            profile_dir=profile_path,
            profile_name=profile_name,
            vocab_size=vocab_size,
            hash_bits=hash_bits,
            decode_token=decode_token,
            excluded_token_ids=excluded_token_ids,
            salt=salt,
            map_mode=map_mode,
            model_id=model_id,
            tokenizer_name=tokenizer_name,
            overwrite=overwrite,
        )
        source = "built_profile"

    return token_hash_map, _hash_profile_info(
        source=source,
        profile_dir=profile_path,
        profile_name=profile_name,
        hash_bits=hash_bits,
        map_mode=map_mode,
        file=hash_map_filename(hash_bits, map_mode),
    )


def load_hash_profile_metadata(profile_dir: str | Path) -> dict[str, Any]:
    path = metadata_path(profile_dir)
    if not path.exists():
        raise FileNotFoundError(f"hash profile metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _new_metadata(
    *,
    profile_name: str,
    token_hash_map: TokenHashMap,
    model_id: str | None,
    tokenizer_name: str | None,
) -> dict[str, Any]:
    return {
        "format_version": HASH_PROFILE_FORMAT_VERSION,
        "profile_name": profile_name,
        "model_id": model_id,
        "tokenizer_name": tokenizer_name,
        "vocab_size": token_hash_map.vocab_size,
        "excluded_token_ids": sorted(token_hash_map.excluded_token_ids),
        "salt": token_hash_map.salt,
        "token_string_source": TOKEN_STRING_SOURCE,
        "maps": {},
    }


def _load_metadata_if_exists(profile_dir: Path) -> dict[str, Any] | None:
    path = metadata_path(profile_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metadata(profile_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path(profile_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_metadata_compatible(
    *,
    metadata: dict[str, Any],
    profile_name: str,
    token_hash_map: TokenHashMap,
    model_id: str | None,
    tokenizer_name: str | None,
) -> None:
    expected = {
        "format_version": HASH_PROFILE_FORMAT_VERSION,
        "profile_name": profile_name,
        "vocab_size": token_hash_map.vocab_size,
        "excluded_token_ids": sorted(token_hash_map.excluded_token_ids),
        "salt": token_hash_map.salt,
        "token_string_source": TOKEN_STRING_SOURCE,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"hash profile metadata mismatch for {key}")
    for key, value in (("model_id", model_id), ("tokenizer_name", tokenizer_name)):
        existing = metadata.get(key)
        if value is not None and existing is not None and existing != value:
            raise ValueError(f"hash profile metadata mismatch for {key}")
        if existing is None and value is not None:
            metadata[key] = value


def _hash_profile_info(
    *,
    source: str,
    profile_dir: Path | None,
    profile_name: str,
    hash_bits: int,
    map_mode: str,
    file: str | None,
) -> dict[str, Any]:
    return {
        "source": source,
        "profile_dir": None if profile_dir is None else str(profile_dir),
        "profile_name": profile_name,
        "map_mode": map_mode,
        "hash_bits": hash_bits,
        "file": file,
        "format_version": HASH_PROFILE_FORMAT_VERSION,
        "algorithm": HASH_ALGORITHM,
        "token_string_source": TOKEN_STRING_SOURCE,
    }


def _array_dtype(hash_bits: int):
    if hash_bits <= 8:
        return np.uint8
    if hash_bits <= 16:
        return np.uint16
    return np.uint32


def _validate_map_mode(map_mode: str) -> None:
    if not isinstance(map_mode, str):
        raise TypeError("map_mode must be a string")
    if not map_mode:
        raise ValueError("map_mode must be non-empty")
    if not map_mode.replace("_", "").replace("-", "").isalnum():
        raise ValueError("map_mode may contain only letters, digits, hyphens, and underscores")
