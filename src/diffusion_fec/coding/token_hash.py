"""Deterministic token-hash maps for vocabulary constraints."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


SUPPORTED_HASH_BITS = frozenset({4, 8, 16})


def _validate_hash_bits(hash_bits: int) -> None:
    if hash_bits not in SUPPORTED_HASH_BITS:
        supported = ", ".join(str(bits) for bits in sorted(SUPPORTED_HASH_BITS))
        raise ValueError(f"hash_bits must be one of: {supported}")


def _validate_token_id(token_id: int, vocab_size: int, field_name: str = "token_id") -> None:
    if not isinstance(token_id, int):
        raise TypeError(f"{field_name} must be an int")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"{field_name} must be in range [0, {vocab_size})")


def _token_bytes(decoded_token: str | bytes) -> bytes:
    if isinstance(decoded_token, bytes):
        return decoded_token
    if isinstance(decoded_token, str):
        return decoded_token.encode("utf-8")
    raise TypeError("decode_token must return str or bytes")


def _resolve_decode_token(
    decode_token: Callable[[int], str | bytes] | object,
) -> Callable[[int], str | bytes]:
    if callable(decode_token):
        return decode_token
    adapter_decode = getattr(decode_token, "decode_token", None)
    if callable(adapter_decode):
        return adapter_decode
    raise TypeError("decode_token must be callable or expose decode_token(token_id)")


def _resolve_batch_decode_token(
    decode_token: Callable[[int], str | bytes] | object,
) -> Callable[[Iterable[int]], Iterable[str | bytes]] | None:
    """Use tokenizer-native batch token conversion when an adapter exposes it."""

    if callable(decode_token):
        return None

    for source in (decode_token, getattr(decode_token, "tokenizer", None)):
        if source is None:
            continue
        convert = getattr(source, "convert_ids_to_tokens", None)
        if callable(convert):
            return lambda token_ids, convert=convert: convert(list(token_ids))
    return None


def _hash_token_to_bucket(
    token_id: int,
    decoded_token: str | bytes,
    *,
    hash_bits: int,
    salt: str,
) -> int:
    payload = b"::".join(
        (
            _token_bytes(decoded_token),
            str(token_id).encode("ascii"),
            salt.encode("utf-8"),
        )
    )
    digest = sha256(payload).digest()
    digest_int = int.from_bytes(digest, byteorder="big", signed=False)
    return digest_int % (1 << hash_bits)


@dataclass(frozen=True)
class TokenHashMap:
    """Vocabulary token IDs mapped into deterministic hash buckets."""

    hash_bits: int
    vocab_size: int
    token_to_bucket: tuple[int, ...]
    bucket_to_token_ids: tuple[tuple[int, ...], ...]
    excluded_token_ids: frozenset[int] = frozenset()
    salt: str = ""

    def __post_init__(self) -> None:
        _validate_hash_bits(self.hash_bits)
        if not isinstance(self.vocab_size, int):
            raise TypeError("vocab_size must be an int")
        if self.vocab_size < 0:
            raise ValueError("vocab_size must be non-negative")
        object.__setattr__(self, "token_to_bucket", tuple(self.token_to_bucket))
        object.__setattr__(
            self,
            "bucket_to_token_ids",
            tuple(tuple(bucket) for bucket in self.bucket_to_token_ids),
        )
        object.__setattr__(self, "excluded_token_ids", frozenset(self.excluded_token_ids))

        bucket_count = 1 << self.hash_bits
        if len(self.token_to_bucket) != self.vocab_size:
            raise ValueError("token_to_bucket length must equal vocab_size")
        if len(self.bucket_to_token_ids) != bucket_count:
            raise ValueError("bucket_to_token_ids length must equal bucket count")

        bucket_token_sets = tuple(set(token_ids) for token_ids in self.bucket_to_token_ids)
        for bucket_id, token_ids in enumerate(self.bucket_to_token_ids):
            if len(bucket_token_sets[bucket_id]) != len(token_ids):
                raise ValueError("bucket_to_token_ids must not contain duplicate token IDs")

        for token_id, bucket_id in enumerate(self.token_to_bucket):
            if not isinstance(bucket_id, int):
                raise TypeError("bucket IDs must be ints")
            if bucket_id < 0 or bucket_id >= bucket_count:
                raise ValueError("bucket IDs must be valid for hash_bits")
            if token_id in self.excluded_token_ids:
                continue
            if token_id not in bucket_token_sets[bucket_id]:
                raise ValueError("bucket_to_token_ids must include every non-excluded token")

        for token_id in self.excluded_token_ids:
            _validate_token_id(token_id, self.vocab_size, "excluded token ID")
        for bucket_id, token_ids in enumerate(self.bucket_to_token_ids):
            for token_id in token_ids:
                _validate_token_id(token_id, self.vocab_size)
                if self.token_to_bucket[token_id] != bucket_id:
                    raise ValueError("bucket_to_token_ids contains a token in the wrong bucket")
                if token_id in self.excluded_token_ids:
                    raise ValueError("bucket_to_token_ids must not contain excluded token IDs")

    @property
    def bucket_count(self) -> int:
        return 1 << self.hash_bits

    def bucket_for_token(self, token_id: int) -> int:
        _validate_token_id(token_id, self.vocab_size)
        return self.token_to_bucket[token_id]

    def candidate_token_ids(self, bucket_id: int) -> tuple[int, ...]:
        self.validate_bucket_id(bucket_id)
        return self.bucket_to_token_ids[bucket_id]

    def allowed_mask(self, bucket_id: int) -> tuple[bool, ...]:
        """Build a Python boolean mask for one bucket on demand."""

        candidates = set(self.candidate_token_ids(bucket_id))
        return tuple(token_id in candidates for token_id in range(self.vocab_size))

    def validate_bucket_id(self, bucket_id: int) -> None:
        if not isinstance(bucket_id, int):
            raise TypeError("bucket_id must be an int")
        if bucket_id < 0 or bucket_id >= self.bucket_count:
            raise ValueError(f"bucket_id must be in range [0, {self.bucket_count})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash_bits": self.hash_bits,
            "vocab_size": self.vocab_size,
            "token_to_bucket": list(self.token_to_bucket),
            "bucket_to_token_ids": [list(bucket) for bucket in self.bucket_to_token_ids],
            "excluded_token_ids": sorted(self.excluded_token_ids),
            "salt": self.salt,
        }


def build_token_hash_map(
    *,
    vocab_size: int,
    hash_bits: int,
    decode_token: Callable[[int], str | bytes] | object,
    excluded_token_ids: Iterable[int] = (),
    salt: str = "",
) -> TokenHashMap:
    """Build a deterministic hash map using a tokenizer-style decode callback."""

    _validate_hash_bits(hash_bits)
    if not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an int")
    if vocab_size < 0:
        raise ValueError("vocab_size must be non-negative")
    excluded = frozenset(excluded_token_ids)
    for token_id in excluded:
        _validate_token_id(token_id, vocab_size, "excluded token ID")

    bucket_count = 1 << hash_bits
    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    token_to_bucket: list[int] = []
    decoded_tokens = _decoded_tokens_for_vocab(
        vocab_size=vocab_size,
        decode_token=decode_token,
    )
    for token_id, decoded_token in enumerate(decoded_tokens):
        bucket_id = _hash_token_to_bucket(
            token_id,
            decoded_token,
            hash_bits=hash_bits,
            salt=salt,
        )
        token_to_bucket.append(bucket_id)
        if token_id not in excluded:
            buckets[bucket_id].append(token_id)

    return TokenHashMap(
        hash_bits=hash_bits,
        vocab_size=vocab_size,
        token_to_bucket=tuple(token_to_bucket),
        bucket_to_token_ids=tuple(tuple(bucket) for bucket in buckets),
        excluded_token_ids=excluded,
        salt=salt,
    )


def _decoded_tokens_for_vocab(
    *,
    vocab_size: int,
    decode_token: Callable[[int], str | bytes] | object,
) -> tuple[str | bytes, ...]:
    batch_decode = _resolve_batch_decode_token(decode_token)
    if batch_decode is not None:
        decoded_tokens = tuple(batch_decode(range(vocab_size)))
        if len(decoded_tokens) != vocab_size:
            raise ValueError("batch token conversion must return one token per token ID")
        if all(token is not None for token in decoded_tokens):
            return decoded_tokens

        decode = _resolve_decode_token(decode_token)
        return tuple(
            decode(token_id) if decoded_token is None else decoded_token
            for token_id, decoded_token in enumerate(decoded_tokens)
        )

    decode = _resolve_decode_token(decode_token)
    return tuple(decode(token_id) for token_id in range(vocab_size))
