from diffusion_fec.coding.token_hash import build_token_hash_map


class TinyTokenizerAdapter:
    def decode_token(self, token_id: int) -> bytes:
        return f"adapter-token-{token_id}".encode("utf-8")


class BatchTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        self.calls.append(tuple(token_ids))
        return [f"batch-token-{token_id}" for token_id in token_ids]


class BatchTokenizerAdapter:
    def __init__(self) -> None:
        self.tokenizer = BatchTokenizer()
        self.fallback_calls: list[int] = []

    def decode_token(self, token_id: int) -> str:
        self.fallback_calls.append(token_id)
        return f"fallback-token-{token_id}"


def test_hash_builder_accepts_tokenizer_like_adapter() -> None:
    token_hash = build_token_hash_map(
        vocab_size=8,
        hash_bits=4,
        decode_token=TinyTokenizerAdapter(),
    )

    assert len(token_hash.token_to_bucket) == 8


def test_hash_builder_prefers_tokenizer_native_batch_conversion() -> None:
    adapter = BatchTokenizerAdapter()

    token_hash = build_token_hash_map(
        vocab_size=8,
        hash_bits=4,
        decode_token=adapter,
    )

    assert len(token_hash.token_to_bucket) == 8
    assert adapter.tokenizer.calls == [tuple(range(8))]
    assert adapter.fallback_calls == []


def test_hash_builder_falls_back_when_batch_token_is_missing() -> None:
    adapter = BatchTokenizerAdapter()

    def convert_ids_to_tokens(token_ids: list[int]) -> list[str | None]:
        adapter.tokenizer.calls.append(tuple(token_ids))
        return [
            None if token_id == 3 else f"batch-token-{token_id}"
            for token_id in token_ids
        ]

    adapter.tokenizer.convert_ids_to_tokens = convert_ids_to_tokens

    token_hash = build_token_hash_map(
        vocab_size=8,
        hash_bits=4,
        decode_token=adapter,
    )

    assert len(token_hash.token_to_bucket) == 8
    assert adapter.fallback_calls == [3]
