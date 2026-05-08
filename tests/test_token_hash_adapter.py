from diffusion_fec.coding.token_hash import build_token_hash_map


class TinyTokenizerAdapter:
    def decode_token(self, token_id: int) -> bytes:
        return f"adapter-token-{token_id}".encode("utf-8")


def test_hash_builder_accepts_tokenizer_like_adapter() -> None:
    token_hash = build_token_hash_map(
        vocab_size=8,
        hash_bits=4,
        decode_token=TinyTokenizerAdapter(),
    )

    assert len(token_hash.token_to_bucket) == 8
