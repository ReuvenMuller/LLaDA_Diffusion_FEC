import os

import pytest

from diffusion_fec.coding.token_hash import build_token_hash_map
from diffusion_fec.models.llada import (
    LLADA_1_5_DEFAULT_EOS_TOKEN_ID,
    LLADA_1_5_DEFAULT_MASK_TOKEN_ID,
    LLADA_1_5_DEFAULT_PAD_TOKEN_ID,
    LLADA_1_5_DEFAULT_VOCAB_SIZE,
    LLADA_1_5_MODEL_ID,
    LLaDAAdapter,
)


pytestmark = pytest.mark.hf


def test_llada_tokenizer_smoke_opt_in() -> None:
    if os.environ.get("RUN_HF_TOKENIZER_SMOKE") != "1":
        pytest.skip("set RUN_HF_TOKENIZER_SMOKE=1 to load the Hugging Face tokenizer")
    pytest.importorskip("transformers")

    adapter = LLaDAAdapter.from_pretrained(LLADA_1_5_MODEL_ID, load_model=False)

    assert adapter.mask_token_id == LLADA_1_5_DEFAULT_MASK_TOKEN_ID
    assert adapter.eos_token_id == LLADA_1_5_DEFAULT_EOS_TOKEN_ID
    assert adapter.pad_token_id == LLADA_1_5_DEFAULT_PAD_TOKEN_ID
    assert adapter.vocab_size == LLADA_1_5_DEFAULT_VOCAB_SIZE
    assert adapter.tokenize("hello", add_special_tokens=False)
    assert adapter.decode_token(0)

    token_hash = build_token_hash_map(
        vocab_size=16,
        hash_bits=4,
        decode_token=adapter.decode_token,
    )
    assert len(token_hash.token_to_bucket) == 16
