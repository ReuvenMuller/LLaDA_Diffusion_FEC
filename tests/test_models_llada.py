from dataclasses import dataclass

import pytest

from diffusion_fec.models.base import MaskedDiffusionModel
from diffusion_fec.models.llada import (
    LLADA_1_5_DEFAULT_EOS_TOKEN_ID,
    LLADA_1_5_DEFAULT_MASK_TOKEN_ID,
    LLADA_1_5_DEFAULT_PAD_TOKEN_ID,
    LLADA_1_5_DEFAULT_VOCAB_SIZE,
    LLaDAAdapter,
)


@dataclass
class FakeConfig:
    mask_token_id: int = 7
    eos_token_id: int = 8
    pad_token_id: int = 9
    vocab_size: int = 16
    max_sequence_length: int = 32


class FakeModel:
    config = FakeConfig()
    device = "fake-device"

    def eval(self):
        self.did_eval = True
        return self


class FakeTokenizer:
    eos_token_id = 8
    pad_token_id = 9
    vocab_size = 16
    model_max_length = 32

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        token_ids = [ord(char) % self.vocab_size for char in text]
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return {"input_ids": token_ids}

    def decode(self, token_ids, skip_special_tokens=False):
        return ",".join(str(token_id) for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id):
        return f"tok-{token_id}"

    def __len__(self):
        return self.vocab_size


def test_llada_adapter_exposes_project_model_contract() -> None:
    adapter = LLaDAAdapter(tokenizer=FakeTokenizer(), model=FakeModel())

    assert isinstance(adapter, MaskedDiffusionModel)
    assert adapter.device == "fake-device"
    assert adapter.mask_token_id == 7
    assert adapter.eos_token_id == 8
    assert adapter.pad_token_id == 9
    assert adapter.vocab_size == 16
    assert adapter.max_sequence_length == 32
    assert adapter.tokenize("ab", add_special_tokens=True) == [1, 2, 8]
    assert adapter.decode([1, 2, 3]) == "1,2,3"
    assert adapter.decode_token(5) == "tok-5"


def test_llada_adapter_can_build_decoding_config_from_constants() -> None:
    adapter = LLaDAAdapter(tokenizer=FakeTokenizer(), model=FakeModel())

    config = adapter.decoding_config(steps=4, block_length=2, banned_token_ids=[10])

    assert config.mask_token_id == 7
    assert config.eos_token_id == 8
    assert config.pad_token_id == 9
    assert config.vocab_size == 16
    assert config.steps == 4
    assert config.block_length == 2
    assert config.banned_token_ids == (10,)


def test_tokenizer_only_adapter_uses_llada_defaults_for_missing_model_config() -> None:
    adapter = LLaDAAdapter(tokenizer=FakeTokenizer(), model=None)

    assert adapter.mask_token_id == LLADA_1_5_DEFAULT_MASK_TOKEN_ID
    assert adapter.eos_token_id == 8
    assert adapter.pad_token_id == 9
    assert adapter.vocab_size == 16


def test_tokenizer_only_adapter_prefers_loaded_model_config_when_available() -> None:
    adapter = LLaDAAdapter(
        tokenizer=FakeTokenizer(),
        model=None,
        model_config=FakeConfig(
            mask_token_id=11,
            eos_token_id=12,
            pad_token_id=13,
            vocab_size=64,
            max_sequence_length=128,
        ),
    )

    assert adapter.mask_token_id == 11
    assert adapter.eos_token_id == 12
    assert adapter.pad_token_id == 13
    assert adapter.vocab_size == 64
    assert adapter.max_sequence_length == 128


def test_llada_defaults_match_documented_model_config() -> None:
    assert LLADA_1_5_DEFAULT_MASK_TOKEN_ID == 126336
    assert LLADA_1_5_DEFAULT_EOS_TOKEN_ID == 126081
    assert LLADA_1_5_DEFAULT_PAD_TOKEN_ID == 126081
    assert LLADA_1_5_DEFAULT_VOCAB_SIZE == 126464


def test_decode_token_rejects_out_of_range_token_id() -> None:
    adapter = LLaDAAdapter(tokenizer=FakeTokenizer(), model=FakeModel())

    with pytest.raises(ValueError, match="token_id must be in range"):
        adapter.decode_token(16)
