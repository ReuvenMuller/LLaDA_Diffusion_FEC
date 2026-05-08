import json

import pytest

from diffusion_fec.data.tokenized_samples import (
    load_tokenized_sample_artifact,
    sha256_file,
    write_tokenized_sample_artifact,
)
from diffusion_fec.experiments.llada_tokenized_artifact import (
    build_llada_tokenized_dataset_artifact,
)


class FakeTokenizerAdapter:
    model_id = "fake-llada"
    vocab_size = 64

    def tokenize(self, text, add_special_tokens=False):
        return fake_tokenize(text)


def fake_tokenize(text: str):
    return [3 + (ord(character) % 29) for character in text]


def test_write_and_load_tokenized_sample_artifact(tmp_path) -> None:
    dataset_path = tmp_path / "genfec_messages.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "wiki_0", "original_message": "alpha", "word_count": 1},
                {"id": "wiki_1", "original_message": "beta", "word_count": 1},
            ]
        ),
        encoding="utf-8",
    )
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(json.dumps({"dataset": "test"}), encoding="utf-8")
    output_path = tmp_path / "llada_tokenized.json"

    result = write_tokenized_sample_artifact(
        dataset_path=dataset_path,
        output_path=output_path,
        tokenize=fake_tokenize,
        tokenizer_name="fake-llada",
        model_id="fake-llada",
        vocab_size=64,
        sample_count=1,
        seed=0,
        max_tokens=4,
        dataset_label="test_wikitext_copy",
        source_dataset_manifest_path=source_manifest,
    )
    samples, info = load_tokenized_sample_artifact(
        output_path,
        expected_vocab_size=64,
        expected_model_id="fake-llada",
        expected_tokenizer_name="fake-llada",
    )

    assert result["artifact_sha256"] == sha256_file(output_path)
    assert info["tokenized_artifact_sha256"] == sha256_file(output_path)
    assert info["dataset_label"] == "test_wikitext_copy"
    assert info["source_dataset"]["sha256"] == sha256_file(dataset_path)
    assert info["source_dataset"]["manifest"]["sha256"] == sha256_file(source_manifest)
    assert info["text_source"] == "pretokenized_token_samples"
    assert len(samples) == 1
    assert samples[0].sample_id == "wiki_0"
    assert len(samples[0].token_ids) == 4
    assert samples[0].tokenizer_name == "fake-llada"


def test_build_llada_tokenized_dataset_artifact_uses_adapter_without_model_weights(tmp_path) -> None:
    dataset_path = tmp_path / "genfec_messages.json"
    dataset_path.write_text(
        json.dumps([{"id": "wiki_0", "original_message": "alpha"}]),
        encoding="utf-8",
    )
    output_path = tmp_path / "llada_tokenized.json"

    build_llada_tokenized_dataset_artifact(
        dataset_path=dataset_path,
        output_path=output_path,
        model_id="fake-llada",
        tokenizer_adapter=FakeTokenizerAdapter(),
        sample_count=1,
        max_tokens=3,
    )
    samples, info = load_tokenized_sample_artifact(output_path)

    assert samples[0].sample_id == "wiki_0"
    assert len(samples[0].token_ids) == 3
    assert info["model_id"] == "fake-llada"
    assert info["tokenizer_name"] == "fake-llada"


def test_load_tokenized_sample_artifact_rejects_out_of_vocab_ids(tmp_path) -> None:
    artifact_path = tmp_path / "bad_tokenized.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_kind": "pretokenized_token_samples",
                "format_version": 1,
                "tokenization": {
                    "tokenizer_name": "fake-llada",
                    "model_id": "fake-llada",
                    "vocab_size": 8,
                    "add_special_tokens": False,
                },
                "source_dataset": {"label": "bad", "path": "bad.json", "sha256": "abc"},
                "selection": {"selection_seed": 0, "min_tokens": 1, "max_tokens": None},
                "samples": [
                    {
                        "sample_id": "wiki_0",
                        "text": "bad",
                        "token_ids": [3, 9],
                        "token_count": 2,
                        "tokenizer_name": "fake-llada",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside vocab_size"):
        load_tokenized_sample_artifact(artifact_path)
