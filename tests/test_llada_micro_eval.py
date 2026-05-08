import csv
import json
from dataclasses import dataclass

import pytest

from diffusion_fec.coding.hash_profiles import build_and_save_hash_profile
from diffusion_fec.data.tokenized_samples import write_tokenized_sample_artifact
from diffusion_fec.experiments.llada_micro_eval import (
    RealLLaDAMicroEvalUnavailable,
    run_real_llada_micro_eval,
)
from diffusion_fec.experiments.micro_eval import MICRO_EVAL_MODEL_HASH, MICRO_EVAL_MODEL_ONLY
from diffusion_fec.experiments.runner import main


MODEL_ID = "fake-llada"


class FakeCuda:
    @staticmethod
    def is_available():
        return True


class FakeTorch:
    cuda = FakeCuda()
    bfloat16 = "bf16"


class FakeLogits(list):
    @property
    def shape(self):
        return (len(self), len(self[0]), len(self[0][0]))


@dataclass
class FakeForwardOutput:
    logits: FakeLogits


class FakeLLaDAAdapter:
    model_id = MODEL_ID
    mask_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    vocab_size = 32
    max_sequence_length = 64

    def tokenize(self, text, add_special_tokens=False):
        return [3 + (ord(character) % (self.vocab_size - 3)) for character in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)

    def decoding_config(self, *, steps, block_length):
        from diffusion_fec.decoding.llada_diffusion import DiffusionDecodingConfig

        return DiffusionDecodingConfig(
            mask_token_id=self.mask_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vocab_size=self.vocab_size,
            steps=steps,
            block_length=block_length,
        )

    def forward(self, input_ids, attention_mask=None):
        sequence_length = len(input_ids[0])
        logits = FakeLogits(
            [
                [
                    [0.0 for _ in range(self.vocab_size)]
                    for _ in range(sequence_length)
                ]
            ]
        )
        for position in range(sequence_length):
            logits[0][position][(position + 3) % self.vocab_size] = 100.0 - position
        return FakeForwardOutput(logits=logits)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_fake_profile(profile_dir) -> None:
    build_and_save_hash_profile(
        profile_dir=profile_dir,
        profile_name="fake-llada-profile",
        vocab_size=FakeLLaDAAdapter.vocab_size,
        hash_bits=4,
        decode_token=lambda token_id: f"tok-{token_id}",
        excluded_token_ids={
            FakeLLaDAAdapter.mask_token_id,
            FakeLLaDAAdapter.eos_token_id,
            FakeLLaDAAdapter.pad_token_id,
        },
        salt="fake-real-micro-eval",
        model_id=MODEL_ID,
        tokenizer_name=MODEL_ID,
    )


def test_real_llada_micro_eval_requires_loaded_profile_for_hash_mode(tmp_path) -> None:
    with pytest.raises(RealLLaDAMicroEvalUnavailable, match="hash_profile_dir is required"):
        run_real_llada_micro_eval(
            output_dir=tmp_path / "run",
            model_id=MODEL_ID,
            sample_lengths=(2,),
            mode=MICRO_EVAL_MODEL_HASH,
            torch_module=FakeTorch(),
            tokenizer_adapter=FakeLLaDAAdapter(),
            model_adapter=FakeLLaDAAdapter(),
        )


def test_real_llada_micro_eval_model_hash_writes_artifacts_from_loaded_profile(tmp_path) -> None:
    profile_dir = tmp_path / "profile"
    output_dir = tmp_path / "run"
    build_fake_profile(profile_dir)

    run_real_llada_micro_eval(
        output_dir=output_dir,
        model_id=MODEL_ID,
        sample_lengths=(2,),
        loss_rate=0.5,
        seed=1,
        tokens_per_packet=1,
        mode=MICRO_EVAL_MODEL_HASH,
        hash_bits=4,
        steps=2,
        hash_profile_dir=profile_dir,
        torch_module=FakeTorch(),
        tokenizer_adapter=FakeLLaDAAdapter(),
        model_adapter=FakeLLaDAAdapter(),
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert manifest["runner"] == "real_llada_synthetic_micro_eval"
    assert manifest["not_a_research_claim"] is True
    assert manifest["preflight"]["tiny_forward_shape"] == [1, 1, 32]
    assert manifest["hash_profile"]["source"] == "loaded_profile"
    assert rows[0]["strategy"] == "LLaDA_MicroEval_LoadedHashLookback1_NoPrompt"
    assert rows[0]["hash_profile_source"] == "loaded_profile"
    assert rows[0]["known_count"] == "1"
    assert rows[0]["hash_guided_count"] == "1"
    assert rows[0]["hash_metadata_count"] == "1"
    assert rows[0]["hash_metadata_bit_count"] == "4"
    assert rows[0]["token_bit_width"] == "5"
    assert float(rows[0]["total_overhead_ratio"]) == 0.4
    assert rows[0]["model_forward_calls"] == "1"
    assert rows[0]["decoder_steps"] == "1"
    assert rows[0]["decode_latency_sec"]
    assert events[0]["event_type"] == "real_llada_micro_eval_case"
    assert events[0]["case"]["oracle_hash_metadata"] is False


def test_real_llada_micro_eval_model_only_does_not_require_profile(tmp_path) -> None:
    output_dir = tmp_path / "run"

    run_real_llada_micro_eval(
        output_dir=output_dir,
        model_id=MODEL_ID,
        sample_lengths=(2,),
        loss_rate=1.0,
        seed=1,
        tokens_per_packet=1,
        mode=MICRO_EVAL_MODEL_ONLY,
        hash_bits=4,
        steps=2,
        torch_module=FakeTorch(),
        tokenizer_adapter=FakeLLaDAAdapter(),
        model_adapter=FakeLLaDAAdapter(),
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")

    assert manifest["config"]["mode"] == MICRO_EVAL_MODEL_ONLY
    assert manifest["hash_profile"]["source"] == "not_used"
    assert rows[0]["protection_mode"] == "none"
    assert rows[0]["hash_guided_count"] == "0"
    assert rows[0]["unguided_count"] == "2"
    assert rows[0]["hash_metadata_count"] == "0"
    assert float(rows[0]["total_overhead_ratio"]) == 0.0


def test_real_llada_micro_eval_can_tokenize_dataset_samples(tmp_path) -> None:
    dataset_path = tmp_path / "genfec_messages.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "wiki_0",
                    "original_message": "abc def",
                    "word_count": 2,
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    run_real_llada_micro_eval(
        output_dir=output_dir,
        model_id=MODEL_ID,
        mode=MICRO_EVAL_MODEL_ONLY,
        loss_rate=0.0,
        seed=1,
        tokens_per_packet=2,
        hash_bits=4,
        steps=2,
        dataset_path=dataset_path,
        dataset_label="test_wikitext_copy",
        dataset_sample_count=1,
        dataset_max_tokens=4,
        torch_module=FakeTorch(),
        tokenizer_adapter=FakeLLaDAAdapter(),
        model_adapter=FakeLLaDAAdapter(),
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")

    assert manifest["config"]["sample_generation"]["type"] == "loaded_text_dataset_tokenized_with_llada"
    assert manifest["config"]["sample_generation"]["dataset"]["sample_ids"] == ["wiki_0"]
    assert manifest["config"]["sample_lengths"] == [4]
    assert rows[0]["sample_id"] == "wiki_0"
    assert rows[0]["source_token_count"] == "4"


def test_real_llada_micro_eval_uses_pretokenized_samples(tmp_path) -> None:
    dataset_path = tmp_path / "genfec_messages.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "wiki_0",
                    "original_message": "abcd",
                    "word_count": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    tokenized_path = tmp_path / "llada_tokenized.json"
    fake_adapter = FakeLLaDAAdapter()
    write_tokenized_sample_artifact(
        dataset_path=dataset_path,
        output_path=tokenized_path,
        tokenize=lambda text: fake_adapter.tokenize(text, add_special_tokens=False),
        tokenizer_name=MODEL_ID,
        model_id=MODEL_ID,
        vocab_size=fake_adapter.vocab_size,
        sample_count=1,
        seed=0,
        max_tokens=4,
        dataset_label="test_llada_tokenized",
    )
    output_dir = tmp_path / "run"

    run_real_llada_micro_eval(
        output_dir=output_dir,
        model_id=MODEL_ID,
        mode=MICRO_EVAL_MODEL_ONLY,
        loss_rate=0.0,
        seed=1,
        tokens_per_packet=2,
        hash_bits=4,
        steps=2,
        tokenized_samples_path=tokenized_path,
        torch_module=FakeTorch(),
        tokenizer_adapter=fake_adapter,
        model_adapter=fake_adapter,
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")

    sample_generation = manifest["config"]["sample_generation"]
    assert sample_generation["type"] == "loaded_pretokenized_llada_samples"
    assert sample_generation["dataset"]["text_source"] == "pretokenized_token_samples"
    assert sample_generation["dataset"]["tokenizer_verification"] == "matched_current_llada_tokenizer"
    assert sample_generation["dataset"]["tokenized_artifact_sha256"]
    assert manifest["config"]["sample_lengths"] == [4]
    assert rows[0]["sample_id"] == "wiki_0"
    assert rows[0]["source_token_count"] == "4"


def test_real_llada_micro_eval_rejects_stale_pretokenized_samples(tmp_path) -> None:
    dataset_path = tmp_path / "genfec_messages.json"
    dataset_path.write_text(
        json.dumps([{"id": "wiki_0", "original_message": "abcd"}]),
        encoding="utf-8",
    )
    tokenized_path = tmp_path / "llada_tokenized.json"
    fake_adapter = FakeLLaDAAdapter()
    write_tokenized_sample_artifact(
        dataset_path=dataset_path,
        output_path=tokenized_path,
        tokenize=lambda text: [3, 4, 5, 6],
        tokenizer_name=MODEL_ID,
        model_id=MODEL_ID,
        vocab_size=fake_adapter.vocab_size,
        sample_count=1,
        seed=0,
        max_tokens=4,
    )

    with pytest.raises(RealLLaDAMicroEvalUnavailable, match="does not match"):
        run_real_llada_micro_eval(
            output_dir=tmp_path / "run",
            model_id=MODEL_ID,
            mode=MICRO_EVAL_MODEL_ONLY,
            loss_rate=0.0,
            seed=1,
            tokens_per_packet=2,
            hash_bits=4,
            steps=2,
            tokenized_samples_path=tokenized_path,
            torch_module=FakeTorch(),
            tokenizer_adapter=fake_adapter,
            model_adapter=fake_adapter,
        )


def test_runner_routes_real_llada_micro_eval_without_loading_model(monkeypatch, tmp_path) -> None:
    import diffusion_fec.experiments.llada_micro_eval as llada_micro_eval

    captured = {}

    def fake_run_real_llada_micro_eval(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        llada_micro_eval,
        "run_real_llada_micro_eval",
        fake_run_real_llada_micro_eval,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--real-llada-micro-eval",
            "--hash-profile-dir",
            str(tmp_path / "profile"),
            "--llada-local-files-only",
            "--tokenized-samples-file",
            str(tmp_path / "llada_tokenized.json"),
        ]
    )

    assert exit_code == 0
    assert captured["sample_lengths"] == (8,)
    assert captured["local_files_only"] is True
    assert captured["mode"] == MICRO_EVAL_MODEL_HASH
    assert captured["tokenized_samples_path"] == str(tmp_path / "llada_tokenized.json")
