import json

from diffusion_fec.data.text_samples import load_text_records, tokenize_text_records


def tokenize(text: str):
    return [ord(char) % 32 for char in text]


def test_load_text_records_from_jsonl(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "alpha", "split": "train"}),
                json.dumps({"sample_id": "b", "text": "beta"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_text_records(path)

    assert [record.record_id for record in records] == ["a", "b"]
    assert records[0].metadata == {"split": "train"}


def test_load_text_records_from_json_container(tmp_path) -> None:
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps({"samples": [{"record_id": "r1", "text": "one"}]}),
        encoding="utf-8",
    )

    records = load_text_records(path)

    assert records[0].record_id == "r1"
    assert records[0].text == "one"


def test_load_text_records_from_plain_text(tmp_path) -> None:
    path = tmp_path / "samples.txt"
    path.write_text("first\n\nsecond\n", encoding="utf-8")

    records = load_text_records(path)

    assert [record.text for record in records] == ["first", "second"]
    assert [record.record_id for record in records] == ["line-000000", "line-000002"]


def test_tokenize_text_records_is_seeded_and_filters_lengths(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"id-{index}", "text": text})
            for index, text in enumerate(["a", "bb", "ccc", "dddd"])
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_text_records(path)

    first = tokenize_text_records(
        records,
        tokenize=tokenize,
        tokenizer_name="fake",
        sample_count=2,
        seed=7,
        min_tokens=2,
        max_tokens=3,
    )
    second = tokenize_text_records(
        records,
        tokenize=tokenize,
        tokenizer_name="fake",
        sample_count=2,
        seed=7,
        min_tokens=2,
        max_tokens=3,
    )

    assert [sample.to_dict() for sample in first] == [
        sample.to_dict()
        for sample in second
    ]
    assert all(2 <= len(sample.token_ids) <= 3 for sample in first)
