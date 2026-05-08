# Frozen Dataset Artifacts

This directory contains small text datasets used for dataset-backed validation.

## `wikitext2_genfec_test_messages.json`

Frozen copy of the GenFEC `test_messages.json` artifact.

- Source project path: `C:\Users\reuve\OneDrive\Documents\GenFEC_Depth_Exam\test_messages.json`
- Source generator: GenFEC `generate_dataset.py`
- Upstream dataset: Hugging Face `wikitext`, `wikitext-2-raw-v1`, `test` split
- Record count: 100
- Text field: `original_message`
- ID field: `id`
- Intended word count per sample: 300
- SHA256: `8CB164B452F7A2BAAEAD9C95274C33CEB116424DB8635C0366CB421841D17A51`

The text artifact is shared for comparability with GenFEC, but tokenization is
not shared. LLaDA runs must tokenize this text with the LLaDA tokenizer and use
LLaDA-specific hash profiles. Do not reuse Qwen token counts or Qwen hash
profiles.
