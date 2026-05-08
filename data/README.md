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

## LLaDA-Tokenized Artifacts

Fair model-vs-classical comparisons must use the same erased units. Build a
pre-tokenized artifact with the LLaDA tokenizer, then pass that artifact to fake,
classical, and real LLaDA runners:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir data `
  --build-llada-tokenized-artifact `
  --dataset-file data\wikitext2_genfec_test_messages.json `
  --dataset-label wikitext2_genfec_test_messages `
  --source-dataset-manifest data\wikitext2_genfec_manifest.json `
  --dataset-sample-count 10 `
  --dataset-seed 0 `
  --dataset-max-tokens 128 `
  --llada-local-files-only `
  --tokenized-output-file data\llada_tokenized_wikitext2_genfec_seed0_max128.json
```

The builder loads only the tokenizer/config, not model weights. The artifact
records the source dataset hash, optional source manifest hash, tokenizer/model
ID, selection seed, token cap, sample IDs, text, and LLaDA token IDs. Run
manifests record the artifact path and artifact SHA256 when
`--tokenized-samples-file` is used.
