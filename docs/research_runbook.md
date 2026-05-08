# Research Runbook

This runbook is the operational handoff for the current framework. It separates
local correctness, small synthetic sweeps, opt-in real LLaDA micro-evals, and
final research runs.

## Status

The framework code path is complete enough to run the intended comparison:

- LLaDA model only
- LLaDA plus transmitted lookback-1 token hashes
- classical matched-overhead baselines:
  - XOR parity
  - LT/fountain-style repair
  - streaming-window repair

The current smoke and micro-eval outputs are still engineering validation. They
must not be described as research results until configs, profiles, datasets, and
sample counts are frozen and run on the server.

## Local Acceptance

Run the default test suite locally:

```powershell
python -m pytest
```

The default suite is model-free and should not load Hugging Face or LLaDA.

## Local Synthetic Sweep

Run a compact model-free sweep across the main comparison set:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\synthetic_sweep `
  --synthetic-sweep `
  --sample-lengths 8,16,32 `
  --tokens-per-packet 4 `
  --loss-rate 0.5 `
  --hash-bits 4 `
  --seed 0
```

Include burst loss and both interleaving families:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\synthetic_sweep_interleaving `
  --synthetic-sweep `
  --sample-lengths 8,16,32 `
  --tokens-per-packet 4 `
  --loss-rate 0.5 `
  --hash-bits 4 `
  --seed 0 `
  --sweep-include-burst `
  --sweep-include-interleaving-variants `
  --burst-length 2
```

The sweep writes:

```text
runs/<sweep>/
  sweep_manifest.json
  sweep_runs.csv
  runs/<child-run>/run_manifest.json
  runs/<child-run>/results.csv
  runs/<child-run>/events.jsonl
  hash_profiles/<fake-profile>/
  analysis/analysis_manifest.json
  analysis/aggregate.csv
  analysis/summary.md
  analysis/*.svg
  analysis/failure_examples.jsonl
```

The sweep is skip-aware by default. A child run is reused only when its existing
`run_manifest.json` matches the current runner config and dataset selection,
including sample IDs, selection seed, and token cap. If those differ, the child
run is treated as stale and rerun with status `completed_replaced_stale`. Add
`--sweep-overwrite` to force all completed child runs to rerun.

## Dataset-Backed Validation

The frozen GenFEC WikiText-derived text artifact is copied into this project:

```text
data/wikitext2_genfec_test_messages.json
data/wikitext2_genfec_manifest.json
```

It contains 100 WikiText-2 raw test samples with IDs such as `wiki_0`. The text
field is `original_message`. This artifact is shared with GenFEC for
comparability, but tokenization is model-specific. Do not reuse Qwen token
counts or Qwen hash profiles.

For fair comparisons, first freeze a LLaDA-tokenized artifact. This loads only
the tokenizer/config, not model weights:

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

Then run local model-free validation on the frozen LLaDA token IDs:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\dataset_validation_fake `
  --synthetic-sweep `
  --tokenized-samples-file data\llada_tokenized_wikitext2_genfec_seed0_max128.json `
  --tokens-per-packet 4 `
  --loss-rate 0.2 `
  --hash-bits 4 `
  --seed 0
```

The older `--dataset-file` fake path is still useful for quick plumbing checks,
but it uses a local deterministic tokenizer and must not be used for fair
model-vs-classical comparison. The frozen tokenized artifact keeps text samples,
LLaDA token IDs, packetization, loss events, scoring, and overhead accounting on
the same problem.

Fake/model-free LLaDA-shaped validation uses a lightweight decoder proposal path
when available, so it does not materialize full `sequence_length x vocab_size`
logits for large LLaDA vocab artifacts. The decoder still builds the legal
candidate set first, so fixed tokens remain fixed, hash-guided positions are
restricted to their hash buckets, and unguided positions still exclude banned
special tokens. This remains engineering validation only, not a research model
baseline.

## Analysis Artifacts

Build analysis artifacts for any directory containing run outputs:

```powershell
python -m diffusion_fec.analysis.report `
  --run-root runs\synthetic_sweep\runs `
  --output-dir runs\synthetic_sweep\analysis
```

The analysis writer discovers `results.csv` and `events.jsonl`, then writes:

- `aggregate.csv`
- `summary.md`
- `exact_match_rate.svg`
- `lost_position_recovery_rate.svg`
- `decode_latency_sec.svg`
- `repair_overhead_ratio.svg`
- `total_overhead_ratio.svg`
- `failure_examples.jsonl`
- `analysis_manifest.json`

Use `total_overhead_ratio` for strategy comparison. It combines classical
repair-token overhead with transmitted hash metadata token-equivalent overhead.
The model+hash path reports `hash_metadata_bit_count`,
`hash_metadata_token_equivalent_overhead_ratio`, and `total_overhead_ratio` so
lookback metadata is not treated as free.

## Frozen Real-Run Conventions

Real LLaDA runs must use loaded tokenizer-specific hash profiles. Do not rebuild
hash maps during real model execution.

Recommended profile convention:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/hash_profiles/
  llada_1_5_real_v1/
    hash_profile_metadata.json
    uniform_hash4_map.npy
    uniform_hash8_map.npy
    uniform_hash16_map.npy
```

Recommended run root:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/
```

Every real run should record:

- model ID and cache/local-files setting
- hash profile directory and map mode
- hash bits
- source layout and packet wire interleaving
- channel mode and channel parameters
- sample lengths or dataset sample IDs
- decoder steps
- latency and model forward calls
- hash metadata bit budget and total token-equivalent overhead

## Server Real LLaDA Micro-Evals

Use `tmux` on the GPU server. Example model-only command:

```bash
python -m diffusion_fec.experiments.runner \
  --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_model_only_micro_eval \
  --real-llada-micro-eval \
  --micro-eval-mode model_only \
  --llada-local-files-only \
  --sample-lengths 8 \
  --tokens-per-packet 1 \
  --hash-bits 4 \
  --steps 2 \
  --seed 0
```

Example model+hash command:

```bash
python -m diffusion_fec.experiments.runner \
  --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_hash4_micro_eval \
  --real-llada-micro-eval \
  --hash-profile-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/hash_profiles/llada_1_5_real_v1 \
  --llada-local-files-only \
  --sample-lengths 8 \
  --tokens-per-packet 1 \
  --hash-bits 4 \
  --steps 2 \
  --seed 0
```

Dataset-backed model-only command:

```bash
python -m diffusion_fec.experiments.runner \
  --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_dataset_model_only_smoke \
  --real-llada-micro-eval \
  --micro-eval-mode model_only \
  --tokenized-samples-file data/llada_tokenized_wikitext2_genfec_seed0_max128.json \
  --llada-local-files-only \
  --tokens-per-packet 4 \
  --hash-bits 4 \
  --steps 2 \
  --seed 0
```

Dataset-backed model+hash command:

```bash
python -m diffusion_fec.experiments.runner \
  --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_dataset_hash4_smoke \
  --real-llada-micro-eval \
  --hash-profile-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/hash_profiles/llada_1_5_real_v1 \
  --tokenized-samples-file data/llada_tokenized_wikitext2_genfec_seed0_max128.json \
  --llada-local-files-only \
  --tokens-per-packet 4 \
  --hash-bits 4 \
  --steps 2 \
  --seed 0
```

Run the analysis writer over the server run root after a group of runs finishes.

## Final Experiment Gate

Before final research claims:

- freeze the LLaDA-tokenized sample artifact and sample IDs,
- freeze LLaDA cache/model revision,
- freeze hash profiles for 4, 8, and 16 bits,
- freeze packet size, interleaving, and channel configs,
- run model-only, model+hash, and classical baselines under matched overhead,
- aggregate all runs through the same analysis artifact path,
- inspect failure examples before drawing conclusions.
