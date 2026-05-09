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

### Completed 10-Sample Validation Pass

The first dataset-backed validation pass completed on the GPU server on
2026-05-09. It used the frozen 10-sample LLaDA-tokenized WikiText artifact:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/tokenized_artifacts/llada_tokenized_wikitext2_genfec_seed0_max128.json
```

The run root was:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-20260508_232013
```

This validation used `tokens_per_packet=4`, IID packet loss at `loss_rate=0.5`,
`steps=2`, `seed=0`, and the loaded real LLaDA hash profile directory:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/hash_profiles/llada_1_5_real_smoke_v1
```

The pass produced fake/model-free validation sweeps, real LLaDA model-only,
real LLaDA hash4/hash8/hash16 runs, and analysis artifacts. Mean real LLaDA
validation results were:

| strategy | lost recovery | token edit distance | total overhead | decode latency sec | run wall sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| model only | 0.2990 | 47.3 | 0.0000 | 1.4137 | 98.6374 |
| hash4 | 0.4472 | 37.6 | 0.2279 | 1.1238 | 94.3607 |
| hash8 | 0.5451 | 31.2 | 0.4559 | 0.9810 | 91.5803 |
| hash16 | 0.6348 | 25.8 | 0.9118 | 1.0216 | 98.2813 |

All real rows preserved known tokens, left zero mask tokens, and used two model
forward calls. This is a healthy dataset-backed validation result, but it is
still not a final research claim.

### Completed Steps=8 Iteration Validation

A follow-up iteration-count validation completed on the GPU server on
2026-05-09. It reused the same frozen 10-sample artifact, hash profile
directory, IID loss rate, packet size, and seed as the steps=2 pass above, but
changed decoder steps from `2` to `8`.

Run root:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-steps8-20260509_031600
```

Mean real LLaDA validation results were:

| strategy | lost recovery | token edit distance | total overhead | decode latency sec | run wall sec | forward calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model only | 0.2736 | 48.7 | 0.0000 | 4.0762 | 124.4773 | 8 |
| hash4 | 0.4347 | 38.3 | 0.2279 | 3.3574 | 116.6101 | 8 |
| hash8 | 0.5871 | 28.3 | 0.4559 | 3.0708 | 112.1183 | 8 |
| hash16 | 0.6451 | 24.6 | 0.9118 | 3.2705 | 116.8302 | 8 |

Compared with steps=2, hash8 and hash16 improved mean recovery/edit distance,
while model-only and hash4 slipped slightly. All rows still preserved known
tokens and left zero mask tokens. This is useful iteration-count validation, not
a final research result.

### Decoder Refinement Ablation

The validated decoder default is `commit_once + always`: erased positions are
filled over multiple steps, and once a position is committed it is no longer
editable. This remains the default and is the baseline for previous validation
runs.

An experimental true refinement mode is available with:

```bash
--editable-update-mode resample_each_step
```

In this mode, all erased positions remain editable across all denoising steps
and are rewritten every step. Known/received tokens and prompt tokens remain
fixed every step. Hash-guided positions can use one of three schedules:

```bash
--hash-constraint-schedule always
--hash-constraint-schedule final_only
--hash-constraint-schedule late_half
```

`always` applies token-hash buckets every step. `final_only` relaxes hash
constraints until the final step. `late_half` relaxes the first half and
enforces hash buckets in the second half. Relaxed steps still exclude mask,
padding, EOS, and explicitly banned tokens. Commit-once decoding intentionally
supports only `hash_constraint_schedule=always` to avoid misleading runs where
positions commit before a final hash constraint can apply.

Result rows and manifests include `editable_update_mode` and
`hash_constraint_schedule`, and analysis groups by those fields so ablation rows
do not collapse together.

The first real LLaDA refinement ablation completed on 2026-05-09:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-refinement-20260509_110123
```

It reused the frozen 10-sample LLaDA-tokenized WikiText artifact, hash profiles,
IID `loss_rate=0.5`, `tokens_per_packet=4`, `seed=0`, and `steps=8`.

| strategy | update mode | schedule | lost recovery | edit distance | overhead | latency sec |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| model only | commit_once | always | 0.2736 | 48.7 | 0.0000 | 3.9034 |
| model only | resample_each_step | always | 0.2302 | 52.4 | 0.0000 | 6.9380 |
| hash4 | commit_once | always | 0.4347 | 38.3 | 0.2279 | 3.2783 |
| hash4 | resample_each_step | always | 0.3997 | 41.4 | 0.2279 | 4.9633 |
| hash4 | resample_each_step | final_only | 0.3323 | 45.9 | 0.2279 | 6.5466 |
| hash4 | resample_each_step | late_half | 0.3305 | 46.3 | 0.2279 | 5.8959 |
| hash8 | commit_once | always | 0.5871 | 28.3 | 0.4559 | 2.9861 |
| hash8 | resample_each_step | always | 0.4851 | 35.8 | 0.4559 | 4.0878 |
| hash8 | resample_each_step | final_only | 0.4137 | 40.3 | 0.4559 | 6.4568 |
| hash8 | resample_each_step | late_half | 0.4283 | 39.4 | 0.4559 | 5.3928 |
| hash16 | commit_once | always | 0.6451 | 24.6 | 0.9118 | 3.1109 |
| hash16 | resample_each_step | always | 0.5587 | 30.9 | 0.9118 | 4.0781 |
| hash16 | resample_each_step | final_only | 0.5338 | 32.4 | 0.9118 | 6.6556 |
| hash16 | resample_each_step | late_half | 0.5355 | 32.3 | 0.9118 | 5.5382 |

All rows preserved known tokens, left zero mask tokens, and used eight model
forward calls. Under this configuration, commit-once remained better than
resampling for all hash levels. Qualitatively, resampling sometimes drifted into
repetitive or locally awkward text, while commit-once tended to preserve a more
stable reconstruction. This remains decoder-design validation, not a final
research result.

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

Run manifests also include `run_started_at`, `run_finished_at`, and
`run_wall_time_sec`. Result rows repeat `run_wall_time_sec` so aggregation can
report whole-run timing alongside per-case `decode_latency_sec`.

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
- per-case decode latency, whole-run wall time, and model forward calls
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
