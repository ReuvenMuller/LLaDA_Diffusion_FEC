# Development Notes

## Environment Assumptions

The first real LLaDA run will likely require a CUDA-capable GPU.

Expected Python dependencies:

```text
torch
transformers
accelerate
numpy
pandas
scipy
datasets
sentence-transformers
pytest
```

Semantic similarity should be optional so smoke tests do not require sentence-transformer setup.

## Reproducibility Defaults

Use explicit seeds for:

- dataset sampling,
- channel loss,
- stochastic decoding if enabled,
- parity/fountain baselines if added.

For deterministic first runs:

```text
temperature = 0.0
remasking = low_confidence
```

## Output Directory Shape

Each run directory should look like:

```text
runs/<run_name>/
  run_manifest.json
  config_snapshot.json
  results.csv
  events.jsonl
  console.log
```

If a run is resumed, append rows but skip completed run IDs.

## Run ID Shape

Recommended run ID:

```text
<model>|<strategy>|<channel>|loss<rate>|sample<id>
```

Example:

```text
LLaDA-1.5|LLaDA_Hash8_NoPrompt|burst|loss0.2|sample0007
```

## Code Style

Prefer clear modules over clever abstractions.

Guidelines:

- keep tensor-heavy code in decoder modules,
- keep experiment orchestration out of decoder modules,
- keep model loading out of tests unless marked slow,
- keep file outputs centralized in experiment logging utilities,
- make configs serializable.

## First Test Suite

Before loading LLaDA, tests should cover:

- token hash bucket construction,
- packetization,
- interleaving,
- channel outputs,
- reconstruction plan building,
- constraint mask construction,
- fake-model constrained denoising.

## First Real-Model Smoke Test

Target:

```text
model = GSAI-ML/LLaDA-1.5
samples = 3
strategy = LLaDA_Hash8_NoPrompt
channel = random
loss_rate = 0.2
steps = 32 first, then 128
```

Start with `steps=32` to verify mechanics quickly. Use `steps=128` only after correctness checks pass.

Current opt-in CLI smoke:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\real_llada_smoke `
  --real-llada-smoke `
  --sample-count 1 `
  --loss-rate 0.5 `
  --seed 1 `
  --tokens-per-packet 1 `
  --hash-bits 4 `
  --steps 2
```

Use `--llada-local-files-only` to require cached Hugging Face files. The command
loads tokenizer/config first, then refuses to load model weights unless CUDA is
available. Passing `--allow-cpu-real-llada` explicitly overrides that guard, but
CPU loading is expected to be impractical for the 8B model.

Current local synthetic micro-eval:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\fake_micro_eval `
  --micro-eval `
  --sample-lengths 8,16,32 `
  --tokens-per-packet 4 `
  --seed 0
```

Optional interleaving knobs:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\fake_micro_eval_interleaved `
  --micro-eval `
  --sample-lengths 8,16,32 `
  --tokens-per-packet 4 `
  --source-layout round_robin_chunks `
  --source-chunk-size 1 `
  --wire-interleaving matrix `
  --wire-interleaving-span 4
```

These micro-evals use the fake deterministic model and are engineering
validation only.

Optional packet-loss channels:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\fake_micro_eval_burst `
  --micro-eval `
  --sample-lengths 8 `
  --tokens-per-packet 1 `
  --channel burst `
  --burst-start-wire-id 0 `
  --burst-length 2
```

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\fake_micro_eval_ge `
  --micro-eval `
  --sample-lengths 8 `
  --tokens-per-packet 1 `
  --channel gilbert_elliott `
  --ge-good-loss-rate 0.0 `
  --ge-bad-loss-rate 1.0 `
  --ge-good-to-bad-rate 0.1 `
  --ge-bad-to-good-rate 0.5
```

The synthetic sweep driver also supports `--channel gilbert_elliott` and passes
the same `--ge-*` parameters into each child run.

Current opt-in real LLaDA synthetic micro-eval:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\real_llada_micro_eval `
  --real-llada-micro-eval `
  --hash-profile-dir runs\hash_profiles\llada_1_5_smoke_v1 `
  --llada-local-files-only `
  --sample-lengths 8 `
  --tokens-per-packet 1 `
  --hash-bits 4 `
  --steps 2
```

Real micro-eval hash mode requires an existing profile and will not build one
while loading/running LLaDA. Use `--micro-eval-mode model_only` for a model-only
comparison path with no hash profile.

## Classical Baseline Notes

The first classical baseline slice is implemented in `src/diffusion_fec/baselines`.
It includes:

- token bit-width and hash-overhead estimation,
- repair-token overhead accounting,
- XOR parity encoding over source packet stripes,
- XOR repair when exactly one source packet in a stripe is missing,
- support for source/token layout and packet-level wire interleaving.

The baseline code is intentionally separate from LLaDA decoder modules. XOR
parity now has an artifact-writing micro-eval runner that uses the same
`run_manifest.json`, `results.csv`, and `events.jsonl` convention as the fake and
real LLaDA runners:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\xor_parity_micro_eval `
  --xor-parity-micro-eval `
  --sample-lengths 8,16,32 `
  --tokens-per-packet 4 `
  --hash-bits 4 `
  --xor-stripe-size 4
```

Use the same `--source-layout`, `--wire-interleaving`, and `--channel` flags as
the fake LLaDA micro-eval runner.

LT/fountain baseline micro-eval:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\lt_fountain_micro_eval `
  --lt-fountain-micro-eval `
  --sample-lengths 8,16,32 `
  --tokens-per-packet 4 `
  --hash-bits 4 `
  --lt-repair-rate 0.25 `
  --lt-random-seed 7
```

Add `--lt-coverage-aware` to force the LT repair scheduler to cover every source
packet when its repair budget allows.

Streaming-window baseline micro-eval:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\streaming_window_micro_eval `
  --streaming-window-micro-eval `
  --sample-lengths 8,16,32 `
  --tokens-per-packet 4 `
  --hash-bits 4 `
  --stream-window-size 5 `
  --stream-window-stride 1
```

Current opt-in pytest smoke:

```powershell
$env:RUN_LLADA_E2E_SMOKE = "1"
python -m pytest tests\test_llada_e2e_smoke.py -q
```

## Dataset And Analysis Utilities

Text records can be loaded from JSONL, JSON, or plain text with
`diffusion_fec.data.load_text_records(...)`. Convert them to `TokenSample`
objects with `tokenize_text_records(...)` by passing a tokenizer callback.
The loader accepts this project's generic `text` schema and the GenFEC
`original_message` schema.

The frozen GenFEC WikiText-derived dataset copy lives at:

```text
data/wikitext2_genfec_test_messages.json
```

For fair model-vs-classical comparisons, first build a LLaDA-tokenized artifact.
This loads tokenizer/config only:

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

Run a local validation sweep on those exact LLaDA token IDs:

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

For real LLaDA dataset-backed validation, prefer the same
`--tokenized-samples-file` path. The real runner verifies the stored token IDs
against the current LLaDA tokenizer before loading model weights. The older
`--dataset-file` fake path remains available for quick local plumbing checks,
but it uses a deterministic local tokenizer and is not a fair comparison basis.

The fake/model-free decoder path is optimized for large LLaDA-tokenized
artifacts. Fake models can implement `propose_token(...)`, and the decoder will
ask that hook to choose from the already-constrained candidate set instead of
materializing full-vocabulary logits. Real LLaDA still uses the normal
`forward(...).logits` path.

For full-vocabulary torch logits, real LLaDA token selection is vectorized over
the legal candidate IDs instead of scanning candidates one token at a time in
Python. This preserves the real adapter behavior while avoiding a CPU-side
bottleneck observed during the first 10-sample dataset-backed validation run.
The older Python candidate scan remains as the fallback for simple fake logits
and non-tensor test doubles.

Result CSVs can be aggregated with:

```python
from diffusion_fec.analysis import aggregate_result_rows, load_result_rows, write_aggregate_csv

rows = load_result_rows([
    "runs/fake_micro_eval/results.csv",
    "runs/xor_parity_micro_eval/results.csv",
])
aggregate = aggregate_result_rows(rows)
write_aggregate_csv(output_path="runs/aggregate.csv", rows=aggregate)
```

The aggregate helper reports case counts, exact-match rate, known-position
preservation rate, mean recovery/edit metrics, mean per-case decode latency,
mean whole-run wall time, mean model forward calls, and mean repair overhead
where those fields are present. `decode_latency_sec` is per case; top-level
manifest fields `run_started_at`, `run_finished_at`, and `run_wall_time_sec`
describe the whole artifact-writing run.

## Synthetic Sweeps And Reports

The synthetic sweep runner executes the main comparison set with model-free
local paths:

- fake LLaDA-shaped model only,
- fake LLaDA-shaped model plus transmitted lookback-1 hashes,
- XOR parity matched overhead,
- LT/fountain matched overhead,
- streaming-window matched overhead.

Compact local sweep:

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

Interleaving and burst-loss coverage:

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

The sweep writes `sweep_manifest.json`, `sweep_runs.csv`, child run artifacts,
a fake hash profile directory when needed, and an `analysis/` bundle.

Build or rebuild an analysis bundle from any run root:

```powershell
python -m diffusion_fec.analysis.report `
  --run-root runs\synthetic_sweep\runs `
  --output-dir runs\synthetic_sweep\analysis
```

The report bundle includes aggregate CSVs, a markdown summary, SVG metric plots,
and compact failure examples. These outputs summarize the provided artifacts;
they do not convert smoke or micro-eval runs into research claims.

For matched-overhead comparisons, use `total_overhead_ratio`. Model+hash rows
report transmitted lookback metadata via `hash_metadata_bit_count`,
`hash_metadata_token_equivalent_overhead_ratio`, and `total_overhead_ratio`.
Classical rows report repair packets through `actual_repair_token_overhead_ratio`
and the same `total_overhead_ratio` column.

## Correctness Checks For Smoke Output

For every run:

- reconstructed token count equals original token count,
- every known received position matches the original token,
- every fixed suffix position matches the forced suffix token,
- every hash-guided position either satisfies the hash constraint or logs a fallback reason,
- no target position remains as the mask token,
- metrics are finite except optional semantic similarity.

## Oracle Hash Metadata In Smoke Harness

The tiny smoke harness may derive hash metadata from dropped source tokens when
`oracle_hash_metadata=True`. This is only for decoder validation: it proves that
hash-guided denoising obeys constraints once the receiver has the right hash values.
It is not a production or experiment strategy. Real runs should use a protection
encoder that transmits metadata before channel loss.

## Memory Notes

LLaDA 1.5 is a dense 8B-class model. Expect GPU memory needs similar to other bf16 8B models, plus full-sequence diffusion forward passes.

The first implementation should:

- use batch size 1,
- keep sequences comfortably below 4096 tokens,
- use `torch.no_grad()`,
- use bf16 on CUDA,
- delete model objects explicitly between large runs if needed.

Quantization and CPU offload can be considered later.
