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

The baseline code is intentionally separate from LLaDA decoder modules. The next
step is an artifact-writing XOR micro-eval runner that uses the same
`run_manifest.json`, `results.csv`, and `events.jsonl` convention as the fake and
real LLaDA runners.

Current opt-in pytest smoke:

```powershell
$env:RUN_LLADA_E2E_SMOKE = "1"
python -m pytest tests\test_llada_e2e_smoke.py -q
```

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
