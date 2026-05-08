# LLaDA Server Model Run Notes

This note records the issues found while moving the project to the GPU server
and running the first real `GSAI-ML/LLaDA-1.5` smoke path.

## Successful Run

Server checkout:

```text
/mnt/bst/a100/yxie2/rmuller7/LLaDA_Diffusion_FEC
```

Successful artifact directory:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_smoke_gpu6_20260508_013312
```

The run used:

- model: `GSAI-ML/LLaDA-1.5`
- protection: `lookback_1`
- oracle hash metadata: `False`
- hash bits: `4`
- steps: `1`
- tiny forward shape: `[1, 1, 126464]`

It wrote `run_manifest.json`, `results.csv`, and `events.jsonl`.

## Issues Resolved

### Git Ignored The Model Adapter Package

The repository originally had `models/` in `.gitignore`. That ignored
`src/diffusion_fec/models`, so the server clone was missing the LLaDA adapter.

Fix:

```text
models/ -> /models/
```

This keeps only a repository-root `models` directory ignored.

### Wrong Hugging Face Loader

The adapter first used `AutoModel`. The LLaDA model repo exposes the model under
the causal-LM auto class.

Fix:

```text
AutoModelForCausalLM.from_pretrained(...)
```

### `device_map="auto"` Compatibility Failure

Using `device_map="auto"` triggered custom remote-code compatibility problems in
the LLaDA class under the installed Transformers stack.

Fix:

- load the model normally in `bfloat16`
- then call `model.to("cuda")` for the smoke path

This is suitable for a single 80GB A100 smoke run.

### Missing `all_tied_weights_keys`

The installed Transformers version expected the remote model class to expose
`all_tied_weights_keys`. LLaDA's downloaded custom model class did not.

Fix:

The adapter patches the remote class with an empty `all_tied_weights_keys` map
when the attribute is missing.

### Old `tie_weights` Signature

Transformers called `tie_weights` with newer keyword arguments. The LLaDA remote
class had an older no-keyword signature.

Fix:

The adapter wraps `tie_weights` for this class so extra loader arguments are
ignored and the original method still runs.

### Missing `config.use_cache`

The remote model forward path read `self.config.use_cache`, but the LLaDA config
did not define it.

Fix:

The adapter sets `use_cache=False` when the loaded config lacks that attribute.

### Hash Map Build Was Too Slow

The first real smoke built the full LLaDA hash map by calling token conversion
one token ID at a time over all `126464` vocabulary entries. The hash math was
not the slow part; repeated tokenizer calls were.

Fix:

- prefer tokenizer-native batch `convert_ids_to_tokens(...)`
- fall back to per-token decoding only for IDs whose batch token string is `None`
- add persisted hash profiles so real experiment runs load `.npy` maps

After this, a full LLaDA `hash_bits=4` map built on the server in about 4.4
seconds, and future runs can load it directly from the profile.

### Server Pytest Behavior

The complete local test suite passed on Windows. On the server, individual
non-HF test files passed, but one combined `pytest` invocation timed out through
the flaky SSH session.

Current interpretation:

- project functionality is verified by local full tests
- server smoke path is verified by real artifact generation
- server per-file non-HF tests pass
- combined server pytest should be revisited when we create the larger runner

## Notes For Future Runs

- Set `HF_HOME` to the project-specific server cache.
- Use `CUDA_VISIBLE_DEVICES` to pick the least busy GPU.
- Prefer `tmux` for model runs because SSH is flaky.
- Use a hash profile directory for repeated runs.
- A Hugging Face token is optional but may improve download behavior and rate
  limits.

## Next Server Micro-Eval Command

After pulling the latest repository state on the server and activating the
project venv, run a tiny loaded-profile micro-eval:

```bash
python -m diffusion_fec.experiments.runner \
  --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_micro_eval_loaded_hash \
  --real-llada-micro-eval \
  --hash-profile-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/hash_profiles/llada_1_5_smoke_v1 \
  --llada-local-files-only \
  --sample-lengths 8 \
  --tokens-per-packet 1 \
  --hash-bits 4 \
  --steps 2
```

Use `--micro-eval-mode model_only` to run the matching model-only path without a
hash profile. These micro-evals are engineering validation only, not research
claims.
