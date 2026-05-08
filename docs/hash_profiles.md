# Hash Profiles

Hash profiles are the preferred path for real experiments. A profile stores the
tokenizer-specific `token_id -> hash_bucket` map once, then experiment runs load
that map instead of rebuilding it from tokenizer strings.

## Why Profiles

The transmitted lookback protection scheme needs a deterministic hash bucket for
each protected token. During decoding, the same map is used to constrain
hash-guided erased positions to candidate token IDs in the matching bucket.

For LLaDA, the vocabulary has `126464` token IDs. Building the full map by asking
the tokenizer for one token string at a time is much too slow for routine runs.
The implementation now supports tokenizer-native batch conversion, but the best
experiment practice is still to build the map once and reuse it.

## Profile Format

A hash profile directory contains:

- `hash_profile_metadata.json`
- one `.npy` map per hash setting, for example `uniform_hash4_map.npy`

The `.npy` array is one-dimensional:

```text
array[token_id] == hash_bucket
```

Metadata records the model/tokenizer identity, vocabulary size, excluded special
tokens, salt, map mode, hash bits, and algorithm. The token-string source is:

```text
tokenizer_native_token_string_plus_token_id
```

This means the hash input is based on the tokenizer-native token string plus the
token ID and salt, not decoded user-facing text bytes alone.

## CLI Usage

Build a profile while running the fake smoke path:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\fake_profile_smoke `
  --hash-profile-dir runs\hash_profiles\fake_smoke_v1 `
  --build-hash-profile `
  --sample-count 1
```

Reuse the same profile:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\fake_profile_smoke_reuse `
  --hash-profile-dir runs\hash_profiles\fake_smoke_v1 `
  --sample-count 1
```

Real LLaDA smoke uses the same flags:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\real_llada_smoke `
  --real-llada-smoke `
  --hash-profile-dir runs\hash_profiles\llada_1_5_smoke_v1 `
  --build-hash-profile `
  --sample-count 1 `
  --steps 1
```

On later runs, omit `--build-hash-profile` to require loading the existing map.
If the requested map is missing, the runner fails clearly instead of silently
creating a different profile.

## Policy

- Use profile loading for real experiment sweeps.
- Use `--build-hash-profile` only in setup or smoke runs.
- Use live batch-building only when no profile directory is provided.
- Keep profile directories tied to one tokenizer/model/salt/exclusion set.
- Build separate maps for `hash_bits` 4, 8, and 16 when comparing overheads.

