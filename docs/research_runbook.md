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

### Completed Interleaving Validation

The first real LLaDA interleaving ablation completed on 2026-05-09:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-interleaving-20260509_115544
```

It reused the frozen 10-sample LLaDA-tokenized WikiText artifact, hash profiles,
IID `loss_rate=0.5`, `tokens_per_packet=4`, `seed=0`, `steps=8`, and the best
validated decoder path: `commit_once + always`.

| strategy | hash bits | source layout | wire order | lost recovery | edit distance | overhead |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| model only | 4 | contiguous | none | 0.2736 | 48.7 | 0.0000 |
| model only | 4 | contiguous | matrix | 0.3305 | 45.1 | 0.0000 |
| model only | 4 | round_robin_chunks | none | 0.5515 | 30.9 | 0.0000 |
| model only | 4 | round_robin_chunks | matrix | 0.5890 | 29.4 | 0.0000 |
| hash4 | 4 | contiguous | none | 0.4347 | 38.3 | 0.2279 |
| hash4 | 4 | contiguous | matrix | 0.5052 | 34.4 | 0.2279 |
| hash4 | 4 | round_robin_chunks | none | 0.6667 | 23.6 | 0.2279 |
| hash4 | 4 | round_robin_chunks | matrix | 0.7173 | 20.7 | 0.2279 |
| hash8 | 8 | contiguous | none | 0.5871 | 28.3 | 0.4559 |
| hash8 | 8 | contiguous | matrix | 0.6280 | 26.1 | 0.4559 |
| hash8 | 8 | round_robin_chunks | none | 0.7567 | 17.2 | 0.4559 |
| hash8 | 8 | round_robin_chunks | matrix | 0.7900 | 15.4 | 0.4559 |
| hash16 | 16 | contiguous | none | 0.6451 | 24.6 | 0.9118 |
| hash16 | 16 | contiguous | matrix | 0.6772 | 22.4 | 0.9118 |
| hash16 | 16 | round_robin_chunks | none | 0.7795 | 15.8 | 0.9118 |
| hash16 | 16 | round_robin_chunks | matrix | 0.8154 | 13.5 | 0.9118 |

All rows preserved known tokens, left zero mask tokens, used eight model forward
calls, and had exact-match rate `0.0`. Source/token-level round-robin
interleaving provided the largest improvement by dispersing erasures across the
sequence. Packet-level matrix wire interleaving also helped under the fixed IID
seed, but the effect was smaller. The best validation cell was hash16 with both
round-robin source layout and matrix wire order: recovery `0.8154`, edit
distance `13.5`.

Qualitatively, round-robin interleaving reduced repeated local collapses. For
example, a contiguous hash16 reconstruction for `wiki_48` began with repeated
`Doug`, while the round-robin plus matrix version recovered the opening location
and age details much more closely. This is validation evidence, not a final
research claim.

### Hybrid Hash + XOR Validation

The framework includes an experimental hybrid validation runner for testing
whether transmitted `hash4` metadata plus XOR parity can approach `hash8`
performance at roughly comparable total overhead.

Two hybrid modes are available:

```text
pre_peel_only
parity_filter
iterative_peel
iterative_rollback
```

`pre_peel_only` transmits lookback-1 hash metadata on data packets, transmits
XOR parity packets through the same channel, peels any directly solvable XOR
equations after loss, hash-validates peeled tokens when received hash metadata
exists for that position, and then runs LLaDA on the remaining erasures.

`parity_filter` does the same initial peeling, then applies a local parity
candidate filter during LLaDA decoding. Hash filtering still happens first. The
parity filter rejects candidates only when a received parity equation is fully
determined by known/committed tokens plus the current candidate. If filtering
empties the candidate set, the default is to fall back to the hash-filtered
candidate set and log `parity_filter_fallback_count`.

`iterative_peel` is the stronger cooperative hybrid mode. It does the initial
peel, uses the parity candidate filter during LLaDA decoding, and after each
`commit_once` decoder step reruns XOR peeling using received equations plus all
currently known/committed tokens. Newly parity-solved tokens are promoted to
fixed only after hash validation when metadata exists and vocab/special-token
legality checks pass. This mode intentionally requires `commit_once + always`.
It should be validated before sweeping alternative XOR layouts.

`iterative_rollback` is experimental and is currently supported only with
`xor_code=sparse_fountain`, `commit_once`, and `hash_constraint_schedule=always`.
It adds conservative parity/hash conflict feedback to iterative peeling. When a
conflict identifies exactly one soft model-committed token, that token is
remasked and the exact token ID is banned for that position; when multiple soft
commits are implicated, all are remasked without new bans. Received tokens are
trusted roots and are never rolled back. Parity-solved tokens carry dependency
provenance and are invalidated if a dependency is rolled back. Rollback runs
report `rollback_*` diagnostics and may use bounded extra repair rounds.

Local fake/model-free smoke:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\hybrid_fake_smoke `
  --hybrid-xor-hash-micro-eval `
  --sample-lengths 12 `
  --tokens-per-packet 3 `
  --vocab-size 64 `
  --steps 2 `
  --hash-bits 4 `
  --hybrid-mode parity_filter `
  --xor-overhead-bits-per-token 4 `
  --source-layout round_robin_chunks `
  --source-chunk-size 1 `
  --wire-interleaving matrix `
  --wire-interleaving-span 4 `
  --channel burst `
  --burst-start-wire-id 0 `
  --burst-length 3
```

Server real-LLaDA hybrid command shape:

```bash
python -m diffusion_fec.experiments.runner \
  --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_hybrid_hash4_xor4 \
  --real-llada-hybrid-xor-hash-micro-eval \
  --hash-profile-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/hash_profiles/llada_1_5_real_smoke_v1 \
  --tokenized-samples-file /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/tokenized_artifacts/llada_tokenized_wikitext2_genfec_seed0_max128.json \
  --llada-local-files-only \
  --tokens-per-packet 4 \
  --hash-bits 4 \
  --xor-overhead-bits-per-token 4 \
  --steps 8 \
  --seed 0 \
  --source-layout round_robin_chunks \
  --source-chunk-size 1 \
  --wire-interleaving matrix \
  --wire-interleaving-span 4 \
  --hybrid-mode parity_filter
```

Use `--channel burst --burst-start-wire-id 0 --burst-length 16` for the burst
validation counterpart. Hybrid artifacts report `xor_overhead_bits_per_token`,
`xor_target_overhead_ratio`, `actual_repair_token_overhead_ratio`,
hash-metadata overhead, `total_overhead_ratio`, parity peel counts, parity/hash
conflicts, parity candidate rejections, parity filter fallbacks, and final
parity audit counts. These are decoder and coding validation outputs, not final
research claims.

For headline recovery comparisons, use
`channel_lost_position_recovery_rate`. It scores against source-token positions
erased by the channel before any XOR repair, parity peeling, or hybrid promotion.
The older `lost_position_recovery_rate` is retained for backward compatibility;
it is plan-state based and can be misleading for classical and hybrid methods
that promote repaired tokens to known positions before metric scoring.

The first 10-sample hybrid validation completed on 2026-05-11:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-hybrid-xor-20260511_162750
```

It used the frozen 10-sample LLaDA-tokenized WikiText artifact, loaded real
LLaDA hash profiles, `steps=8`, `tokens_per_packet=4`, `seed=0`,
`commit_once + always`, `source_layout=round_robin_chunks`,
`source_chunk_size=1`, `wire_interleaving=matrix`, and
`wire_interleaving_span=4`. Both IID `loss_rate=0.5` and deterministic burst
loss with `burst_start_wire_id=0`, `burst_length=16` were run.

The run was reanalysed after adding `channel_lost_position_recovery_rate`.
Corrected mean validation results are:

| strategy | channel | channel recovery | legacy plan recovery | edit distance | total overhead | latency sec | parity recovered | parity violations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XOR-only hash4 budget | IID | 0.0709 | 0.0000 | 61.2 | 0.2500 | 0.0000 |  |  |
| XOR-only hash8 budget | IID | 0.2244 | 0.0000 | 51.6 | 0.5000 | 0.0000 |  |  |
| LLaDA hash4 | IID | 0.7173 | 0.7173 | 20.7 | 0.2279 | 7.7662 |  |  |
| LLaDA hash8 | IID | 0.7901 | 0.7901 | 15.4 | 0.4559 | 7.3250 |  |  |
| Hybrid hash4+xor4 pre-peel | IID | 0.7216 | 0.7025 | 19.0 | 0.4779 | 6.7901 | 4.0 | 4.9 |
| Hybrid hash4+xor4 parity-filter | IID | 0.7556 | 0.7402 | 16.9 | 0.4779 | 19.2920 | 4.0 | 1.9 |
| XOR-only hash4 budget | burst | 0.1429 | 0.0000 | 48.0 | 0.2500 | 0.0000 |  |  |
| XOR-only hash8 budget | burst | 0.3333 | 0.0000 | 32.0 | 0.5000 | 0.0000 |  |  |
| LLaDA hash4 | burst | 0.7938 | 0.7938 | 13.2 | 0.2279 | 6.9070 |  |  |
| LLaDA hash8 | burst | 0.8563 | 0.8563 | 9.2 | 0.4559 | 7.2438 |  |  |
| Hybrid hash4+xor4 pre-peel | burst | 0.8321 | 0.8042 | 9.4 | 0.4779 | 3.5525 | 8.0 | 4.2 |
| Hybrid hash4+xor4 parity-filter | burst | 0.8804 | 0.8604 | 6.7 | 0.4779 | 12.3244 | 8.0 | 1.0 |

On IID loss, the hybrid parity-filter run improved over hash4 but did not beat
hash8 at similar overhead. Under burst loss, the hybrid parity-filter run
slightly beat hash8 on lost-token recovery and substantially improved token edit
distance, at higher decode latency. This is promising validation evidence, not a
final research claim.

The stronger `iterative_peel` mode was validated on the same frozen setup after
implementation:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-iterative-peel-20260511_213841
```

It used `hash4 + xor4`, `steps=8`, `commit_once + always`,
`source_layout=round_robin_chunks`, `source_chunk_size=1`,
`wire_interleaving=matrix`, and `wire_interleaving_span=4`. It reuses the same
10-sample LLaDA-tokenized artifact and loaded hash profile directory as the
earlier hybrid validation. Mean results:

| strategy | channel | channel recovery | edit distance | normalized edit | exact match | known preserved | masks left | total overhead | latency sec | wall time sec | forwards | iterative recovered | hash conflicts | parity violations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid hash4+xor4 iterative-peel | IID | 0.7561 | 16.7 | 0.1305 | 0.0000 | 1.0000 | 0.0 | 0.4779 | 13.1460 | 131.5964 | 8.0 | 8.5 | 0.4 | 1.3 |
| Hybrid hash4+xor4 iterative-peel | burst | 0.8893 | 6.2 | 0.0484 | 0.0000 | 1.0000 | 0.0 | 0.4779 | 7.6727 | 76.8622 | 8.0 | 14.0 | 0.4 | 0.7 |

Compared with the earlier `parity_filter` validation, iterative peeling made a
small IID improvement (`0.7556` to `0.7561` recovery, edit distance `16.9` to
`16.7`) and a clearer burst improvement (`0.8804` to `0.8893`, edit distance
`6.7` to `6.2`). The burst cell remains above the hash8 validation cell
(`0.8563` recovery, edit distance `9.2`) at similar total overhead. Treat this
as decoder-design validation only; the next step is still controlled sweeps over
XOR layouts and more samples before making research claims.

### Sparse Fountain XOR Validation

The next XOR option is `xor_code=sparse_fountain`, documented as
**Raptor/LT-inspired Sparse Fountain XOR with bounded GF(2) component solving**.
It is not RaptorQ. The existing stripe path remains the default:

```text
xor_code=stripe
```

Sparse fountain XOR generates deterministic overlapping token-level XOR
equations from shared config and seed. The transmitted repair budget is strict:

```text
repair_token_budget = floor(total_tokens * xor_overhead_bits_per_token / token_bit_width)
equation_count <= repair_token_budget
```

Each sparse repair equation transmits one token-equivalent XOR value. Equation
structure is reconstructed from manifest config and seed; position lists in
events/packet metadata are audit metadata, not additional modeled transmitted
overhead. Sparse artifacts report `repair_token_budget`, `sparse_equation_count`,
`actual_repair_token_overhead_ratio`, `sparse_budget_exhausted`,
`sparse_coverage_pass_degree`, `sparse_coverage_zero_count`,
`sparse_coverage_mean`, `sparse_actual_mean_degree`, and
`sparse_degree_histogram`.

The sparse solver first runs normal XOR peeling. If peeling stalls, it builds
small connected components over remaining unknown token positions and received
equations. Components up to `sparse_xor_max_component_unknowns` are solved as a
true GF(2) system: the coefficient matrix is binary over token positions and the
token-ID RHS is carried bit-plane-wise through XOR row operations. Only unique
full-rank solutions are promoted. Rank-deficient or too-large components are
skipped, and solved tokens must pass vocab, special-token, and hash validation
before being fixed.

Local fake sparse smoke:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\sparse_fountain_fake_smoke `
  --hybrid-xor-hash-micro-eval `
  --xor-code sparse_fountain `
  --hybrid-mode iterative_peel `
  --sample-lengths 8 `
  --tokens-per-packet 1 `
  --vocab-size 64 `
  --steps 2 `
  --hash-bits 4 `
  --xor-overhead-bits-per-token 4 `
  --sparse-xor-seed 7 `
  --sparse-xor-enable-linear-solve on `
  --channel burst `
  --burst-start-wire-id 0 `
  --burst-length 2
```

Local fake sparse rollback smoke:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\sparse_fountain_rollback_fake_smoke `
  --hybrid-xor-hash-micro-eval `
  --xor-code sparse_fountain `
  --hybrid-mode iterative_rollback `
  --sample-lengths 8 `
  --tokens-per-packet 1 `
  --vocab-size 64 `
  --steps 2 `
  --hash-bits 4 `
  --xor-overhead-bits-per-token 4 `
  --sparse-xor-seed 7 `
  --sparse-xor-enable-linear-solve on `
  --rollback-extra-steps 4 `
  --rollback-max-total-steps 16 `
  --rollback-max-per-position 3 `
  --channel burst `
  --burst-start-wire-id 0 `
  --burst-length 2
```

Focused real validation should compare only:

```text
hash8
hash4 + stripe XOR iterative
hash4 + sparse_fountain XOR iterative
sparse_fountain XOR only, if cheap
```

Use the frozen 10-sample LLaDA-tokenized artifact, loaded hash profiles,
`steps=8`, `commit_once + always`, `tokens_per_packet=4`,
`source_layout=round_robin_chunks`, `source_chunk_size=1`,
`wire_interleaving=matrix`, `wire_interleaving_span=4`, and both IID
`loss_rate=0.5` and burst `burst_start_wire_id=0`, `burst_length=16`. Use
`channel_lost_position_recovery_rate` as the headline recovery metric.

For rollback validation, keep the matrix focused: compare `hash8`,
`hash4 + sparse_fountain XOR iterative_peel`, and
`hash4 + sparse_fountain XOR iterative_rollback` on the same IID and burst
settings. This is decoder-design validation only.

Future TODO: when the unresolved or conflicted set is small, add an endgame
search that tries top-k hash-valid candidates and reranks candidate
combinations by parity satisfaction. That search is not part of the current
rollback implementation.

The first focused sparse-fountain validation completed on the GPU server:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-sparse-fountain-20260512_011610
```

It used the frozen 10-sample LLaDA-tokenized artifact, loaded hash4 profile,
`steps=8`, `commit_once + always`, token and packet interleaving, sparse seed
`7`, coverage enabled, and bounded component size `8`.

| strategy | channel | channel recovery | edit distance | normalized edit | exact match | known preserved | masks left | total overhead | latency sec | forwards | initial peel | initial linear | iterative peel | iterative linear | sparse mean degree | parity violations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse-only xor4 | IID | 0.0753 | 53.7 | 0.4195 | 0.0000 | 1.0000 | 53.7 | 0.2344 | 0.0000 | 0.0 | 3.9 | 0.0 |  |  | 4.6667 |  |
| Sparse-only xor4 | burst | 0.2500 | 24.0 | 0.1875 | 0.0000 | 1.0000 | 24.0 | 0.2344 | 0.0000 | 0.0 | 8.0 | 0.0 |  |  | 4.6667 |  |
| Hybrid hash4+sparse-xor4 iterative | IID | 0.8343 | 10.6 | 0.0828 | 0.1000 | 1.0000 | 0.0 | 0.4623 | 10.3379 | 8.0 | 3.9 | 0.0 | 7.6 | 0.0 | 4.6667 | 0.8 |
| Hybrid hash4+sparse-xor4 iterative | burst | 0.9781 | 0.7 | 0.0055 | 0.6000 | 1.0000 | 0.0 | 0.4623 | 1.2336 | 7.7 | 8.0 | 0.0 | 6.4 | 0.0 | 4.6667 | 0.2 |

Compared with the earlier 10-sample validations, sparse hybrid improved over
hash8 (`0.7901` IID, `0.8563` burst) and stripe XOR iterative (`0.7561` IID,
`0.8893` burst) at similar total overhead. In this first run the bounded GF(2)
linear solver recovered `0.0` mean tokens; the improvement came from sparse
parity structure, initial/iterative peeling, and parity candidate filtering.
This is strong validation evidence, but still not a final research claim.

The first focused rollback validation completed on the GPU server:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-rollback-20260512_030702
```

It used the same frozen 10-sample artifact, loaded hash profiles, sparse seed
`7`, `hash4 + xor4`, `steps=8`, `commit_once + always`, token and packet
interleaving, IID `loss_rate=0.5`, and burst `burst_start_wire_id=0`,
`burst_length=16`. The detached server run required:

```bash
export HF_HOME=/mnt/bst/a100/yxie2/rmuller7/.hf-cache-llada-diffusion-fec
```

Without that cache setting, `--llada-local-files-only` cannot find the cached
tokenizer/config in a fresh `tmux` session.

| strategy | channel | channel recovery | edit distance | normalized edit | exact match | known preserved | masks left | total overhead | latency sec | forwards | decoder steps | iterative peel | rollback events | rollback positions | rollback bans | parity violations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hash8 | IID | 0.7901 | 15.4 | 0.1203 | 0.0000 | 1.0000 | 0.0 | 0.4559 | 2.8268 | 8.0 | 8.0 |  |  |  |  |  |
| Sparse iterative peel | IID | 0.8343 | 10.6 | 0.0828 | 0.1000 | 1.0000 | 0.0 | 0.4623 | 10.2744 | 8.0 | 8.0 | 7.6 | 0.0 | 0.0 | 0.0 | 0.8 |
| Sparse iterative rollback | IID | 0.8397 | 10.2 | 0.0797 | 0.1000 | 1.0000 | 0.1 | 0.4623 | 10.9446 | 9.2 | 9.2 | 7.9 | 3.4 | 3.5 | 0.1 | 0.4 |
| Hash8 | burst | 0.8563 | 9.2 | 0.0719 | 0.0000 | 1.0000 | 0.0 | 0.4559 | 2.7289 | 8.0 | 8.0 |  |  |  |  |  |
| Sparse iterative peel | burst | 0.9781 | 0.7 | 0.0055 | 0.6000 | 1.0000 | 0.0 | 0.4623 | 1.1851 | 7.7 | 7.7 | 6.4 | 0.0 | 0.0 | 0.0 | 0.2 |
| Sparse iterative rollback | burst | 0.9844 | 0.5 | 0.0039 | 0.6000 | 1.0000 | 0.1 | 0.4623 | 1.2284 | 8.1 | 8.1 | 6.5 | 0.9 | 0.5 | 0.2 | 0.1 |

Rollback modestly improved sparse iterative peel in both IID and burst, and it
cut mean parity violations roughly in half. The cost was a small increase in
forward calls/latency and a mean `0.1` remaining mask tokens after the rollback
budget. This is promising decoder-design validation, not a final result.

### Sparse Hybrid Diagnostics

After the rollback validation, the sparse-hybrid diagnostic slice added a
behavior-equivalent parity-filter optimization and finer timing fields. The
filter now computes the required token for determined XOR equations directly
instead of scanning every hash-bucket candidate. Runs after this slice report:

- `model_forward_time_sec`
- `candidate_construction_time_sec`
- `parity_candidate_filter_time_sec`
- `xor_peel_time_sec`
- `linear_solver_time_sec`
- `post_commit_hook_time_sec`
- `rollback_time_sec`
- `total_decode_time_sec`
- `parity_filter_required_token_checks`
- `parity_filter_full_scan_count`
- `parity_filter_candidate_membership_checks`

Old rollback artifacts can still be reanalysed for failure taxonomy, but they
do not contain the timing breakdown above. Use:

```bash
python -m diffusion_fec.analysis.diagnostics \
  --run-root /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-rollback-20260512_030702 \
  --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-rollback-20260512_030702/diagnostics
```

The command writes `case_summary.csv`, `step_timing.csv`,
`error_taxonomy.csv`, `failure_examples.md`, and `diagnostic_summary.md`. If
`diagnostic_summary.md` says timing is missing, rerun only the focused IID and
burst `sparse_fountain iterative_rollback` cells with the same frozen
10-sample settings. Do not run a large sweep for this diagnostic pass.

The old rollback run was reanalysed successfully:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-rollback-20260512_030702/diagnostics
```

It produced 60 case summaries and 466 wrong channel-lost token rows, but no
timing breakdown because those fields did not exist in the old artifacts.

The focused instrumented rerun completed here:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-diagnostics-20260513_165302
```

It reran only `sparse_fountain iterative_rollback` for IID and burst with the
same frozen 10-sample settings. The headline timing split showed that IID
latency is dominated by candidate construction, not LLaDA forward time:

| channel | recovery | edit | latency | model forward | candidate construction | parity filter | hook | XOR peel | rollback | mean candidates | max candidates | full scans |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IID | 0.8397 | 10.2 | 5.4393 | 0.2958 | 5.0833 | 1.1493 | 0.0030 | 0.0006 | 0.0002 | 74760.2 | 126462.0 | 0.0 |
| burst | 0.9844 | 0.5 | 0.8710 | 0.2677 | 0.5442 | 0.0476 | 0.0024 | 0.0006 | 0.0001 | 7883.0 | 8035.4 | 0.0 |

The parity-filter shortcut worked as intended: `full scans` stayed at `0.0`.
The remaining IID latency is therefore mostly from scoring/selecting over very
large candidate sets. The error taxonomy also shows that most IID wrong tokens
are not parity-constrained by surviving equations: 69 of 102 wrong IID
channel-lost positions had zero surviving sparse parity equations, and only 17
of 102 had hash metadata. Burst had only 5 wrong channel-lost positions, all
with hash metadata. This points to the next method work: improve transmitted
hash coverage and parity coverage for IID-scattered erasures, and consider
top-k/endgame search only after candidate-set size is reduced.

The channel-loss reanalysis artifacts are written under:

```text
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-hybrid-xor-20260511_162750/channel_reanalysis
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-hybrid-xor-20260511_162750/analysis_channel_loss
```

To re-run the correction:

```bash
python -m diffusion_fec.analysis.channel_recovery \
  --run-root /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-hybrid-xor-20260511_162750 \
  --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-hybrid-xor-20260511_162750/channel_reanalysis \
  --patch-results
python -m diffusion_fec.analysis.report \
  --run-root /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-hybrid-xor-20260511_162750 \
  --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/dataset-validation-hybrid-xor-20260511_162750/analysis_channel_loss
```

Use `mean_channel_lost_position_recovery_rate` in summary tables for classical
and hybrid comparison.

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
- `channel_lost_position_recovery_rate.svg`
- `lost_position_recovery_rate.svg`
- `decode_latency_sec.svg`
- `repair_overhead_ratio.svg`
- `total_overhead_ratio.svg`
- `failure_examples.jsonl`
- `analysis_manifest.json`

Use `channel_lost_position_recovery_rate` as the headline recovery metric and
`total_overhead_ratio` for strategy comparison. Total overhead combines
classical repair-token overhead with transmitted hash metadata token-equivalent
overhead. The model+hash path reports `hash_metadata_bit_count`,
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
