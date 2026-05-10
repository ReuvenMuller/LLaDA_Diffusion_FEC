# Roadmap To Research-Ready LLaDA Diffusion FEC

This roadmap is the source of truth after the first profile-backed real LLaDA GPU
smoke. Older planning docs are still useful background, but some pre-GPU
assumptions have been superseded.

## Current State

Done:

- core serializable recovery dataclasses
- contiguous packetization and reconstruction planning
- seeded IID packet loss
- transmitted lookback-1 token-hash protection
- tokenizer-specific hash profiles
- model-free constrained diffusion decoder
- fake deterministic smoke runner
- token metrics and local artifacts
- Hugging Face LLaDA adapter
- opt-in real LLaDA smoke on the GPU server
- profile-backed real LLaDA smoke on the GPU server
- source/token layout modes: `contiguous` and `round_robin_chunks`
- packet-level matrix/span wire interleaving
- deterministic burst-loss helper over wire IDs for interleaving validation
- model-free synthetic micro-eval runner with model-only and model+hash modes
- opt-in real LLaDA synthetic micro-eval runner with loaded-profile enforcement
- runner-integrated packet-loss channels: IID, burst, and Gilbert-Elliott
- profile-backed real LLaDA micro-eval executed on the GPU server for model-only
  and model+hash modes
- classical overhead accounting helpers
- XOR parity baseline codec with source-layout and wire-interleaving support
- XOR parity synthetic micro-eval runner with manifest/results/events artifacts
- LT/fountain-style baseline codec and synthetic micro-eval runner
- streaming-window matched-overhead baseline codec and synthetic micro-eval runner
- deterministic text-record loading utilities
- frozen GenFEC WikiText-derived validation artifact copied into `data/`
- dataset-backed token sample loading for fake, classical, and real LLaDA
  micro-eval paths
- LLaDA-tokenized sample artifact writer/loader so fake, classical, and real
  LLaDA paths can consume the same token IDs
- result aggregation helpers for sweep CSVs
- deterministic synthetic sweep orchestration for the main comparison set
- analysis report artifacts: aggregate CSV, markdown summary, SVG metric plots,
  and qualitative failure examples
- comparable overhead accounting across model+hash metadata and classical repair
  packets via `total_overhead_ratio`
- experimental hybrid hash+XOR validation path with initial XOR peeling,
  optional parity candidate filtering during LLaDA decoding, final parity audit,
  and IID/burst channel support
- research runbook with local, server, profile, and final experiment conventions

Not done:

- larger server sweeps using frozen datasets and profiles
- final paper-ready plots/tables generated from completed server sweeps

Current smoke and micro-eval outputs are engineering validation only. They are
not research claims.

## Run Types

Smoke tests are tiny correctness checks. They prove code paths run and core
invariants hold.

Micro-evals are small profile-backed runs over synthetic or tiny fixed samples.
They help debug strategy behavior and server readiness, but they are not final
research evidence.

Real experiments use fixed configs, loaded hash profiles, larger sample sets,
matched-overhead baselines, server-backed execution, and frozen manifests.

## Research Story

The main comparison is:

- LLaDA model only
- LLaDA plus transmitted token hashes
- classical matched-overhead methods from the GenFEC line of work

Unigram, frequency, and hash-only baselines may be used for debugging, but they
are not part of the main research story.

## Hard Rules

- Do not import from `GenFEC_Depth_Exam`; use it only as a reference benchmark.
- Do not modify `GenFEC_Depth_Exam`.
- Real LLaDA runs must load existing hash profiles.
- Real LLaDA runs must not live-build hash maps during model execution.
- Fair model-vs-classical dataset comparisons must use frozen LLaDA-tokenized
  sample artifacts, not the fake local text tokenizer.
- No oracle hash metadata in real strategy paths.
- Default local tests stay model-free and fast.
- Real LLaDA tests and runs stay opt-in and server-backed.

## Server Assumptions

Expected server layout:

```text
/mnt/bst/a100/yxie2/rmuller7/LLaDA_Diffusion_FEC
/mnt/bst/a100/yxie2/rmuller7/.venvs/llada-diffusion-fec
/mnt/bst/a100/yxie2/rmuller7/.hf-cache-llada-diffusion-fec
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs
```

Run assumptions:

- CUDA GPU is available.
- Hugging Face config/tokenizer/model files are cached or downloadable.
- Hash profiles exist before real LLaDA experiment runs.
- Long model runs use `tmux` because SSH is flaky.
- Manifests record cache/profile/model assumptions clearly.

## Phased Roadmap

### Phase 1: Interleaving Foundation

Reimplement GenFEC's two interleaving ideas cleanly:

- source/token layout: `contiguous` and `round_robin_chunks`
- packet-level wire interleaving: disabled and matrix/span ordering

Absolute token positions, source packet identity, and wire IDs must remain
explicit in every packet.

Status: implemented and covered by local tests.

### Phase 2: Profile-Backed Synthetic Micro-Evals

Add a model-free micro-eval runner that supports:

- configurable synthetic sample lengths
- model-only and model+hash modes
- loaded or fake-built hash profiles
- source layout config
- packet wire interleaving config
- `run_manifest.json`, `results.csv`, and `events.jsonl`

The fake model remains deterministic and local-only.

Status: fake/profile-backed path implemented. The default remains model-free and
fast.

### Phase 3: Real LLaDA Micro-Evals

Add opt-in real LLaDA micro-evals using the same strategy/config surface.

Requirements:

- load profiles only
- fail clearly for missing profile/cache/CUDA/model path
- write manifest/results/events
- record latency, model forward calls, token counts, known/hash-guided/unguided
  counts, and profile metadata

Status: implemented at the framework level. It is opt-in, loads profiles only,
records latency/model-forward metadata, and keeps default tests model-free. The
next operational step is running it on the GPU server.

### Phase 4: Channels

Add packet-erasure channels:

- burst loss over wire IDs
- Gilbert-Elliott burst-state loss

Packet-level interleaving should affect burst geometry in wire order.
Source/token interleaving should affect the geometry of erased target positions.

Status: implemented for IID, burst, and Gilbert-Elliott in the smoke/micro-eval
runner path. Local tests verify the packet-level and source-level interleaving
effects through runner artifacts.

### Phase 5: Classical Matched-Overhead Baselines

Reimplement, in this codebase:

- XOR parity matched overhead
- LT/fountain-style matched overhead
- streaming-window matched overhead

Baseline outputs must use the same artifact interface as LLaDA runs. Baseline
code should not live in decoder modules.

Status: overhead accounting, XOR parity, LT/fountain, and streaming-window baselines are
implemented with artifact-writing synthetic micro-evals and tests.

### Phase 5B: Hybrid Hash + XOR Validation

Add an experimental integrated path:

- transmit lookback-1 token hashes on data packets
- transmit XOR parity packets through the same channel
- peel directly solvable parity equations after loss
- promote peeled tokens only when hash-consistent if hash metadata exists
- run LLaDA on remaining erasures
- optionally apply parity-aware candidate filtering during decoding
- audit final parity equation satisfaction

Status: implemented for validation. XOR-only remains a separate baseline.
Hybrid artifacts report hash overhead, XOR repair overhead, total overhead,
peel statistics, parity/hash conflicts, candidate rejections, fallbacks, and
final parity audit counts.

### Phase 6: Datasets, Sweeps, And Analysis

Add:

- deterministic real text sample loading
- server-ready run commands/scripts
- resumable or skip-aware sweeps
- result aggregation
- recovery, exact-match, known-preservation, hash-guided recovery, latency, and
  overhead plots/tables
- qualitative failure examples

Freeze configs and hash profiles before final research runs.

Status: framework support is implemented. The code can run skip-aware synthetic
sweeps, aggregate run outputs, write summary tables and SVG metric plots, and
extract failure examples. The remaining work is operational: run larger
server-backed experiments with frozen data/profile/config choices.

## Immediate Next Phase

The implementation framework is ready for first frozen server sweeps.

Recommended next operations:

- run dataset-backed fake/classical validation on the frozen WikiText artifact
- build or verify LLaDA hash profiles for 4, 8, and 16 bits
- run model-only and model+hash real LLaDA dataset-backed micro-evals under
  frozen configs
- run classical matched-overhead dataset-backed baselines with the same
  channel/interleaving settings
- aggregate all run roots with `python -m diffusion_fec.analysis.report`
- compare overhead with `total_overhead_ratio`, not repair overhead alone
- inspect `failure_examples.jsonl` before making research claims

Acceptance:

```powershell
python -m pytest
```
