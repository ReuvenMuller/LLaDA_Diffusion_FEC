# LLaDA Diffusion FEC

Documentation-first research codebase for testing masked-diffusion language models as token-erasure recovery engines.

This project is intentionally separate from the original GenFEC codebase. It keeps the core research idea, token recovery with lightweight integrity constraints, but changes the model interface from causal next-token generation to LLaDA-style masked diffusion.

## Research Question

Can a masked-diffusion language model recover erased token positions more naturally than a causal LLM when:

- received token positions are frozen exactly,
- erased positions are represented as mask tokens,
- token hashes restrict candidate vocabularies,
- suffixes, delimiters, or protocol markers can be fixed during denoising,
- recovery quality is measured under packet-erasure channels and matched overhead?

## Why A Separate Codebase

The GenFEC depth-exam project grew into a full experiment harness around Qwen GGUF models and causal decoding. That was useful, but LLaDA changes the core decoding abstraction.

This project starts clean so the implementation can be shaped around:

- tensor-level masked-token recovery,
- bidirectional conditioning over known prefix, middle, and suffix positions,
- per-position logit masks,
- diffusion-step diagnostics,
- cleaner separation between model backend, channel simulation, integrity metadata, and evaluation.

## Initial Model Target

The first target is `GSAI-ML/LLaDA-1.5`.

Useful known constants from the model configuration:

- mask token ID: `126336`
- end-of-text token ID: `126081`
- vocabulary size: `126464`
- maximum sequence length: `4096`

See [docs/reference_sources.md](docs/reference_sources.md) for model and code references.

## Documentation Index

- [Project overview](docs/project_overview.md)
- [Lessons from GenFEC](docs/lessons_from_genfec.md)
- [System architecture](docs/architecture.md)
- [LLaDA decoding design](docs/llada_decoding_design.md)
- [Implementation plan](docs/implementation_plan.md)
- [Experiment plan](docs/experiment_plan.md)
- [Development notes](docs/development_notes.md)
- [Roadmap](docs/roadmap.md)
- [Research runbook](docs/research_runbook.md)
- [Hash profiles](docs/hash_profiles.md)
- [LLaDA server model run notes](docs/llada_server_model_notes.md)
- [Reference sources](docs/reference_sources.md)
- [Server SSH workflow](docs/server_ssh_workflow.md)
- [Agent handoff](docs/AGENT_HANDOFF.md)

## Planned Package Shape

```text
src/
  diffusion_fec/
    models/
    coding/
    decoding/
    channels/
    metrics/
    experiments/
tests/
docs/
```

The first prototype slices are now implemented: packetization, transmitted
lookback hash metadata, source and wire interleaving, constrained fake-model
decoding, metrics, artifact logging, opt-in real LLaDA smoke, persisted hash
profiles, and a deterministic fake micro-eval runner.

Minimal local micro-eval:

```powershell
python -m diffusion_fec.experiments.runner `
  --output-dir runs\fake_micro_eval `
  --micro-eval `
  --sample-lengths 8,16,32 `
  --tokens-per-packet 4 `
  --seed 0
```

This command is model-free and is not a research result.

Compact local framework sweep:

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

The sweep writes child run artifacts plus `analysis/aggregate.csv`,
`analysis/summary.md`, SVG metric plots, and `analysis/failure_examples.jsonl`.
It is still synthetic engineering validation, not a research result.
