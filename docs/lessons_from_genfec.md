# Lessons From GenFEC

This project should keep the useful research lessons from GenFEC while avoiding implementation debt from a causal-LLM-specific design.

## What Worked

### Keep Token Positions Explicit

GenFEC improved once packets carried explicit source token positions. This is even more important for diffusion recovery because the tensor layout is the reconstruction surface.

Every packet, metadata block, decoded token, and metric should reference absolute target-token positions.

### Separate Wire Order From Source Layout

Packet wire interleaving and source token packing are different ideas:

- wire interleaving changes the transmission order,
- source layout changes which token positions share a packet.

Keep them separate in this project. A burst channel should operate over wire order, while recovery should map back to source positions.

### Make Overhead First-Class

GenFEC's most useful comparisons came from logging metadata bits, repair bits, source payload bits, and total transmitted bits.

This project should log:

- source payload token count,
- source payload bit estimate,
- hash metadata bits,
- classical repair payload bits,
- total transmitted bits,
- overhead ratio,
- metadata bits per token.

### Include Matched Baselines Early

Classical baselines should not be bolted on at the end. They define whether the neural method is doing useful work for its redundancy budget.

Start with simple matched-overhead baselines:

- unprotected masked diffusion,
- no-LLM erasure fill baseline,
- XOR parity,
- streaming-window parity,
- optional fountain-style baseline later.

### Log Failure Modes, Not Just Scores

The GenFEC logs became valuable because they included ambiguity, fallback, confidence, and per-token decisions.

For diffusion decoding, log:

- number of masked positions,
- number of hash-constrained positions,
- number of unconstrained erased positions,
- candidate count per missing position,
- selected token probability,
- commit step per token,
- whether the token was fixed, hash-guided, or unconstrained,
- whether fixed-token restoration ever had to overwrite a model proposal.

### Keep Smoke Tests Tiny

Large LLM runs are slow and expensive. The first testing path should use:

- 1 to 3 samples,
- one loss rate,
- one hash width,
- one model,
- one or two channels,
- minimal semantic scoring.

## What To Improve

### Do Not Make Prompt Text The Recovery Surface

GenFEC needed prompts with `_` placeholders because causal models generate text strings. LLaDA does not need that.

For this project, the authoritative recovery state should be token tensors:

```text
known token IDs + mask token IDs + fixed masks + editable masks
```

Natural-language prompts can be an ablation, not the core mechanism.

### Do Not Reuse Token Hash Assets Across Tokenizers

Token hashes are vocabulary-specific. The Qwen optimized hash profile from GenFEC cannot be reused for LLaDA.

Start with uniform hash maps for LLaDA. Build optimized LLaDA profiles only after the first prototype works.

### Avoid Backend Lock-In

GenFEC's causal implementation was tied closely to `llama_cpp`.

This project should define a model adapter interface first, then implement a LLaDA/Hugging Face adapter behind it.

### Keep Strategy Definitions Declarative

Strategy definitions should live in config objects or files, not inside decoding functions.

Example strategy axes:

- hash bits: none, 4, 8, 16,
- channel: random, burst, Gilbert-Elliott,
- source layout: contiguous, round-robin chunks,
- diffusion steps: 32, 64, 128,
- block length: 32, 64, full sequence,
- prompt mode: none, instruction, left-context.

### Keep Analysis Paths Stable

GenFEC benefited from stable output paths and manifests.

Every run directory should include:

- `run_manifest.json`,
- `results.csv`,
- `events.jsonl`,
- `config_snapshot.json`,
- optional `console.log`.

## Design Principles For This Project

1. Token tensors are the source of truth.
2. Fixed positions are enforced every denoising step.
3. Hash constraints are hard logit masks, not soft preferences.
4. Prompt/context text is optional side information.
5. Every result row must include enough metadata to reproduce the condition.
6. Model-specific tokenization is isolated behind adapters.
7. The first prototype favors clarity over speed.
