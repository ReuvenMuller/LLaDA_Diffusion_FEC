# Project Overview

## Purpose

LLaDA Diffusion FEC is a research framework for recovering missing text tokens after packet erasures using a masked-diffusion language model and lightweight token-level integrity metadata.

The project starts from one observation:

```text
packet loss recovery is naturally a masked-token denoising problem
```

The original GenFEC project used a causal language model to generate the entire reconstructed sequence left to right while forcing known tokens and filtering missing-token candidates by hash. That works, but it is awkward for fixed suffixes, middle gaps, and bidirectional context.

LLaDA-style masked diffusion gives a cleaner primitive:

```text
[known token] [MASK] [MASK] [known token] [MASK] [known suffix]
```

The model predicts all masked positions under full-sequence context, and the decoder decides which positions to commit at each denoising step.

## Core Hypothesis

For token-erasure recovery, a masked-diffusion LLM should provide a better algorithmic match than a causal LLM because:

- known received tokens can remain fixed throughout the denoising process,
- missing positions are explicit editable variables,
- the model can condition on both left and right context,
- hard suffix or delimiter constraints are straightforward,
- token-hash constraints can be applied as per-position logit masks,
- uncertainty can be tracked over diffusion steps instead of only next-token choices.

## Scope

This project focuses on text-token recovery, not production networking.

In scope:

- tokenization and packetization,
- packet-erasure channels,
- lightweight token-hash metadata,
- LLaDA masked-diffusion decoding,
- fixed-position and suffix constraints,
- matched-overhead baselines,
- exact, token-level, semantic, latency, and ambiguity metrics.

Out of scope for the first prototype:

- production-grade forward error correction,
- streaming deployment,
- neural training or finetuning,
- large optimized hash-map assets,
- multi-node experiment orchestration.

## First Milestone

The first runnable milestone should answer:

Can `GSAI-ML/LLaDA-1.5` reconstruct erased token positions when given:

- a clean token sequence with some positions replaced by the LLaDA mask token,
- received-token positions frozen,
- optional 8-bit or 16-bit token hashes for erased positions,
- deterministic low-confidence remasking,
- a small WikiText-derived evaluation set?

## Non-Goals From GenFEC

Do not port the GenFEC implementation directly.

Avoid carrying over:

- llama.cpp-specific interfaces,
- prompt strings with underscore placeholders as the primary recovery surface,
- causal logits processors,
- root-level module sprawl,
- experiment choices encoded inside decoder internals.

Keep:

- careful overhead accounting,
- matched baselines,
- reproducible experiment manifests,
- JSONL logs with enough detail for later analysis,
- clear separation between strategy definitions and algorithm code.

## Primary Research Contributions To Test

The project should test these candidate contributions:

1. Masked-diffusion recovery is a better fit for non-contiguous token erasures than causal reconstruction.
2. Token hashes can be applied as hard per-position vocabulary constraints during denoising.
3. Fixed suffixes, delimiters, and received tokens can be guaranteed exactly by restoring them every step.
4. Diffusion-step confidence provides a useful signal for recovery ambiguity and failure analysis.
5. Hash-constrained diffusion can trade small metadata overhead for improved exact and token-level recovery.
