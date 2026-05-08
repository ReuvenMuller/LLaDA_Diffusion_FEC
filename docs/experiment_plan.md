# Experiment Plan

## Goals

The experiment plan should answer three questions in order:

1. Does constrained LLaDA recovery work mechanically?
2. Do token hashes improve recovery over unprotected masked diffusion?
3. Does masked diffusion provide advantages over causal GenFEC-style recovery patterns, especially for non-contiguous erasures and fixed suffixes?

Do not start with a large matrix. Start with tight validation runs.

## Primary Metrics

Use these metrics from the beginning:

- exact token-sequence match,
- token edit distance,
- normalized token edit distance,
- lost-position token recovery rate,
- decode latency,
- candidate count after constraints,
- number of hash-guided positions,
- number of unguided positions.

Optional after smoke tests:

- semantic similarity,
- per-step convergence plots,
- qualitative example extraction.

## Initial Dataset

Use a small text dataset derived from WikiText or a local JSON file.

Target sample shape:

```json
{
  "id": 0,
  "original_message": "...",
  "word_count": 300
}
```

First smoke set:

- 3 samples,
- roughly 100 to 300 words each,
- keep token length below LLaDA's context limit.

Later study set:

- 60 samples,
- stable sample IDs,
- same input messages across strategies.

## Initial Strategy Matrix

### P0 Smoke Strategies

| Strategy | Prompt | Hash | Purpose |
| --- | --- | --- | --- |
| `NoLLM_Erasure` | none | none | Lower bound, leaves erased positions unresolved. |
| `LLaDA_Unprotected_NoPrompt` | none | none | Tests diffusion fill from known tokens only. |
| `LLaDA_Hash8_NoPrompt` | none | 8-bit | First core method. |
| `LLaDA_Hash16_NoPrompt` | none | 16-bit | Tests stronger hash guidance. |

### P1 Prompt Ablations

| Strategy | Prompt | Hash | Purpose |
| --- | --- | --- | --- |
| `LLaDA_Unprotected_Instruction` | instruction | none | Tests whether instruction helps. |
| `LLaDA_Hash8_Instruction` | instruction | 8-bit | Hash plus simple instruction. |
| `LLaDA_Hash8_LeftContext` | left context | 8-bit | Tests context-assisted recovery. |

### P2 Layout Ablations

| Strategy | Layout | Purpose |
| --- | --- | --- |
| `Hash8_Contiguous` | contiguous source packets | Baseline packet loss geometry. |
| `Hash8_SourceRRChunk1` | round-robin single-token chunks | Disperses burst losses across positions. |
| `Hash8_SourceRRChunk4` | round-robin 4-token chunks | Tests less aggressive dispersion. |

## Initial Channel Matrix

Smoke:

- random IID loss at `0.2`,
- burst loss at `0.2`.

Core:

- random IID loss at `0.1`, `0.2`, `0.3`, `0.4`,
- burst loss at `0.1`, `0.2`, `0.3`, `0.4`,
- Gilbert-Elliott loss after random/burst are stable.

Do not include `0.5` or `0.6` loss until the decoder mechanics are validated. Those levels are useful stress tests, but they can obscure basic implementation bugs.

## Initial Decoding Config

Use the official LLaDA defaults as the starting point:

```text
steps = 128
block_length = 32
temperature = 0.0
remasking = low_confidence
cfg_scale = 0.0
```

For speed ablations:

```text
steps = 32, 64, 128
block_length = 32
```

For long targets:

```text
block_length = 64 or 128
```

Only change one decoding parameter at a time.

## Prompt Modes

### `none`

No natural-language prompt. The model sees only the masked target token sequence.

This is the primary scientific baseline because it isolates the recovery mechanism.

### `instruction`

Prepend a short instruction as fixed prompt tokens:

```text
Recover the original text. Fill only the masked positions.
```

Metrics still score only target positions.

### `left_context`

Prepend known previous text as fixed context. This is useful for studying applications where the receiver has already decoded earlier blocks.

Do not mix this into the main comparison until the no-prompt baseline is stable.

## Expected Run Sizes

Tiny smoke:

```text
3 samples x 2 channels x 1 loss rate x 4 strategies = 24 rows
```

Small validation:

```text
10 samples x 2 channels x 2 loss rates x 4 strategies = 160 rows
```

Core first matrix:

```text
60 samples x 2 channels x 4 loss rates x 4 strategies = 1,920 rows
```

Prompt ablation:

```text
60 samples x 2 channels x 2 loss rates x 4 prompt/hash strategies = 960 rows
```

## Diagnostics To Log

Per run:

- run ID,
- model name,
- model repo,
- tokenizer name,
- strategy name,
- hash bits,
- prompt mode,
- channel,
- loss rate,
- sample ID,
- token count,
- missing token count,
- known token count,
- hash-guided missing token count,
- unguided missing token count,
- diffusion steps,
- block length,
- latency.

Per target position:

- original token ID,
- reconstructed token ID,
- state,
- hash value,
- candidate count,
- selected probability,
- margin,
- commit step,
- correct or incorrect.

Per denoising step:

- step index,
- still-masked count,
- committed count,
- average confidence,
- hash-guided committed count,
- unguided committed count.

## Comparisons To Make

### Core Method Comparison

Compare:

- no-LLM erasure,
- unprotected LLaDA,
- Hash8 LLaDA,
- Hash16 LLaDA.

Report:

- exact match,
- token recovery,
- latency,
- ambiguity.

### Hash Width Comparison

Compare:

- Hash4,
- Hash8,
- Hash16.

Expectations:

- Hash4 has more collisions and more language-model ambiguity,
- Hash16 has fewer collisions but higher overhead,
- Hash8 may be the best practical midpoint.

### Prompt Comparison

Compare:

- no prompt,
- instruction prompt,
- left-context prompt.

Report separately. Prompt/context changes the information available to the model and should not be mixed into the pure FEC result.

### Source Layout Comparison

Compare:

- contiguous packetization,
- wire interleaving,
- source round-robin chunks.

Hypothesis:

Masked diffusion should benefit from dispersed erasures because many small gaps may be easier than one long missing span.

### Step Count Comparison

Compare:

- 32 steps,
- 64 steps,
- 128 steps.

Report quality/latency tradeoff.

## Failure-Mode Questions

Use qualitative examples to answer:

- Does LLaDA make locally plausible but globally wrong substitutions?
- Does Hash16 improve exact match or simply reduce ambiguity?
- Do source-dispersed erasures improve token recovery while hurting exact match?
- Does the model rely more on right context than causal GenFEC could?
- Are errors concentrated around punctuation, whitespace, numbers, or rare words?

## Reporting Rules

Every figure/table should state:

- model,
- tokenizer,
- prompt mode,
- hash width,
- channel,
- loss rates,
- number of samples,
- whether semantic similarity was enabled.

Avoid comparing LLaDA token-level results directly to Qwen token-level results without noting the tokenizer difference.
