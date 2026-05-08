# Agent Handoff For Implementation

## Start Here

You are implementing a new, separate codebase:

```text
C:\Users\reuve\OneDrive\Documents\LLaDA_Diffusion_FEC
```

This project is intentionally separate from the original GenFEC repository. Do not import or modify the GenFEC codebase unless the user explicitly asks. You may use it only as conceptual reference.

Your first responsibility is to review the documentation in this project before writing code.

Recommended reading order:

1. `README.md`
2. `docs/project_overview.md`
3. `docs/lessons_from_genfec.md`
4. `docs/architecture.md`
5. `docs/llada_decoding_design.md`
6. `docs/implementation_plan.md`
7. `docs/experiment_plan.md`
8. `docs/development_notes.md`
9. `docs/roadmap.md`
10. `docs/research_runbook.md`
11. `docs/hash_profiles.md`
12. `docs/llada_server_model_notes.md`
13. `docs/reference_sources.md`

The most important implementation file is `docs/llada_decoding_design.md`. It defines the actual constrained denoising algorithm.

## Current State

The core framework is now implemented:

- core recovery dataclasses,
- source and wire packetization/interleaving,
- IID, burst, and Gilbert-Elliott packet loss,
- transmitted lookback-1 token-hash protection,
- tokenizer-specific persisted hash profiles,
- fake LLaDA-shaped constrained diffusion decoder,
- Hugging Face LLaDA adapter,
- opt-in real LLaDA smoke and micro-eval paths,
- model-only and model+hash synthetic micro-evals,
- XOR, LT/fountain, and streaming-window matched-overhead baselines,
- deterministic text sample loading,
- sweep orchestration,
- aggregate/report artifacts and failure-example extraction.

Default tests are model-free. Real LLaDA paths are opt-in and should be run on
the GPU server with loaded hash profiles.

## Before Executing

Before making code changes, create an implementation plan and share it with the user.

The plan should include:

- which phase from `docs/implementation_plan.md` you will implement first,
- which files/modules you expect to create,
- which tests you will add,
- what you will deliberately leave for a later phase,
- how you will verify the work without loading LLaDA unless necessary.

Do not jump directly into the real model integration. The intended path is:

1. core dataclasses,
2. packetization/channels,
3. hash maps,
4. fake-model constrained diffusion decoder,
5. real LLaDA adapter and smoke test.

## Project Goal

Build a research framework for token-erasure recovery using LLaDA-style masked diffusion.

The core recovery problem is:

```text
[known received token] [MASK] [MASK] [known received token] [MASK] [fixed suffix]
```

The decoder must:

- freeze received tokens,
- freeze optional prompt/context tokens,
- freeze optional suffix/delimiter tokens,
- update only erased positions,
- apply token-hash constraints as hard logit masks,
- restore fixed tokens after every denoising step,
- return normal token-level metrics and diagnostics.

## Core Algorithm

The first real decoder should adapt the official LLaDA generation loop.

At a high level:

```python
x, masks, fixed_token_ids = build_initial_tensor(plan, prompt_tokens)

for step in range(num_steps):
    logits = model(x).logits
    logits = apply_special_token_bans(logits)
    logits = apply_hash_bucket_masks(logits, plan, hash_map)

    proposals, confidence = select_tokens(logits)
    transfer_index = select_high_confidence_editable_positions(confidence)
    x[transfer_index] = proposals[transfer_index]

    x[masks.fixed_mask] = fixed_token_ids[masks.fixed_mask]
```

Known tokens and forced suffix tokens must never change. Hash-guided positions should only select token IDs whose hash bucket matches the received hash value, unless a fallback is explicitly logged.

## Initial Model Target

Use:

```text
GSAI-ML/LLaDA-1.5
```

Known config values:

```text
mask_token_id = 126336
eos_token_id = 126081
pad_token_id = 126081
vocab_size = 126464
max_sequence_length = 4096
```

The project should use Hugging Face `AutoModel` and `AutoTokenizer` with `trust_remote_code=True` when real model loading is added.

Do not make model loading part of the first unit-test path. Use fake-model tests first.

## Lessons To Preserve From GenFEC

Keep:

- explicit token-position metadata,
- separate source layout and wire interleaving,
- matched overhead accounting,
- small smoke tests,
- detailed JSONL logs,
- candidate ambiguity and confidence diagnostics,
- tokenizer-specific hash maps.

Avoid:

- causal LLM prompt-first decoding,
- llama.cpp-specific APIs,
- reusing Qwen hash profiles,
- embedding experiment matrices inside decoder code,
- large runs before correctness tests pass.

## Expected First Implementation Slice

A good first slice is Phase 1 plus part of Phase 2:

```text
src/diffusion_fec/
  __init__.py
  types.py
  coding/
    __init__.py
    packetizer.py
  channels/
    __init__.py
    random_loss.py
tests/
  test_types.py
  test_packetizer.py
  test_random_loss.py
```

This first slice should not load LLaDA.

Verification should prove:

- dataclasses serialize cleanly,
- packetization covers each token position exactly once,
- wire IDs are assigned deterministically,
- random loss is seeded and reproducible,
- reconstruction plans mark known and missing tokens correctly.

## Expected Second Implementation Slice

Add token hash maps and constraint-mask construction:

```text
src/diffusion_fec/coding/token_hash.py
src/diffusion_fec/decoding/constraints.py
tests/test_token_hash.py
tests/test_constraints.py
```

Verification should prove:

- every vocab token maps to a valid bucket,
- bucket masks have the expected shape,
- fixed masks preserve known tokens,
- hash-guided positions produce the correct allowed-token set.

## Expected Third Implementation Slice

Add a fake-model diffusion decoder:

```text
src/diffusion_fec/decoding/llada_diffusion.py
src/diffusion_fec/decoding/diagnostics.py
tests/test_fake_diffusion_decoder.py
```

Verification should prove:

- known tokens are never changed,
- suffix tokens are never changed,
- hash-guided positions only commit valid hash-bucket tokens,
- all editable positions are eventually filled,
- confidence/commit-step diagnostics are recorded.

Only after this should the real LLaDA adapter be implemented.

## Verification Discipline

Prefer fast local tests first:

```powershell
python -m pytest
```

When real model loading is added, keep those tests opt-in or slow-marked. The first real-model smoke test should be tiny:

```text
model = GSAI-ML/LLaDA-1.5
samples = 1 to 3
strategy = LLaDA_Hash8_NoPrompt
channel = random
loss_rate = 0.2
steps = 32 first, then 128
```

## Final Notes

This project is documentation-first on purpose. Let the docs drive the implementation.

If a design choice is unclear, prefer the simplest version that preserves the central invariant:

```text
fixed positions remain fixed; only erased positions denoise; token hashes are hard constraints
```

Keep the first implementation boring, tested, and small. The interesting research only becomes trustworthy once the basic mechanics are airtight.
