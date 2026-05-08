# Implementation Plan

This plan builds the project in small layers. Each layer should have tests before moving to the next.

## Phase 0: Documentation And Skeleton

Status: in progress.

Deliverables:

- README,
- architecture document,
- LLaDA decoding design,
- experiment plan,
- reference links,
- empty `src` and `tests` folders.

## Phase 1: Core Types And Utilities

Goal: represent the recovery problem without loading a model.

Implement:

- `diffusion_fec.types`
- `TokenSample`
- `Packet`
- `ReconstructionEntry`
- `ReconstructionPlan`
- `DecodingResult`
- `ConfidenceStat`
- `StepSummary`

Tests:

- dataclass construction,
- serialization to dict/JSON,
- simple reconstruction plan counts.

## Phase 2: Packetization And Channels

Goal: create reproducible erasure conditions.

Implement:

- contiguous packetization,
- optional source round-robin chunk layout,
- wire interleaving,
- random IID channel,
- burst channel,
- Gilbert-Elliott channel.

Tests:

- every token position appears exactly once in source packets,
- wire IDs are assigned deterministically,
- burst channel drops contiguous wire IDs,
- reconstruction plan labels known and missing positions correctly.

## Phase 3: LLaDA Tokenizer/Model Adapter

Goal: isolate model-specific APIs.

Implement:

- `LLaDAAdapter`,
- model loading,
- tokenizer loading,
- `tokenize`,
- `decode`,
- `forward`,
- constants from config.

Initial loading target:

```text
GSAI-ML/LLaDA-1.5
```

Tests:

- tokenizer round-trip smoke test,
- mask/eos/vocab values read from model config,
- optional model-load smoke test behind a slow marker.

## Phase 4: Uniform Token Hash Maps

Goal: create token-hash constraints for the LLaDA vocabulary.

Implement:

- uniform hash map builder,
- bucket candidate lists,
- bucket boolean masks,
- special-token ban mask,
- cache/save/load hash maps.

Hash input should include token ID and decoded token bytes:

```text
decoded_token_bytes + "::" + token_id + salt
```

Current implementation note: the LLaDA adapter's `decode_token(token_id)` prefers the
tokenizer-native token string from `convert_ids_to_tokens(token_id)`. This is acceptable
as long as encoding and decoding sides build hashes with the same adapter and salt.

Tests:

- hash IDs are in range,
- every token maps to one bucket,
- bucket masks have correct shape,
- special tokens can be excluded from candidate lists.

## Phase 5: Fake-Model Diffusion Decoder

Goal: test the constrained denoising loop without LLaDA.

Implement:

- `decode_masked_diffusion` with a fake model interface,
- fixed-token restoration,
- hash-bucket logit masking,
- low-confidence commit schedule,
- diagnostics.

Tests:

- known tokens never change,
- suffix tokens never change,
- hash-guided positions only select matching hash tokens,
- unguided positions can select any non-banned token,
- all editable positions eventually fill.

This phase is important. It proves the algorithm before model weight downloads enter the loop.

## Phase 6: LLaDA Diffusion Decoder

Goal: run real LLaDA recovery on tiny samples.

Implement:

- tensor construction from `ReconstructionPlan`,
- optional prompt/context prefix,
- official LLaDA-style block/step schedule,
- deterministic argmax,
- low-confidence remasking,
- result decoding,
- per-position confidence logs.

Default config:

```text
steps = 128
block_length = 32
temperature = 0.0
remasking = low_confidence
prompt_mode = none
```

Smoke command target:

```text
1 model, 3 samples, hash8, random loss 0.2
```

## Phase 7: Experiment Runner

Goal: produce CSV/JSONL artifacts.

Implement:

- dataset loading,
- strategy matrix,
- run IDs,
- manifest writing,
- results CSV,
- detailed JSONL logging,
- resume/skip completed runs.

Initial strategies:

- `LLaDA_Unprotected_NoPrompt`,
- `LLaDA_Hash8_NoPrompt`,
- `LLaDA_Hash16_NoPrompt`,
- `LLaDA_Hash8_InstructionPrompt`,
- `NoLLM_Erasure`.

Initial channels:

- random loss at `0.2`,
- burst loss at `0.2`.

## Phase 8: Baselines

Goal: avoid neural-only claims.

Implement:

- no-LLM erasure baseline,
- unprotected LLaDA fill,
- XOR parity,
- streaming-window parity.

Later:

- fountain-style baseline,
- parity plus diffusion reranking.

## Phase 9: Analysis Scripts

Goal: make results interpretable.

Implement:

- aggregate result tables,
- exact match by loss/channel/hash,
- token recovery by loss/channel/hash,
- latency summaries,
- candidate ambiguity summaries,
- qualitative example extraction.

## Suggested Initial Commands

After code exists, the first commands should look like:

```powershell
python -m diffusion_fec.experiments.runner --smoke-test
```

Then:

```powershell
python -m diffusion_fec.experiments.runner `
  --models LLaDA-1.5 `
  --strategies LLaDA_Hash8_NoPrompt LLaDA_Unprotected_NoPrompt `
  --channels random burst `
  --loss-rates 0.2 `
  --sample-limit 3 `
  --output-dir runs\llada_smoke_hash8
```

## First Success Criteria

The first prototype is successful when:

- known tokens are exactly preserved,
- all erased positions are filled,
- hash-guided predictions always satisfy token hash constraints,
- result rows and JSONL logs are written,
- a fake-model test suite passes,
- a tiny LLaDA run completes on GPU.

## Do Not Optimize Yet

Avoid these until the first prototype works:

- quantization,
- optimized hash profiles,
- custom CUDA kernels,
- multi-GPU scheduling,
- large experiment matrices,
- semantic scoring as a default.
