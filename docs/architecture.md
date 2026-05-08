# System Architecture

## High-Level Flow

```text
dataset sample
  -> tokenizer adapter
  -> token packetizer
  -> protection encoder
  -> packet-erasure channel
  -> reconstruction plan
  -> LLaDA diffusion decoder
  -> metrics and logs
```

The architecture should be modular enough that the LLaDA decoder can be tested without channels, and the channel/protection harness can be tested without loading the model.

## Package Layout

Planned package structure:

```text
src/
  diffusion_fec/
    __init__.py
    config.py
    types.py
    models/
      __init__.py
      base.py
      llada.py
    coding/
      __init__.py
      packetizer.py
      token_hash.py
      protection.py
      baselines.py
    decoding/
      __init__.py
      llada_diffusion.py
      constraints.py
      diagnostics.py
    channels/
      __init__.py
      random_loss.py
      burst_loss.py
      gilbert_elliott.py
    metrics/
      __init__.py
      token_metrics.py
      semantic_metrics.py
    experiments/
      __init__.py
      runner.py
      strategies.py
      logging.py
```

## Core Data Types

The implementation should begin with a small set of explicit dataclasses.

### TokenSample

Represents one clean source message after tokenization.

Fields:

- `sample_id`
- `text`
- `token_ids`
- `tokenizer_name`

### Packet

Represents a transmitted packet.

Fields:

- `source_id`
- `wire_id`
- `kind`
- `token_ids`
- `token_positions`
- `metadata`

`token_positions` must always be absolute positions in the original target token sequence.

### ReconstructionEntry

Represents one target token position after channel loss.

Fields:

- `position`
- `state`: `known`, `missing`, or `unguided`
- `token_id`: set for known positions
- `hash_value`: set when hash metadata exists for a missing position
- `fixed`: true for positions that must never change

### ReconstructionPlan

Represents the complete target sequence state.

Fields:

- `entries`
- `total_tokens`
- `known_count`
- `missing_count`
- `hash_guided_count`
- `unguided_count`

### DecodingResult

Represents model output and diagnostics.

Fields:

- `reconstructed_text`
- `reconstructed_tokens`
- `decode_latency_sec`
- `steps`
- `fixed_token_count`
- `editable_token_count`
- `hash_guided_token_count`
- `confidence_stats`
- `step_summaries`
- `diagnostics`

## Model Adapter

The model adapter isolates Hugging Face/LLaDA details from experiment code.

Required methods:

```python
class MaskedDiffusionModel:
    @property
    def device(self): ...

    @property
    def mask_token_id(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str: ...

    def forward(self, input_ids, attention_mask=None): ...
```

First adapter:

```text
GSAI-ML/LLaDA-1.5 via transformers AutoModel and AutoTokenizer
```

The adapter should load with:

```python
AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-1.5",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-1.5",
    trust_remote_code=True,
)
```

## Coding Layer

The coding layer builds the same abstract recovery problem regardless of model.

### Packetizer

Responsibilities:

- split target tokens into packets,
- preserve absolute token positions,
- optionally use contiguous or source-dispersed layouts,
- assign wire IDs,
- expose packet count and token coverage.

### Token Hash Map

Responsibilities:

- build a hash bucket for every LLaDA vocabulary token,
- support 4-bit, 8-bit, and 16-bit hash widths,
- avoid special/control tokens where appropriate,
- provide fast boolean masks or candidate ID lists per bucket.

Start with uniform CRC-style maps.

Hash-map construction must use one consistent tokenizer-derived token representation on
both the sender and receiver. The current adapter uses the tokenizer-native token string
from `convert_ids_to_tokens(token_id)` plus the numeric token ID and salt. If that changes
to decoded text bytes later, both sides must be regenerated together.

Later add optimized/frequency-aware maps after the base LLaDA prototype works.

### Protection Encoder

First strategy:

```text
single-token-hash lookback
```

For each data packet, attach hash metadata for a nearby source packet. The default can mirror GenFEC's lookback-1 design, but this should be configurable.

Later strategies:

- source-dispersed hash protection,
- direct per-packet checksum,
- parity plus diffusion reranking,
- fixed delimiter or suffix protection.

## Channel Layer

Channels operate on packets in wire order and return surviving/dropped packet lists.

Initial channels:

- random IID packet loss,
- contiguous burst packet loss,
- Gilbert-Elliott burst-state loss.

The first implementation can port the channel logic conceptually, but should use fresh code and tests.

## Decoding Layer

The decoder receives:

- model adapter,
- reconstruction plan,
- optional prompt/context token IDs,
- hash map,
- decoding config.

It constructs:

```text
[optional prompt/context tokens] [target recovery tokens]
```

Prompt/context positions are fixed. Known target positions are fixed. Missing target positions are initialized to the LLaDA mask token.

The decoder then runs a constrained denoising loop.

## Metrics Layer

Initial metrics:

- exact token sequence match,
- token edit distance,
- normalized token edit distance,
- lost-position token recovery rate,
- optional semantic similarity.

Semantic similarity should be optional in smoke tests because it can dominate setup time.

## Experiment Runner

The experiment runner should:

- load dataset,
- load model once per model config,
- build strategy matrix,
- run all sample/channel/strategy combinations,
- write results incrementally,
- write detailed JSONL events,
- save run manifest before the first model call.

Keep the runner thin. It should orchestrate modules, not implement algorithms inline.

## Testing Strategy

Tests should start without model loading.

Required early tests:

- packet positions survive contiguous packetization,
- source-dispersed layout covers every position exactly once,
- burst channel drops expected wire positions,
- reconstruction plans label known/missing/unguided correctly,
- hash maps return valid bucket IDs,
- constraint masks preserve known tokens,
- a fake diffusion model can exercise the denoising loop deterministically.

Model-dependent tests should be opt-in.
