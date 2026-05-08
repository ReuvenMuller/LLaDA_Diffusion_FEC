# LLaDA Decoding Design

## Design Goal

Implement GenFEC-style token-hash recovery using LLaDA 1.5 as a masked-diffusion model.

The decoder should not ask LLaDA to print a reconstructed string. It should give LLaDA a token tensor where erased positions are `<|mdm_mask|>`, then constrain the denoising loop.

## LLaDA Generation Loop

The official LLaDA generator follows this shape:

```python
x = full(prompt_tokens + mask_tokens)
prompt_index = x != mask_id

for block in blocks:
    for step in steps:
        logits = model(x).logits
        x0 = argmax(add_gumbel_noise(logits))
        confidence = probability_of_selected_token(logits, x0)
        transfer_index = top_confidence_masked_positions(confidence)
        x[transfer_index] = x0[transfer_index]
```

This project adds three pieces:

1. fixed-token restoration,
2. hash-bucket logit masking,
3. recovery diagnostics.

## Tensor Layout

There are two supported layouts.

### No-Prompt Layout

This is the cleanest scientific baseline.

```text
[target token 0] [MASK] [target token 2] [MASK] ...
```

Input length equals target token length.

Use this as the primary recovery result unless prompt/context is explicitly being evaluated.

### Prompt/Context Layout

This is an ablation for using natural-language or left-context side information.

```text
[prompt/context tokens] [target known/mask/fixed tokens]
```

The prompt/context span is fixed. Metrics score only the target span.

## Masks

The decoder should construct these boolean masks over the full input tensor:

- `prompt_mask`: prompt/context positions, always fixed,
- `known_mask`: received target tokens, always fixed,
- `editable_mask`: erased target positions that may be denoised,
- `hash_guided_mask`: editable positions with token-hash metadata,
- `unguided_mask`: editable positions without token-hash metadata,
- `target_mask`: positions belonging to the target message,
- `fixed_mask`: `prompt_mask | known_mask | forced_suffix_mask`.

The fixed token IDs are stored separately:

```python
fixed_token_ids = x.clone()
```

At every denoising step:

```python
x[fixed_mask] = fixed_token_ids[fixed_mask]
```

This is the hard-constraint guarantee.

## Hash-Bucket Logit Masking

For a missing position with hash value `h`, only tokens whose hash bucket equals `h` are valid.

Naive implementation:

```python
bucket_mask = hash_map == h
logits[position, ~bucket_mask] = -inf
```

Efficient implementation:

Precompute bucket masks:

```python
bucket_allowed[bucket_id] -> bool tensor of shape [vocab_size]
```

Then apply per position:

```python
allowed = bucket_allowed[hash_value]
logits[batch, position, ~allowed] = -inf
```

Special tokens should also be banned unless explicitly fixed:

- mask token,
- padding token,
- end-of-text token during normal recovery,
- reserved/control tokens,
- empty or placeholder-like tokens if they cause decode artifacts.

## Candidate Selection

Initial deterministic policy:

```python
proposal = argmax(filtered_logits)
confidence = softmax(filtered_logits)[proposal]
```

This matches the deterministic GenFEC style and makes results reproducible.

Later stochastic policy:

```python
proposal = sample(filtered_logits, temperature)
```

Stochastic sampling should only be used for candidate generation or reranking experiments, not the first baseline.

## Remasking / Commit Policy

Use LLaDA's low-confidence remasking policy first.

At each step, commit a fixed number of still-masked positions:

```python
num_transfer_tokens = get_num_transfer_tokens(mask_index, steps)
select highest confidence masked positions
x[transfer_index] = proposal[transfer_index]
```

Important modification:

Only editable target positions are eligible for transfer. Prompt and known positions are never eligible.

```python
eligible = editable_mask & (x == mask_id)
confidence = where(eligible, confidence, -inf)
```

## Full Algorithm

```python
def decode_llada_hash_diffusion(model, plan, hash_map, config, prompt_tokens=None):
    x, masks, fixed_token_ids = build_initial_tensor(plan, prompt_tokens)
    bucket_allowed = build_bucket_allowed_masks(hash_map, config.hash_bits)
    banned_mask = build_banned_token_mask(model.tokenizer)

    for block in range(num_blocks):
        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps_per_block)

        for step in range(steps_per_block):
            logits = model.forward(x, attention_mask=attention_mask).logits

            logits = apply_special_token_bans(logits, banned_mask, masks.editable_mask)
            logits = apply_hash_constraints(logits, plan, bucket_allowed, masks.hash_guided_mask)

            proposal, confidence, candidate_count = select_proposals(logits)
            confidence = restrict_transfer_confidence(confidence, x, masks.editable_mask)

            transfer_index = select_transfer_positions(confidence, num_transfer_tokens[step])
            x[transfer_index] = proposal[transfer_index]

            x[masks.fixed_mask] = fixed_token_ids[masks.fixed_mask]
            record_step_diagnostics(...)

    return build_decoding_result(x, masks.target_mask)
```

## Confidence Statistics

Record per target position:

- position,
- state: known, hash_guided, unguided,
- selected token ID,
- top-1 probability,
- top-2 probability,
- margin,
- candidate count after masks,
- commit step,
- was fixed from input,
- was restored after model proposal,
- hash value if present.

For positions committed early, confidence should come from the step where the token was committed.

For known positions, confidence is `1.0`.

## Important Edge Cases

### No Missing Positions

Return the known target tokens immediately.

### Hash Bucket Empty After Banning

Fallback options:

1. allow banned special tokens only if the hash bucket contains no normal tokens,
2. use unguided logits for that position,
3. emit a fixed fallback token.

Default first prototype:

```text
fall back to unguided logits and log hash_bucket_empty=true
```

### Sequence Too Long

LLaDA 1.5 has a 4096-token maximum sequence length.

For longer sequences:

- split into windows,
- overlap windows if needed,
- score only target positions,
- log truncation/windowing.

First prototype should keep samples below the context limit.

### Prompt Length Consumes Context

If using prompt/context layout, the target plus prompt must fit.

Default priority:

1. preserve target,
2. include short instruction,
3. include left context only if space remains.

### Fixed Suffix

Suffix constraints are just known fixed positions at the end:

```text
[known prefix] [MASK] [MASK] [fixed suffix]
```

Use the same `fixed_mask` restoration mechanism.

## Expected Advantages Over Causal GenFEC

The LLaDA decoder should be better aligned with:

- arbitrary missing spans,
- source-dispersed losses,
- fixed suffixes,
- multiple separated gaps,
- future parity/hash global search,
- direct uncertainty tracking over all erased positions.

## Expected Risks

Risks to test early:

- tokenizer mismatch makes comparison to Qwen results indirect,
- LLaDA may underperform strong causal Qwen models on raw language modeling,
- full forward passes over the whole sequence can be expensive,
- hash masks may overconstrain if tokenization boundaries differ from semantic units,
- deterministic argmax may reduce diversity needed to escape early wrong commits.
