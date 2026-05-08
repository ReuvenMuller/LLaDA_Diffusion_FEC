# Reference Sources

This file tracks the external sources that guided the initial documentation.

## LLaDA 1.5

Primary model target:

- Hugging Face model: https://huggingface.co/GSAI-ML/LLaDA-1.5
- Model config: https://huggingface.co/GSAI-ML/LLaDA-1.5/raw/main/config.json
- Tokenizer config: https://huggingface.co/GSAI-ML/LLaDA-1.5/raw/main/tokenizer_config.json

Useful configuration values:

```text
model_type = llada
architecture = LLaDAModelLM
vocab_size = 126464
embedding_size = 126464
mask_token_id = 126336
eos_token_id = 126081
pad_token_id = 126081
max_sequence_length = 4096
use_cache = false
```

## Official LLaDA Generation Code

Repository:

- https://github.com/ML-GSAI/LLaDA

Official generator:

- https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

Key implementation ideas to preserve:

- initialize generation region with mask token IDs,
- keep prompt positions fixed,
- predict masked tokens with the model,
- compute selected-token confidence,
- commit only a subset of positions each step,
- use `low_confidence` remasking as the default policy,
- use `steps=128`, `block_length=32`, `temperature=0.0` as the starting config.

## Related Diffusion LLMs Considered

### Dream

- Repository: https://github.com/DreamLM/Dream
- Useful because it exposes token/logit hooks in generation demos.
- Good reference for hard suffix token control.
- Not the first target because the user chose LLaDA 1.5 and LLaDA is the cleaner research framing for masked diffusion.

### LLaDA-MoE

- Hugging Face model: https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct
- Interesting future target because it has lower active parameters than dense LLaDA-style models.
- Not first target because support and integration may be thinner than LLaDA 1.5.

### WeDLM

- Hugging Face model: https://huggingface.co/tencent/WeDLM-8B-Instruct
- Interesting future target because it is Qwen3-based and may preserve more tokenizer/ecosystem compatibility with prior GenFEC work.
- Not first target because the project direction is LLaDA-style masked diffusion.

## Internal Reference

Original project used for lessons only:

```text
C:\Users\reuve\OneDrive\Documents\GenFEC_Depth_Exam
```

Do not import code from this project directly unless explicitly choosing to port a small, tested idea.

Useful ideas from the original project:

- packet erasure channels,
- token hash maps,
- explicit packet/token position metadata,
- matched overhead accounting,
- JSONL detailed logs,
- smoke-test discipline,
- source-dispersed layouts,
- confidence and failure-mode reporting.

Implementation should remain independent.
