import os

import pytest

from diffusion_fec.models.llada import LLADA_1_5_MODEL_ID, LLaDAAdapter


pytestmark = [pytest.mark.hf, pytest.mark.slow]


def test_llada_real_model_forward_smoke_opt_in() -> None:
    if os.environ.get("RUN_LLADA_MODEL_SMOKE") != "1":
        pytest.skip("set RUN_LLADA_MODEL_SMOKE=1 to load LLaDA model weights")

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    allow_cpu = os.environ.get("RUN_LLADA_MODEL_SMOKE_ALLOW_CPU") == "1"
    if not torch.cuda.is_available() and not allow_cpu:
        pytest.skip(
            "CUDA is not available; set RUN_LLADA_MODEL_SMOKE_ALLOW_CPU=1 to try CPU"
        )

    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    adapter = LLaDAAdapter.from_pretrained(
        LLADA_1_5_MODEL_ID,
        load_model=True,
        model_kwargs=model_kwargs,
    )
    if torch.cuda.is_available() and adapter.model is not None:
        adapter.model.to("cuda")
    input_ids = [[adapter.mask_token_id]]
    output = adapter.forward(input_ids, attention_mask=[[1]])

    logits = output.logits
    assert logits.shape[0] == 1
    assert logits.shape[1] == 1
    assert logits.shape[2] == adapter.vocab_size
