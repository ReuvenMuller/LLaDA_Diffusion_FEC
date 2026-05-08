"""Hugging Face adapter for GSAI-ML/LLaDA-1.5."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import inspect
from typing import Any

from diffusion_fec.decoding.llada_diffusion import DiffusionDecodingConfig


LLADA_1_5_MODEL_ID = "GSAI-ML/LLaDA-1.5"
LLADA_1_5_DEFAULT_MASK_TOKEN_ID = 126336
LLADA_1_5_DEFAULT_EOS_TOKEN_ID = 126081
LLADA_1_5_DEFAULT_PAD_TOKEN_ID = 126081
LLADA_1_5_DEFAULT_VOCAB_SIZE = 126464
LLADA_1_5_DEFAULT_MAX_SEQUENCE_LENGTH = 4096


@dataclass
class LLaDAAdapter:
    """Adapter that presents LLaDA through the project model interface."""

    tokenizer: Any
    model: Any | None = None
    model_id: str = LLADA_1_5_MODEL_ID
    model_config: Any | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = LLADA_1_5_MODEL_ID,
        *,
        load_model: bool = True,
        config_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> "LLaDAAdapter":
        """Load tokenizer and optionally model weights from Hugging Face."""

        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install Hugging Face dependencies with `pip install -e .[hf]` "
                "before loading LLaDA."
            ) from exc

        config_options = {"trust_remote_code": True}
        config_options.update(config_kwargs or {})
        config = AutoConfig.from_pretrained(model_id, **config_options)
        _ensure_llada_config_defaults(config)
        tokenizer_options = {"trust_remote_code": True}
        tokenizer_options.update(tokenizer_kwargs or {})
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_options)

        model = None
        model_config = config
        if load_model:
            load_options = {"trust_remote_code": True}
            load_options.update(_default_model_kwargs())
            load_options["config"] = config
            load_options.update(model_kwargs or {})
            model_config = load_options.get("config", config)
            _ensure_llada_config_defaults(model_config)
            _patch_remote_llada_model_class(
                config=model_config,
                model_id=model_id,
                local_files_only=bool(load_options.get("local_files_only", False)),
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_options)
            eval_method = getattr(model, "eval", None)
            if callable(eval_method):
                eval_method()

        return cls(
            tokenizer=tokenizer,
            model=model,
            model_id=model_id,
            model_config=model_config,
        )

    @property
    def device(self) -> Any:
        if self.model is None:
            return "cpu"
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            try:
                return next(parameters()).device
            except StopIteration:
                return "cpu"
        return "cpu"

    @property
    def mask_token_id(self) -> int:
        return int(self._config_value("mask_token_id", LLADA_1_5_DEFAULT_MASK_TOKEN_ID))

    @property
    def eos_token_id(self) -> int | None:
        value = _first_non_none(
            self._config_value("eos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            LLADA_1_5_DEFAULT_EOS_TOKEN_ID,
        )
        return None if value is None else int(value)

    @property
    def pad_token_id(self) -> int | None:
        value = _first_non_none(
            self._config_value("pad_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
            LLADA_1_5_DEFAULT_PAD_TOKEN_ID,
        )
        return None if value is None else int(value)

    @property
    def vocab_size(self) -> int:
        value = _first_non_none(
            self._config_value("vocab_size", None),
            len(self.tokenizer) if hasattr(self.tokenizer, "__len__") else None,
            getattr(self.tokenizer, "vocab_size", None),
            LLADA_1_5_DEFAULT_VOCAB_SIZE,
        )
        return int(value)

    @property
    def max_sequence_length(self) -> int | None:
        value = _first_non_none(
            self._config_value("max_sequence_length", None),
            getattr(self.tokenizer, "model_max_length", None),
            LLADA_1_5_DEFAULT_MAX_SEQUENCE_LENGTH,
        )
        return None if value is None else int(value)

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
        )
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids]

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def decode_token(self, token_id: int) -> str:
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token_id must be in range [0, {self.vocab_size})")
        convert = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if callable(convert):
            token = convert(int(token_id))
            if token is not None:
                return str(token)
        return self.decode([int(token_id)], skip_special_tokens=False)

    def forward(self, input_ids, attention_mask=None):
        """Run a LLaDA forward pass using batch-first Python lists or tensors."""

        if self.model is None:
            raise RuntimeError("LLaDAAdapter.forward requires load_model=True")

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Install Hugging Face dependencies with `pip install -e .[hf]` "
                "before running model forward."
            ) from exc

        input_tensor = _as_torch_long_tensor(input_ids, device=self.device, torch_module=torch)
        attention_tensor = None
        if attention_mask is not None:
            attention_tensor = _as_torch_long_tensor(
                attention_mask,
                device=self.device,
                torch_module=torch,
            )
        with torch.no_grad():
            return self.model(input_ids=input_tensor, attention_mask=attention_tensor)

    def decoding_config(
        self,
        *,
        steps: int = 128,
        block_length: int = 32,
        banned_token_ids: Sequence[int] = (),
        fallback_on_empty_hash_bucket: bool = True,
    ) -> DiffusionDecodingConfig:
        """Build decoder config from loaded tokenizer/model constants."""

        return DiffusionDecodingConfig(
            mask_token_id=self.mask_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vocab_size=self.vocab_size,
            steps=steps,
            block_length=block_length,
            banned_token_ids=tuple(banned_token_ids),
            fallback_on_empty_hash_bucket=fallback_on_empty_hash_bucket,
        )

    def _config_value(self, name: str, default: Any) -> Any:
        for source in (
            getattr(self.model, "config", None),
            self.model_config,
        ):
            if source is not None and hasattr(source, name):
                return getattr(source, name)
        return default


def _default_model_kwargs() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    return {"torch_dtype": torch.bfloat16}


def _patch_remote_llada_model_class(
    *,
    config: Any,
    model_id: str,
    local_files_only: bool,
) -> None:
    """Patch older remote LLaDA code for newer Transformers loaders."""

    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel")
    if not class_ref:
        return

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        return

    model_class = get_class_from_dynamic_module(
        class_ref,
        model_id,
        local_files_only=local_files_only,
    )
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}

    tie_weights = getattr(model_class, "tie_weights", None)
    if not callable(tie_weights) or _accepts_transformers_tie_kwargs(tie_weights):
        return
    if getattr(tie_weights, "_diffusion_fec_compat_wrapper", False):
        return

    original_tie_weights = tie_weights

    def tie_weights_compat(self, *args, **kwargs):
        return original_tie_weights(self)

    tie_weights_compat._diffusion_fec_compat_wrapper = True
    model_class.tie_weights = tie_weights_compat


def _ensure_llada_config_defaults(config: Any) -> None:
    if config is not None and not hasattr(config, "use_cache"):
        config.use_cache = False


def _accepts_transformers_tie_kwargs(tie_weights: Any) -> bool:
    try:
        signature = inspect.signature(tie_weights)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return True
    return {
        "missing_keys",
        "recompute_mapping",
    }.issubset(signature.parameters)


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_torch_long_tensor(value, *, device: Any, torch_module):
    if hasattr(value, "to") and hasattr(value, "dtype"):
        return value.to(device=device, dtype=torch_module.long)
    return torch_module.tensor(value, dtype=torch_module.long, device=device)
