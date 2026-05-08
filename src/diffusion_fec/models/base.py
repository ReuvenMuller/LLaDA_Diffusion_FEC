"""Shared interface for masked-diffusion model adapters."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class MaskedDiffusionModel(Protocol):
    """Minimal model contract used by the constrained decoder."""

    @property
    def device(self) -> Any:
        """Device-like value where model inputs should be placed."""

    @property
    def mask_token_id(self) -> int:
        """Token ID used to mark editable erased positions."""

    @property
    def eos_token_id(self) -> int | None:
        """End-of-text token ID, when known."""

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID, when known."""

    @property
    def vocab_size(self) -> int:
        """Vocabulary size expected by model logits."""

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenize text into model token IDs."""

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode model token IDs into text."""

    def decode_token(self, token_id: int) -> str | bytes:
        """Decode one token ID for tokenizer-specific hash construction."""

    def forward(self, input_ids, attention_mask=None) -> Any:
        """Run a batch-first forward pass and return an object exposing logits."""
