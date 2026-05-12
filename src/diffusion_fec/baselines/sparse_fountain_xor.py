"""Sparse fountain-style XOR repair over absolute token positions."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, floor
from typing import Any

from diffusion_fec.baselines.overhead import OverheadSummary, token_bit_width_for_vocab
from diffusion_fec.coding.packetizer import (
    WIRE_INTERLEAVING_MATRIX,
    SourceLayoutConfig,
    WireInterleavingConfig,
    packetize_sample,
)
from diffusion_fec.types import Packet, TokenSample


SPARSE_FOUNTAIN_XOR_SCHEME = "sparse_fountain_xor"
SPARSE_FOUNTAIN_XOR_METADATA_KEY = "sparse_fountain_xor"
DEFAULT_SPARSE_XOR_RANDOM_SEED = 7
DEFAULT_SPARSE_XOR_DEGREE_DISTRIBUTION = ((2, 0.5), (3, 0.35), (4, 0.15))
DEFAULT_SPARSE_XOR_MAX_COVERAGE_DEGREE = 8


@dataclass(frozen=True)
class SparseFountainXorConfig:
    """Seeded sparse XOR equation configuration."""

    xor_overhead_bits_per_token: float = 4.0
    vocab_size: int = 2
    random_seed: int = DEFAULT_SPARSE_XOR_RANDOM_SEED
    coverage_enabled: bool = True
    degree_distribution: tuple[tuple[int, float], ...] = field(
        default_factory=lambda: DEFAULT_SPARSE_XOR_DEGREE_DISTRIBUTION
    )
    max_coverage_degree: int = DEFAULT_SPARSE_XOR_MAX_COVERAGE_DEGREE

    def __post_init__(self) -> None:
        if self.xor_overhead_bits_per_token < 0.0:
            raise ValueError("xor_overhead_bits_per_token must be non-negative")
        if not isinstance(self.vocab_size, int):
            raise TypeError("vocab_size must be an int")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an int")
        if not isinstance(self.max_coverage_degree, int):
            raise TypeError("max_coverage_degree must be an int")
        if self.max_coverage_degree <= 0:
            raise ValueError("max_coverage_degree must be positive")
        distribution = _normalize_degree_distribution(self.degree_distribution)
        object.__setattr__(self, "degree_distribution", distribution)

    @property
    def token_bit_width(self) -> int:
        return token_bit_width_for_vocab(self.vocab_size)

    @property
    def target_overhead_ratio(self) -> float:
        return self.xor_overhead_bits_per_token / self.token_bit_width

    def repair_token_budget(self, total_tokens: int) -> int:
        if total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        return floor(total_tokens * self.target_overhead_ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": SPARSE_FOUNTAIN_XOR_SCHEME,
            "xor_overhead_bits_per_token": self.xor_overhead_bits_per_token,
            "vocab_size": self.vocab_size,
            "token_bit_width": self.token_bit_width,
            "target_overhead_ratio": self.target_overhead_ratio,
            "random_seed": self.random_seed,
            "coverage_enabled": self.coverage_enabled,
            "degree_distribution": [
                {"degree": degree, "weight": weight}
                for degree, weight in self.degree_distribution
            ],
            "max_coverage_degree": self.max_coverage_degree,
        }


@dataclass(frozen=True)
class SparseEquationSpec:
    """Deterministic sparse equation graph entry."""

    equation_index: int
    positions: tuple[int, ...]
    source: str

    @property
    def degree(self) -> int:
        return len(self.positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation_index": self.equation_index,
            "positions": list(self.positions),
            "degree": self.degree,
            "source": self.source,
        }


@dataclass(frozen=True)
class SparseFountainDiagnostics:
    """Coverage and budget diagnostics for one sparse encoding."""

    repair_token_budget: int
    equation_count: int
    actual_repair_token_overhead_ratio: float
    budget_exhausted: bool
    coverage_enabled: bool
    coverage_possible: bool
    coverage_pass_degree: int
    coverage_zero_count: int
    coverage_min: int
    coverage_mean: float
    actual_mean_degree: float
    degree_histogram: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_token_budget": self.repair_token_budget,
            "equation_count": self.equation_count,
            "actual_repair_token_overhead_ratio": self.actual_repair_token_overhead_ratio,
            "budget_exhausted": self.budget_exhausted,
            "coverage_enabled": self.coverage_enabled,
            "coverage_possible": self.coverage_possible,
            "coverage_pass_degree": self.coverage_pass_degree,
            "coverage_zero_count": self.coverage_zero_count,
            "coverage_min": self.coverage_min,
            "coverage_mean": self.coverage_mean,
            "actual_mean_degree": self.actual_mean_degree,
            "degree_histogram": dict(sorted(self.degree_histogram.items())),
        }


@dataclass(frozen=True)
class SparseFountainXorEncoded:
    """Encoded source packets plus sparse repair equations."""

    packets: tuple[Packet, ...]
    source_packets: tuple[Packet, ...]
    parity_packets: tuple[Packet, ...]
    overhead: OverheadSummary
    config: SparseFountainXorConfig
    equation_specs: tuple[SparseEquationSpec, ...]
    diagnostics: SparseFountainDiagnostics

    @property
    def source_packet_count(self) -> int:
        return len(self.source_packets)

    @property
    def extra_packet_count(self) -> int:
        return len(self.parity_packets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "packets": [packet.to_dict() for packet in self.packets],
            "source_packets": [packet.to_dict() for packet in self.source_packets],
            "parity_packets": [packet.to_dict() for packet in self.parity_packets],
            "source_packet_count": self.source_packet_count,
            "extra_packet_count": self.extra_packet_count,
            "overhead": self.overhead.to_dict(),
            "config": self.config.to_dict(),
            "equation_specs": [spec.to_dict() for spec in self.equation_specs],
            "diagnostics": self.diagnostics.to_dict(),
        }


def parse_degree_distribution(value: str | Mapping[int, float] | Sequence[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    """Parse degree distribution config from CLI-style text or mappings."""

    if isinstance(value, str):
        items: list[tuple[int, float]] = []
        for item in value.split(","):
            raw = item.strip()
            if not raw:
                continue
            if ":" not in raw:
                raise ValueError("degree distribution entries must look like degree:weight")
            degree_text, weight_text = raw.split(":", 1)
            items.append((int(degree_text), float(weight_text)))
        return _normalize_degree_distribution(items)
    if isinstance(value, Mapping):
        return _normalize_degree_distribution(tuple((int(k), float(v)) for k, v in value.items()))
    return _normalize_degree_distribution(value)


def build_sparse_equation_specs(
    *,
    total_tokens: int,
    config: SparseFountainXorConfig,
) -> tuple[SparseEquationSpec, ...]:
    """Build the deterministic sparse equation graph from shared config."""

    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if total_tokens == 0:
        return ()
    budget = config.repair_token_budget(total_tokens)
    if budget <= 0:
        return ()

    rng = random.Random(config.random_seed)
    specs: list[SparseEquationSpec] = []
    used_position_sets: set[tuple[int, ...]] = set()

    if config.coverage_enabled:
        target_degree = _coverage_pass_degree(total_tokens=total_tokens, budget=budget, config=config)
        shuffled_positions = list(range(total_tokens))
        rng.shuffle(shuffled_positions)
        cursor = 0
        while cursor < total_tokens and len(specs) < budget:
            raw_positions = shuffled_positions[cursor:cursor + target_degree]
            cursor += target_degree
            if len(raw_positions) < target_degree and len(raw_positions) < total_tokens:
                fillers = [position for position in range(total_tokens) if position not in raw_positions]
                rng.shuffle(fillers)
                raw_positions.extend(fillers[: target_degree - len(raw_positions)])
            _append_spec(
                specs=specs,
                used_position_sets=used_position_sets,
                positions=raw_positions,
                source="coverage",
            )

    while len(specs) < budget:
        degree = _sample_degree(config=config, rng=rng, total_tokens=total_tokens)
        positions = _sample_unique_position_set(
            rng=rng,
            total_tokens=total_tokens,
            degree=degree,
            used_position_sets=used_position_sets,
        )
        _append_spec(
            specs=specs,
            used_position_sets=used_position_sets,
            positions=positions,
            source="random",
        )

    return tuple(specs)


def encode_sparse_fountain_xor(
    sample: TokenSample,
    *,
    tokens_per_packet: int,
    config: SparseFountainXorConfig,
    source_layout: SourceLayoutConfig | None = None,
    wire_interleaving: WireInterleavingConfig | None = None,
) -> SparseFountainXorEncoded:
    """Encode source packets plus sparse fountain-style XOR repair packets."""

    source_layout = source_layout or SourceLayoutConfig()
    wire_interleaving = wire_interleaving or WireInterleavingConfig()
    source_packets = packetize_sample(
        sample,
        tokens_per_packet=tokens_per_packet,
        source_layout=source_layout,
        wire_interleaving=WireInterleavingConfig(),
    )
    specs = build_sparse_equation_specs(
        total_tokens=len(sample.token_ids),
        config=config,
    )
    parity_packets = tuple(
        _repair_packet_for_spec(
            sample=sample,
            spec=spec,
            config=config,
            source_packet_count=len(source_packets),
        )
        for spec in specs
    )
    packets = _assign_wire_ids(
        packets=[*source_packets, *parity_packets],
        wire_interleaving=wire_interleaving,
    )
    source_count = len(source_packets)
    source_packets = tuple(packets[:source_count])
    parity_packets = tuple(packets[source_count:])
    diagnostics = _diagnostics(
        total_tokens=len(sample.token_ids),
        specs=specs,
        config=config,
    )
    return SparseFountainXorEncoded(
        packets=tuple(packets),
        source_packets=source_packets,
        parity_packets=parity_packets,
        overhead=_overhead_summary(
            total_tokens=len(sample.token_ids),
            equation_count=len(specs),
            repair_token_budget=diagnostics.repair_token_budget,
            config=config,
        ),
        config=config,
        equation_specs=specs,
        diagnostics=diagnostics,
    )


def sparse_fountain_diagnostics_for_specs(
    *,
    total_tokens: int,
    specs: Sequence[SparseEquationSpec],
    config: SparseFountainXorConfig,
) -> SparseFountainDiagnostics:
    """Return diagnostics for reconstructed sparse equation specs."""

    return _diagnostics(total_tokens=total_tokens, specs=tuple(specs), config=config)


def _repair_packet_for_spec(
    *,
    sample: TokenSample,
    spec: SparseEquationSpec,
    config: SparseFountainXorConfig,
    source_packet_count: int,
) -> Packet:
    parity_value = 0
    for position in spec.positions:
        parity_value ^= sample.token_ids[position]
    return Packet(
        source_id=sample.sample_id,
        wire_id=source_packet_count + spec.equation_index,
        kind=SPARSE_FOUNTAIN_XOR_SCHEME,
        token_ids=(parity_value,),
        token_positions=(spec.equation_index,),
        metadata={
            SPARSE_FOUNTAIN_XOR_METADATA_KEY: {
                "scheme": SPARSE_FOUNTAIN_XOR_SCHEME,
                "equation_index": spec.equation_index,
                "positions": list(spec.positions),
                "degree": spec.degree,
                "equation_source": spec.source,
                "random_seed": config.random_seed,
                "config": config.to_dict(),
                "positions_are_artifact_audit_metadata": True,
            }
        },
    )


def _diagnostics(
    *,
    total_tokens: int,
    specs: tuple[SparseEquationSpec, ...],
    config: SparseFountainXorConfig,
) -> SparseFountainDiagnostics:
    budget = config.repair_token_budget(total_tokens)
    coverage = [0 for _ in range(total_tokens)]
    for spec in specs:
        for position in spec.positions:
            coverage[position] += 1
    degree_histogram = Counter(spec.degree for spec in specs)
    coverage_min = min(coverage) if coverage else 0
    coverage_mean = (sum(coverage) / total_tokens) if total_tokens else 0.0
    actual_mean_degree = (
        sum(spec.degree for spec in specs) / len(specs)
        if specs
        else 0.0
    )
    return SparseFountainDiagnostics(
        repair_token_budget=budget,
        equation_count=len(specs),
        actual_repair_token_overhead_ratio=(len(specs) / total_tokens if total_tokens else 0.0),
        budget_exhausted=(budget > 0 and len(specs) >= budget),
        coverage_enabled=config.coverage_enabled,
        coverage_possible=_coverage_possible(total_tokens=total_tokens, budget=budget, config=config),
        coverage_pass_degree=_coverage_pass_degree(total_tokens=total_tokens, budget=budget, config=config),
        coverage_zero_count=sum(1 for count in coverage if count == 0),
        coverage_min=coverage_min,
        coverage_mean=coverage_mean,
        actual_mean_degree=actual_mean_degree,
        degree_histogram=dict(degree_histogram),
    )


def _overhead_summary(
    *,
    total_tokens: int,
    equation_count: int,
    repair_token_budget: int,
    config: SparseFountainXorConfig,
) -> OverheadSummary:
    return OverheadSummary(
        target_hash_bits=None,
        vocab_size=config.vocab_size,
        token_bit_width=config.token_bit_width,
        target_overhead_ratio=config.target_overhead_ratio,
        repair_packet_count=equation_count,
        repair_token_budget=repair_token_budget,
        actual_repair_token_overhead_ratio=(equation_count / total_tokens if total_tokens else 0.0),
        metadata_bits_total=0,
    )


def _coverage_pass_degree(
    *,
    total_tokens: int,
    budget: int,
    config: SparseFountainXorConfig,
) -> int:
    if total_tokens <= 0 or budget <= 0:
        return 0
    return min(
        total_tokens,
        config.max_coverage_degree,
        max(1, 2 if total_tokens > 1 else 1, ceil(total_tokens / budget)),
    )


def _coverage_possible(
    *,
    total_tokens: int,
    budget: int,
    config: SparseFountainXorConfig,
) -> bool:
    if total_tokens == 0:
        return True
    if not config.coverage_enabled or budget <= 0:
        return False
    return budget * min(total_tokens, config.max_coverage_degree) >= total_tokens


def _append_spec(
    *,
    specs: list[SparseEquationSpec],
    used_position_sets: set[tuple[int, ...]],
    positions: Sequence[int],
    source: str,
) -> None:
    normalized = tuple(sorted({int(position) for position in positions}))
    if not normalized:
        return
    used_position_sets.add(normalized)
    specs.append(
        SparseEquationSpec(
            equation_index=len(specs),
            positions=normalized,
            source=source,
        )
    )


def _sample_degree(
    *,
    config: SparseFountainXorConfig,
    rng: random.Random,
    total_tokens: int,
) -> int:
    draw = rng.random()
    cumulative = 0.0
    selected = config.degree_distribution[-1][0]
    for degree, weight in config.degree_distribution:
        cumulative += weight
        if draw <= cumulative:
            selected = degree
            break
    return min(total_tokens, max(1, selected))


def _sample_unique_position_set(
    *,
    rng: random.Random,
    total_tokens: int,
    degree: int,
    used_position_sets: set[tuple[int, ...]],
) -> tuple[int, ...]:
    degree = min(total_tokens, max(1, degree))
    for _ in range(20):
        positions = tuple(sorted(rng.sample(range(total_tokens), degree)))
        if positions not in used_position_sets:
            return positions
    return tuple(sorted(rng.sample(range(total_tokens), degree)))


def _normalize_degree_distribution(
    distribution: Sequence[tuple[int, float]],
) -> tuple[tuple[int, float], ...]:
    items = tuple((int(degree), float(weight)) for degree, weight in distribution)
    if not items:
        raise ValueError("degree_distribution must not be empty")
    for degree, weight in items:
        if degree <= 0:
            raise ValueError("degree values must be positive")
        if weight < 0.0:
            raise ValueError("degree weights must be non-negative")
    total = sum(weight for _, weight in items)
    if total <= 0.0:
        raise ValueError("degree distribution weight total must be positive")
    return tuple((degree, weight / total) for degree, weight in items)


def _assign_wire_ids(
    *,
    packets: Sequence[Packet],
    wire_interleaving: WireInterleavingConfig,
) -> tuple[Packet, ...]:
    wire_order = _build_wire_order(len(packets), wire_interleaving)
    packet_index_to_wire_id = {
        packet_index: wire_id
        for wire_id, packet_index in enumerate(wire_order)
    }
    return tuple(
        Packet(
            source_id=packet.source_id,
            wire_id=packet_index_to_wire_id[packet_index],
            kind=packet.kind,
            token_ids=packet.token_ids,
            token_positions=packet.token_positions,
            metadata=dict(packet.metadata),
        )
        for packet_index, packet in enumerate(packets)
    )


def _build_wire_order(
    packet_count: int,
    wire_interleaving: WireInterleavingConfig,
) -> list[int]:
    if (
        wire_interleaving.mode != WIRE_INTERLEAVING_MATRIX
        or wire_interleaving.span <= 1
        or packet_count <= 1
    ):
        return list(range(packet_count))
    span = min(wire_interleaving.span, packet_count)
    rows = ceil(packet_count / span)
    matrix: list[list[int | None]] = [[None for _ in range(span)] for _ in range(rows)]
    packet_index = 0
    for row in range(rows):
        for column in range(span):
            if packet_index < packet_count:
                matrix[row][column] = packet_index
                packet_index += 1
    wire_order: list[int] = []
    for column in range(span):
        for row in range(rows):
            value = matrix[row][column]
            if value is not None:
                wire_order.append(value)
    return wire_order
