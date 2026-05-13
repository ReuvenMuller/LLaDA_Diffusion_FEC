from diffusion_fec.baselines.xor_equations import (
    ParityCandidateFilter,
    XorTokenEquation,
    audit_xor_equations,
    equations_from_parity_packets,
    peel_xor_equations,
)
from diffusion_fec.baselines.xor_parity import XorParityConfig, encode_xor_parity
from diffusion_fec.coding.token_hash import token_hash_map_from_token_to_bucket
from diffusion_fec.types import ReconstructionEntry, TokenSample


def make_sample(token_ids=(10, 11, 12, 13)) -> TokenSample:
    return TokenSample(
        sample_id="sample",
        text="sample",
        token_ids=tuple(token_ids),
        tokenizer_name="test-tokenizer",
    )


def modulo_hash_map(vocab_size: int = 32):
    return token_hash_map_from_token_to_bucket(
        hash_bits=4,
        token_to_bucket=tuple(token_id % 16 for token_id in range(vocab_size)),
    )


def test_xor_equations_are_extracted_from_parity_metadata() -> None:
    encoded = encode_xor_parity(
        make_sample(),
        tokens_per_packet=2,
        config=XorParityConfig(data_packets_per_stripe=2, stripe_stride=2),
    )

    equations = equations_from_parity_packets(encoded.parity_packets)

    assert len(equations) == 2
    assert equations[0].positions == (0, 2)
    assert equations[0].parity_value == 10 ^ 12
    assert equations[1].positions == (1, 3)
    assert equations[1].parity_value == 11 ^ 13


def test_dropped_parity_packets_do_not_create_equations() -> None:
    encoded = encode_xor_parity(
        make_sample(),
        tokens_per_packet=2,
        config=XorParityConfig(data_packets_per_stripe=2, stripe_stride=2),
    )

    assert equations_from_parity_packets(encoded.source_packets) == ()


def test_peeling_recovers_single_unknown() -> None:
    equation = XorTokenEquation(
        equation_id="e0",
        parity_packet_wire_id=4,
        stripe_id=0,
        parity_offset=0,
        positions=(0, 2),
        parity_value=10 ^ 12,
    )

    result = peel_xor_equations(equations=(equation,), known_tokens={0: 10})

    assert result.recovered_tokens == {2: 12}
    assert result.known_tokens[2] == 12
    assert result.peel_iteration_count == 1
    assert result.recovery_provenance[2].method == "peel"
    assert result.recovery_provenance[2].equation_ids == ("e0",)
    assert result.recovery_provenance[2].dependency_positions == (0,)


def test_repeated_peeling_unlocks_later_equations() -> None:
    equations = (
        XorTokenEquation("e0", 4, 0, 0, (0, 1), 1 ^ 2),
        XorTokenEquation("e1", 5, 1, 0, (1, 2), 2 ^ 4),
    )

    result = peel_xor_equations(equations=equations, known_tokens={0: 1})

    assert result.recovered_tokens == {1: 2, 2: 4}
    assert result.peel_iteration_count == 1


def test_multi_unknown_equations_remain_unsolved() -> None:
    equation = XorTokenEquation("e0", 4, 0, 0, (0, 1, 2), 7)

    result = peel_xor_equations(equations=(equation,), known_tokens={0: 1})

    assert result.recovered_tokens == {}
    assert result.peel_iteration_count == 0


def test_hash_validation_accepts_consistent_peeled_tokens() -> None:
    equation = XorTokenEquation("e0", 4, 0, 0, (0, 2), 10 ^ 12)

    result = peel_xor_equations(
        equations=(equation,),
        known_tokens={0: 10},
        hash_metadata={2: 12},
        token_hash_map=modulo_hash_map(),
    )

    assert result.recovered_tokens == {2: 12}
    assert result.conflict_count == 0


def test_hash_validation_rejects_conflicting_peeled_tokens() -> None:
    equation = XorTokenEquation("e0", 4, 0, 0, (0, 2), 10 ^ 12)

    result = peel_xor_equations(
        equations=(equation,),
        known_tokens={0: 10},
        hash_metadata={2: 13},
        token_hash_map=modulo_hash_map(),
    )

    assert result.recovered_tokens == {}
    assert result.conflict_count == 1
    assert result.conflicts[0].reason == "parity_hash_conflict"
    assert result.conflicts[0].equation_positions == (0, 2)
    assert result.conflicts[0].dependency_positions == (0,)


def test_parity_candidate_filter_rejects_incompatible_candidates() -> None:
    equation = XorTokenEquation("e0", 4, 0, 0, (0, 1), 2 ^ 5)
    entry = ReconstructionEntry(position=1, state="unguided", fixed=False)
    candidate_filter = ParityCandidateFilter(
        equations=(equation,),
        known_tokens={0: 2},
        mask_token_id=0,
    )

    candidates = candidate_filter(
        entry=entry,
        candidate_token_ids=(4, 5, 6),
        input_ids=(2, 0),
        step=0,
        full_position=1,
    )

    assert candidates == (5,)
    assert candidate_filter.diagnostics()["parity_candidate_rejections"] == 2
    assert candidate_filter.diagnostics()["parity_filter_required_token_checks"] == 1
    assert candidate_filter.diagnostics()["parity_filter_full_scan_count"] == 0


def test_parity_candidate_filter_falls_back_when_empty() -> None:
    equation = XorTokenEquation("e0", 4, 0, 0, (0, 1), 2 ^ 5)
    entry = ReconstructionEntry(position=1, state="unguided", fixed=False)
    candidate_filter = ParityCandidateFilter(
        equations=(equation,),
        known_tokens={0: 2},
        mask_token_id=0,
        fallback_on_empty=True,
    )

    candidates = candidate_filter(
        entry=entry,
        candidate_token_ids=(4, 6),
        input_ids=(2, 0),
        step=0,
        full_position=1,
    )

    assert candidates == (4, 6)
    assert candidate_filter.diagnostics()["parity_filter_fallback_count"] == 1


def test_parity_candidate_filter_matches_old_scan_for_determined_required_present() -> None:
    equations = (XorTokenEquation("e0", 4, 0, 0, (0, 1), 2 ^ 5),)
    entry = ReconstructionEntry(position=1, state="unguided", fixed=False)
    candidates = tuple(range(1000))
    candidate_filter = ParityCandidateFilter(
        equations=equations,
        known_tokens={0: 2},
        mask_token_id=0,
    )

    result = candidate_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(2, 0),
        step=0,
        full_position=1,
    )

    assert result == _old_parity_candidate_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(2, 0),
        equations=equations,
        known_tokens={0: 2},
        mask_token_id=0,
        fallback_on_empty=True,
    )
    assert result == (5,)
    assert candidate_filter.diagnostics()["parity_filter_full_scan_count"] == 0
    assert candidate_filter.diagnostics()["parity_filter_candidate_membership_checks"] == 1


def test_parity_candidate_filter_matches_old_scan_for_required_absent_fallback_on_and_off() -> None:
    equations = (XorTokenEquation("e0", 4, 0, 0, (0, 1), 2 ^ 5),)
    entry = ReconstructionEntry(position=1, state="unguided", fixed=False)
    candidates = (4, 6, 8)

    fallback_filter = ParityCandidateFilter(
        equations=equations,
        known_tokens={0: 2},
        mask_token_id=0,
        fallback_on_empty=True,
    )
    strict_filter = ParityCandidateFilter(
        equations=equations,
        known_tokens={0: 2},
        mask_token_id=0,
        fallback_on_empty=False,
    )

    fallback_result = fallback_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(2, 0),
        step=0,
        full_position=1,
    )
    strict_result = strict_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(2, 0),
        step=0,
        full_position=1,
    )

    assert fallback_result == _old_parity_candidate_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(2, 0),
        equations=equations,
        known_tokens={0: 2},
        mask_token_id=0,
        fallback_on_empty=True,
    )
    assert strict_result == _old_parity_candidate_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(2, 0),
        equations=equations,
        known_tokens={0: 2},
        mask_token_id=0,
        fallback_on_empty=False,
    )


def test_parity_candidate_filter_matches_old_scan_for_conflicting_determined_equations() -> None:
    equations = (
        XorTokenEquation("e0", 4, 0, 0, (0, 1), 2 ^ 5),
        XorTokenEquation("e1", 5, 0, 0, (2, 1), 3 ^ 7),
    )
    entry = ReconstructionEntry(position=1, state="unguided", fixed=False)
    candidates = (5, 7, 9)
    candidate_filter = ParityCandidateFilter(
        equations=equations,
        known_tokens={0: 2, 2: 3},
        mask_token_id=0,
        fallback_on_empty=True,
    )

    result = candidate_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(2, 0, 3),
        step=0,
        full_position=1,
    )

    assert result == _old_parity_candidate_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(2, 0, 3),
        equations=equations,
        known_tokens={0: 2, 2: 3},
        mask_token_id=0,
        fallback_on_empty=True,
    )
    assert result == candidates


def test_parity_candidate_filter_matches_old_scan_for_undetermined_and_mixed_equations() -> None:
    equations = (
        XorTokenEquation("undetermined", 4, 0, 0, (0, 1, 2), 2 ^ 5 ^ 9),
        XorTokenEquation("determined", 5, 0, 0, (3, 1), 6 ^ 5),
    )
    entry = ReconstructionEntry(position=1, state="unguided", fixed=False)
    candidates = (4, 5, 6)
    candidate_filter = ParityCandidateFilter(
        equations=equations,
        known_tokens={3: 6},
        mask_token_id=0,
    )

    result = candidate_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(0, 0, 0, 6),
        step=0,
        full_position=1,
    )

    assert result == _old_parity_candidate_filter(
        entry=entry,
        candidate_token_ids=candidates,
        input_ids=(0, 0, 0, 6),
        equations=equations,
        known_tokens={3: 6},
        mask_token_id=0,
        fallback_on_empty=True,
    )
    assert result == (5,)


def _old_parity_candidate_filter(
    *,
    entry: ReconstructionEntry,
    candidate_token_ids,
    input_ids,
    equations,
    known_tokens,
    mask_token_id: int,
    fallback_on_empty: bool,
):
    kept = tuple(
        token_id
        for token_id in candidate_token_ids
        if _old_candidate_satisfies(
            position=entry.position,
            candidate_token_id=token_id,
            input_ids=input_ids,
            equations=equations,
            known_tokens=known_tokens,
            mask_token_id=mask_token_id,
        )
    )
    if kept or not fallback_on_empty:
        return kept
    return tuple(candidate_token_ids)


def _old_candidate_satisfies(
    *,
    position,
    candidate_token_id,
    input_ids,
    equations,
    known_tokens,
    mask_token_id,
) -> bool:
    for equation in equations:
        accumulator = equation.parity_value
        determined = True
        for other_position in equation.positions:
            if other_position == position:
                continue
            if other_position in known_tokens:
                token_id = known_tokens[other_position]
            elif other_position >= len(input_ids) or input_ids[other_position] == mask_token_id:
                determined = False
                break
            else:
                token_id = input_ids[other_position]
            accumulator ^= token_id
        if determined and candidate_token_id != accumulator:
            return False
    return True


def test_xor_audit_counts_satisfied_and_violated_equations() -> None:
    equations = (
        XorTokenEquation("ok", 4, 0, 0, (0, 1), 1 ^ 2),
        XorTokenEquation("bad", 5, 1, 0, (1, 2), 2 ^ 3),
        XorTokenEquation("open", 6, 2, 0, (2, 3), 3 ^ 4),
    )

    audit = audit_xor_equations(
        equations=equations,
        token_by_position={0: 1, 1: 2, 2: 9},
    )

    assert audit.satisfied_count == 1
    assert audit.violated_count == 1
    assert audit.unresolved_count == 1
