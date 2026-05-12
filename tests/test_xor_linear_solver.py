from diffusion_fec.baselines.xor_equations import XorTokenEquation, solve_xor_equations
from diffusion_fec.coding.token_hash import TokenHashMap


def _equation(equation_id: str, positions: tuple[int, ...], parity: int) -> XorTokenEquation:
    return XorTokenEquation(
        equation_id=equation_id,
        parity_packet_wire_id=0,
        stripe_id=0,
        parity_offset=0,
        positions=positions,
        parity_value=parity,
    )


def _full_rank_stuck_system() -> tuple[XorTokenEquation, ...]:
    a, b, c = 3, 5, 6
    return (
        _equation("ab", (0, 1), a ^ b),
        _equation("bc", (1, 2), b ^ c),
        _equation("abc", (0, 1, 2), a ^ b ^ c),
    )


def test_linear_solver_solves_component_that_peeling_cannot() -> None:
    result = solve_xor_equations(
        equations=_full_rank_stuck_system(),
        known_tokens={},
        vocab_size=16,
        enable_linear_solve=True,
        max_component_unknowns=8,
    )

    assert result.recovered_tokens == {0: 3, 1: 5, 2: 6}
    assert result.peel_iteration_count == 0
    assert result.linear_solver_diagnostics["linear_solver_components_solved"] == 1
    assert result.linear_solver_diagnostics["linear_solver_tokens_recovered"] == 3


def test_rank_deficient_component_is_not_promoted() -> None:
    result = solve_xor_equations(
        equations=(
            _equation("ab1", (0, 1), 3 ^ 5),
            _equation("ab2", (0, 1), 3 ^ 5),
        ),
        known_tokens={},
        vocab_size=16,
        enable_linear_solve=True,
        max_component_unknowns=8,
    )

    assert result.recovered_tokens == {}
    assert result.linear_solver_diagnostics["linear_solver_rank_deficient_count"] == 1


def test_large_component_is_skipped() -> None:
    equations = tuple(
        _equation(f"e{i}", (i, i + 1), i ^ (i + 1))
        for i in range(8)
    )

    result = solve_xor_equations(
        equations=equations,
        known_tokens={},
        vocab_size=32,
        enable_linear_solve=True,
        max_component_unknowns=4,
    )

    assert result.recovered_tokens == {}
    assert result.linear_solver_diagnostics["linear_solver_too_large_count"] == 1


def test_outside_vocab_solution_is_rejected() -> None:
    result = solve_xor_equations(
        equations=_full_rank_stuck_system(),
        known_tokens={},
        vocab_size=5,
        enable_linear_solve=True,
        max_component_unknowns=8,
    )

    assert result.recovered_tokens == {}
    assert result.linear_solver_diagnostics["linear_solver_validation_conflict_count"] > 0
    assert any(conflict.reason == "solved_token_outside_vocab" for conflict in result.conflicts)


def test_banned_token_solution_is_rejected() -> None:
    result = solve_xor_equations(
        equations=_full_rank_stuck_system(),
        known_tokens={},
        vocab_size=16,
        banned_token_ids={5},
        enable_linear_solve=True,
        max_component_unknowns=8,
    )

    assert result.recovered_tokens == {}
    assert any(conflict.reason == "solved_token_is_banned" for conflict in result.conflicts)


def test_hash_mismatch_solution_is_rejected() -> None:
    token_hash_map = TokenHashMap(
        hash_bits=4,
        vocab_size=16,
        token_to_bucket=tuple(0 for _ in range(16)),
        bucket_to_token_ids=((tuple(range(16))), *(() for _ in range(15))),
    )

    result = solve_xor_equations(
        equations=_full_rank_stuck_system(),
        known_tokens={},
        hash_metadata={1: 1},
        token_hash_map=token_hash_map,
        vocab_size=16,
        enable_linear_solve=True,
        max_component_unknowns=8,
    )

    assert result.recovered_tokens == {}
    assert any(conflict.reason == "parity_hash_conflict" for conflict in result.conflicts)
