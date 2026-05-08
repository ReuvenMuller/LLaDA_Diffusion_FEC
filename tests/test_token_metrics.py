from diffusion_fec.metrics.token_metrics import compute_token_metrics, token_edit_distance
from diffusion_fec.types import (
    ReconstructionEntry,
    ReconstructionPlan,
    STATE_KNOWN,
    STATE_MISSING,
    STATE_UNGUIDED,
)


def make_plan() -> ReconstructionPlan:
    return ReconstructionPlan(
        entries=[
            ReconstructionEntry(position=0, state=STATE_KNOWN, token_id=10, fixed=True),
            ReconstructionEntry(position=1, state=STATE_MISSING, hash_value=2),
            ReconstructionEntry(position=2, state=STATE_UNGUIDED),
            ReconstructionEntry(position=3, state=STATE_KNOWN, token_id=13, fixed=True),
        ],
        total_tokens=4,
    )


def test_exact_token_sequence_match_metrics() -> None:
    metrics = compute_token_metrics(
        original_tokens=[10, 11, 12, 13],
        reconstructed_tokens=[10, 11, 12, 13],
        reconstruction_plan=make_plan(),
        mask_token_id=0,
    )

    assert metrics.exact_match is True
    assert metrics.token_edit_distance == 0
    assert metrics.normalized_token_edit_distance == 0.0
    assert metrics.lost_position_recovery_rate == 1.0
    assert metrics.lost_position_count == 2
    assert metrics.lost_position_recovered_count == 2
    assert metrics.known_position_preserved is True
    assert metrics.remaining_mask_token_count == 0


def test_token_edit_distance_counts_substitution_insertion_and_deletion() -> None:
    assert token_edit_distance([1, 2, 3], [1, 9, 3]) == 1
    assert token_edit_distance([1, 2, 3], [1, 2, 8, 3]) == 1
    assert token_edit_distance([1, 2, 3], [1, 3]) == 1
    assert token_edit_distance([1, 2, 3], [1, 9, 8, 3]) == 2


def test_normalized_token_edit_distance_uses_longer_sequence_length() -> None:
    metrics = compute_token_metrics(
        original_tokens=[1, 2, 3],
        reconstructed_tokens=[1, 9, 8, 3],
    )

    assert metrics.token_edit_distance == 2
    assert metrics.normalized_token_edit_distance == 0.5


def test_lost_position_recovery_only_scores_erased_positions() -> None:
    metrics = compute_token_metrics(
        original_tokens=[10, 11, 12, 13],
        reconstructed_tokens=[10, 99, 12, 13],
        reconstruction_plan=make_plan(),
        mask_token_id=0,
    )

    assert metrics.exact_match is False
    assert metrics.lost_position_count == 2
    assert metrics.lost_position_recovered_count == 1
    assert metrics.lost_position_recovery_rate == 0.5
    assert metrics.known_position_preserved is True


def test_known_position_preservation_detects_changed_or_missing_known_tokens() -> None:
    changed = compute_token_metrics(
        original_tokens=[10, 11, 12, 13],
        reconstructed_tokens=[99, 11, 12, 13],
        reconstruction_plan=make_plan(),
        mask_token_id=0,
    )
    truncated = compute_token_metrics(
        original_tokens=[10, 11, 12, 13],
        reconstructed_tokens=[10, 11, 12],
        reconstruction_plan=make_plan(),
        mask_token_id=0,
    )

    assert changed.known_position_preserved is False
    assert truncated.known_position_preserved is False


def test_remaining_mask_tokens_are_counted() -> None:
    metrics = compute_token_metrics(
        original_tokens=[10, 11, 12, 13],
        reconstructed_tokens=[10, 0, 12, 0],
        reconstruction_plan=make_plan(),
        mask_token_id=0,
    )

    assert metrics.remaining_mask_token_count == 2


def test_no_lost_positions_are_vacuously_recovered() -> None:
    plan = ReconstructionPlan(
        entries=[
            ReconstructionEntry(position=0, state=STATE_KNOWN, token_id=10, fixed=True),
            ReconstructionEntry(position=1, state=STATE_KNOWN, token_id=11, fixed=True),
        ],
        total_tokens=2,
    )
    metrics = compute_token_metrics(
        original_tokens=[10, 11],
        reconstructed_tokens=[10, 11],
        reconstruction_plan=plan,
        mask_token_id=0,
    )

    assert metrics.lost_position_count == 0
    assert metrics.lost_position_recovery_rate == 1.0
    assert metrics.to_dict()["exact_match"] is True
