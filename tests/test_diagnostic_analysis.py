import csv
import json

from diffusion_fec.analysis.diagnostics import build_diagnostic_artifacts


def test_diagnostic_analysis_writes_artifacts_and_handles_missing_timing(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    event = _synthetic_event(include_timing=False)
    (run_root / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    result = build_diagnostic_artifacts(run_root=run_root)

    output_dir = run_root / "diagnostics"
    for filename in (
        "case_summary.csv",
        "step_timing.csv",
        "error_taxonomy.csv",
        "failure_examples.md",
        "diagnostic_summary.md",
    ):
        assert (output_dir / filename).exists()
    assert result["case_count"] == 1
    assert result["timing_breakdown_available"] is False
    summary = (output_dir / "diagnostic_summary.md").read_text(encoding="utf-8")
    assert "Timing breakdown is missing" in summary


def test_diagnostic_analysis_classifies_wrong_channel_lost_position(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "events.jsonl").write_text(
        json.dumps(_synthetic_event(include_timing=True)) + "\n",
        encoding="utf-8",
    )

    result = build_diagnostic_artifacts(run_root=run_root)

    assert result["timing_breakdown_available"] is True
    with (run_root / "diagnostics" / "error_taxonomy.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["position"] == "1"
    assert row["hash_metadata_available"] == "True"
    assert row["sparse_parity_coverage_count"] == "1"
    assert row["surviving_parity_equation_count"] == "1"
    assert row["was_model_committed"] == "True"
    assert row["was_rolled_back"] == "True"
    assert row["was_token_banned"] == "True"
    assert row["involved_in_final_violated_parity_equation"] == "True"


def _synthetic_event(*, include_timing: bool):
    diagnostics = {
        "model_forward_calls": 1,
        "rollback_positions": (1,),
        "rollback_banned_tokens_by_position": {"1": (9,)},
        "rollback_no_progress_stop": False,
        "rollback_remaining_masks_after_budget": 0,
        "iterative_linear_solver_rank_deficient_count": 2,
        "iterative_linear_solver_too_large_count": 1,
    }
    if include_timing:
        diagnostics.update(
            {
                "total_decode_time_sec": 1.0,
                "model_forward_time_sec": 0.4,
                "candidate_construction_time_sec": 0.2,
                "parity_candidate_filter_time_sec": 0.1,
                "xor_peel_time_sec": 0.05,
                "linear_solver_time_sec": 0.02,
                "post_commit_hook_time_sec": 0.3,
                "rollback_time_sec": 0.01,
                "mean_candidate_count": 5,
                "max_candidate_count": 8,
                "parity_filter_required_token_checks": 3,
                "parity_filter_full_scan_count": 0,
                "parity_filter_candidate_membership_checks": 3,
                "parity_filter_time_sec": 0.1,
                "step_diagnostics": (
                    {
                        "step": 0,
                        "remaining_masks_before": 1,
                        "remaining_masks_after": 0,
                        "model_committed_count": 1,
                        "parity_peeled_count": 0,
                        "rollback_count": 1,
                    },
                ),
            }
        )
    return {
        "run_id": "run",
        "case_id": "case0000",
        "strategy": "SparseHybrid",
        "case": {
            "sample": {"sample_id": "sample", "text": "hello", "token_ids": [1, 2, 3]},
            "channel_config": {"mode": "random_iid"},
            "hybrid_mode": "iterative_rollback",
            "xor_code": "sparse_fountain",
            "metrics": {
                "token_edit_distance": 1,
                "normalized_token_edit_distance": 1 / 3,
                "exact_match": False,
                "remaining_mask_token_count": 0,
            },
            "channel_lost_positions": [1],
            "channel_lost_metrics": {
                "channel_lost_position_count": 1,
                "channel_lost_position_recovered_count": 0,
                "channel_lost_position_recovery_rate": 0.0,
            },
            "hash_metadata": {"1": 2},
            "encoded": {"equation_specs": [{"positions": [1, 2]}]},
            "loss_result": {
                "received": [
                    {
                        "kind": "sparse_fountain_xor",
                        "metadata": {"sparse_fountain_xor": {"positions": [1, 2]}},
                    }
                ]
            },
            "initial_peel": {"recovered_tokens": {}},
            "final_audit": {"violations": [{"equation_id": "e", "positions": [1, 2]}]},
            "reconstruction_plan": {
                "entries": [
                    {"position": 0, "state": "known"},
                    {"position": 1, "state": "missing"},
                    {"position": 2, "state": "known"},
                ]
            },
            "decoding_result": {
                "decode_latency_sec": 1.0,
                "steps": 1,
                "reconstructed_tokens": [1, 9, 3],
                "confidence_stats": [
                    {"position": 1, "commit_step": 0, "top1_probability": 0.8, "was_fixed": False}
                ],
                "step_summaries": [{"step": 0, "still_masked_count": 0}],
                "diagnostics": diagnostics,
            },
        },
    }
