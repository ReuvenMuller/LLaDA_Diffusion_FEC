import csv
import json

from artifact_helpers import (
    assert_manifest_has_run_timing,
    assert_row_has_run_timing,
    normalized_artifact_text,
)
from diffusion_fec.channels.packet_loss import CHANNEL_BURST, PacketLossChannelConfig
from diffusion_fec.baselines.xor_equations import XorTokenEquation
from diffusion_fec.baselines.xor_parity import XorParityConfig
from diffusion_fec.coding.packetizer import (
    SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS,
    WIRE_INTERLEAVING_MATRIX,
    SourceLayoutConfig,
    WireInterleavingConfig,
)
from diffusion_fec.coding.token_hash import build_token_hash_map
from diffusion_fec.decoding.llada_diffusion import DiffusionDecodingConfig
from diffusion_fec.experiments.hybrid_eval import (
    HYBRID_MODE_ITERATIVE_PEEL,
    HYBRID_MODE_PARITY_FILTER,
    HYBRID_MODE_PRE_PEEL_ONLY,
    IterativeXorPeelHook,
    XOR_CODE_SPARSE_FOUNTAIN,
    XOR_CODE_STRIPE,
    run_hybrid_recovery_case,
    run_hybrid_xor_hash_micro_eval,
)
from diffusion_fec.experiments.runner import main as runner_main
from diffusion_fec.types import TokenSample


class PositionChoiceModel:
    def __init__(self, choices):
        self.choices = dict(choices)
        self.proposal_calls = []

    def propose_token(
        self,
        *,
        position,
        full_position,
        candidate_token_ids,
        input_ids,
        step,
    ):
        self.proposal_calls.append((position, step, tuple(input_ids)))
        preferred = self.choices.get((position, step), self.choices.get(position, 3))
        token_id = preferred if preferred in candidate_token_ids else candidate_token_ids[0]
        return {"token_id": token_id, "top1_probability": 1.0, "top2_probability": 0.0}

    def forward(self, input_ids, attention_mask=None):
        raise AssertionError("test model should use propose_token")

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_token_hash(vocab_size=64):
    return build_token_hash_map(
        vocab_size=vocab_size,
        hash_bits=4,
        decode_token=lambda token_id: f"token-{token_id}",
        excluded_token_ids={0, 1, 2},
    )


def test_hybrid_micro_eval_writes_artifacts_and_overhead_fields(tmp_path) -> None:
    output_dir = tmp_path / "hybrid"

    run_hybrid_xor_hash_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        loss_rate=0.5,
        seed=0,
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=64,
        steps=2,
        hybrid_mode=HYBRID_MODE_PARITY_FILTER,
        xor_overhead_bits_per_token=4.0,
    )

    manifest = read_json(output_dir / "run_manifest.json")
    rows = read_csv(output_dir / "results.csv")
    events = read_jsonl(output_dir / "events.jsonl")

    assert_manifest_has_run_timing(manifest)
    assert manifest["runner"] == "hybrid_xor_hash_synthetic_micro_eval"
    assert manifest["config"]["hybrid_mode"] == HYBRID_MODE_PARITY_FILTER
    assert manifest["config"]["protection_mode"] == "lookback_1+xor_parity"
    assert manifest["config"]["oracle_hash_metadata"] is False
    assert rows[0]["hybrid_mode"] == HYBRID_MODE_PARITY_FILTER
    assert rows[0]["xor_overhead_bits_per_token"] == "4.0"
    assert float(rows[0]["hash_metadata_token_equivalent_overhead_ratio"]) > 0.0
    assert float(rows[0]["actual_repair_token_overhead_ratio"]) > 0.0
    assert abs(
        float(rows[0]["total_overhead_ratio"])
        - (
            float(rows[0]["hash_metadata_token_equivalent_overhead_ratio"])
            + float(rows[0]["actual_repair_token_overhead_ratio"])
        )
    ) < 1e-12
    assert "parity_candidate_rejections" in rows[0]
    assert "parity_filter_fallback_count" in rows[0]
    assert_row_has_run_timing(rows[0])
    assert events[0]["event_type"] == "hybrid_xor_hash_micro_eval_case"
    assert events[0]["case"]["hybrid_mode"] == HYBRID_MODE_PARITY_FILTER
    assert events[0]["case"]["decoding_result"]["diagnostics"]["hybrid_mode"] == HYBRID_MODE_PARITY_FILTER
    assert "channel_lost_position_recovery_rate" in rows[0]
    assert "channel_lost_metrics" in events[0]["case"]


def test_hybrid_pre_peel_mode_disables_candidate_filter(tmp_path) -> None:
    output_dir = tmp_path / "hybrid_pre_peel"

    run_hybrid_xor_hash_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        loss_rate=0.5,
        seed=0,
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=64,
        steps=2,
        hybrid_mode=HYBRID_MODE_PRE_PEEL_ONLY,
    )

    row = read_csv(output_dir / "results.csv")[0]
    event = read_jsonl(output_dir / "events.jsonl")[0]

    assert row["hybrid_mode"] == HYBRID_MODE_PRE_PEEL_ONLY
    assert row["parity_candidate_rejections"] == "0"
    assert event["case"]["parity_filter_diagnostics"]["parity_candidate_filter_calls"] == 0
    assert row["iterative_peel_enabled"] == "False"


def test_hybrid_initial_xor_peel_keeps_channel_loss_denominator(tmp_path) -> None:
    output_dir = tmp_path / "hybrid_channel_denominator"

    run_hybrid_xor_hash_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        loss_rate=0.5,
        seed=0,
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=64,
        steps=2,
        hybrid_mode=HYBRID_MODE_PRE_PEEL_ONLY,
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=1,
        ),
    )

    row = read_csv(output_dir / "results.csv")[0]
    event_case = read_jsonl(output_dir / "events.jsonl")[0]["case"]

    assert int(row["parity_peel_recovered_count"]) > 0
    assert row["lost_position_count"] == "0"
    assert row["channel_lost_position_count"] == "2"
    assert event_case["channel_lost_positions"] == [0, 1]
    assert event_case["channel_lost_metrics"]["channel_lost_position_count"] == 2


def test_hybrid_iterative_peel_writes_artifact_diagnostics(tmp_path) -> None:
    output_dir = tmp_path / "hybrid_iterative"

    run_hybrid_xor_hash_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        loss_rate=0.5,
        seed=0,
        tokens_per_packet=2,
        hash_bits=4,
        vocab_size=64,
        steps=2,
        hybrid_mode=HYBRID_MODE_ITERATIVE_PEEL,
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=1,
        ),
    )

    row = read_csv(output_dir / "results.csv")[0]
    event_case = read_jsonl(output_dir / "events.jsonl")[0]["case"]

    assert row["hybrid_mode"] == HYBRID_MODE_ITERATIVE_PEEL
    assert row["iterative_peel_enabled"] == "True"
    assert "iterative_peel_recovered_count" in row
    assert "iterative_peel_recovered_positions" in row
    assert event_case["decoding_result"]["diagnostics"]["iterative_peel_enabled"] is True


def test_iterative_peel_recovers_token_after_llada_commit_and_locks_it() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="fake",
        token_ids=(10, 11, 12, 13),
        tokenizer_name="fake-tokenizer",
    )
    token_hash = make_token_hash()
    model = PositionChoiceModel(
        {
            (0, 0): 10,
            (1, 1): 44,
        }
    )

    case = run_hybrid_recovery_case(
        sample=sample,
        model=model,
        config=DiffusionDecodingConfig(
            mask_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            vocab_size=64,
            steps=2,
            block_length=1,
        ),
        tokens_per_packet=1,
        token_hash_map=token_hash,
        xor_config=XorParityConfig(data_packets_per_stripe=2),
        source_layout=SourceLayoutConfig(),
        wire_interleaving=WireInterleavingConfig(),
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=2,
        ),
        hybrid_mode=HYBRID_MODE_ITERATIVE_PEEL,
        parity_filter_fallback=True,
    )

    diagnostics = case.decoding_result.diagnostics
    assert case.initial_peel.recovered_count == 0
    assert case.decoding_result.reconstructed_tokens == sample.token_ids
    assert model.proposal_calls == [
        (0, 0, (0, 0, 12, 13)),
        (1, 0, (0, 0, 12, 13)),
    ]
    assert diagnostics["iterative_peel_enabled"] is True
    assert diagnostics["iterative_peel_recovered_count"] == 1
    assert diagnostics["iterative_peel_recovered_positions"] == (1,)
    assert diagnostics["iterative_peel_recovered_count_by_step"] == (1,)
    assert diagnostics["post_commit_fixed_positions_per_step"] == (1,)


def test_iterative_peel_hash_mismatch_prevents_promotion() -> None:
    token_hash = make_token_hash()
    true_bucket = token_hash.bucket_for_token(11)
    wrong_bucket = (true_bucket + 1) % 16
    hook = IterativeXorPeelHook(
        equations=[
            XorTokenEquation(
                equation_id="eq0",
                parity_packet_wire_id=2,
                stripe_id=0,
                parity_offset=0,
                positions=(0, 1),
                parity_value=10 ^ 11,
            )
        ],
        hash_metadata={1: wrong_bucket},
        token_hash_map=token_hash,
        mask_token_id=0,
        vocab_size=64,
        banned_token_ids={0, 1, 2},
    )
    result = hook(
        input_ids=(10, 0),
        step=0,
        prompt_length=0,
        committed_positions=(0,),
        fixed_token_ids={},
        plan=_two_token_empty_plan(),
        config=DiffusionDecodingConfig(mask_token_id=0, vocab_size=64),
    )

    assert result["fixed_tokens"] == {}
    assert hook.diagnostics()["iterative_peel_recovered_count"] == 0
    assert hook.diagnostics()["iterative_peel_hash_conflict_count"] == 1


def test_iterative_peel_rejects_special_token_solution() -> None:
    token_hash = make_token_hash()
    hook = IterativeXorPeelHook(
        equations=[
            XorTokenEquation(
                equation_id="eq0",
                parity_packet_wire_id=2,
                stripe_id=0,
                parity_offset=0,
                positions=(0, 1),
                parity_value=10 ^ 0,
            )
        ],
        hash_metadata={},
        token_hash_map=token_hash,
        mask_token_id=0,
        vocab_size=64,
        banned_token_ids={0, 1, 2},
    )

    result = hook(
        input_ids=(10, 0),
        step=0,
        prompt_length=0,
        committed_positions=(0,),
        fixed_token_ids={},
        plan=_two_token_empty_plan(),
        config=DiffusionDecodingConfig(mask_token_id=0, vocab_size=64),
    )

    assert result["fixed_tokens"] == {}
    assert hook.diagnostics()["iterative_peel_special_token_conflict_count"] == 1


def test_iterative_peel_requires_validated_commit_once_hash_schedule() -> None:
    sample = TokenSample(
        sample_id="sample-1",
        text="fake",
        token_ids=(10, 11),
        tokenizer_name="fake-tokenizer",
    )

    try_config = DiffusionDecodingConfig(
        mask_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vocab_size=64,
        steps=2,
        block_length=1,
        editable_update_mode="resample_each_step",
        hash_constraint_schedule="always",
    )

    import pytest

    with pytest.raises(ValueError, match="iterative_peel.*commit_once"):
        run_hybrid_recovery_case(
            sample=sample,
            model=PositionChoiceModel({}),
            config=try_config,
            tokens_per_packet=1,
            token_hash_map=make_token_hash(),
            xor_config=XorParityConfig(data_packets_per_stripe=2),
            source_layout=SourceLayoutConfig(),
            wire_interleaving=WireInterleavingConfig(),
            channel_config=PacketLossChannelConfig(
                mode=CHANNEL_BURST,
                burst_start_wire_id=0,
                burst_length=1,
            ),
            hybrid_mode=HYBRID_MODE_ITERATIVE_PEEL,
            parity_filter_fallback=True,
        )


def _two_token_empty_plan():
    from diffusion_fec.coding.packetizer import build_reconstruction_plan

    return build_reconstruction_plan(total_tokens=2, received_packets=[])


def test_hybrid_burst_channel_output_is_deterministic(tmp_path) -> None:
    source_layout = SourceLayoutConfig(mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS, chunk_size=1)
    wire_interleaving = WireInterleavingConfig(mode=WIRE_INTERLEAVING_MATRIX, span=4)
    channel_config = PacketLossChannelConfig(
        mode=CHANNEL_BURST,
        loss_rate=0.5,
        seed=0,
        burst_start_wire_id=0,
        burst_length=3,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    for output_dir in (first, second):
        run_hybrid_xor_hash_micro_eval(
            output_dir=output_dir,
            sample_lengths=(12,),
            loss_rate=0.5,
            seed=0,
            tokens_per_packet=3,
            hash_bits=4,
            vocab_size=64,
            steps=2,
            hybrid_mode=HYBRID_MODE_PARITY_FILTER,
            source_layout=source_layout,
            wire_interleaving=wire_interleaving,
            channel_config=channel_config,
        )

    assert normalized_artifact_text(first / "results.csv") == normalized_artifact_text(
        second / "results.csv"
    )
    assert normalized_artifact_text(first / "events.jsonl") == normalized_artifact_text(
        second / "events.jsonl"
    )


def test_hybrid_cli_entrypoint_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "hybrid_cli"

    exit_code = runner_main(
        [
            "--output-dir",
            str(output_dir),
            "--hybrid-xor-hash-micro-eval",
            "--sample-lengths",
            "8",
            "--tokens-per-packet",
            "2",
            "--vocab-size",
            "64",
            "--steps",
            "2",
            "--hybrid-mode",
            "pre_peel_only",
            "--xor-overhead-bits-per-token",
            "4",
        ]
    )

    assert exit_code == 0
    assert read_json(output_dir / "run_manifest.json")["config"]["hybrid_mode"] == "pre_peel_only"
    assert read_csv(output_dir / "results.csv")[0]["protection_mode"] == "lookback_1+xor_parity"


def test_default_hybrid_xor_code_is_stripe(tmp_path) -> None:
    output_dir = tmp_path / "stripe_default"

    run_hybrid_xor_hash_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=2,
        vocab_size=64,
        steps=2,
        hybrid_mode=HYBRID_MODE_ITERATIVE_PEEL,
    )

    manifest = read_json(output_dir / "run_manifest.json")
    row = read_csv(output_dir / "results.csv")[0]
    assert manifest["config"]["xor_code"] == XOR_CODE_STRIPE
    assert row["xor_code"] == XOR_CODE_STRIPE
    assert row["sparse_equation_count"] == "0"


def test_sparse_hybrid_artifacts_include_sparse_and_linear_diagnostics(tmp_path) -> None:
    output_dir = tmp_path / "sparse_hybrid"

    run_hybrid_xor_hash_micro_eval(
        output_dir=output_dir,
        sample_lengths=(8,),
        tokens_per_packet=1,
        vocab_size=64,
        steps=2,
        hybrid_mode=HYBRID_MODE_ITERATIVE_PEEL,
        xor_code=XOR_CODE_SPARSE_FOUNTAIN,
        sparse_xor_seed=2,
        sparse_xor_enable_linear_solve=True,
        source_layout=SourceLayoutConfig(mode=SOURCE_LAYOUT_ROUND_ROBIN_CHUNKS, chunk_size=1),
        wire_interleaving=WireInterleavingConfig(mode=WIRE_INTERLEAVING_MATRIX, span=4),
        channel_config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=0,
            burst_length=2,
        ),
    )

    manifest = read_json(output_dir / "run_manifest.json")
    row = read_csv(output_dir / "results.csv")[0]
    event = json.loads((output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert manifest["config"]["xor_code"] == XOR_CODE_SPARSE_FOUNTAIN
    assert manifest["config"]["sparse_fountain_xor"]["random_seed"] == 2
    assert int(row["sparse_equation_count"]) > 0
    assert row["linear_solver_enabled"] == "True"
    assert row["iterative_linear_solver_enabled"] == "True"
    assert event["case"]["xor_code"] == XOR_CODE_SPARSE_FOUNTAIN
    assert event["case"]["sparse_diagnostics"]["coverage_pass_degree"] >= 1
