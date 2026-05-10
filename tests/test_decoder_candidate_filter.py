from diffusion_fec.baselines.xor_equations import ParityCandidateFilter, XorTokenEquation
from diffusion_fec.decoding.llada_diffusion import DiffusionDecodingConfig, decode_masked_diffusion
from diffusion_fec.types import (
    ReconstructionEntry,
    ReconstructionPlan,
    STATE_KNOWN,
    STATE_UNGUIDED,
)


class FirstCandidateModel:
    def propose_token(
        self,
        *,
        position,
        full_position,
        candidate_token_ids,
        input_ids,
        step,
    ):
        return {
            "token_id": candidate_token_ids[0],
            "top1_probability": 1.0,
            "top2_probability": 0.0,
        }

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)


def test_decoder_candidate_filter_can_narrow_proposal_candidates() -> None:
    plan = ReconstructionPlan(
        total_tokens=2,
        entries=(
            ReconstructionEntry(position=0, state=STATE_KNOWN, token_id=1, fixed=True),
            ReconstructionEntry(position=1, state=STATE_UNGUIDED, fixed=False),
        ),
    )
    candidate_filter = ParityCandidateFilter(
        equations=(
            XorTokenEquation(
                equation_id="e0",
                parity_packet_wire_id=2,
                stripe_id=0,
                parity_offset=0,
                positions=(0, 1),
                parity_value=1 ^ 6,
            ),
        ),
        known_tokens={0: 1},
        mask_token_id=0,
    )

    result = decode_masked_diffusion(
        model=FirstCandidateModel(),
        plan=plan,
        config=DiffusionDecodingConfig(mask_token_id=0, vocab_size=8, steps=1),
        candidate_filter=candidate_filter,
    )

    assert result.reconstructed_tokens == (1, 6)
    assert result.diagnostics["candidate_filter_used"] is True
    assert result.diagnostics["candidate_filter_rejections"] == 6
    assert result.diagnostics["candidate_filter_diagnostics"]["parity_candidate_rejections"] == 6
