from diffusion_fec.channels.packet_loss import CHANNEL_BURST, PacketLossChannelConfig, apply_packet_loss_channel
from diffusion_fec.metrics.loss_metrics import compute_packet_loss_diagnostics
from diffusion_fec.types import Packet


def test_packet_loss_diagnostics_include_data_and_repair_denominators() -> None:
    packets = (
        Packet(source_id="s", wire_id=0, kind="data", token_ids=(10,), token_positions=(0,)),
        Packet(source_id="s", wire_id=1, kind="data", token_ids=(11,), token_positions=(1,)),
        Packet(source_id="s", wire_id=2, kind="parity", token_ids=(1,), token_positions=(0,)),
        Packet(source_id="s", wire_id=3, kind="parity", token_ids=(2,), token_positions=(1,)),
    )
    result = apply_packet_loss_channel(
        packets,
        config=PacketLossChannelConfig(
            mode=CHANNEL_BURST,
            burst_start_wire_id=1,
            burst_length=2,
        ),
    )

    diagnostics = compute_packet_loss_diagnostics(
        loss_result=result,
        source_token_count=2,
        channel_lost_position_count=1,
    )

    assert diagnostics["total_transmitted_packet_count"] == 4
    assert diagnostics["dropped_packet_count"] == 2
    assert diagnostics["received_packet_count"] == 2
    assert diagnostics["actual_wire_packet_loss_rate"] == 0.5
    assert diagnostics["source_packet_count"] == 2
    assert diagnostics["dropped_data_packet_count"] == 1
    assert diagnostics["dropped_repair_packet_count"] == 1
    assert diagnostics["actual_data_packet_loss_rate"] == 0.5
    assert diagnostics["actual_repair_packet_loss_rate"] == 0.5
    assert diagnostics["actual_source_token_loss_rate"] == 0.5
