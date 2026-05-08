import os

import pytest

from diffusion_fec.experiments.llada_smoke import run_real_llada_smoke


pytestmark = [pytest.mark.hf, pytest.mark.slow]


def test_real_llada_end_to_end_smoke_opt_in(tmp_path) -> None:
    if os.environ.get("RUN_LLADA_E2E_SMOKE") != "1":
        pytest.skip("set RUN_LLADA_E2E_SMOKE=1 to run real LLaDA end-to-end smoke")

    run_real_llada_smoke(
        output_dir=tmp_path / "real_llada_smoke",
        seed=1,
        steps=2,
        hash_bits=4,
        local_files_only=os.environ.get("LLADA_LOCAL_FILES_ONLY") == "1",
        allow_cpu=os.environ.get("RUN_LLADA_E2E_SMOKE_ALLOW_CPU") == "1",
    )

    assert (tmp_path / "real_llada_smoke" / "run_manifest.json").exists()
    assert (tmp_path / "real_llada_smoke" / "results.csv").exists()
    assert (tmp_path / "real_llada_smoke" / "events.jsonl").exists()
