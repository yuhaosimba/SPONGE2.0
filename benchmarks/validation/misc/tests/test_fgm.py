import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.utils import Extractor, Outputer, Runner


ATOM_NUMBERS = 3000
ION_SERIAL_START = 2880


def _resolve_binary(env_key, fallback_name=None):
    explicit = os.environ.get(env_key)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        found = shutil.which(explicit)
        if found:
            return found

    if fallback_name:
        found = shutil.which(fallback_name)
        if found:
            return found
    return None


def _require_gpu_binary():
    binary = _resolve_binary("SPONGE_FGM_GPU_BIN")
    if binary is None:
        pytest.skip("FGM GPU validation requires SPONGE_FGM_GPU_BIN")
    return binary


def _require_cpu_binary():
    binary = _resolve_binary("SPONGE_FGM_CPU_BIN", "SPONGE")
    if binary is None:
        pytest.skip("FGM CPU validation requires SPONGE_FGM_CPU_BIN or SPONGE")
    return binary


def _write_fgm_mdin(case_dir, *, use_fft, mode="nve"):
    mdin = (
        'md_name = "validation fgm double layer"\n'
        f'mode = "{mode}"\n'
        "step_limit = 1\n"
        "dt = 0.0\n"
        "cutoff = 8.0\n"
        "skin = 2.0\n"
        "print_zeroth_frame = 1\n"
        "write_mdout_interval = 1\n"
        "write_information_interval = 1\n"
        'frc = "frc.dat"\n'
        'amber_parm7 = "parm7"\n'
        'amber_rst7 = "rst2.txt"\n'
        "\n"
    )
    if mode == "npt":
        mdin += (
            'thermostat = "middle_langevin"\n'
            "target_temperature = 300.0\n"
            'barostat = "berendsen_barostat"\n'
            "target_pressure = 1.0\n"
            "\n"
        )
    mdin += (
        "[FGM_Double_Layer]\n"
        "enable = 1\n"
        f"ion_serial_start = {ION_SERIAL_START}\n"
        "z1 = 10.0\n"
        "z2 = 50.0\n"
        "ep1 = 0.0\n"
        "ep2 = 0.0\n"
        "first_gamma = 1.9\n"
        "first_iteration_steps = 20\n"
        "second_gamma = 1.4\n"
        "second_iteration_steps = 20\n"
        "green_force_refresh_interval = 1\n"
        f"FFT = {1 if use_fft else 0}\n"
        'sphere_pos_file_name = "Discretize_Sphere_500_points.dat"\n'
        "\n"
    )
    Path(case_dir, "mdin.spg.toml").write_text(mdin, encoding="utf-8")


def _run_fgm_case(statics_path, outputs_path, run_name, *, sponge_bin, use_fft):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="fgm_two_chain",
        run_name=run_name,
    )
    _write_fgm_mdin(case_dir, use_fft=use_fft)
    Runner.run_sponge(
        case_dir,
        timeout=1800,
        sponge_cmd=sponge_bin,
    )
    rows = Extractor.parse_mdout_rows(
        case_dir / "mdout.txt",
        ["step", "FGM_Double_Layer", "potential"],
    )
    forces = Extractor.extract_sponge_forces(case_dir, ATOM_NUMBERS)
    return case_dir, rows[-1], forces


@pytest.mark.parametrize(
    ("run_name", "use_fft"),
    [("fgm_fft", True), ("fgm_sor", False)],
)
def test_fgm_two_chain_emits_energy_and_force(
    statics_path, outputs_path, run_name, use_fft
):
    gpu_bin = _require_gpu_binary()
    _, last_row, forces = _run_fgm_case(
        statics_path,
        outputs_path,
        run_name,
        sponge_bin=gpu_bin,
        use_fft=use_fft,
    )

    ion_force_norm = np.linalg.norm(forces[ION_SERIAL_START:], axis=1)
    max_ion_force = float(np.max(ion_force_norm))

    Outputer.print_table(
        ["Metric", "Value"],
        [
            ["Run", run_name],
            ["Step", str(last_row["step"])],
            ["FGM_Double_Layer", f'{last_row["FGM_Double_Layer"]:.6f}'],
            ["Potential", f'{last_row["potential"]:.6f}'],
            ["MaxIonForce", f"{max_ion_force:.6f}"],
        ],
        title=f"Misc Validation: {run_name}",
    )

    assert np.isfinite(last_row["FGM_Double_Layer"])
    assert abs(last_row["FGM_Double_Layer"]) > 1e-3
    assert np.isfinite(last_row["potential"])
    assert max_ion_force > 1e-4


def test_fgm_rejects_npt_mode(statics_path, outputs_path):
    gpu_bin = _require_gpu_binary()
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="fgm_two_chain",
        run_name="fgm_npt_reject",
    )
    _write_fgm_mdin(case_dir, use_fft=True, mode="npt")

    with pytest.raises(RuntimeError):
        Runner.run_sponge(case_dir, timeout=600, sponge_cmd=gpu_bin)
    mdinfo = Path(case_dir, "mdinfo.txt").read_text(encoding="utf-8")
    assert "FGM_Double_Layer does not support NPT mode" in mdinfo


def test_fgm_cpu_backend_reports_not_supported(statics_path, outputs_path):
    cpu_bin = _require_cpu_binary()
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="fgm_two_chain",
        run_name="fgm_cpu_reject",
    )
    _write_fgm_mdin(case_dir, use_fft=True)

    with pytest.raises(RuntimeError):
        Runner.run_sponge(case_dir, timeout=600, sponge_cmd=cpu_bin)
    mdinfo = Path(case_dir, "mdinfo.txt").read_text(encoding="utf-8")
    assert "FGM_Double_Layer" in mdinfo
