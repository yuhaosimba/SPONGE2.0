import math

from benchmarks.performance.gmx_vs_sponge.tests import utils


def test_build_gromacs_mdp_nvt_has_aligned_cutoffs_and_no_trajectory():
    mdp = utils.build_gromacs_mdp(
        mode="NVT",
        steps=200000,
        dt_ps=0.004,
        log_interval=10000,
        nstlist=50,
        rcut_nm=1.0,
        target_temperature=300.0,
        target_pressure=1.0,
        seed=2026,
    )
    assert "integrator      = sd" in mdp
    assert "pcoupl          = no" in mdp
    assert "nstxout         = 0" in mdp
    assert "nstxout-compressed = 0" in mdp
    assert "rlist           = 1.0" in mdp
    assert "rcoulomb        = 1.0" in mdp
    assert "rvdw            = 1.0" in mdp
    assert "nstlist         = 50" in mdp
    assert "constraints     = all-bonds" in mdp
    assert "DispCorr        = no" in mdp


def test_build_sponge_mdin_npt_has_langevin_berendsen_and_no_trajectory():
    mdin = utils.build_sponge_mdin(
        mode="NPT",
        steps=200000,
        dt_ps=0.004,
        log_interval=10000,
        refresh_interval=50,
        cutoff_angstrom=10.0,
        target_temperature=300.0,
        target_pressure=1.0,
        seed=2026,
        amber_parm7="/tmp/a.prmtop",
        amber_rst7="/tmp/b.rst7",
    )
    assert 'mode = "npt"' in mdin
    assert 'constrain_mode = "SETTLE"' in mdin
    assert 'thermostat = "middle_langevin"' in mdin
    assert 'barostat = "berendsen_barostat"' in mdin
    assert "cutoff = 10.0" in mdin
    assert "write_information_interval = 10000" in mdin
    assert "write_mdout_interval = 10000" in mdin
    assert "refresh_interval = 50" in mdin
    assert "grid_spacing = 1.0" in mdin
    assert 'amber_parm7 = "/tmp/a.prmtop"' in mdin
    assert 'amber_rst7 = "/tmp/b.rst7"' in mdin
    assert "crd =" not in mdin
    assert "vel =" not in mdin
    assert "frc =" not in mdin
    assert "box =" not in mdin


def test_parse_gromacs_ns_per_day_from_log():
    log_text = """
    Some header
    Performance:      255.7        3.753
    More lines
    """
    ns_day = utils.parse_gromacs_ns_per_day(log_text)
    assert math.isclose(ns_day, 255.7, rel_tol=0.0, abs_tol=1e-9)


def test_summarize_samples_uses_median_and_population_std():
    stats = utils.summarize_samples([10.0, 20.0, 30.0])
    assert stats["count"] == 3
    assert math.isclose(stats["median"], 20.0)
    assert math.isclose(stats["mean"], 20.0)
    assert math.isclose(stats["std"], math.sqrt(200.0 / 3.0))
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0
