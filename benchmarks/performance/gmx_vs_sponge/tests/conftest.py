import pytest

from benchmarks.performance.gmx_vs_sponge.tests import utils


def pytest_addoption(parser):
    group = parser.getgroup("gmx-vs-sponge")
    group.addoption(
        "--case-dir",
        action="store",
        default=None,
        help="Case directory containing BENCH_TOP_FILE and BENCH_GRO_FILE.",
    )
    group.addoption(
        "--warmup",
        action="store",
        type=int,
        default=None,
        help="Warmup repeats for each mode/engine benchmark.",
    )
    group.addoption(
        "--repeats",
        action="store",
        type=int,
        default=None,
        help="Measured repeats for each mode/engine benchmark.",
    )
    group.addoption(
        "--steps",
        action="store",
        type=int,
        default=None,
        help="Steps per benchmark run.",
    )
    group.addoption(
        "--gpu-id",
        action="store",
        type=int,
        default=None,
        help="GPU id passed through CUDA_VISIBLE_DEVICES.",
    )
    group.addoption(
        "--gmx-ntmpi",
        action="store",
        type=int,
        default=None,
        help="GROMACS mdrun -ntmpi.",
    )
    group.addoption(
        "--gmx-ntomp",
        action="store",
        type=int,
        default=None,
        help="GROMACS mdrun -ntomp.",
    )


@pytest.fixture(scope="session")
def bench_config(pytestconfig):
    return utils.config_from_sources(
        case_dir=pytestconfig.getoption("case_dir"),
        warmup=pytestconfig.getoption("warmup"),
        repeats=pytestconfig.getoption("repeats"),
        steps=pytestconfig.getoption("steps"),
        gpu_id=pytestconfig.getoption("gpu_id"),
        gmx_ntmpi=pytestconfig.getoption("gmx_ntmpi"),
        gmx_ntomp=pytestconfig.getoption("gmx_ntomp"),
    )
