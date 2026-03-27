import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.utils import Outputer

SECONDS_PER_DAY = 86400.0
DEFAULT_CASE_DIR = Path(
    "/home/ylj/SPONGE_GITHUB/SPONGE/benchmarks/performance/gmx_vs_sponge/statics/water_160k"
)


@dataclass(frozen=True)
class BenchConfig:
    case_dir: Path
    warmup: int
    repeats: int
    steps: int
    gpu_id: int
    gmx_ntmpi: int
    gmx_ntomp: int
    top_file: str
    gro_file_nvt: str
    gro_file_npt: str
    sponge_parm7_file: str
    sponge_rst7_file_nvt: str
    sponge_rst7_file_npt: str
    dt_ps: float
    temperature_k: float
    pressure_bar: float
    log_interval: int
    nstlist: int
    cutoff_angstrom: float
    pme_grid_spacing_angstrom: float
    coordinate_tolerance_angstrom: float
    box_tolerance_angstrom: float
    seed: int
    sponge_bin: str
    gmx_bin: str

    @property
    def rcut_nm(self) -> float:
        return self.cutoff_angstrom / 10.0


class CommandError(RuntimeError):
    pass


def build_gromacs_mdp(
    *,
    mode: str,
    steps: int,
    dt_ps: float,
    log_interval: int,
    nstlist: int,
    rcut_nm: float,
    target_temperature: float,
    target_pressure: float,
    seed: int,
) -> str:
    mode_upper = mode.upper()
    if mode_upper not in {"NVT", "NPT"}:
        raise ValueError(f"Unsupported mode: {mode}")

    lines = [
        "; ===== RUN CONTROL =====",
        f"nsteps          = {int(steps)}",
        f"dt              = {float(dt_ps):.6f}",
        "",
        "; ===== OUTPUT CONTROL =====",
        "nstxout         = 0",
        "nstvout         = 0",
        "nstfout         = 0",
        "nstxout-compressed = 0",
        f"nstenergy       = {int(log_interval)}",
        f"nstlog          = {int(log_interval)}",
        "nstcheckpoint   = 0",
        "",
        "; ===== NEIGHBOR SEARCHING =====",
        "cutoff-scheme   = Verlet",
        f"nstlist         = {int(nstlist)}",
        f"rlist           = {float(rcut_nm):.4f}",
        "",
        "; ===== ELECTROSTATICS =====",
        "coulombtype     = PME",
        f"rcoulomb        = {float(rcut_nm):.4f}",
        "pme_order       = 4",
        "fourierspacing  = 0.10",
        "",
        "; ===== VAN DER WAALS =====",
        "vdwtype         = cutoff",
        f"rvdw            = {float(rcut_nm):.4f}",
        "DispCorr        = no",
        "",
        "; ===== CONSTRAINTS =====",
        "constraints     = all-bonds",
        "constraint_algorithm = lincs",
        "lincs_iter      = 1",
        "lincs_order     = 4",
        "",
        "; ===== PBC / COM =====",
        "pbc             = xyz",
        "comm-mode       = Linear",
        "comm-grps       = System",
        "gen-vel         = no",
        "",
    ]

    if mode_upper == "NVT":
        lines += [
            "; ===== NVT (Langevin-like, stochastic dynamics) =====",
            "integrator      = sd",
            f"ref_t           = {float(target_temperature):.4f}",
            "tau_t           = 1.0",
            "tc-grps         = System",
            "bd-fric         = 1.0",
            f"ld-seed         = {int(seed)}",
            "pcoupl          = no",
        ]
    else:
        lines += [
            "; ===== NPT (Berendsen barostat) =====",
            "integrator      = md",
            "tcoupl          = v-rescale",
            "tc-grps         = System",
            "tau_t           = 1.0",
            f"ref_t           = {float(target_temperature):.4f}",
            f"ld-seed         = {int(seed)}",
            "pcoupl          = Berendsen",
            "pcoupltype      = isotropic",
            "tau_p           = 1.0",
            f"ref_p           = {float(target_pressure):.4f}",
            "compressibility = 4.5e-5",
        ]

    return "\n".join(lines) + "\n"


def build_sponge_mdin(
    *,
    mode: str,
    steps: int,
    dt_ps: float,
    log_interval: int,
    refresh_interval: int,
    cutoff_angstrom: float,
    target_temperature: float,
    target_pressure: float,
    seed: int,
    amber_parm7: str = "water.prmtop",
    amber_rst7: str = "water.inpcrd",
) -> str:
    mode_lower = mode.lower()
    if mode_lower not in {"nvt", "npt"}:
        raise ValueError(f"Unsupported mode: {mode}")

    lines = [
        'md_name = "gmx_vs_sponge_160k"',
        f'mode = "{mode_lower}"',
        f"step_limit = {int(steps)}",
        f"dt = {float(dt_ps):.6f}",
        f"cutoff = {float(cutoff_angstrom):.4f}",
        'constrain_mode = "SETTLE"',
        f"target_temperature = {float(target_temperature):.4f}",
        'thermostat = "middle_langevin"',
        "thermostat_tau = 1.0",
        f"thermostat_seed = {int(seed)}",
        f"write_information_interval = {int(log_interval)}",
        f"write_mdout_interval = {int(log_interval)}",
        "write_trajectory_interval = 0",
        "write_restart_file_interval = 0",
        "print_zeroth_frame = 0",
        f'amber_parm7 = "{amber_parm7}"',
        f'amber_rst7 = "{amber_rst7}"',
    ]

    if mode_lower == "npt":
        lines.extend(
            [
                f"target_pressure = {float(target_pressure):.4f}",
                'barostat = "berendsen_barostat"',
                "barostat_tau = 1.0",
                "barostat_update_interval = 10",
                "barostat_isotropy = \"isotropic\"",
                "barostat_compressibility = 4.5e-5",
            ]
        )

    lines.extend(
        [
            "",
            "[neighbor_list]",
            f"refresh_interval = {int(refresh_interval)}",
            "",
            "[PME]",
            "grid_spacing = 1.0",
            "update_interval = 1",
        ]
    )

    return "\n".join(lines) + "\n"


def parse_gromacs_ns_per_day(log_text: str) -> float:
    patterns = [
        r"Performance:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*ns/day",
    ]
    for pattern in patterns:
        match = re.search(pattern, log_text)
        if match:
            return float(match.group(1))
    raise ValueError("Failed to parse ns/day from GROMACS log")


def summarize_samples(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("No samples to summarize")
    vals = [float(v) for v in samples]
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "count": len(vals),
        "median": float(statistics.median(vals)),
        "mean": float(statistics.fmean(vals)),
        "std": float(std),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def ensure_requirements(cfg: BenchConfig) -> None:
    for exe in [cfg.sponge_bin, cfg.gmx_bin]:
        if shutil.which(exe) is None:
            raise FileNotFoundError(f"Required executable not found in PATH: {exe}")

    case_dir = Path(cfg.case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"BENCH_CASE_DIR does not exist: {case_dir}")

    for filename in [cfg.top_file, cfg.gro_file_nvt, cfg.gro_file_npt]:
        path = case_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required input file: {path}")

    for path in [
        Path(cfg.sponge_parm7_file),
        Path(cfg.sponge_rst7_file_nvt),
        Path(cfg.sponge_rst7_file_npt),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required SPONGE AMBER input file: {path}")



def parse_gro_meta(gro_path: Path) -> dict[str, Any]:
    lines = gro_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {gro_path}")
    natom = int(lines[1].split()[0])
    box_fields = [float(v) for v in lines[-1].split()]
    if len(box_fields) < 3:
        raise ValueError(f"Invalid box line in {gro_path}: {lines[-1]}")
    return {
        "natom": natom,
        "box_nm": box_fields,
    }


def _parse_rst7_meta_and_coordinates(rst7_path: Path) -> tuple[int, list[float], list[float]]:
    lines = rst7_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid rst7 file: {rst7_path}")
    natom = int(lines[1].split()[0])
    values: list[float] = []
    for line in lines[2:]:
        for i in range(0, len(line), 12):
            token = line[i : i + 12].strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
    coord_count = natom * 3
    if len(values) < coord_count:
        raise ValueError(
            f"rst7 has insufficient coordinate values: expected {coord_count}, got {len(values)}"
        )
    coords = values[:coord_count]
    box = values[-6:] if len(values) >= 6 else []
    return natom, coords, box


def _parse_gro_coordinates_angstrom(gro_path: Path) -> tuple[int, list[float]]:
    lines = gro_path.read_text(encoding="utf-8").splitlines()
    natom = int(lines[1].split()[0])
    coords: list[float] = []
    for line in lines[2 : 2 + natom]:
        coords.append(float(line[20:28]) * 10.0)
        coords.append(float(line[28:36]) * 10.0)
        coords.append(float(line[36:44]) * 10.0)
    return natom, coords


def assert_coordinate_and_box_alignment(
    *,
    gro_path: Path,
    rst7_path: Path,
    coord_tol_angstrom: float,
    box_tol_angstrom: float,
) -> dict[str, float]:
    natom_gro, gro_coords = _parse_gro_coordinates_angstrom(gro_path)
    natom_rst7, rst7_coords, rst7_box = _parse_rst7_meta_and_coordinates(rst7_path)
    if natom_gro != natom_rst7:
        raise ValueError(
            f"Atom count mismatch: gro={natom_gro}, rst7={natom_rst7}. "
            "Please regenerate aligned coordinates."
        )

    sum_sq = 0.0
    max_abs = 0.0
    for x, y in zip(gro_coords, rst7_coords):
        d = x - y
        sum_sq += d * d
        abs_d = abs(d)
        if abs_d > max_abs:
            max_abs = abs_d
    coord_rmsd = (sum_sq / len(gro_coords)) ** 0.5
    if coord_rmsd > coord_tol_angstrom:
        raise ValueError(
            f"Coordinate mismatch exceeds tolerance: rmsd={coord_rmsd:.6f} A, "
            f"tol={coord_tol_angstrom:.6f} A. Please use a matched gro/rst7 pair."
        )

    gro_box = parse_gro_meta(gro_path)["box_nm"][:3]
    if len(rst7_box) >= 3:
        rst7_box_nm = [rst7_box[0] / 10.0, rst7_box[1] / 10.0, rst7_box[2] / 10.0]
        for idx, (gx, rx) in enumerate(zip(gro_box, rst7_box_nm)):
            delta_a = abs(gx - rx) * 10.0
            if delta_a > box_tol_angstrom:
                axis = "xyz"[idx]
                raise ValueError(
                    f"Box mismatch on axis {axis}: |gro-rst7|={delta_a:.6f} A > "
                    f"{box_tol_angstrom:.6f} A"
                )

    return {
        "coord_rmsd_angstrom": coord_rmsd,
        "coord_max_abs_angstrom": max_abs,
    }


def _run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise CommandError(
            f"Command failed ({result.returncode}) in {cwd}: {' '.join(cmd)}\n{output}"
        )
    return output


def _base_env(gpu_id: int, ntomp: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = str(ntomp)
    return env


def run_sponge_once(
    run_dir: Path, cfg: BenchConfig, mode: str, *, sponge_rst7_file: str
) -> dict[str, float]:
    mdin = build_sponge_mdin(
        mode=mode,
        steps=cfg.steps,
        dt_ps=cfg.dt_ps,
        log_interval=cfg.log_interval,
        refresh_interval=cfg.nstlist,
        cutoff_angstrom=cfg.cutoff_angstrom,
        target_temperature=cfg.temperature_k,
        target_pressure=cfg.pressure_bar,
        seed=cfg.seed,
        amber_parm7=str(Path(cfg.sponge_parm7_file).resolve()),
        amber_rst7=str(Path(sponge_rst7_file).resolve()),
    )
    (run_dir / "mdin.spg.toml").write_text(mdin, encoding="utf-8")

    start = time.perf_counter()
    _run_command(
        [cfg.sponge_bin, "-mdin", "mdin.spg.toml"],
        cwd=run_dir,
        env=_base_env(cfg.gpu_id, cfg.gmx_ntomp),
    )
    elapsed = time.perf_counter() - start

    steps_per_s = float(cfg.steps) / elapsed
    ns_per_day = float(cfg.steps) * cfg.dt_ps * SECONDS_PER_DAY / elapsed / 1000.0

    sponge_outputs = _detect_existing_files(
        run_dir, ["mdcrd.dat", "mdvel.dat", "mdfrc.dat", "mdbox.txt"]
    )

    return {
        "elapsed_s": elapsed,
        "steps_per_s": steps_per_s,
        "ns_per_day": ns_per_day,
        "unavoidable_output_files": sponge_outputs,
    }


def run_gromacs_once(
    run_dir: Path, cfg: BenchConfig, mode: str, *, gro_file: str
) -> dict[str, float]:
    mdp = build_gromacs_mdp(
        mode=mode,
        steps=cfg.steps,
        dt_ps=cfg.dt_ps,
        log_interval=cfg.log_interval,
        nstlist=cfg.nstlist,
        rcut_nm=cfg.rcut_nm,
        target_temperature=cfg.temperature_k,
        target_pressure=cfg.pressure_bar,
        seed=cfg.seed,
    )
    (run_dir / "run.mdp").write_text(mdp, encoding="utf-8")

    env = _base_env(cfg.gpu_id, cfg.gmx_ntomp)

    _run_command(
        [
            cfg.gmx_bin,
            "grompp",
            "-f",
            "run.mdp",
            "-c",
            str((cfg.case_dir / gro_file).resolve()),
            "-p",
            str((cfg.case_dir / cfg.top_file).resolve()),
            "-o",
            "run.tpr",
            "-maxwarn",
            "1" if mode.upper() == "NPT" else "0",
        ],
        cwd=run_dir,
        env=env,
    )

    mdrun_base = [
        cfg.gmx_bin,
        "mdrun",
        "-s",
        "run.tpr",
        "-deffnm",
        "run",
        "-ntmpi",
        str(cfg.gmx_ntmpi),
        "-ntomp",
        str(cfg.gmx_ntomp),
        "-pin",
        "on",
        "-noconfout",
    ]

    gpu_mode = True
    start = time.perf_counter()
    try:
        _run_command(
            mdrun_base + ["-gpu_id", "0", "-nb", "gpu", "-pme", "gpu"],
            cwd=run_dir,
            env=env,
        )
    except CommandError as exc:
        message = str(exc)
        no_gpu_patterns = [
            "0 detected device(s)",
            "did not correspond to any of the 0 detected device(s)",
            "GPU support: disabled",
        ]
        if any(token in message for token in no_gpu_patterns):
            gpu_mode = False
            _run_command(mdrun_base, cwd=run_dir, env=env)
        else:
            raise
    elapsed = time.perf_counter() - start

    log_path = run_dir / "run.log"
    if not log_path.exists():
        raise FileNotFoundError(f"GROMACS log not found: {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    try:
        ns_per_day = parse_gromacs_ns_per_day(log_text)
    except ValueError:
        ns_per_day = float(cfg.steps) * cfg.dt_ps * SECONDS_PER_DAY / elapsed / 1000.0

    steps_per_s = float(cfg.steps) / elapsed

    _assert_no_trajectory_outputs(run_dir, engine="gromacs")

    return {
        "elapsed_s": elapsed,
        "steps_per_s": steps_per_s,
        "ns_per_day": ns_per_day,
        "gpu_mode": gpu_mode,
    }


def _assert_no_trajectory_outputs(run_dir: Path, *, engine: str) -> None:
    if engine == "gromacs":
        banned = ["run.xtc", "run.trr", "run.trn"]
    else:
        raise ValueError(f"Unknown or unsupported strict trajectory check for engine: {engine}")

    hits = _detect_existing_files(run_dir, banned)
    if hits:
        raise AssertionError(f"Trajectory outputs must be disabled, found: {hits}")


def _detect_existing_files(run_dir: Path, names: list[str]) -> list[str]:
    return [name for name in names if (run_dir / name).exists()]


def benchmark_mode(
    *,
    output_root: Path,
    cfg: BenchConfig,
    mode: str,
    engine: str,
) -> dict[str, Any]:
    mode_upper = mode.upper()
    engine_lower = engine.lower()
    if mode_upper == "NVT":
        gro_file = cfg.gro_file_nvt
        sponge_rst7 = cfg.sponge_rst7_file_nvt
    elif mode_upper == "NPT":
        gro_file = cfg.gro_file_npt
        sponge_rst7 = cfg.sponge_rst7_file_npt
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    warmup_results: list[dict[str, float]] = []
    measure_results: list[dict[str, float]] = []

    for idx in range(cfg.warmup + cfg.repeats):
        is_warmup = idx < cfg.warmup
        phase = "warmup" if is_warmup else "measure"
        run_index = idx if is_warmup else (idx - cfg.warmup)
        run_dir = output_root / mode_upper / engine_lower / f"{phase}_{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if engine_lower == "sponge":
            result = run_sponge_once(
                run_dir, cfg, mode_upper, sponge_rst7_file=sponge_rst7
            )
        else:
            result = run_gromacs_once(run_dir, cfg, mode_upper, gro_file=gro_file)
        if is_warmup:
            warmup_results.append(result)
        else:
            measure_results.append(result)

    ns_samples = [r["ns_per_day"] for r in measure_results]
    steps_samples = [r["steps_per_s"] for r in measure_results]

    return {
        "mode": mode_upper,
        "engine": engine_lower,
        "warmup_count": cfg.warmup,
        "measure_count": cfg.repeats,
        "warmup": warmup_results,
        "measure": measure_results,
        "ns_per_day": summarize_samples(ns_samples),
        "steps_per_s": summarize_samples(steps_samples),
    }


def write_summary_files(output_root: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "mode",
            "engine",
            "metric",
            "count",
            "median",
            "mean",
            "std",
            "min",
            "max",
        ])
        for row in summary["results"]:
            for metric in ["ns_per_day", "steps_per_s"]:
                stats = row[metric]
                writer.writerow(
                    [
                        row["mode"],
                        row["engine"],
                        metric,
                        stats["count"],
                        f"{stats['median']:.8f}",
                        f"{stats['mean']:.8f}",
                        f"{stats['std']:.8f}",
                        f"{stats['min']:.8f}",
                        f"{stats['max']:.8f}",
                    ]
                )

    return json_path, csv_path


def print_result_table(results: list[dict[str, Any]]) -> None:
    rows = []
    for item in results:
        rows.append(
            [
                item["mode"],
                item["engine"],
                f"{item['ns_per_day']['median']:.3f}",
                f"{item['steps_per_s']['median']:.3f}",
                str(item["measure_count"]),
            ]
        )
    Outputer.print_table(
        ["Mode", "Engine", "Median ns/day", "Median steps/s", "Repeats"],
        rows,
        title="SPONGE vs GROMACS GPU Throughput",
    )


def config_from_sources(
    *,
    case_dir: str | None = None,
    warmup: int | None = None,
    repeats: int | None = None,
    steps: int | None = None,
    gpu_id: int | None = None,
    gmx_ntmpi: int | None = None,
    gmx_ntomp: int | None = None,
) -> BenchConfig:
    env = os.environ
    resolved_case_dir = Path(
        case_dir or env.get("BENCH_CASE_DIR", str(DEFAULT_CASE_DIR))
    ).resolve()

    def _resolve_sponge_input(
        env_key: str,
        default_name: str,
        *,
        search_case_parent: bool = True,
    ) -> str:
        env_value = env.get(env_key)
        if env_value:
            return str(Path(env_value).resolve())
        local = resolved_case_dir / default_name
        if local.exists():
            return str(local.resolve())
        if search_case_parent:
            parent = resolved_case_dir.parent / default_name
            if parent.exists():
                return str(parent.resolve())
        return str(local.resolve())

    cfg = BenchConfig(
        case_dir=resolved_case_dir,
        warmup=int(warmup if warmup is not None else env.get("BENCH_WARMUP", 3)),
        repeats=int(repeats if repeats is not None else env.get("BENCH_REPEATS", 5)),
        steps=int(steps if steps is not None else env.get("BENCH_STEPS", 200000)),
        gpu_id=int(gpu_id if gpu_id is not None else env.get("BENCH_GPU_ID", 0)),
        gmx_ntmpi=int(gmx_ntmpi if gmx_ntmpi is not None else env.get("BENCH_GMX_NTMPI", 1)),
        gmx_ntomp=int(gmx_ntomp if gmx_ntomp is not None else env.get("BENCH_GMX_NTOMP", 8)),
        top_file=env.get("BENCH_TOP_FILE", "water.top"),
        gro_file_nvt=env.get("BENCH_GRO_FILE_NVT", "water_nvt_eq.gro"),
        gro_file_npt=env.get("BENCH_GRO_FILE_NPT", "water_npt_eq.gro"),
        sponge_parm7_file=_resolve_sponge_input("BENCH_SPONGE_PARM7_FILE", "water.prmtop"),
        sponge_rst7_file_nvt=_resolve_sponge_input(
            "BENCH_SPONGE_RST7_FILE_NVT", "water_nvt_eq.rst7"
        ),
        sponge_rst7_file_npt=_resolve_sponge_input(
            "BENCH_SPONGE_RST7_FILE_NPT", "water_npt_eq.rst7"
        ),
        dt_ps=float(env.get("BENCH_DT_PS", 0.004)),
        temperature_k=float(env.get("BENCH_TEMPERATURE_K", 300.0)),
        pressure_bar=float(env.get("BENCH_PRESSURE_BAR", 1.0)),
        log_interval=int(env.get("BENCH_LOG_INTERVAL", 10000)),
        nstlist=int(env.get("BENCH_NSTLIST", 50)),
        cutoff_angstrom=float(env.get("BENCH_CUTOFF_ANGSTROM", 10.0)),
        pme_grid_spacing_angstrom=float(env.get("BENCH_PME_GRID_ANGSTROM", 1.0)),
        coordinate_tolerance_angstrom=float(env.get("BENCH_COORD_TOL_ANGSTROM", 0.02)),
        box_tolerance_angstrom=float(env.get("BENCH_BOX_TOL_ANGSTROM", 0.05)),
        seed=int(env.get("BENCH_SEED", 2026)),
        sponge_bin=env.get("BENCH_SPONGE_BIN", "SPONGE"),
        gmx_bin=env.get("BENCH_GMX_BIN", "gmx"),
    )

    if cfg.warmup < 0:
        raise ValueError("BENCH_WARMUP must be >= 0")
    for key, value in [
        ("BENCH_REPEATS", cfg.repeats),
        ("BENCH_STEPS", cfg.steps),
        ("BENCH_GMX_NTMPI", cfg.gmx_ntmpi),
        ("BENCH_GMX_NTOMP", cfg.gmx_ntomp),
        ("BENCH_LOG_INTERVAL", cfg.log_interval),
        ("BENCH_NSTLIST", cfg.nstlist),
    ]:
        if value <= 0:
            raise ValueError(f"{key} must be positive")

    return cfg


def build_summary(
    *,
    cfg: BenchConfig,
    gro_meta: dict[str, Any],
    output_root: Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "config": {
            **asdict(cfg),
            "case_dir": str(cfg.case_dir),
        },
        "case": {
            "gro_file_nvt": cfg.gro_file_nvt,
            "gro_file_npt": cfg.gro_file_npt,
            "top_file": cfg.top_file,
            "natom": gro_meta["natom"],
            "box_nm": gro_meta["box_nm"],
        },
        "paths": {
            "output_root": str(output_root),
        },
        "parameter_snapshot": {
            "cutoff": {
                "gromacs_nm": cfg.rcut_nm,
                "sponge_angstrom": cfg.cutoff_angstrom,
            },
            "pme": {
                "gromacs_fourierspacing_nm": 0.10,
                "sponge_grid_spacing_angstrom": cfg.pme_grid_spacing_angstrom,
            },
            "neighbor_refresh": {
                "gromacs_nstlist": cfg.nstlist,
                "sponge_refresh_interval": cfg.nstlist,
            },
            "output_interval": cfg.log_interval,
            "mode_coupling": {
                "NVT": "langevin",
                "NPT": "berendsen_barostat",
            },
            "sponge_input": {
                "type": "amber",
                "parm7": str(cfg.sponge_parm7_file),
                "rst7_nvt": str(cfg.sponge_rst7_file_nvt),
                "rst7_npt": str(cfg.sponge_rst7_file_npt),
            },
        },
        "results": results,
    }
