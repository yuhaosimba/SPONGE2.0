from pathlib import Path

from benchmarks.performance.gmx_vs_sponge.tests import utils


def test_water160k_gpu_throughput_compare(outputs_path, bench_config):
    cfg = bench_config
    utils.ensure_requirements(cfg)

    gro_meta_nvt = utils.parse_gro_meta(cfg.case_dir / cfg.gro_file_nvt)
    gro_meta_npt = utils.parse_gro_meta(cfg.case_dir / cfg.gro_file_npt)
    assert gro_meta_nvt["natom"] == gro_meta_npt["natom"]

    align_meta = {
        "NVT": utils.assert_coordinate_and_box_alignment(
            gro_path=cfg.case_dir / cfg.gro_file_nvt,
            rst7_path=Path(cfg.sponge_rst7_file_nvt),
            coord_tol_angstrom=cfg.coordinate_tolerance_angstrom,
            box_tol_angstrom=cfg.box_tolerance_angstrom,
        ),
        "NPT": utils.assert_coordinate_and_box_alignment(
            gro_path=cfg.case_dir / cfg.gro_file_npt,
            rst7_path=Path(cfg.sponge_rst7_file_npt),
            coord_tol_angstrom=cfg.coordinate_tolerance_angstrom,
            box_tol_angstrom=cfg.box_tolerance_angstrom,
        ),
    }

    output_root = Path(outputs_path) / "water160k_gpu"
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in ("NVT", "NPT"):
        for engine in ("sponge", "gromacs"):
            result = utils.benchmark_mode(
                output_root=output_root,
                cfg=cfg,
                mode=mode,
                engine=engine,
            )
            results.append(result)

    summary = utils.build_summary(
        cfg=cfg,
        gro_meta=gro_meta_npt,
        output_root=output_root,
        results=results,
    )
    summary["alignment_check"] = align_meta
    summary_json, summary_csv = utils.write_summary_files(output_root, summary)

    utils.print_result_table(results)
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV : {summary_csv}")

    # Basic consistency checks for required protocol.
    for item in results:
        assert item["measure_count"] == cfg.repeats
        assert item["warmup_count"] == cfg.warmup
        assert item["ns_per_day"]["count"] == cfg.repeats
        assert item["steps_per_s"]["count"] == cfg.repeats
