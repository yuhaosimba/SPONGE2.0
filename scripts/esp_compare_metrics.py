#!/usr/bin/env python3

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.comparison.utils import force_stats
from benchmarks.utils import Extractor, Outputer


DEFAULT_MDOUT_NAME = "mdout.txt"
DEFAULT_FORCE_NAME = "frc.dat"
DEFAULT_ENERGY_COLUMNS = [
    "potential",
    "PM",
    "PM_direct",
    "PM_reciprocal",
    "PM_self",
    "PM_correction",
]
DEFAULT_AUX_COLUMNS = [
    "pressure",
    "Pxx",
    "Pyy",
    "Pzz",
    "Pxy",
    "Pxz",
    "Pyz",
]
ATOM_COUNT_CANDIDATES = [
    "atom_numbers.txt",
    "mass.txt",
    "charge.txt",
]


@dataclass
class CaseSpec:
    label: str
    case_dir: Path | None
    mdout_path: Path
    force_path: Path
    atom_count: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two SPONGE runs in force and electrostatic energy "
            "decomposition, with an optional third reference run."
        )
    )
    add_case_arguments(parser, "a", "PME")
    add_case_arguments(parser, "b", "ESP")
    add_case_arguments(parser, "reference", "REF", required=False)
    parser.add_argument(
        "--mdout-name",
        default=DEFAULT_MDOUT_NAME,
        help=f"Default mdout file name inside case directories. Default: {DEFAULT_MDOUT_NAME}",
    )
    parser.add_argument(
        "--frc-name",
        default=DEFAULT_FORCE_NAME,
        help=f"Default force file name inside case directories. Default: {DEFAULT_FORCE_NAME}",
    )
    parser.add_argument(
        "--atom-count",
        type=int,
        default=None,
        help="Global atom count override for all compared cases.",
    )
    parser.add_argument(
        "--atom-count-file",
        default=None,
        help=(
            "Optional file name inside each case directory used to read the atom "
            "count before falling back to built-in candidates."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON summary output path.",
    )
    return parser.parse_args()


def add_case_arguments(parser, prefix, default_label, *, required=True):
    parser.add_argument(
        f"--case-{prefix}",
        default=None,
        required=False,
        help=f"Case directory for {prefix}.",
    )
    parser.add_argument(
        f"--mdout-{prefix}",
        default=None,
        required=False,
        help=f"mdout path for {prefix}. Overrides case directory lookup.",
    )
    parser.add_argument(
        f"--frc-{prefix}",
        default=None,
        required=False,
        help=f"Force file path for {prefix}. Overrides case directory lookup.",
    )
    parser.add_argument(
        f"--label-{prefix}",
        default=default_label,
        help=f"Display label for {prefix}. Default: {default_label}",
    )
    if required:
        parser.set_defaults(**{f"{prefix}_required": True})


def require_existing_file(path_like, *, label):
    path = Path(path_like).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_existing_dir(path_like, *, label):
    path = Path(path_like).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def resolve_case_spec(args, prefix):
    label = getattr(args, f"label_{prefix}")
    case_dir_arg = getattr(args, f"case_{prefix}")
    mdout_arg = getattr(args, f"mdout_{prefix}")
    frc_arg = getattr(args, f"frc_{prefix}")

    if case_dir_arg is None and mdout_arg is None and frc_arg is None:
        if prefix == "reference":
            return None
        raise ValueError(f"Missing inputs for '{prefix}': provide --case-{prefix} or explicit mdout/frc paths")

    case_dir = None
    if case_dir_arg is not None:
        case_dir = require_existing_dir(case_dir_arg, label=f"{label} case directory")

    if mdout_arg is None:
        if case_dir is None:
            raise ValueError(f"Missing --mdout-{prefix} because --case-{prefix} was not provided")
        mdout_path = require_existing_file(
            case_dir / args.mdout_name,
            label=f"{label} mdout",
        )
    else:
        mdout_path = require_existing_file(mdout_arg, label=f"{label} mdout")

    if frc_arg is None:
        if case_dir is None:
            raise ValueError(f"Missing --frc-{prefix} because --case-{prefix} was not provided")
        force_path = require_existing_file(
            case_dir / args.frc_name,
            label=f"{label} force file",
        )
    else:
        force_path = require_existing_file(frc_arg, label=f"{label} force file")

    atom_count = resolve_atom_count(
        case_dir=case_dir,
        explicit_atom_count=args.atom_count,
        atom_count_file=args.atom_count_file,
        label=label,
    )
    return CaseSpec(
        label=label,
        case_dir=case_dir,
        mdout_path=mdout_path,
        force_path=force_path,
        atom_count=atom_count,
    )


def resolve_atom_count(*, case_dir, explicit_atom_count, atom_count_file, label):
    if explicit_atom_count is not None:
        if explicit_atom_count <= 0:
            raise ValueError(f"Invalid atom count for {label}: {explicit_atom_count}")
        return int(explicit_atom_count)

    if case_dir is None:
        raise ValueError(
            f"Atom count for {label} could not be inferred without a case directory. "
            "Pass --atom-count."
        )

    candidates = []
    if atom_count_file:
        candidates.append(case_dir / atom_count_file)
    for name in ATOM_COUNT_CANDIDATES:
        candidates.append(case_dir / name)
    candidates.extend(sorted(case_dir.glob("*_mass.txt")))
    candidates.extend(sorted(case_dir.glob("*_charge.txt")))

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            atom_count = Extractor.read_first_field_int(candidate)
        except Exception:
            continue
        if atom_count > 0:
            return atom_count

    raise ValueError(
        f"Failed to infer atom count for {label} under {case_dir}. Pass --atom-count or --atom-count-file."
    )


def load_mdout_rows(mdout_path):
    mdout = Extractor._read_mdout(mdout_path.parent, mdout_path.name)
    available_columns = [
        column
        for column in ["step", *DEFAULT_ENERGY_COLUMNS, *DEFAULT_AUX_COLUMNS]
        if hasattr(mdout, column)
    ]
    rows = Extractor.parse_mdout_rows(
        mdout_path,
        available_columns,
        int_columns=("step",),
    )
    return rows, available_columns


def load_case_data(spec):
    force = Extractor.extract_sponge_forces(
        spec.force_path.parent,
        spec.atom_count,
        frc_name=spec.force_path.name,
    )
    rows, available_columns = load_mdout_rows(spec.mdout_path)
    final_row = rows[-1]
    return {
        "label": spec.label,
        "case_dir": str(spec.case_dir) if spec.case_dir is not None else None,
        "mdout_path": str(spec.mdout_path),
        "force_path": str(spec.force_path),
        "atom_count": spec.atom_count,
        "forces": force,
        "mdout_rows": rows,
        "available_columns": available_columns,
        "final_row": final_row,
    }


def vector_norm(values):
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64).ravel()))


def compute_force_metrics(reference_force, predicted_force):
    reference_force = np.asarray(reference_force, dtype=np.float64)
    predicted_force = np.asarray(predicted_force, dtype=np.float64)
    stats = force_stats(reference_force, predicted_force)
    diff = predicted_force - reference_force
    ref_l2 = vector_norm(reference_force)
    pred_l2 = vector_norm(predicted_force)
    diff_l2 = vector_norm(diff)
    per_atom_l2 = np.linalg.norm(diff, axis=1)
    per_atom_inf = np.max(np.abs(diff), axis=1)
    denom = ref_l2 if ref_l2 > 1.0e-30 else 1.0
    stats.update(
        {
            "reference_l2": ref_l2,
            "predicted_l2": pred_l2,
            "diff_l2": diff_l2,
            "rel_l2_diff": float(diff_l2 / denom),
            "max_atom_l2_diff": float(np.max(per_atom_l2)),
            "max_atom_inf_diff": float(np.max(per_atom_inf)),
        }
    )
    return stats


def compute_energy_component_summary(case_a, case_b, reference_case):
    common = sorted(
        set(case_a["final_row"]).intersection(case_b["final_row"]).intersection(
            DEFAULT_ENERGY_COLUMNS + DEFAULT_AUX_COLUMNS
        )
    )
    result = []
    for column in common:
        row = {
            "component": column,
            case_a["label"]: float(case_a["final_row"][column]),
            case_b["label"]: float(case_b["final_row"][column]),
            f"{case_b['label']}-{case_a['label']}": float(
                case_b["final_row"][column] - case_a["final_row"][column]
            ),
        }
        if reference_case is not None and column in reference_case["final_row"]:
            ref_value = float(reference_case["final_row"][column])
            row[reference_case["label"]] = ref_value
            denom = max(1.0, abs(ref_value))
            row[f"{case_a['label']}-ref_abs"] = abs(
                float(case_a["final_row"][column]) - ref_value
            )
            row[f"{case_b['label']}-ref_abs"] = abs(
                float(case_b["final_row"][column]) - ref_value
            )
            row[f"{case_a['label']}-ref_rel"] = row[f"{case_a['label']}-ref_abs"] / denom
            row[f"{case_b['label']}-ref_rel"] = row[f"{case_b['label']}-ref_abs"] / denom
        result.append(row)
    return result


def build_series(case_data, component):
    values = [row[component] for row in case_data["mdout_rows"] if component in row]
    return np.asarray(values, dtype=np.float64)


def compute_delta_metrics(series_reference, series_predicted):
    if series_reference.size < 2 or series_predicted.size < 2:
        return None
    count = min(series_reference.size, series_predicted.size)
    ref_delta = np.diff(series_reference[:count])
    pred_delta = np.diff(series_predicted[:count])
    diff = pred_delta - ref_delta
    denom = max(vector_norm(ref_delta), 1.0)
    return {
        "frames_compared": int(count),
        "max_abs_delta_diff": float(np.max(np.abs(diff))),
        "rms_delta_diff": float(np.sqrt(np.mean(diff * diff))),
        "rel_l2_delta_diff": float(vector_norm(diff) / denom),
    }


def compute_delta_summary(case_a, case_b, reference_case):
    common = sorted(
        set(case_a["available_columns"]).intersection(case_b["available_columns"]).intersection(
            DEFAULT_ENERGY_COLUMNS + DEFAULT_AUX_COLUMNS
        )
    )
    result = {}
    for component in common:
        component_result = {
            f"{case_b['label']}_vs_{case_a['label']}": compute_delta_metrics(
                build_series(case_a, component),
                build_series(case_b, component),
            )
        }
        if reference_case is not None and component in reference_case["available_columns"]:
            ref_series = build_series(reference_case, component)
            component_result[f"{case_a['label']}_vs_{reference_case['label']}"] = compute_delta_metrics(
                ref_series,
                build_series(case_a, component),
            )
            component_result[f"{case_b['label']}_vs_{reference_case['label']}"] = compute_delta_metrics(
                ref_series,
                build_series(case_b, component),
            )
        result[component] = component_result
    return result


def print_case_overview(case_a, case_b, reference_case):
    rows = []
    for case_data in [case_a, case_b, reference_case]:
        if case_data is None:
            continue
        rows.append(
            [
                case_data["label"],
                case_data["atom_count"],
                case_data["mdout_path"],
                case_data["force_path"],
            ]
        )
    Outputer.print_table(
        ["Label", "Atoms", "mdout", "force"],
        rows,
        title="Case Overview",
    )


def print_force_tables(summary, case_a, case_b, reference_case):
    pair_rows = [[key, format_metric_value(value)] for key, value in summary["pair"].items()]
    Outputer.print_table(
        ["Metric", f"{case_b['label']} vs {case_a['label']}"],
        pair_rows,
        title="Force Comparison",
    )
    if reference_case is None:
        return

    ref_rows = []
    for label, metrics in summary["vs_reference"].items():
        for key, value in metrics.items():
            ref_rows.append([label, key, format_metric_value(value)])
    Outputer.print_table(
        ["Case", "Metric", "Value"],
        ref_rows,
        title=f"Force Comparison Against {reference_case['label']}",
    )


def print_energy_table(energy_rows):
    if not energy_rows:
        print("\nNo shared PM/energy components were found in the compared mdout files.")
        return
    headers = list(energy_rows[0].keys())
    rows = [[format_metric_value(row[key]) if isinstance(row[key], float) else row[key] for key in headers] for row in energy_rows]
    Outputer.print_table(headers, rows, title="Final-Frame Energy Comparison")


def print_delta_summary(delta_summary):
    rows = []
    for component, comparisons in delta_summary.items():
        for label, metrics in comparisons.items():
            if metrics is None:
                rows.append([component, label, "n/a", "n/a", "n/a"])
                continue
            rows.append(
                [
                    component,
                    label,
                    metrics["frames_compared"],
                    format_metric_value(metrics["max_abs_delta_diff"]),
                    format_metric_value(metrics["rel_l2_delta_diff"]),
                ]
            )
    if not rows:
        return
    Outputer.print_table(
        ["Component", "Comparison", "Frames", "max_abs_delta_diff", "rel_l2_delta_diff"],
        rows,
        title="Frame-to-Frame Delta Comparison",
    )


def format_metric_value(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return str(float(value))
        return f"{float(value):.10e}"
    return str(value)


def make_json_ready(value):
    if isinstance(value, dict):
        return {key: make_json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    args = parse_args()
    spec_a = resolve_case_spec(args, "a")
    spec_b = resolve_case_spec(args, "b")
    spec_reference = resolve_case_spec(args, "reference")

    if spec_a.atom_count != spec_b.atom_count:
        raise ValueError(
            f"Atom count mismatch: {spec_a.label}={spec_a.atom_count}, "
            f"{spec_b.label}={spec_b.atom_count}"
        )
    if spec_reference is not None and spec_reference.atom_count != spec_a.atom_count:
        raise ValueError(
            f"Atom count mismatch: {spec_reference.label}={spec_reference.atom_count}, "
            f"{spec_a.label}={spec_a.atom_count}"
        )

    case_a = load_case_data(spec_a)
    case_b = load_case_data(spec_b)
    reference_case = load_case_data(spec_reference) if spec_reference is not None else None

    pair_metrics = compute_force_metrics(case_a["forces"], case_b["forces"])
    summary = {
        "case_a": {
            "label": case_a["label"],
            "case_dir": case_a["case_dir"],
            "mdout_path": case_a["mdout_path"],
            "force_path": case_a["force_path"],
            "atom_count": case_a["atom_count"],
            "available_columns": case_a["available_columns"],
        },
        "case_b": {
            "label": case_b["label"],
            "case_dir": case_b["case_dir"],
            "mdout_path": case_b["mdout_path"],
            "force_path": case_b["force_path"],
            "atom_count": case_b["atom_count"],
            "available_columns": case_b["available_columns"],
        },
        "pair": pair_metrics,
        "energy_final": compute_energy_component_summary(
            case_a,
            case_b,
            reference_case,
        ),
        "energy_delta": compute_delta_summary(case_a, case_b, reference_case),
    }

    if reference_case is not None:
        ref_force = reference_case["forces"]
        summary["reference"] = {
            "label": reference_case["label"],
            "case_dir": reference_case["case_dir"],
            "mdout_path": reference_case["mdout_path"],
            "force_path": reference_case["force_path"],
            "atom_count": reference_case["atom_count"],
            "available_columns": reference_case["available_columns"],
        }
        summary["vs_reference"] = {
            case_a["label"]: compute_force_metrics(ref_force, case_a["forces"]),
            case_b["label"]: compute_force_metrics(ref_force, case_b["forces"]),
            f"{case_b['label']}_pair_relative_to_{reference_case['label']}": {
                "rel_l2_pair_diff": float(
                    vector_norm(case_b["forces"] - case_a["forces"])
                    / max(vector_norm(ref_force), 1.0)
                )
            },
        }

    print_case_overview(case_a, case_b, reference_case)
    print_force_tables(summary, case_a, case_b, reference_case)
    print_energy_table(summary["energy_final"])
    print_delta_summary(summary["energy_delta"])

    if args.json_out:
        json_path = Path(args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(make_json_ready(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote JSON summary to {json_path}")


if __name__ == "__main__":
    main()
