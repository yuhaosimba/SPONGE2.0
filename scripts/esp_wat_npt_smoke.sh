#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPONGE_BIN="${SPONGE_BIN:-${ROOT_DIR}/build-dev-cuda13/SPONGE}"
PM_BACKEND="${PM_BACKEND:-esp}"
STEP_LIMIT="${STEP_LIMIT:-4}"
BAROSTAT_UPDATE_INTERVAL="${BAROSTAT_UPDATE_INTERVAL:-1}"

if [[ ! -x "${SPONGE_BIN}" ]]; then
  echo "SPONGE executable not found: ${SPONGE_BIN}" >&2
  exit 1
fi
if [[ ! -d "${ROOT_DIR}/benchmarks/performance/nonortho/statics/WAT_nonortho" ]]; then
  echo "Static WAT_nonortho case not found: ${ROOT_DIR}/benchmarks/performance/nonortho/statics/WAT_nonortho" >&2
  exit 1
fi
if [[ "${PM_BACKEND}" != "esp" && "${PM_BACKEND}" != "pme" ]]; then
  echo "Unsupported PM_BACKEND: ${PM_BACKEND}" >&2
  exit 1
fi

WORK_ROOT="$(mktemp -d /tmp/esp_wat_npt.XXXXXX)"
trap 'rm -rf "${WORK_ROOT}"' EXIT
CASE_DIR="${WORK_ROOT}/WAT_nonortho_${PM_BACKEND}_npt"
cp -R "${ROOT_DIR}/benchmarks/performance/nonortho/statics/WAT_nonortho" "${CASE_DIR}"

python3 - <<'PY' "${CASE_DIR}/WAT_mass.txt" "${CASE_DIR}/initial_velocity.txt"
import math
import random
import sys
from pathlib import Path

mass_path = Path(sys.argv[1])
velocity_path = Path(sys.argv[2])

lines = mass_path.read_text(encoding="utf-8").splitlines()
masses = [float(x) for x in lines[1:] if x.strip()]
if not masses:
    raise SystemExit(f"Empty mass file: {mass_path}")

rng = random.Random(2026)
k_b = 0.00198716
temperature = 300.0
velocities = []
mobile_indices = []
for idx, mass in enumerate(masses):
    if mass > 0.0:
        sigma = math.sqrt(k_b * temperature / mass)
        velocity = [rng.gauss(0.0, sigma) for _ in range(3)]
        velocities.append(velocity)
        mobile_indices.append(idx)
    else:
        velocities.append([0.0, 0.0, 0.0])

total_mass = sum(masses[idx] for idx in mobile_indices)
if total_mass <= 0.0:
    raise SystemExit("No positive-mass atoms found for velocity initialization")

for axis in range(3):
    center_velocity = (
        sum(masses[idx] * velocities[idx][axis] for idx in mobile_indices)
        / total_mass
    )
    for idx in mobile_indices:
        velocities[idx][axis] -= center_velocity

kinetic = 0.5 * sum(
    masses[idx] * sum(component * component for component in velocities[idx])
    for idx in mobile_indices
)
dof = 3 * len(mobile_indices) - 3
scale = math.sqrt(temperature * dof * k_b / (2.0 * kinetic))
for idx in mobile_indices:
    for axis in range(3):
        velocities[idx][axis] *= scale

velocity_path.write_text(
    "\n".join(
        [str(len(masses))]
        + [f"{v[0]:.7f} {v[1]:.7f} {v[2]:.7f}" for v in velocities]
    )
    + "\n",
    encoding="utf-8",
)
PY

cat > "${CASE_DIR}/mdin.spg.toml" <<EOF
md_name = "WAT_nonortho ${PM_BACKEND} NPT smoke"
mode = "npt"
step_limit = ${STEP_LIMIT}
dt = 0.001
cutoff = 8.0
default_in_file_prefix = "WAT"
constrain_mode = "SETTLE"
crd = "mdcrd.dat"
frc = "frc.dat"
box = "mdbox.txt"
mdout = "mdout.txt"
mdinfo = "mdinfo.txt"
velocity_in_file = "initial_velocity.txt"
thermostat = "middle_langevin"
thermostat_tau = 0.1
thermostat_seed = 2026
target_temperature = 300.0
barostat = "andersen_barostat"
barostat_tau = 0.1
barostat_update_interval = ${BAROSTAT_UPDATE_INTERVAL}
target_pressure = 1.0
print_zeroth_frame = 1
write_information_interval = 1
write_mdout_interval = 1
write_trajectory_interval = 1
write_restart_file_interval = 1
dont_check_input = 1

[PM]
backend = "${PM_BACKEND}"
Direct_Tolerance = 1e-4
print_detail = true
EOF

if [[ "${PM_BACKEND}" == "esp" ]]; then
  cat >> "${CASE_DIR}/mdin.spg.toml" <<'EOF'
esp_tolerance = 1e-4
esp_table_mode = "poly"
esp_table_points = 1024
esp_print_detail = true
EOF
fi

(
  cd "${CASE_DIR}"
  "${SPONGE_BIN}" -mdin mdin.spg.toml > sponge.stdout 2> sponge.stderr
)

if [[ ! -s "${CASE_DIR}/mdout.txt" ]]; then
  echo "WAT_nonortho ${PM_BACKEND} NPT smoke failed: mdout.txt was not generated" >&2
  cat "${CASE_DIR}/sponge.stdout" >&2 || true
  cat "${CASE_DIR}/sponge.stderr" >&2 || true
  exit 1
fi
if [[ ! -s "${CASE_DIR}/mdbox.txt" ]]; then
  echo "WAT_nonortho ${PM_BACKEND} NPT smoke failed: mdbox.txt was not generated" >&2
  cat "${CASE_DIR}/sponge.stdout" >&2 || true
  cat "${CASE_DIR}/sponge.stderr" >&2 || true
  exit 1
fi

python3 - <<'PY' "${CASE_DIR}/mdout.txt" "${CASE_DIR}/mdbox.txt" "${CASE_DIR}/sponge.stdout" "${CASE_DIR}/sponge.stderr"
import math
import re
import sys
from pathlib import Path

pattern = re.compile(r"(?<![A-Za-z])[-+]?(?:nan(?:\([^)]*\))?|inf(?:inity)?)(?![A-Za-z])", re.I)
for name in sys.argv[1:]:
    text = Path(name).read_text(encoding="utf-8", errors="replace")
    if pattern.search(text):
        print(f"NPT smoke failed: NaN/Inf detected in {name}", file=sys.stderr)
        raise SystemExit(1)

box_rows = []
for line in Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace").splitlines():
    fields = line.split()
    if len(fields) < 6:
        continue
    box_rows.append([float(x) for x in fields[:6]])
if len(box_rows) < 2:
    print("NPT smoke failed: expected at least two box frames", file=sys.stderr)
    raise SystemExit(1)

def triclinic_volume(row):
    lx, ly, lz, alpha_deg, beta_deg, gamma_deg = row
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        return -1.0
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    metric_det = 1.0 + 2.0 * cos_a * cos_b * cos_g - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g
    if metric_det < 0.0 and metric_det > -1.0e-12:
        metric_det = 0.0
    if metric_det <= 0.0:
        return -1.0
    return lx * ly * lz * math.sqrt(metric_det)

volumes = [triclinic_volume(row) for row in box_rows]
if any(volume <= 0.0 for volume in volumes):
    print("NPT smoke failed: non-positive triclinic volume detected", file=sys.stderr)
    raise SystemExit(1)

if max(abs(a - b) for a, b in zip(box_rows[0], box_rows[-1])) <= 1.0e-8:
    print("NPT smoke failed: box did not change across the run", file=sys.stderr)
    raise SystemExit(1)
PY

echo "WAT_nonortho ${PM_BACKEND} NPT smoke passed"
tail -n 6 "${CASE_DIR}/mdout.txt"
echo
echo "Final box samples:"
tail -n 4 "${CASE_DIR}/mdbox.txt"
