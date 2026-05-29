#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPONGE_BIN="${SPONGE_BIN:-${ROOT_DIR}/build-dev-cuda13/SPONGE}"
STATIC_CASE="${ROOT_DIR}/benchmarks/performance/nonortho/statics/WAT_nonortho"

if [[ ! -x "${SPONGE_BIN}" ]]; then
  echo "SPONGE executable not found: ${SPONGE_BIN}" >&2
  exit 1
fi
if [[ ! -d "${STATIC_CASE}" ]]; then
  echo "Static WAT_nonortho case not found: ${STATIC_CASE}" >&2
  exit 1
fi

WORK_ROOT="$(mktemp -d /tmp/esp_wat_single_step.XXXXXX)"
trap 'rm -rf "${WORK_ROOT}"' EXIT
CASE_DIR="${WORK_ROOT}/WAT_nonortho_esp"
cp -R "${STATIC_CASE}" "${CASE_DIR}"

cat > "${CASE_DIR}/mdin.spg.toml" <<'EOF'
md_name = "ESP WAT_nonortho single-step smoke"
mode = "nve"
step_limit = 1
dt = 0.001
cutoff = 8.0
default_in_file_prefix = "WAT"
constrain_mode = "SETTLE"
crd = "mdcrd.dat"
frc = "frc.dat"
box = "mdbox.txt"
mdout = "mdout.txt"
mdinfo = "mdinfo.txt"
print_zeroth_frame = 1
write_information_interval = 1
write_mdout_interval = 1
write_trajectory_interval = 1
write_restart_file_interval = 1
dont_check_input = 1

[PM]
backend = "esp"
Direct_Tolerance = 1e-4
grid_spacing = 1.0
esp_tolerance = 1e-4
esp_order = 6
esp_grid_spacing = 1.6
esp_parameter_mode = "manual"
esp_table_mode = "poly"
esp_table_points = 1024
esp_print_detail = true
EOF

(
  cd "${CASE_DIR}"
  "${SPONGE_BIN}" -mdin mdin.spg.toml > sponge.stdout 2> sponge.stderr
)

if [[ ! -s "${CASE_DIR}/mdout.txt" ]]; then
  echo "ESP WAT smoke failed: mdout.txt was not generated" >&2
  cat "${CASE_DIR}/sponge.stdout" >&2 || true
  cat "${CASE_DIR}/sponge.stderr" >&2 || true
  exit 1
fi

python3 - <<'PY' "${CASE_DIR}/mdout.txt" "${CASE_DIR}/sponge.stdout" "${CASE_DIR}/sponge.stderr"
import re
import sys
from pathlib import Path

pattern = re.compile(r"(?<![A-Za-z])[-+]?(?:nan(?:\([^)]*\))?|inf(?:inity)?)(?![A-Za-z])", re.I)
for name in sys.argv[1:]:
    text = Path(name).read_text(encoding="utf-8", errors="replace")
    if pattern.search(text):
        print(f"ESP WAT smoke failed: NaN/Inf detected in {name}", file=sys.stderr)
        print(Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
        raise SystemExit(1)
PY

echo "ESP WAT single-step smoke passed"
tail -n 5 "${CASE_DIR}/mdout.txt"
