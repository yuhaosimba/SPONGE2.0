from pathlib import Path
import subprocess
import textwrap


def test_lj_launch_block_y_sanitize_helper(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    header = (
        repo_root
        / "SPONGE"
        / "Lennard_Jones_force"
        / "Lennard_Jones_force.h"
    )
    source = tmp_path / "lj_launch_block_y.cpp"
    binary = tmp_path / "lj_launch_block_y"

    source.write_text(
        textwrap.dedent(
            f"""
            #include "{header}"

            int CONTROLLER::MPI_rank = 0;

            int main() {{
                if (Sanitize_LJ_Block_Y(0, 32, 1024, 32) != 32) return 1;
                if (Sanitize_LJ_Block_Y(8, 32, 1024, 32) != 8) return 2;
                if (Sanitize_LJ_Block_Y(128, 32, 1024, 32) != 32) return 3;
                if (Sanitize_LJ_Block_Y(-5, 8, 256, 32) != 8) return 4;
                if (Sanitize_LJ_Block_Y(9, 8, 256, 32) != 8) return 5;
                return 0;
            }}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-fopenmp",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
