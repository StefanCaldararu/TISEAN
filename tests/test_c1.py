import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
C1_BIN = REPO_ROOT / "bin" / "c1"
C1_INPUT = REPO_ROOT / "tests" / "refs" / "ar-run_l1000.txt"


def run_c1(tmp_path, *args):
    # c1.f declares file/fout as character*72, and c1 (unlike addnoise or
    # surrogates) always writes to a file rather than stdout when -o is
    # omitted. Run with short, relative filenames in a scratch cwd so long
    # tmp_path/repo paths never get silently truncated by the Fortran
    # runtime.
    shutil.copy(C1_INPUT, tmp_path / "input.txt")

    cmd = [str(C1_BIN), "-d1", "-t0", "-l300", *args, "-o", "out.txt", "input.txt"]
    subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=tmp_path)
    return (tmp_path / "out.txt").read_text()


def parse_c1(text):
    """Returns (dims, radii, masses): dims[i] is the embedding dimension
    (from the '#m=' header) that row i belongs to."""
    dims, radii, masses = [], [], []
    current_m = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#m="):
            current_m = int(s.split("=")[1])
            continue
        parts = s.split()
        if len(parts) == 2:
            try:
                r, m = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            dims.append(current_m)
            radii.append(r)
            masses.append(m)
    return np.array(dims), np.array(radii), np.array(masses)


def test_c1_mass_column_is_seed_independent(tmp_path):
    for ncmin in (10, 50, 200):
        out1 = run_c1(tmp_path, "-m1", "-M3", "-n%d" % ncmin, "-I1")
        out2 = run_c1(tmp_path, "-m1", "-M3", "-n%d" % ncmin, "-I2")

        _, _, mass1 = parse_c1(out1)
        _, _, mass2 = parse_c1(out2)

        assert len(mass1) == 48
        assert len(mass2) == 48
        np.testing.assert_allclose(mass1, mass2, rtol=1e-6)


def test_c1_radius_agrees_across_seeds_with_enough_references(tmp_path):
    out1 = run_c1(tmp_path, "-m1", "-M3", "-n200", "-I1")
    out2 = run_c1(tmp_path, "-m1", "-M3", "-n200", "-I2")

    _, radius1, _ = parse_c1(out1)
    _, radius2, _ = parse_c1(out2)

    rel_diff = np.abs(radius1 - radius2) / radius1
    assert rel_diff.max() < 0.25


def test_c1_m_and_M_select_embedding_dimensions(tmp_path):
    out = run_c1(tmp_path, "-m2", "-M4", "-n10", "-I1")
    dims, _, _ = parse_c1(out)

    assert sorted(set(dims.tolist())) == [2, 3, 4]
