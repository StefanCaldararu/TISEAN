import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
C2D_BIN = REPO_ROOT / "bin" / "c2d"
C2D_INPUT_D1M3T1 = REPO_ROOT / "tests" / "refs" / "c2naive_d1M3t1l300.txt"
C2D_INPUT_M2M4 = REPO_ROOT / "tests" / "refs" / "c2naive_m2M4d1t1l300.txt"


def run_c2d():
    subprocess.run(
        ["./bin/c2d", "-o", "./tests/refs/c2d_out.txt", "./tests/refs/c2naive_ref.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/c2d_out.txt") as f:
        return f.read()


def run_c2d_tmp(tmp_path, input_path, *args):
    # c2d.f declares file/fout as character*72 (source_f/c2d.f:29), so run in
    # a scratch cwd with short relative filenames -- see run_c1 at
    # tests/test_c1.py:12.
    shutil.copy(input_path, tmp_path / "input.txt")

    cmd = [str(C2D_BIN), *args, "-o", "out.txt", "input.txt"]
    subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=tmp_path)
    return (tmp_path / "out.txt").read_text()


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers/blank lines)
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def parse_blocks(text, ncols):
    """Returns (dims, rows): dims[i] is the '#m=' block row i belongs to,
    rows[i] is that row's ncols values."""
    dims, rows = [], []
    current_m = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#m="):
            current_m = int(s.split("=")[1])
            continue
        parts = s.split()
        if len(parts) == ncols:
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            dims.append(current_m)
            rows.append(vals)
    return np.array(dims), np.array(rows)


def block_sizes(dims):
    return {m: int(np.sum(dims == m)) for m in sorted(set(dims.tolist()))}


def test_c2d_regression():
    out = run_c2d()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2d_ref.txt", comments="#")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)


def test_c2d_a1_d1M3t1(tmp_path):
    out = run_c2d_tmp(tmp_path, C2D_INPUT_D1M3T1, "-a1")
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2d_a1_d1M3t1.txt", comments="#")
    np.testing.assert_allclose(data, ref, rtol=1e-6)


def test_c2d_a2_d1M3t1(tmp_path):
    out = run_c2d_tmp(tmp_path, C2D_INPUT_D1M3T1, "-a2")
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2d_a2_d1M3t1.txt", comments="#")
    np.testing.assert_allclose(data, ref, rtol=1e-6)


def test_c2d_a5_d1M3t1(tmp_path):
    out = run_c2d_tmp(tmp_path, C2D_INPUT_D1M3T1, "-a5")
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2d_a5_d1M3t1.txt", comments="#")
    np.testing.assert_allclose(data, ref, rtol=1e-6)


def test_c2d_a_flag_changes_output(tmp_path):
    # -a is c2d's only substantive flag (besides -o/-V/-h); -a1, -a2 and -a5
    # must each produce a different file.
    out_a1 = run_c2d_tmp(tmp_path, C2D_INPUT_D1M3T1, "-a1")
    out_a2 = run_c2d_tmp(tmp_path, C2D_INPUT_D1M3T1, "-a2")
    out_a5 = run_c2d_tmp(tmp_path, C2D_INPUT_D1M3T1, "-a5")

    assert out_a1 != out_a2
    assert out_a1 != out_a5
    assert out_a2 != out_a5


def test_c2d_writes_two_columns(tmp_path):
    out = run_c2d_tmp(tmp_path, C2D_INPUT_D1M3T1, "-a1")
    for line in out.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        assert len(s.split()) == 2


def test_c2d_row_count_per_block(tmp_path):
    # rows_out == rows_in - 2*iav per block, when no slope is filtered
    # (source_f/c2d.f:57). Input block sizes are 36, 20, 14 (70 rows total
    # across three blocks); checked for -a1, -a2, -a3 and -a5.
    input_dims, _ = parse_blocks(C2D_INPUT_D1M3T1.read_text(), 2)
    in_sizes = block_sizes(input_dims)

    for iav in (1, 2, 3, 5):
        out = run_c2d_tmp(tmp_path, C2D_INPUT_D1M3T1, "-a%d" % iav)
        out_dims, _ = parse_blocks(out, 2)
        out_sizes = block_sizes(out_dims)

        assert sorted(out_sizes) == sorted(in_sizes)
        for m, n_in in in_sizes.items():
            assert out_sizes[m] == n_in - 2 * iav


def test_c2d_labels_pass_through(tmp_path):
    out = run_c2d_tmp(tmp_path, C2D_INPUT_D1M3T1, "-a1")
    dims, _ = parse_blocks(out, 2)

    assert sorted(set(dims.tolist())) == [1, 2, 3]


def test_c2d_labels_pass_through_m2M4(tmp_path):
    out = run_c2d_tmp(tmp_path, C2D_INPUT_M2M4, "-a1")
    dims, _ = parse_blocks(out, 2)

    assert sorted(set(dims.tolist())) == [2, 3, 4]
