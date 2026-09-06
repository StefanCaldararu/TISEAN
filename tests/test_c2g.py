import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
C2G_BIN = REPO_ROOT / "bin" / "c2g"
C2G_INPUT_D1M3T1 = REPO_ROOT / "tests" / "refs" / "c2naive_d1M3t1l300.txt"
C2G_INPUT_M2M4 = REPO_ROOT / "tests" / "refs" / "c2naive_m2M4d1t1l300.txt"


def run_c2g():
    subprocess.run(
        ["./bin/c2g", "-o", "./tests/refs/c2g_out.txt", "./tests/refs/c2naive_ref.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/c2g_out.txt") as f:
        return f.read()


def run_c2g_tmp(tmp_path, input_path, *args):
    # c2g.f declares file/fout as character*72 (source_f/c2g.f:28), so run in
    # a scratch cwd with short relative filenames -- see run_c1 at
    # tests/test_c1.py:12.
    shutil.copy(input_path, tmp_path / "input.txt")

    cmd = [str(C2G_BIN), *args, "-o", "out.txt", "input.txt"]
    subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=tmp_path)
    return (tmp_path / "out.txt").read_text()


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers/blank lines)
        if len(parts) == 3:
            try:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
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


def test_c2g_regression():
    out = run_c2g()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2g_ref.txt", comments="#")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)


def test_c2g_d1M3t1(tmp_path):
    out = run_c2g_tmp(tmp_path, C2G_INPUT_D1M3T1)
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2g_d1M3t1.txt", comments="#")
    np.testing.assert_allclose(data, ref, rtol=1e-6)


def test_c2g_m2M4t1(tmp_path):
    out = run_c2g_tmp(tmp_path, C2G_INPUT_M2M4)
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2g_m2M4t1.txt", comments="#")
    np.testing.assert_allclose(data, ref, rtol=1e-6)


def test_c2g_writes_three_columns(tmp_path):
    out = run_c2g_tmp(tmp_path, C2G_INPUT_D1M3T1)
    for line in out.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        assert len(s.split()) == 3


def test_c2g_row_count_and_radii_per_block(tmp_path):
    # c2g emits exactly one output row per input row (source_f/c2g.f:58),
    # and its first column is the input's radius column sorted ascending
    # (c2g.f:55-57 sorts by e = log(radius) before the loop). Input block
    # sizes are 36, 20, 14.
    input_dims, input_rows = parse_blocks(C2G_INPUT_D1M3T1.read_text(), 2)
    in_sizes = block_sizes(input_dims)

    out = run_c2g_tmp(tmp_path, C2G_INPUT_D1M3T1)
    out_dims, out_rows = parse_blocks(out, 3)
    out_sizes = block_sizes(out_dims)

    assert sorted(out_sizes) == sorted(in_sizes)
    for m, n_in in in_sizes.items():
        assert out_sizes[m] == n_in

        in_radii_sorted = np.sort(input_rows[input_dims == m][:, 0])
        out_radii = out_rows[out_dims == m][:, 0]
        np.testing.assert_allclose(out_radii, in_radii_sorted, rtol=1e-6)


def test_c2g_labels_pass_through(tmp_path):
    out = run_c2g_tmp(tmp_path, C2G_INPUT_D1M3T1)
    dims, _ = parse_blocks(out, 3)

    assert sorted(set(dims.tolist())) == [1, 2, 3]


def test_c2g_labels_pass_through_m2M4(tmp_path):
    out = run_c2g_tmp(tmp_path, C2G_INPUT_M2M4)
    dims, _ = parse_blocks(out, 3)

    assert sorted(set(dims.tolist())) == [2, 3, 4]
