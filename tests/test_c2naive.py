import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
C2NAIVE_BIN = REPO_ROOT / "bin" / "c2naive"
C2NAIVE_INPUT = REPO_ROOT / "tests" / "refs" / "ar-run_l1000.txt"

# sc = max(x) - min(x) over the points actually read. Every reference file
# below is generated with -l300 and no -x/-c, so they all read the same
# first 300 points of the series and share this scale.
_SERIES_L300 = np.loadtxt(C2NAIVE_INPUT)[:300]
SC_L300 = _SERIES_L300.max() - _SERIES_L300.min()


def run_c2naive():
    subprocess.run(
        ["./bin/c2naive", "-d1", "-M3", "-t0", "-l300",
         "-o", "./tests/refs/c2naive_out.txt", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/c2naive_out.txt") as f:
        return f.read()


def run_c2naive_tmp(tmp_path, *args):
    # c2naive.f declares file/fout as character*72 and never writes to
    # stdout without -o (source_f/c2naive.f:50/58 fall back to
    # <inputfile>_c2 silently). Run with short, relative filenames in a
    # scratch cwd so long tmp_path/repo paths never get silently
    # truncated by the Fortran runtime.
    shutil.copy(C2NAIVE_INPUT, tmp_path / "input.txt")

    cmd = [str(C2NAIVE_BIN), *args, "-o", "out.txt", "input.txt"]
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


def parse_blocks(text):
    """Returns (dims, radii, values): dims[i] is the embedding dimension
    (from the '#m=' header) that row i belongs to."""
    dims, radii, values = [], [], []
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
                r, v = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            dims.append(current_m)
            radii.append(r)
            values.append(v)
    return np.array(dims), np.array(radii), np.array(values)


def assert_structural_properties(dims, radii, values, res, sc):
    """Properties that hold for every flag combination (source_f/c2naive.f:62-75),
    checked per '#m=' block."""
    for m in sorted(set(dims.tolist())):
        mask = dims == m
        block_radii = radii[mask]
        block_values = values[mask]

        # c(0,m)/c(0,m) is exactly 1.0
        assert block_values[0] == 1.0

        # cumulated downward from the top bin: non-increasing
        assert np.all(np.diff(block_values) <= 1e-9)

        assert np.all(block_values > 0)
        assert np.all(block_values <= 1)

        j = np.arange(len(block_radii))
        expected_radii = sc * 2.0 ** (-j / res)
        np.testing.assert_allclose(block_radii, expected_radii, rtol=1e-6)


def test_c2naive_regression():
    out = run_c2naive()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2naive_ref.txt", comments="#")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)


def test_c2naive_d1M3t1l300(tmp_path):
    out = run_c2naive_tmp(tmp_path, "-d1", "-M3", "-t1", "-l300")
    dims, radii, values = parse_blocks(out)

    ref = np.loadtxt("tests/refs/c2naive_d1M3t1l300.txt", comments="#")
    np.testing.assert_allclose(np.column_stack([radii, values]), ref, rtol=1e-6)

    assert_structural_properties(dims, radii, values, res=2.0, sc=SC_L300)


def test_c2naive_d2M3t1l300(tmp_path):
    out = run_c2naive_tmp(tmp_path, "-d2", "-M3", "-t1", "-l300")
    dims, radii, values = parse_blocks(out)

    ref = np.loadtxt("tests/refs/c2naive_d2M3t1l300.txt", comments="#")
    np.testing.assert_allclose(np.column_stack([radii, values]), ref, rtol=1e-6)

    assert_structural_properties(dims, radii, values, res=2.0, sc=SC_L300)


def test_c2naive_m2M4d1t1l300(tmp_path):
    out = run_c2naive_tmp(tmp_path, "-m2", "-M4", "-d1", "-t1", "-l300")
    dims, radii, values = parse_blocks(out)

    ref = np.loadtxt("tests/refs/c2naive_m2M4d1t1l300.txt", comments="#")
    np.testing.assert_allclose(np.column_stack([radii, values]), ref, rtol=1e-6)

    # -m2 -M4 labels blocks 2, 3, 4 -- not 1, 2, 3
    assert sorted(set(dims.tolist())) == [2, 3, 4]

    assert_structural_properties(dims, radii, values, res=2.0, sc=SC_L300)


def test_c2naive_res4d1M3t1l300(tmp_path):
    out = run_c2naive_tmp(tmp_path, "-#4", "-d1", "-M3", "-t1", "-l300")
    dims, radii, values = parse_blocks(out)

    ref = np.loadtxt("tests/refs/c2naive_res4d1M3t1l300.txt", comments="#")
    np.testing.assert_allclose(np.column_stack([radii, values]), ref, rtol=1e-6)

    assert_structural_properties(dims, radii, values, res=4.0, sc=SC_L300)


def test_c2naive_T20d1M3t1l300(tmp_path):
    out = run_c2naive_tmp(tmp_path, "-T20", "-d1", "-M3", "-t1", "-l300")
    dims, radii, values = parse_blocks(out)

    ref = np.loadtxt("tests/refs/c2naive_T20d1M3t1l300.txt", comments="#")
    np.testing.assert_allclose(np.column_stack([radii, values]), ref, rtol=1e-6)

    assert_structural_properties(dims, radii, values, res=2.0, sc=SC_L300)


def test_c2naive_flags_change_output(tmp_path):
    # -d, -m, -M, -t, -T, -#, -l, -x and -c each affect the output. A port
    # that silently ignores one of these will fail this test even though it
    # might still pass the golden comparisons above (which only vary a few
    # flags at a time).
    baseline = run_c2naive_tmp(tmp_path, "-d1", "-M3", "-t1", "-l300")

    variations = {
        "-d": ["-d2", "-M3", "-t1", "-l300"],
        "-m": ["-d1", "-m2", "-M3", "-t1", "-l300"],
        "-M": ["-d1", "-M4", "-t1", "-l300"],
        "-t": ["-d1", "-M3", "-t0", "-l300"],
        "-T": ["-d1", "-M3", "-t1", "-l300", "-T20"],
        "-#": ["-d1", "-M3", "-t1", "-l300", "-#4"],
        "-l": ["-d1", "-M3", "-t1", "-l200"],
        "-x": ["-d1", "-M3", "-t1", "-l300", "-x50"],
        "-c": ["-d1", "-M3", "-t1", "-l300", "-c2"],
    }

    for name, args in variations.items():
        out = run_c2naive_tmp(tmp_path, *args)
        assert out != baseline, name
