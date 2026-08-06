import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints everything through "%e" (6 digits after the point, so ~7
# significant digits), so comparisons against numbers parsed from its
# stdout can't be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

VARIANCE_VAR_EQ_ZERO = 23
SOLVELE_SINGULAR_MATRIX = 19

POLYNOMP_BIN = os.path.abspath("./bin/polynomp")
POLYPAR_BIN = os.path.abspath("./bin/polypar")


def run_cli(args, check=True):
    return subprocess.run(
        [POLYNOMP_BIN] + list(args),
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        check=check,
    )


def generate_order_file(tmp_path, dim, order, suffix=""):
    """Uses the real polypar CLI (not the Python bindings under test) to
    produce a parameter file, so this test doesn't depend on polypar's own
    Python binding being correct. Returns (path, order array of shape
    (plength, dim))."""
    path = tmp_path / f"order{suffix}.pol"
    subprocess.run(
        [POLYPAR_BIN, f"-m{dim}", f"-p{order}", "-o", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = np.loadtxt(path, dtype=np.uint32).reshape(-1, dim)
    return path, rows


def load_series(path, column=1, length=None, exclude=0):
    """Replicates get_series()'s -x/-l/-c handling: skip `exclude` lines,
    then keep up to `length` of the rest, then pick `column` (1-indexed)."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


def parse_cli_output(text, dim):
    """Parses polynomp's stdout/output-file format:
    '#FCE: <in> <out>' then `plength` '# e1 .. edim coeff' lines, then one
    forecasted value per remaining line."""
    fce_insample = fce_outsample = None
    order_rows = []
    coeffs = []
    forecast = []
    for line in text.splitlines():
        if line.startswith("#FCE:"):
            _, a, b = line.split()
            fce_insample, fce_outsample = float(a), float(b)
        elif line.startswith("# "):
            parts = line.split()[1:]
            order_rows.append([int(x) for x in parts[:dim]])
            coeffs.append(float(parts[dim]))
        else:
            s = line.strip()
            if s:
                forecast.append(float(s))
    return (
        fce_insample,
        fce_outsample,
        np.array(order_rows),
        np.array(coeffs),
        np.array(forecast),
    )


CASES = [
    # label, datafile, column, dim, poly_order, delay, insample, step, length, exclude
    ("defaults_all_data", "tests/refs/ar-run_l1000.txt", 1, 2, 3, 1, None, 1000, None, 0),
    ("small_dim1_order2", "tests/refs/ar-run_l1000.txt", 1, 1, 2, 1, 300, 50, None, 0),
    ("dim3_order2_delay2", "tests/refs/ar-run_l1000.txt", 1, 3, 2, 2, 400, 30, None, 0),
    ("order0_constant_only", "tests/refs/ar-run_l1000.txt", 1, 2, 0, 1, 200, 20, None, 0),
    ("insample_equal_length_no_outsample", "tests/refs/ar-run_l1000.txt", 1, 2, 3, 1, 1000, 20, None, 0),
    ("length_and_exclude", "tests/refs/ar-run_l1000.txt", 1, 2, 3, 1, 300, 40, 500, 100),
    ("column2_henon", "tests/refs/henon_l1000.txt", 2, 2, 2, 1, 300, 25, None, 0),
    ("step_zero", "tests/refs/ar-run_l1000.txt", 1, 2, 2, 1, 300, 0, None, 0),
]


@pytest.mark.parametrize(
    "label,datafile,column,dim,poly_order,delay,insample,step,length,exclude",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_fit_matches_cli(
    tmp_path, label, datafile, column, dim, poly_order, delay, insample, step, length, exclude
):
    order_path, order = generate_order_file(tmp_path, dim, poly_order)

    args = [f"-m{dim}", f"-d{delay}", f"-L{step}", f"-c{column}", "-p", str(order_path)]
    if insample is not None:
        args.append(f"-n{insample}")
    if length is not None:
        args.append(f"-l{length}")
    if exclude:
        args.append(f"-x{exclude}")
    args.append(datafile)

    result = run_cli(args)
    cli_fce_in, cli_fce_out, cli_order, cli_coeffs, cli_forecast = parse_cli_output(
        result.stdout, dim
    )
    np.testing.assert_array_equal(cli_order, order)

    series = load_series(datafile, column=column, length=length, exclude=exclude)

    kwargs = dict(delay=delay, step=step)
    if insample is not None:
        kwargs["insample"] = insample
    py_result = tisean.polynomp.fit(series, order, **kwargs)

    assert py_result.dim == dim
    assert py_result.plength == order.shape[0]
    expected_has_outsample = insample is not None and insample < len(series)
    assert py_result.has_outsample == expected_has_outsample

    np.testing.assert_allclose(py_result.param, cli_coeffs, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.fce_insample, cli_fce_in, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.fce_outsample, cli_fce_out, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.forecast, cli_forecast, **CLI_TEXT_TOL)


def test_defaults_match_cli_defaults():
    # No -m/-d/-n/-L at all: CLI defaults are -m2, -d1, -n unset (all data
    # used, no out-of-sample error), -L1000.
    datafile = "tests/refs/ar-run_l1000.txt"
    order_path = "tests/refs/parameter_m2p3.pol"
    order = np.loadtxt(order_path, dtype=np.uint32)

    result = run_cli(["-p", order_path, datafile])
    cli_fce_in, cli_fce_out, cli_order, cli_coeffs, cli_forecast = parse_cli_output(
        result.stdout, dim=2
    )

    series = load_series(datafile)
    py_result = tisean.polynomp.fit(series, order)

    assert py_result.dim == 2
    assert py_result.delay == 1
    assert py_result.step == 1000
    assert py_result.has_outsample is False
    np.testing.assert_allclose(py_result.param, cli_coeffs, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.fce_insample, cli_fce_in, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.forecast, cli_forecast, **CLI_TEXT_TOL)


def test_verbosity_does_not_change_output(tmp_path):
    order_path, _ = generate_order_file(tmp_path, dim=2, order=3)
    datafile = "tests/refs/ar-run_l1000.txt"

    out0 = run_cli(
        ["-m2", "-d1", "-n300", "-L50", "-V0", "-p", str(order_path), datafile]
    ).stdout
    out1 = run_cli(
        ["-m2", "-d1", "-n300", "-L50", "-V1", "-p", str(order_path), datafile]
    ).stdout
    assert out0 == out1


def test_dash_o_output_file_matches_python(tmp_path):
    order_path, order = generate_order_file(tmp_path, dim=2, order=3)
    datafile = "tests/refs/ar-run_l1000.txt"
    outfile = tmp_path / "out.pbf"

    run_cli(
        ["-m2", "-d1", "-n300", "-L50", "-p", str(order_path), "-o", str(outfile), datafile]
    )
    cli_fce_in, cli_fce_out, cli_order, cli_coeffs, cli_forecast = parse_cli_output(
        outfile.read_text(), dim=2
    )

    series = load_series(datafile)
    py_result = tisean.polynomp.fit(series, order, delay=1, insample=300, step=50)

    np.testing.assert_allclose(py_result.param, cli_coeffs, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.forecast, cli_forecast, **CLI_TEXT_TOL)


def test_zero_variance_rejected_like_cli(tmp_path):
    order_path, order = generate_order_file(tmp_path, dim=2, order=3)
    series_path = tmp_path / "const.txt"
    series = np.full(50, 3.0)
    np.savetxt(series_path, series)

    result = run_cli(
        ["-m2", "-d1", "-L5", "-p", str(order_path), str(series_path)], check=False
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    with pytest.raises(ValueError):
        tisean.polynomp.fit(series, order, step=5)


def test_singular_matrix_rejected_like_cli(tmp_path):
    # Two polynomial terms (exponents 150 and 200) that underflow to exactly
    # 0.0 for every point of a small-magnitude series make the
    # normal-equations matrix singular in a row Gaussian elimination with
    # partial pivoting actually checks (not the last, unchecked row - see
    # source_c/api/polynomp_api.c's solvele_core()).
    n = 200
    series = np.empty(n)
    series[0::2] = 0.001
    series[1::2] = -0.001
    series_path = tmp_path / "tiny_alt.txt"
    np.savetxt(series_path, series)

    order = np.array([[0], [150], [200]], dtype=np.uint32)
    order_path = tmp_path / "singular_order.pol"
    np.savetxt(order_path, order, fmt="%d")

    result = run_cli(
        ["-m1", "-d1", "-L5", "-p", str(order_path), str(series_path)], check=False
    )
    assert result.returncode == SOLVELE_SINGULAR_MATRIX

    with pytest.raises(ValueError):
        tisean.polynomp.fit(series, order, step=5)


def test_rejects_plength_zero():
    series = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    order = np.empty((0, 1), dtype=np.uint32)
    with pytest.raises(ValueError):
        tisean.polynomp.fit(series, order)


def test_rejects_dim_zero():
    series = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    order = np.empty((2, 0), dtype=np.uint32)
    with pytest.raises(ValueError):
        tisean.polynomp.fit(series, order)


def test_rejects_series_too_short_for_dim_delay():
    order = np.array([[0, 0], [1, 0]], dtype=np.uint32)
    series = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        tisean.polynomp.fit(series, order, delay=5)


def test_result_shapes_and_dtype(tmp_path):
    order_path, order = generate_order_file(tmp_path, dim=2, order=3)
    series = load_series("tests/refs/ar-run_l1000.txt")

    result = tisean.polynomp.fit(series, order, delay=1, insample=300, step=40)

    assert result.param.shape == (result.plength,)
    assert result.forecast.shape == (result.step,) == (40,)
    assert result.param.dtype == np.float64
    assert result.forecast.dtype == np.float64
