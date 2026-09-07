import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

SAV_GOL__UNDERDETERMINED = 68
SAV_GOL__TOO_LARGE_DERIVATIVE = 69

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
SAV_GOL_BIN = os.path.abspath("./bin/sav_gol")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [SAV_GOL_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Rows of dim %e-formatted values, one row per time step, shape
    (length, dim)."""
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip()
    ]
    return np.array(rows)


def load_columns(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Returns a (dim, length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    return raw.T.copy()


CASES = [
    # label, args, datafile, columns, length, exclude, nb, nf, power, deriv
    ("defaults", [], AR_RUN, [1], None, 0, 2, 2, 2, 0),
    ("length", ["-l300"], AR_RUN, [1], 300, 0, 2, 2, 2, 0),
    ("exclude", ["-x50"], AR_RUN, [1], None, 50, 2, 2, 2, 0),
    ("length_and_exclude", ["-l300", "-x50"], AR_RUN, [1], 300, 50, 2, 2, 2, 0),
    ("dim2_column_order", ["-m2", "-c2,1"], HENON, [2, 1], None, 0, 2, 2, 2, 0),
    ("dim3", ["-m3", "-c1,2,3"], LORENZ, [1, 2, 3], None, 0, 2, 2, 2, 0),
    ("n_wide", ["-n5,5"], AR_RUN, [1], None, 0, 5, 5, 2, 0),
    ("n_asymmetric", ["-n1,3"], AR_RUN, [1], None, 0, 1, 3, 2, 0),
    ("power_1", ["-p1"], AR_RUN, [1], None, 0, 2, 2, 1, 0),
    ("power_4_wider_window", ["-n5,5", "-p4"], AR_RUN, [1], None, 0, 5, 5, 4, 0),
    ("deriv_1", ["-n5,5", "-p3", "-D1"], AR_RUN, [1], None, 0, 5, 5, 3, 1),
    ("deriv_2", ["-n5,5", "-p4", "-D2"], AR_RUN, [1], None, 0, 5, 5, 4, 2),
    ("verbosity_0", ["-V0"], AR_RUN, [1], None, 0, 2, 2, 2, 0),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,nb,nf,power,deriv",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_filter_matches_cli(label, args, datafile, columns, length, exclude, nb, nf, power, deriv):
    out = run_cli(args + [datafile])
    cli_result = parse_output(out)

    series = load_columns(datafile, columns, length=length, exclude=exclude)
    py_result = tisean.sav_gol.filter(series, nb=nb, nf=nf, power=power, deriv=deriv)

    assert py_result.dim == len(columns)
    assert py_result.length == series.shape[1]
    np.testing.assert_allclose(py_result.data.T, cli_result, **CLI_TEXT_TOL)


def test_filter_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.sg"
    run_cli(["-n3,3", "-p2", "-o" + str(outfile), AR_RUN])

    cli_result = parse_output(outfile.read_text())

    series = load_columns(AR_RUN, [1])
    py_result = tisean.sav_gol.filter(series, nb=3, nf=3, power=2)

    np.testing.assert_allclose(py_result.data.T, cli_result, **CLI_TEXT_TOL)


def test_filter_default_keyword_args_match_cli_defaults():
    series = load_columns(AR_RUN, [1])

    default = tisean.sav_gol.filter(series)
    explicit = tisean.sav_gol.filter(series, nb=2, nf=2, power=2, deriv=0)

    np.testing.assert_array_equal(default.data, explicit.data)


def test_filter_edges_are_left_unfiltered_when_deriv_is_zero():
    series = load_columns(AR_RUN, [1])

    result = tisean.sav_gol.filter(series, nb=2, nf=2, power=2, deriv=0)

    np.testing.assert_array_equal(result.data[:, :2], series[:, :2])
    np.testing.assert_array_equal(result.data[:, -2:], series[:, -2:])


def test_filter_edges_are_zeroed_when_deriv_is_nonzero():
    series = load_columns(AR_RUN, [1])

    result = tisean.sav_gol.filter(series, nb=2, nf=2, power=2, deriv=1)

    np.testing.assert_array_equal(result.data[:, :2], np.zeros((1, 2)))
    np.testing.assert_array_equal(result.data[:, -2:], np.zeros((1, 2)))


def test_filter_rejects_underdetermined_system_like_cli():
    result = subprocess.run(
        [SAV_GOL_BIN, "-n1,1", "-p3", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == SAV_GOL__UNDERDETERMINED

    series = load_columns(AR_RUN, [1])
    with pytest.raises(ValueError):
        tisean.sav_gol.filter(series, nb=1, nf=1, power=3)


def test_filter_rejects_derivative_order_larger_than_power_like_cli():
    result = subprocess.run(
        [SAV_GOL_BIN, "-n5,5", "-p2", "-D3", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == SAV_GOL__TOO_LARGE_DERIVATIVE

    series = load_columns(AR_RUN, [1])
    with pytest.raises(ValueError):
        tisean.sav_gol.filter(series, nb=5, nf=5, power=2, deriv=3)
