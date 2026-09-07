import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

VARIANCE_VAR_EQ_ZERO = 23
POINCARE_WRONG_COMPONENT = 63
POINCARE_OUTSIDE_REGION = 64

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
POINCARE_BIN = os.path.abspath("./bin/poincare")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [POINCARE_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip()
    ]
    if not rows:
        return np.empty((0, 0))
    return np.array(rows)


def load_column(path, column, length=None, exclude=0):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


def variance_mean(series):
    """Mirrors poincare_api.c's average: a plain sequential sum divided by
    the count, not numpy's (differently-rounding) pairwise .mean()."""
    total = 0.0
    for v in series:
        total += v
    return total / len(series)


def expected_rows(result):
    """Replicates the CLI's own row layout: dim-1 interpolated coordinates
    followed by the time since the previous crossing."""
    return np.column_stack([result.point, result.dt])


CASES = [
    # label, args, datafile, column, length, exclude, dim, comp, delay, from_above, where
    ("defaults", [], AR_RUN, 1, None, 0, 2, None, 1, False, None),
    ("dim3", ["-m3"], AR_RUN, 1, None, 0, 3, None, 1, False, None),
    ("dim3_comp1", ["-m3", "-q1"], AR_RUN, 1, None, 0, 3, 1, 1, False, None),
    ("delay2", ["-m2", "-d2"], AR_RUN, 1, None, 0, 2, None, 2, False, None),
    ("from_above", ["-C1"], AR_RUN, 1, None, 0, 2, None, 1, True, None),
    ("explicit_where", ["-a0.0"], AR_RUN, 1, None, 0, 2, None, 1, False, 0.0),
    ("column_2", ["-c2"], HENON, 2, None, 0, 2, None, 1, False, None),
    ("length", ["-l500"], AR_RUN, 1, 500, 0, 2, None, 1, False, None),
    ("exclude", ["-x50"], AR_RUN, 1, None, 50, 2, None, 1, False, None),
    ("verbosity_0", ["-V0"], AR_RUN, 1, None, 0, 2, None, 1, False, None),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,dim,comp,delay,from_above,where",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_find_matches_cli(
    label, args, datafile, column, length, exclude, dim, comp, delay, from_above, where
):
    out = run_cli(args + [datafile])
    cli_rows = parse_output(out)

    series = load_column(datafile, column, length=length, exclude=exclude)
    result = tisean.poincare.find(
        series, dim=dim, comp=comp, delay=delay, from_above=from_above, where=where
    )

    assert result.dim == dim
    py_rows = expected_rows(result)
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_find_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.poin"
    run_cli(["-m2", "-o" + str(outfile), AR_RUN])

    cli_rows = parse_output(outfile.read_text())

    series = load_column(AR_RUN, 1)
    result = tisean.poincare.find(series, dim=2)

    py_rows = expected_rows(result)
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_find_default_kwargs_match_documented_cli_defaults():
    series = load_column(AR_RUN, 1)

    default = tisean.poincare.find(series)
    explicit = tisean.poincare.find(
        series, dim=2, comp=2, delay=1, from_above=False, where=variance_mean(series)
    )

    assert default.count == explicit.count
    np.testing.assert_array_equal(default.point, explicit.point)
    np.testing.assert_array_equal(default.dt, explicit.dt)


def test_find_comp_defaults_to_dim():
    series = load_column(AR_RUN, 1)

    default = tisean.poincare.find(series, dim=3)
    explicit = tisean.poincare.find(series, dim=3, comp=3)

    assert default.count == explicit.count
    np.testing.assert_array_equal(default.point, explicit.point)
    np.testing.assert_array_equal(default.dt, explicit.dt)


def test_find_rejects_comp_greater_than_dim_like_cli():
    result = subprocess.run(
        [POINCARE_BIN, "-m2", "-q3", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == POINCARE_WRONG_COMPONENT

    series = load_column(AR_RUN, 1)
    with pytest.raises(ValueError):
        tisean.poincare.find(series, dim=2, comp=3)


def test_find_rejects_where_outside_data_range_like_cli():
    result = subprocess.run(
        [POINCARE_BIN, "-a1000000", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == POINCARE_OUTSIDE_REGION

    series = load_column(AR_RUN, 1)
    with pytest.raises(ValueError):
        tisean.poincare.find(series, where=1000000.0)


def test_find_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")

    result = subprocess.run(
        [POINCARE_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series = np.full(20, 1.5)
    with pytest.raises(ValueError):
        tisean.poincare.find(series)
