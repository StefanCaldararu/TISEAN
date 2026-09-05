import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

EXTREMA_STRANGE_COMPONENT = 53

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
EXTREMA_BIN = os.path.abspath("./bin/extrema")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [EXTREMA_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text, dim):
    """Every non-empty line has dim interpolated values followed by dt -
    dim+1 numbers total, one row per extremum found."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows).reshape(-1, dim + 1)


def load_multi(path, columns, length=None, exclude=0):
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
    # label, args, datafile, columns, length, exclude, which, maxima, mintime
    ("defaults", [], AR_RUN, [1], None, 0, 0, True, 0.0),
    ("minima", ["-z"], AR_RUN, [1], None, 0, 0, False, 0.0),
    ("mintime", ["-t2.0"], AR_RUN, [1], None, 0, 0, True, 2.0),
    ("dim2_w2", ["-m2", "-w2"], HENON, [1, 2], None, 0, 1, True, 0.0),
    ("dim3", ["-m3"], LORENZ, [1, 2, 3], None, 0, 0, True, 0.0),
    (
        "dim3_reordered_columns",
        ["-m3", "-c3,1,2"],
        LORENZ,
        [3, 1, 2],
        None,
        0,
        0,
        True,
        0.0,
    ),
    ("length", ["-l300"], AR_RUN, [1], 300, 0, 0, True, 0.0),
    ("exclude", ["-x50"], AR_RUN, [1], None, 50, 0, True, 0.0),
    ("length_and_exclude", ["-l300", "-x50"], AR_RUN, [1], 300, 50, 0, True, 0.0),
    ("verbosity_0", ["-V0"], AR_RUN, [1], None, 0, 0, True, 0.0),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,which,maxima,mintime",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_find_matches_cli(
    label, args, datafile, columns, length, exclude, which, maxima, mintime
):
    dim = len(columns)
    out = run_cli(args + [datafile])
    cli_rows = parse_output(out, dim)

    series = load_multi(datafile, columns, length=length, exclude=exclude)
    result = tisean.extrema.find(series, which=which, maxima=maxima, mintime=mintime)

    assert result.dim == dim
    py_rows = np.hstack([result.point, result.dt.reshape(-1, 1)])

    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_find_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.ext"
    run_cli(["-o" + str(outfile), AR_RUN])

    cli_rows = parse_output(outfile.read_text(), 1)

    series = load_multi(AR_RUN, [1])
    result = tisean.extrema.find(series)

    py_rows = np.hstack([result.point, result.dt.reshape(-1, 1)])
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_find_default_kwargs_match_cli_defaults():
    series = load_multi(AR_RUN, [1])

    default = tisean.extrema.find(series)
    explicit = tisean.extrema.find(series, which=0, maxima=True, mintime=0.0)

    np.testing.assert_array_equal(default.point, explicit.point)
    np.testing.assert_array_equal(default.dt, explicit.dt)


def test_find_rejects_which_out_of_range_like_cli():
    result = subprocess.run(
        [EXTREMA_BIN, "-w2", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == EXTREMA_STRANGE_COMPONENT

    series = load_multi(AR_RUN, [1])
    with pytest.raises(ValueError):
        tisean.extrema.find(series, which=1)


def test_find_on_too_short_series_returns_empty_result_instead_of_crashing():
    series = np.zeros((1, 1))
    result = tisean.extrema.find(series)

    assert result.count == 0
    assert result.point.shape == (0, 1)
    assert result.dt.shape == (0,)
