import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# figures), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
NRLAZY_BIN = os.path.abspath("./bin/nrlazy")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [NRLAZY_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_plain(text, comp):
    """Rows of `comp` %e-formatted values, one per line (default output,
    no -V's VER_USR2/4 bit)."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != comp:
            continue
        rows.append([float(x) for x in parts])
    return np.array(rows)


def parse_with_neighbors(text, comp):
    """Rows of `comp` %e-formatted values followed by an integer neighbor
    count (-V's VER_USR2/4 bit)."""
    values, neighbors = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != comp + 1:
            continue
        values.append([float(x) for x in parts[:comp]])
        neighbors.append(int(parts[comp]))
    return np.array(values), np.array(neighbors)


def load_columns(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Returns a (comp, length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    return raw.T.copy()


# label, cli_args, datafile, columns, embed, delay, iterations, r, v, length, exclude
CASES = [
    ("defaults", [], AR_RUN, [1], 5, 1, 1, None, None, None, 0),
    ("embed3", ["-m1,3"], AR_RUN, [1], 3, 1, 1, None, None, None, 0),
    ("embed7_delay2", ["-m1,7", "-d2"], AR_RUN, [1], 7, 2, 1, None, None, None, 0),
    ("iterations3", ["-m1,3", "-i3"], AR_RUN, [1], 3, 1, 3, None, None, None, 0),
    ("r_small", ["-m1,3", "-r0.05"], AR_RUN, [1], 3, 1, 1, 0.05, None, None, 0),
    ("r_large", ["-m1,3", "-r0.2"], AR_RUN, [1], 3, 1, 1, 0.2, None, None, 0),
    ("v_set", ["-m1,3", "-v0.5"], AR_RUN, [1], 3, 1, 1, None, 0.5, None, 0),
    (
        "v_overwrites_r",
        ["-m1,3", "-r0.05", "-v0.5"],
        AR_RUN,
        [1],
        3,
        1,
        1,
        0.05,
        0.5,
        None,
        0,
    ),
    (
        "length_and_exclude",
        ["-m1,3", "-l400", "-x100"],
        AR_RUN,
        [1],
        3,
        1,
        1,
        None,
        None,
        400,
        100,
    ),
    (
        "bivariate_henon",
        ["-m2,3", "-c1,2"],
        HENON,
        [1, 2],
        3,
        1,
        1,
        None,
        None,
        None,
        0,
    ),
    (
        "bivariate_reordered_columns",
        ["-m2,3", "-c2,1"],
        HENON,
        [2, 1],
        3,
        1,
        1,
        None,
        None,
        None,
        0,
    ),
    (
        "trivariate_lorenz",
        ["-m3,2", "-c1,2,3"],
        LORENZ,
        [1, 2, 3],
        2,
        1,
        1,
        None,
        None,
        None,
        0,
    ),
]


@pytest.mark.parametrize(
    "label,cli_args,datafile,columns,embed,delay,iterations,r,v,length,exclude",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_correct_matches_cli_across_options(
    label, cli_args, datafile, columns, embed, delay, iterations, r, v, length, exclude
):
    comp = len(columns)
    out = run_cli(cli_args + ["-V0", datafile])
    cli_values = parse_plain(out, comp)

    series = load_columns(datafile, columns, length=length, exclude=exclude)
    result = tisean.nrlazy.correct(
        series, embed=embed, delay=delay, iterations=iterations, r=r, v=v
    )

    assert result.comp == comp
    assert result.length == series.shape[1]
    np.testing.assert_allclose(result.series.T, cli_values, **CLI_TEXT_TOL)


NEIGHBOR_CASES = [
    ("univariate", ["-m1,3"], AR_RUN, [1], 3, 1, 1),
    ("bivariate_henon", ["-m2,3", "-c1,2"], HENON, [1, 2], 3, 1, 1),
]


@pytest.mark.parametrize(
    "label,cli_args,datafile,columns,embed,delay,iterations",
    NEIGHBOR_CASES,
    ids=[c[0] for c in NEIGHBOR_CASES],
)
def test_neighbors_matches_cli_dash_v4(
    label, cli_args, datafile, columns, embed, delay, iterations
):
    comp = len(columns)
    out = run_cli(cli_args + ["-V4", datafile])
    cli_values, cli_neighbors = parse_with_neighbors(out, comp)

    series = load_columns(datafile, columns)
    result = tisean.nrlazy.correct(series, embed=embed, delay=delay, iterations=iterations)

    np.testing.assert_allclose(result.series.T, cli_values, **CLI_TEXT_TOL)
    np.testing.assert_array_equal(result.neighbors, cli_neighbors)


def test_correct_default_kwargs_match_cli_defaults():
    # No -m/-d/-i/-r/-v at all: defaults are embed=5, delay=1, iterations=1,
    # eps=(rescaled interval)/1000, i.e. r=None, v=None.
    out = run_cli(["-V0", AR_RUN])
    cli_values = parse_plain(out, 1)

    series = load_columns(AR_RUN, [1])

    default = tisean.nrlazy.correct(series)
    explicit = tisean.nrlazy.correct(
        series, embed=5, delay=1, iterations=1, r=None, v=None
    )

    np.testing.assert_array_equal(default.series, explicit.series)
    np.testing.assert_allclose(default.series.T, cli_values, **CLI_TEXT_TOL)


def test_correct_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 50) + "\n")

    result = subprocess.run(
        [NRLAZY_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL

    series = np.full((1, 50), 1.5)
    with pytest.raises(ValueError):
        tisean.nrlazy.correct(series)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(embed=0), "embed"),
        (dict(delay=0), "delay"),
        (dict(iterations=0), "iterations"),
    ],
)
def test_correct_rejects_invalid_scalar_options(kwargs, match):
    series = load_columns(AR_RUN, [1])
    with pytest.raises(ValueError, match=match):
        tisean.nrlazy.correct(series, **kwargs)


def test_correct_rejects_series_too_short_for_embed_delay():
    series = np.arange(30.0).reshape(1, -1)
    with pytest.raises(ValueError, match="too short"):
        tisean.nrlazy.correct(series, embed=10, delay=5)


def test_correct_rejects_empty_component_axis():
    series = np.empty((0, 100))
    with pytest.raises(ValueError, match="component"):
        tisean.nrlazy.correct(series)
