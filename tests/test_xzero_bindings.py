import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11

HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
XZERO_BIN = os.path.abspath("./bin/xzero")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [XZERO_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Returns a (steps, 2) array of [step, error], matching what the CLI
    actually prints: one 'step error' row per forecast horizon, no header
    lines at all."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return np.array(rows)


def load_two_columns(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling for the two
    columns xzero reads: skip `exclude` lines, keep up to `length` of the
    rest, then pick the two 1-indexed `columns` in the given order.
    Returns (series1, series2), both raw/uncentered."""
    raw = np.loadtxt(path)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    col1, col2 = columns
    return raw[:, col1 - 1].copy(), raw[:, col2 - 1].copy()


CASES = [
    # label, args, datafile, columns, length, exclude,
    # dim, delay, n_ref, minn, eps0, epsf, step, epsset
    ("defaults", [], HENON, (1, 2), None, 0, 3, 1, None, 30, 1.0e-3, 1.2, 1, False),
    ("dim2", ["-m2"], HENON, (1, 2), None, 0, 2, 1, None, 30, 1.0e-3, 1.2, 1, False),
    ("delay2", ["-d2"], HENON, (1, 2), None, 0, 3, 2, None, 30, 1.0e-3, 1.2, 1, False),
    ("n_ref50", ["-n50"], HENON, (1, 2), None, 0, 3, 1, 50, 30, 1.0e-3, 1.2, 1, False),
    ("minn10", ["-k10"], HENON, (1, 2), None, 0, 3, 1, None, 10, 1.0e-3, 1.2, 1, False),
    ("epsf1.5", ["-f1.5"], HENON, (1, 2), None, 0, 3, 1, None, 30, 1.0e-3, 1.5, 1, False),
    ("step5", ["-s5"], HENON, (1, 2), None, 0, 3, 1, None, 30, 1.0e-3, 1.2, 5, False),
    ("eps0_raw_scale", ["-r0.05"], HENON, (1, 2), None, 0, 3, 1, None, 30, 0.05, 1.2, 1, True),
    ("columns_reordered", ["-c2,1"], HENON, (2, 1), None, 0, 3, 1, None, 30, 1.0e-3, 1.2, 1, False),
    ("columns_lorenz_subset", ["-c1,3"], LORENZ, (1, 3), None, 0, 3, 1, None, 30, 1.0e-3, 1.2, 1, False),
    ("length", ["-l300"], HENON, (1, 2), 300, 0, 3, 1, None, 30, 1.0e-3, 1.2, 1, False),
    ("exclude", ["-x50"], HENON, (1, 2), None, 50, 3, 1, None, 30, 1.0e-3, 1.2, 1, False),
    (
        "combo",
        ["-m3", "-d1", "-n50", "-k10", "-s5"],
        HENON,
        (1, 2),
        None,
        0,
        3,
        1,
        50,
        10,
        1.0e-3,
        1.2,
        5,
        False,
    ),
    ("verbosity_0", ["-V0"], HENON, (1, 2), None, 0, 3, 1, None, 30, 1.0e-3, 1.2, 1, False),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,dim,delay,n_ref,minn,eps0,epsf,step,epsset",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_forecast_matches_cli(
    label, args, datafile, columns, length, exclude, dim, delay, n_ref, minn, eps0, epsf, step, epsset
):
    out = run_cli(args + [datafile])
    cli_rows = parse_output(out)

    series1, series2 = load_two_columns(datafile, columns, length=length, exclude=exclude)
    result = tisean.xzero.forecast(
        series1,
        series2,
        dim=dim,
        delay=delay,
        n_ref=n_ref,
        minn=minn,
        eps0=eps0,
        epsf=epsf,
        step=step,
        epsset=epsset,
    )

    assert result.steps == step
    np.testing.assert_array_equal(cli_rows[:, 0], np.arange(1, step + 1))
    np.testing.assert_allclose(result.error, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_forecast_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.cze"
    run_cli(["-n50", "-k10", "-s5", "-o" + str(outfile), HENON])

    cli_rows = parse_output(outfile.read_text())

    series1, series2 = load_two_columns(HENON, (1, 2))
    result = tisean.xzero.forecast(series1, series2, n_ref=50, minn=10, step=5)

    np.testing.assert_allclose(result.error, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_forecast_default_keyword_arguments_match_cli_defaults():
    # No -m/-d/-n/-k/-r/-f/-s at all: xzero's own documented defaults are
    # dim=3, delay=1, n_ref=length, minn=30, eps0=1e-3 (unscaled, i.e.
    # epsset=False), epsf=1.2, step=1.
    series1, series2 = load_two_columns(HENON, (1, 2))

    default = tisean.xzero.forecast(series1, series2)
    explicit = tisean.xzero.forecast(
        series1,
        series2,
        dim=3,
        delay=1,
        n_ref=None,
        minn=30,
        eps0=1.0e-3,
        epsf=1.2,
        step=1,
        epsset=False,
    )

    assert default.steps == explicit.steps == 1
    np.testing.assert_array_equal(default.error, explicit.error)


def test_forecast_n_ref_none_matches_explicit_length():
    series1, series2 = load_two_columns(HENON, (1, 2))

    default = tisean.xzero.forecast(series1, series2, step=3)
    explicit = tisean.xzero.forecast(series1, series2, step=3, n_ref=len(series1))

    np.testing.assert_array_equal(default.error, explicit.error)


def test_forecast_clength_reflects_n_ref_and_step():
    series1, series2 = load_two_columns(HENON, (1, 2))

    result = tisean.xzero.forecast(series1, series2, n_ref=50, step=5)

    assert result.clength == 50 - 5


def test_forecast_rejects_constant_series1_like_cli(tmp_path):
    datafile = tmp_path / "constant1.txt"
    lines = [f"{1.5} {i * 0.1}" for i in range(20)]
    datafile.write_text("\n".join(lines) + "\n")

    result = subprocess.run(
        [XZERO_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL

    series1 = np.full(20, 1.5)
    series2 = np.arange(20) * 0.1
    with pytest.raises(ValueError):
        tisean.xzero.forecast(series1, series2)


def test_forecast_rejects_constant_series2_like_cli(tmp_path):
    datafile = tmp_path / "constant2.txt"
    lines = [f"{i * 0.1} {1.5}" for i in range(20)]
    datafile.write_text("\n".join(lines) + "\n")

    result = subprocess.run(
        [XZERO_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL

    series1 = np.arange(20) * 0.1
    series2 = np.full(20, 1.5)
    with pytest.raises(ValueError):
        tisean.xzero.forecast(series1, series2)


def test_forecast_rejects_mismatched_lengths():
    series1, series2 = load_two_columns(HENON, (1, 2))

    with pytest.raises(ValueError):
        tisean.xzero.forecast(series1, series2[:-1])


def test_forecast_rejects_non_1d_series():
    series1, series2 = load_two_columns(HENON, (1, 2))

    with pytest.raises(ValueError):
        tisean.xzero.forecast(series1.reshape(-1, 1), series2.reshape(-1, 1))


def test_forecast_rejects_dim_less_than_one():
    series1, series2 = load_two_columns(HENON, (1, 2))

    with pytest.raises(ValueError):
        tisean.xzero.forecast(series1, series2, dim=0)


def test_forecast_rejects_step_larger_than_n_ref():
    series1, series2 = load_two_columns(HENON, (1, 2))

    with pytest.raises(ValueError):
        tisean.xzero.forecast(series1, series2, n_ref=10, step=20)
