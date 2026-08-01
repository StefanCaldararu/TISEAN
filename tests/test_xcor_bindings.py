import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

VARIANCE_VAR_EQ_ZERO = 23

HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
XCOR_BIN = os.path.abspath("./bin/xcor")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [XCOR_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Returns (average1, stddev1, average2, stddev2, rows) where rows is a
    (2*tau+1, 2) array of [lag, value], matching what the CLI actually
    prints: four '# ...=' comment lines, then one 'lag value' row per
    lag."""
    lines = [line for line in text.splitlines() if line.strip()]
    average1 = float(lines[0].split("=")[1])
    stddev1 = float(lines[1].split("=")[1])
    average2 = float(lines[2].split("=")[1])
    stddev2 = float(lines[3].split("=")[1])
    rows = [[float(x) for x in line.split()] for line in lines[4:]]
    return average1, stddev1, average2, stddev2, np.array(rows)


def load_two_columns(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling for the two
    columns xcor reads: skip `exclude` lines, keep up to `length` of the
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
    # label, args, datafile, columns, length, exclude, tau
    ("defaults", [], HENON, (1, 2), None, 0, 100),
    ("tau_20", ["-D20"], HENON, (1, 2), None, 0, 20),
    ("tau_5", ["-D5"], HENON, (1, 2), None, 0, 5),
    ("columns_reordered", ["-c2,1"], HENON, (2, 1), None, 0, 100),
    ("columns_lorenz_subset", ["-c1,3"], LORENZ, (1, 3), None, 0, 100),
    ("length", ["-l300"], HENON, (1, 2), 300, 0, 100),
    ("exclude", ["-x50"], HENON, (1, 2), None, 50, 100),
    ("length_and_exclude", ["-l300", "-x50", "-D20"], HENON, (1, 2), 300, 50, 20),
    ("verbosity_0", ["-V0"], HENON, (1, 2), None, 0, 100),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,tau",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(label, args, datafile, columns, length, exclude, tau):
    out = run_cli(args + [datafile])
    cli_average1, cli_stddev1, cli_average2, cli_stddev2, cli_rows = parse_output(out)

    series1, series2 = load_two_columns(
        datafile, columns, length=length, exclude=exclude
    )
    result = tisean.xcor.compute(series1, series2, tau=tau)

    assert result.tau == tau
    assert result.length == len(series1)
    np.testing.assert_allclose(result.average1, cli_average1, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.stddev1, cli_stddev1, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.average2, cli_average2, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.stddev2, cli_stddev2, **CLI_TEXT_TOL)

    np.testing.assert_array_equal(cli_rows[:, 0], np.arange(-tau, tau + 1))
    np.testing.assert_allclose(result.values, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.ccr"
    run_cli(["-D10", "-o" + str(outfile), HENON])

    cli_average1, cli_stddev1, cli_average2, cli_stddev2, cli_rows = parse_output(
        outfile.read_text()
    )

    series1, series2 = load_two_columns(HENON, (1, 2))
    result = tisean.xcor.compute(series1, series2, tau=10)

    np.testing.assert_allclose(result.average1, cli_average1, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.stddev1, cli_stddev1, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.average2, cli_average2, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.stddev2, cli_stddev2, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.values, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_compute_clamps_tau_to_length_minus_one_like_cli():
    length = 50

    out = run_cli(["-l50", "-D1000", HENON])
    cli_average1, cli_stddev1, cli_average2, cli_stddev2, cli_rows = parse_output(out)
    assert cli_rows.shape[0] == 2 * (length - 1) + 1

    series1, series2 = load_two_columns(HENON, (1, 2), length=length)
    result = tisean.xcor.compute(series1, series2, tau=1000)

    assert result.tau == length - 1
    np.testing.assert_allclose(result.average1, cli_average1, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.stddev1, cli_stddev1, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.average2, cli_average2, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.stddev2, cli_stddev2, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.values, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_compute_default_tau_is_100():
    series1, series2 = load_two_columns(HENON, (1, 2))

    default = tisean.xcor.compute(series1, series2)
    explicit = tisean.xcor.compute(series1, series2, tau=100)

    assert default.tau == explicit.tau == 100
    np.testing.assert_array_equal(default.values, explicit.values)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    lines = [f"{1.5} {i * 0.1}" for i in range(20)]
    datafile.write_text("\n".join(lines) + "\n")

    result = subprocess.run(
        [XCOR_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series1 = np.full(20, 1.5)
    series2 = np.arange(20) * 0.1
    with pytest.raises(ValueError):
        tisean.xcor.compute(series1, series2)


def test_compute_rejects_mismatched_lengths():
    series1, series2 = load_two_columns(HENON, (1, 2))

    with pytest.raises(ValueError):
        tisean.xcor.compute(series1, series2[:-1])
