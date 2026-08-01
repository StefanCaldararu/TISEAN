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

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
CORR_BIN = os.path.abspath("./bin/corr")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [CORR_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Returns (average, stddev, rows) where rows is a (tau+1, 2) array of
    [lag, value], matching what the CLI actually prints: a '# average=' and
    a '# standard deviation=' comment line, then one 'lag value' row per
    lag."""
    lines = [line for line in text.splitlines() if line.strip()]
    average = float(lines[0].split("=")[1])
    stddev = float(lines[1].split("=")[1])
    rows = [[float(x) for x in line.split()] for line in lines[2:]]
    return average, stddev, np.array(rows)


def load_column(path, column, length=None, exclude=0):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


CASES = [
    # label, args, datafile, column, length, exclude, tau, normalize
    ("defaults", [], AR_RUN, 1, None, 0, 100, True),
    ("tau_20", ["-D20"], AR_RUN, 1, None, 0, 20, True),
    ("tau_5", ["-D5"], AR_RUN, 1, None, 0, 5, True),
    ("no_normalize", ["-n"], AR_RUN, 1, None, 0, 100, False),
    ("column_2", ["-c2"], HENON, 2, None, 0, 100, True),
    ("length", ["-l300"], AR_RUN, 1, 300, 0, 100, True),
    ("exclude", ["-x50"], AR_RUN, 1, None, 50, 100, True),
    ("length_and_exclude", ["-l300", "-x50", "-D20"], AR_RUN, 1, 300, 50, 20, True),
    ("verbosity_0", ["-V0"], AR_RUN, 1, None, 0, 100, True),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,tau,normalize",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(label, args, datafile, column, length, exclude, tau, normalize):
    out = run_cli(args + [datafile])
    cli_average, cli_stddev, cli_rows = parse_output(out)

    series = load_column(datafile, column, length=length, exclude=exclude)
    result = tisean.corr.compute(series, tau=tau, normalize=normalize)

    assert result.tau == tau
    assert result.length == len(series)
    np.testing.assert_allclose(result.average, cli_average, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.stddev, cli_stddev, **CLI_TEXT_TOL)

    np.testing.assert_array_equal(cli_rows[:, 0], np.arange(tau + 1))
    np.testing.assert_allclose(result.values, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.cor"
    run_cli(["-D10", "-o" + str(outfile), AR_RUN])

    cli_average, cli_stddev, cli_rows = parse_output(outfile.read_text())

    series = load_column(AR_RUN, 1)
    result = tisean.corr.compute(series, tau=10)

    np.testing.assert_allclose(result.average, cli_average, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.stddev, cli_stddev, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.values, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_compute_clamps_tau_to_length_minus_one_like_cli():
    length = 50

    out = run_cli(["-l50", "-D1000", AR_RUN])
    cli_average, cli_stddev, cli_rows = parse_output(out)
    assert cli_rows.shape[0] == length

    series = load_column(AR_RUN, 1, length=length)
    result = tisean.corr.compute(series, tau=1000)

    assert result.tau == length - 1
    np.testing.assert_allclose(result.average, cli_average, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.stddev, cli_stddev, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.values, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_compute_default_tau_and_normalize():
    series = load_column(AR_RUN, 1)

    default = tisean.corr.compute(series)
    explicit = tisean.corr.compute(series, tau=100, normalize=True)

    assert default.tau == explicit.tau == 100
    np.testing.assert_array_equal(default.values, explicit.values)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")

    result = subprocess.run(
        [CORR_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series = np.full(20, 1.5)
    with pytest.raises(ValueError):
        tisean.corr.compute(series)
