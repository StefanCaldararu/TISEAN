import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
MUTUAL_BIN = os.path.abspath("./bin/mutual")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [MUTUAL_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Returns (shannon, rows) where rows is a (corrlength+1, 2) array of
    [tau, value], matching what the CLI actually prints: a '#shannon='
    comment line, then one 'tau value' row per lag starting at 0."""
    lines = [line for line in text.splitlines() if line.strip()]
    shannon = float(lines[0].split("=")[1])
    rows = [[float(x) for x in line.split()] for line in lines[1:]]
    return shannon, np.array(rows)


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
    # label, args, datafile, column, length, exclude, partitions, corrlength
    ("defaults", [], AR_RUN, 1, None, 0, 16, 20),
    ("partitions_8", ["-b8"], AR_RUN, 1, None, 0, 8, 20),
    ("partitions_32", ["-b32"], AR_RUN, 1, None, 0, 32, 20),
    ("corrlength_5", ["-D5"], AR_RUN, 1, None, 0, 16, 5),
    ("corrlength_50", ["-D50"], AR_RUN, 1, None, 0, 16, 50),
    ("column_2", ["-c2"], HENON, 2, None, 0, 16, 20),
    ("length", ["-l300"], AR_RUN, 1, 300, 0, 16, 20),
    ("exclude", ["-x50"], AR_RUN, 1, None, 50, 16, 20),
    (
        "length_and_exclude",
        ["-l300", "-x50", "-b8", "-D10"],
        AR_RUN,
        1,
        300,
        50,
        8,
        10,
    ),
    ("verbosity_0", ["-V0"], AR_RUN, 1, None, 0, 16, 20),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,partitions,corrlength",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(
    label, args, datafile, column, length, exclude, partitions, corrlength
):
    out = run_cli(args + [datafile])
    cli_shannon, cli_rows = parse_output(out)

    series = load_column(datafile, column, length=length, exclude=exclude)
    result = tisean.mutual.compute(series, partitions=partitions, corrlength=corrlength)

    assert result.partitions == partitions
    assert result.corrlength == corrlength
    assert result.length == len(series)

    np.testing.assert_allclose(result.values[0], cli_shannon, **CLI_TEXT_TOL)
    np.testing.assert_array_equal(cli_rows[:, 0], np.arange(corrlength + 1))
    np.testing.assert_allclose(result.values, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.mut"
    run_cli(["-b8", "-D10", "-o" + str(outfile), AR_RUN])

    cli_shannon, cli_rows = parse_output(outfile.read_text())

    series = load_column(AR_RUN, 1)
    result = tisean.mutual.compute(series, partitions=8, corrlength=10)

    np.testing.assert_allclose(result.values[0], cli_shannon, **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.values, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_compute_clamps_corrlength_to_length_minus_one_like_cli():
    length = 50

    out = run_cli(["-l50", "-D1000", AR_RUN])
    _, cli_rows = parse_output(out)
    assert cli_rows.shape[0] == length

    series = load_column(AR_RUN, 1, length=length)
    result = tisean.mutual.compute(series, corrlength=1000)

    assert result.corrlength == length - 1
    np.testing.assert_allclose(result.values, cli_rows[:, 1], **CLI_TEXT_TOL)


def test_compute_default_partitions_and_corrlength():
    series = load_column(AR_RUN, 1)

    default = tisean.mutual.compute(series)
    explicit = tisean.mutual.compute(series, partitions=16, corrlength=20)

    assert default.partitions == explicit.partitions == 16
    assert default.corrlength == explicit.corrlength == 20
    np.testing.assert_array_equal(default.values, explicit.values)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")

    result = subprocess.run(
        [MUTUAL_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL

    series = np.full(20, 1.5)
    with pytest.raises(ValueError):
        tisean.mutual.compute(series)


def test_compute_rejects_non_positive_partitions():
    series = load_column(AR_RUN, 1)
    with pytest.raises(ValueError):
        tisean.mutual.compute(series, partitions=0)


def test_compute_rejects_negative_corrlength():
    series = load_column(AR_RUN, 1)
    with pytest.raises(ValueError):
        tisean.mutual.compute(series, corrlength=-1)


def test_compute_rejects_non_1d_series():
    series = load_column(AR_RUN, 1)
    with pytest.raises(ValueError):
        tisean.mutual.compute(series.reshape(-1, 1))
